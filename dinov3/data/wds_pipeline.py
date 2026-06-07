# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
WebDataset data pipeline builder.

Supports streaming packed multi-channel TIFF/NPY shards (``packwds:`` and
``packwds_chvit:``) via webdataset>=1.0.  In webdataset 1.0.x, DataPipeline
does NOT support method-chaining (.map / .select); all stages must be passed
at construction time.
"""

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch

logger = logging.getLogger("dinov3")


@dataclass
class WdsConfig:
    """WebDataset pipeline configuration.

    Attributes:
        shard_urls: tar shard URL list, brace expression string, or explicit
            list of URLs.
        shuffle_buffer: shuffle buffer size.
        batch_size: optional batch size (usually set in DataLoader).
        num_workers: number of worker processes.
    """
    shard_urls: Union[str, List[str]]
    shuffle_buffer: int = 1000
    batch_size: Optional[int] = None
    num_workers: int = 4
    target_channels: Optional[int] = None


def _make_shard_source(wds, shard_urls):
    """Create an infinite shard source for training.

    We prefer ResampledShards so the stream never exhausts after one pass
    through the finite shard list.
    """
    if hasattr(wds, "ResampledShards"):
        return wds.ResampledShards(shard_urls)
    raise RuntimeError(
        "webdataset.ResampledShards is required for infinite streaming, "
        "but it was not found in the installed webdataset package."
    )


def build_packed_wds_pipeline(
    config: WdsConfig,
    transform: Optional[Callable] = None,
) -> torch.utils.data.IterableDataset:
    """Build a WebDataset pipeline for packed multi-channel shards (``packwds:``).

    Packed shards are produced by ``data/repackage``.  Each tar sample
    contains one ``ch<N>.tif`` file per available channel plus a
    ``meta.json``.  All channels are decoded and assembled into a single
    ``(target_channels, H, W)`` float32 tensor; missing channels are
    zero-padded so the tensor shape is always fixed.

    Args:
        config: ``WdsConfig``; ``target_channels`` is **required** and
            defines the output tensor channel count (e.g. 8).
        transform: DINOv3-style transform applied to each image tensor.

    Returns:
        An IterableDataset suitable for use with DataLoader.
    """
    try:
        import webdataset as wds
    except ImportError:
        logger.error("webdataset not installed — run: pip install webdataset")
        raise

    from .wds_decoder import decode_packed_sample

    target_ch = config.target_channels or 8

    def decode_sample(sample: dict) -> Optional[dict]:
        tensor = decode_packed_sample(sample, target_channels=target_ch)
        if tensor is None:
            return None
        return {
            "image": tensor,
            "__key__": sample.get("__key__", ""),
            "__url__": sample.get("__url__", ""),
        }

    stages = [
        _make_shard_source(wds, config.shard_urls),
        wds.tarfile_to_samples(),
        wds.shuffle(config.shuffle_buffer),
        wds.map(decode_sample),
        wds.select(lambda x: x is not None),
    ]

    if transform is not None:
        def apply_transform(sample: dict) -> tuple:
            transformed = transform(sample["image"])
            transformed["__key__"] = sample.get("__key__", "")
            transformed["__url__"] = sample.get("__url__", "")
            return transformed, ()

        stages.append(wds.map(apply_transform))

    pipeline = wds.DataPipeline(*stages)
    logger.info(
        "Packed WebDataset pipeline built (resampled infinite): target_channels=%d  urls=%s",
        target_ch,
        config.shard_urls,
    )
    return pipeline


def build_packed_robust_wds_pipeline(
    config: WdsConfig,
    transform: Optional[Callable] = None,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> torch.utils.data.IterableDataset:
    """``packwds_robust:`` variant of :func:`build_packed_wds_pipeline`.

    Uses :func:`decode_packed_sample_robust` (per-channel percentile
    normalization + single-channel replication).  The original
    :func:`build_packed_wds_pipeline` is unchanged.
    """
    try:
        import webdataset as wds
    except ImportError:
        logger.error("webdataset not installed — run: pip install webdataset")
        raise

    from .wds_decoder import decode_packed_sample_robust

    target_ch = config.target_channels or 8

    def decode_sample(sample: dict) -> Optional[dict]:
        tensor = decode_packed_sample_robust(
            sample, target_channels=target_ch, p_low=p_low, p_high=p_high
        )
        if tensor is None:
            return None
        return {
            "image": tensor,
            "__key__": sample.get("__key__", ""),
            "__url__": sample.get("__url__", ""),
        }

    stages = [
        _make_shard_source(wds, config.shard_urls),
        wds.tarfile_to_samples(),
        wds.shuffle(config.shuffle_buffer),
        wds.map(decode_sample),
        wds.select(lambda x: x is not None),
    ]

    if transform is not None:
        def apply_transform(sample: dict) -> tuple:
            transformed = transform(sample["image"])
            transformed["__key__"] = sample.get("__key__", "")
            transformed["__url__"] = sample.get("__url__", "")
            return transformed, ()

        stages.append(wds.map(apply_transform))

    pipeline = wds.DataPipeline(*stages)
    logger.info(
        "Packed ROBUST WebDataset pipeline built: target_channels=%d pct=[%s,%s] urls=%s",
        target_ch,
        p_low,
        p_high,
        config.shard_urls,
    )
    return pipeline


def build_packed_channelvit_wds_pipeline(
    config: WdsConfig,
    transform: Optional[Callable] = None,
    sample_channels: Optional[int] = None,
) -> torch.utils.data.IterableDataset:
    """Build packed WebDataset pipeline for true ChannelViT inputs.

    Each sample returns a fixed-size random subset of channels that are actually
    present in the tar entry plus their 0-based channel ids.  Missing channels
    are not padded, so absent channels do not become tokens.
    """
    try:
        import webdataset as wds
    except ImportError:
        logger.error("webdataset not installed — run: pip install webdataset")
        raise

    from .wds_decoder import decode_packed_channelvit_sample

    max_ch = config.target_channels or 8
    sample_ch = sample_channels

    def decode_sample(sample: dict) -> Optional[dict]:
        decoded = decode_packed_channelvit_sample(
            sample,
            max_channels=max_ch,
            sample_channels=sample_ch,
        )
        if decoded is None:
            return None
        decoded["__key__"] = sample.get("__key__", "")
        decoded["__url__"] = sample.get("__url__", "")
        return decoded

    stages = [
        _make_shard_source(wds, config.shard_urls),
        wds.tarfile_to_samples(),
        wds.shuffle(config.shuffle_buffer),
        wds.map(decode_sample),
        wds.select(lambda x: x is not None),
    ]

    if transform is not None:
        def apply_transform(sample: dict) -> tuple:
            transformed = transform(sample["image"], channel_ids=sample["channel_ids"])
            transformed["channel_ids"] = sample["channel_ids"]
            transformed["__key__"] = sample.get("__key__", "")
            transformed["__url__"] = sample.get("__url__", "")
            return transformed, ()

        stages.append(wds.map(apply_transform))

    pipeline = wds.DataPipeline(*stages)
    logger.info(
        "Packed ChannelViT WebDataset pipeline built: max_channels=%d sample_channels=%s urls=%s",
        max_ch,
        sample_ch if sample_ch is not None else "all-present",
        config.shard_urls,
    )
    return pipeline


def is_webdataset(dataset) -> bool:
    """Return True if dataset is a WebDataset IterableDataset."""
    return isinstance(dataset, torch.utils.data.IterableDataset)
