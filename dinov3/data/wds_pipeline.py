# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
WebDataset data pipeline builder.

Supports streaming packed multi-channel TIFF/NPY shards (``packwds:`` and
``packwds_chvit:``) via webdataset>=1.0.  Robust variants add #4-style
per-channel percentile normalization.  In webdataset 1.0.x, DataPipeline does
NOT support method-chaining (.map / .select); all stages must be passed at
construction time.
"""

import logging
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Union

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


class WeightedIterableDataset(torch.utils.data.IterableDataset):
    """Randomly interleave multiple infinite iterable datasets by weight."""

    def __init__(
        self,
        datasets: Iterable[torch.utils.data.IterableDataset],
        weights: Iterable[float],
        *,
        seed: int = 0,
        names: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self.datasets = list(datasets)
        self.weights = [float(w) for w in weights]
        self.names = list(names) if names is not None else [str(i) for i in range(len(self.datasets))]
        self.seed = int(seed)
        if not self.datasets:
            raise ValueError("WeightedIterableDataset requires at least one dataset")
        if len(self.datasets) != len(self.weights):
            raise ValueError("datasets and weights must have the same length")
        if len(self.names) != len(self.datasets):
            raise ValueError("names and datasets must have the same length")
        if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
            raise ValueError(f"weights must be non-negative and sum to > 0, got {self.weights}")

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        try:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except Exception:
            rank = 0
        rng = random.Random(self.seed + 1009 * worker_id + 9176 * rank)
        iterators = [iter(dataset) for dataset in self.datasets]
        choices = list(range(len(iterators)))
        while True:
            idx = rng.choices(choices, weights=self.weights, k=1)[0]
            try:
                yield next(iterators[idx])
            except StopIteration:
                iterators[idx] = iter(self.datasets[idx])
                yield next(iterators[idx])


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
    min_channels: int = 1,
) -> torch.utils.data.IterableDataset:
    """Build packed WebDataset pipeline for true ChannelViT inputs.

    Each sample returns real channels from the tar entry plus their 0-based
    channel ids. ``sample_channels`` optionally caps the selected count and
    ``min_channels`` can filter out low-channel samples. Missing channels are
    not padded, so absent channels do not become tokens.
    """
    try:
        import webdataset as wds
    except ImportError:
        logger.error("webdataset not installed — run: pip install webdataset")
        raise

    from .wds_decoder import decode_packed_channelvit_sample

    max_ch = config.target_channels or 8
    sample_ch = sample_channels
    min_ch = min_channels

    def decode_sample(sample: dict) -> Optional[dict]:
        decoded = decode_packed_channelvit_sample(
            sample,
            max_channels=max_ch,
            sample_channels=sample_ch,
            min_channels=min_ch,
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
        "Packed ChannelViT WebDataset pipeline built: max_channels=%d min_channels=%d sample_channels=%s urls=%s",
        max_ch,
        min_ch,
        sample_ch if sample_ch is not None else "all-present",
        config.shard_urls,
    )
    return pipeline


def build_packed_channelvit_robust_wds_pipeline(
    config: WdsConfig,
    transform: Optional[Callable] = None,
    sample_channels: Optional[int] = None,
    min_channels: int = 1,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> torch.utils.data.IterableDataset:
    """Build true multi-channel samples with robust per-channel normalization.

    This is the combination needed by the dual-route + #4 recipe: keep only
    real channels and channel ids (``packwds_chvit`` behavior), optionally
    filter by ``min_channels``, and normalize each selected channel using #4's
    percentile window before augmentation.
    """
    try:
        import webdataset as wds
    except ImportError:
        logger.error("webdataset not installed — run: pip install webdataset")
        raise

    from .wds_decoder import decode_packed_channelvit_sample_robust

    max_ch = config.target_channels or 8
    sample_ch = sample_channels
    min_ch = min_channels

    def decode_sample(sample: dict) -> Optional[dict]:
        decoded = decode_packed_channelvit_sample_robust(
            sample,
            max_channels=max_ch,
            sample_channels=sample_ch,
            min_channels=min_ch,
            p_low=p_low,
            p_high=p_high,
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
        "Packed ChannelViT ROBUST WebDataset pipeline built: max_channels=%d min_channels=%d sample_channels=%s pct=[%s,%s] urls=%s",
        max_ch,
        min_ch,
        sample_ch if sample_ch is not None else "all-present",
        p_low,
        p_high,
        config.shard_urls,
    )
    return pipeline


def is_webdataset(dataset) -> bool:
    """Return True if dataset is a WebDataset IterableDataset."""
    return isinstance(dataset, torch.utils.data.IterableDataset)
