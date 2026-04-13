# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
WebDataset data pipeline builder.

Supports streaming multi-channel TIFF and NPY images via webdataset>=1.0.
In webdataset 1.0.x, DataPipeline does NOT support method-chaining
(.map / .select); all stages must be passed at construction time.
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
            list of URLs (e.g. pipe: URLs for S3 streaming).
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


def _find_image_key(sample: dict) -> Optional[str]:
    """Return the first supported image key found in a WebDataset sample dict."""
    supported_keys = ("tiff", "tif", "npy")
    for extension in supported_keys:
        for key in sample:
            lowered = key.lower()
            if lowered == extension or lowered.endswith(f".{extension}"):
                return key
    return None


def _infer_image_format(image_key: str) -> Optional[str]:
    """Infer normalized image format from a WebDataset sample key."""
    lowered = image_key.lower()
    for extension in ("tiff", "tif", "npy"):
        if lowered == extension or lowered.endswith(f".{extension}"):
            return extension
    return None


def build_wds_pipeline(
    config: WdsConfig,
    transform: Optional[Callable] = None,
) -> torch.utils.data.IterableDataset:
    """Build a WebDataset pipeline for multi-format WebDataset shards.

    All stages are passed to DataPipeline() at construction — this is
    required by webdataset 1.0.x which does not support chained .map()
    calls on DataPipeline objects.

    Args:
        config: WdsConfig with shard URLs and shuffle settings.
        transform: DINOv3-style transform applied to each image tensor.

    Returns:
        An IterableDataset suitable for use with DataLoader.
    """
    try:
        import webdataset as wds
    except ImportError:
        logger.error("webdataset not installed — run: pip install webdataset")
        raise

    from .wds_decoder import decode_npy_bytes, decode_tiff_bytes

    def decode_sample(sample: dict) -> Optional[dict]:
        """Decode supported image bytes from a raw WebDataset sample."""
        image_key = _find_image_key(sample)
        if image_key is None:
            logger.warning(f"No supported image key found in sample keys: {list(sample.keys())}")
            return None
        image_format = _infer_image_format(image_key)
        if image_format == "npy":
            image_tensor = decode_npy_bytes(sample[image_key], target_channels=config.target_channels)
        else:
            image_tensor = decode_tiff_bytes(sample[image_key], target_channels=config.target_channels)
        if image_tensor is None:
            return None
        return {"image": image_tensor, "__key__": sample.get("__key__", "")}

    stages = [
        _make_shard_source(wds, config.shard_urls),
        wds.tarfile_to_samples(),
        wds.shuffle(config.shuffle_buffer),
        wds.map(decode_sample),
        wds.select(lambda x: x is not None),
    ]

    if transform is not None:
        def apply_transform(sample: dict) -> tuple:
            """Apply DINOv3 transform and return (image, target) pair.

            The TIFF/NPY decoder produces float32 tensors in [0, 1].
            DataAugmentationDINO is constructed with float_input=True when
            the dataset path is a WebDataset, so RandomSolarize threshold and
            other dtype-sensitive ops are configured for float32 inputs.
            No uint8 conversion is needed — full 16-bit precision is preserved.
            """
            transformed = transform(sample["image"])  # float32, (C, H, W), [0, 1]
            return transformed, ()  # no labels in self-supervised training

        stages.append(wds.map(apply_transform))

    pipeline = wds.DataPipeline(*stages)
    logger.info(f"WebDataset pipeline built (resampled infinite): {config.shard_urls}")
    return pipeline


class _MultiChannelWdsDataset(torch.utils.data.IterableDataset):
    """Zip N single-channel WebDataset streams into multi-channel samples.

    Each per-channel stream independently shuffles and yields 1-channel
    float32 tensors.  This dataset round-robin reads one sample from each
    stream and stacks them along the channel dimension, producing an
    (N, H, W) tensor.  A DINOv3 transform is then applied.

    Because per-channel shards are shuffled independently, paired channels
    in one sample may NOT originate from the same physical specimen.  For
    self-supervised pretraining this is acceptable.
    """

    def __init__(
        self,
        per_channel_pipelines: List[torch.utils.data.IterableDataset],
        channel_names: List[str],
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.pipelines = per_channel_pipelines
        self.channel_names = channel_names
        self.transform = transform

    def __iter__(self):
        iters = [iter(p) for p in self.pipelines]
        while True:
            channels = []
            try:
                for it in iters:
                    sample = next(it)
                    channels.append(sample["image"])  # (1, H, W)
            except StopIteration:
                return

            stacked = torch.cat(channels, dim=0)  # (N, H, W)

            if self.transform is not None:
                transformed = self.transform(stacked)
                yield transformed, ()
            else:
                yield stacked, ()


def build_multichannel_wds_pipeline(
    shard_patterns: List[str],
    channel_names: List[str],
    transform: Optional[Callable] = None,
    shuffle_buffer: int = 1000,
) -> torch.utils.data.IterableDataset:
    """Build a multi-channel WebDataset by zipping per-channel streams.

    Args:
        shard_patterns: One shard pattern (brace expression) per channel.
        channel_names: Human-readable channel names for logging.
        transform: DINOv3-style transform applied to the stacked N-channel tensor.
        shuffle_buffer: Per-stream shuffle buffer size.

    Returns:
        An IterableDataset that yields (transformed_image, ()) tuples.
    """
    try:
        import webdataset as wds
    except ImportError:
        logger.error("webdataset not installed — run: pip install webdataset")
        raise

    from .wds_decoder import decode_npy_bytes, decode_tiff_bytes

    per_channel_pipelines = []
    for i, pattern in enumerate(shard_patterns):
        def _make_decoder():
            def decode_sample(sample: dict) -> Optional[dict]:
                image_key = _find_image_key(sample)
                if image_key is None:
                    return None
                fmt = _infer_image_format(image_key)
                if fmt == "npy":
                    tensor = decode_npy_bytes(sample[image_key], target_channels=1)
                else:
                    tensor = decode_tiff_bytes(sample[image_key], target_channels=1)
                if tensor is None:
                    return None
                return {"image": tensor, "__key__": sample.get("__key__", "")}
            return decode_sample

        stages = [
            _make_shard_source(wds, pattern),
            wds.tarfile_to_samples(),
            wds.shuffle(shuffle_buffer),
            wds.map(_make_decoder()),
            wds.select(lambda x: x is not None),
        ]
        pipeline = wds.DataPipeline(*stages)
        per_channel_pipelines.append(pipeline)
        logger.info(f"  channel {channel_names[i]}: {pattern}")

    return _MultiChannelWdsDataset(per_channel_pipelines, channel_names, transform)


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

    Fully backward-compatible: old single-channel shards (``wds:``) are
    unaffected — they continue to use :func:`build_wds_pipeline`.

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
        return {"image": tensor, "__key__": sample.get("__key__", "")}

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
            return transformed, ()

        stages.append(wds.map(apply_transform))

    pipeline = wds.DataPipeline(*stages)
    logger.info(
        "Packed WebDataset pipeline built (resampled infinite): target_channels=%d  urls=%s",
        target_ch,
        config.shard_urls,
    )
    return pipeline


def is_webdataset(dataset) -> bool:
    """Return True if dataset is a WebDataset IterableDataset."""
    return isinstance(dataset, torch.utils.data.IterableDataset)
