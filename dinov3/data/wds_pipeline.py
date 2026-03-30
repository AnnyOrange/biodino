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
from typing import Callable, Optional

import torch

logger = logging.getLogger("dinov3")


@dataclass
class WdsConfig:
    """WebDataset pipeline configuration.

    Attributes:
        shard_urls: tar shard URL list or brace expression.
        shuffle_buffer: shuffle buffer size.
        batch_size: optional batch size (usually set in DataLoader).
        num_workers: number of worker processes.
    """
    shard_urls: str
    shuffle_buffer: int = 1000
    batch_size: Optional[int] = None
    num_workers: int = 4
    target_channels: Optional[int] = None


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
        wds.SimpleShardList(config.shard_urls),
        wds.split_by_node,
        wds.split_by_worker,
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
    logger.info(f"WebDataset pipeline built: {config.shard_urls}")
    return pipeline


def is_webdataset(dataset) -> bool:
    """Return True if dataset is a WebDataset IterableDataset."""
    return isinstance(dataset, torch.utils.data.IterableDataset)
