# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
WebDataset 数据管道构建模块。

提供 WebDataset 管道构建功能，支持多通道 TIFF 图像的流式读取。
"""

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

logger = logging.getLogger("dinov3")


@dataclass
class WdsConfig:
    """WebDataset 管道配置。

    Attributes:
        shard_urls: tar 分片 URL 列表或 brace 表达式。
        shuffle_buffer: shuffle 缓冲区大小。
        batch_size: 批次大小（可选，在 DataLoader 中设置）。
        num_workers: 工作进程数。
    """
    shard_urls: str
    shuffle_buffer: int = 1000
    batch_size: Optional[int] = None
    num_workers: int = 4


def build_wds_pipeline(
    config: WdsConfig,
    transform: Optional[Callable] = None,
) -> torch.utils.data.IterableDataset:
    """
    构建 WebDataset 数据管道。

    Args:
        config: WebDataset 配置对象。
        transform: 应用于图像的变换函数（如 DINOv3 Transforms）。

    Returns:
        可迭代的 WebDataset Pipeline。

    Example:
        >>> config = WdsConfig(shard_urls="/data/shards-{0000..0099}.tar")
        >>> pipeline = build_wds_pipeline(config, transform=my_transform)
        >>> loader = DataLoader(pipeline, batch_size=32)
    """
    try:
        import webdataset as wds
    except ImportError:
        logger.error("webdataset 未安装，请运行: pip install webdataset")
        raise

    from .wds_decoder import create_tiff_decoder

    pipeline = _create_base_pipeline(config.shard_urls, config.shuffle_buffer)
    pipeline = _add_decoder_stage(pipeline)
    pipeline = _add_transform_stage(pipeline, transform)

    logger.info(f"WebDataset 管道已构建: {config.shard_urls}")
    return pipeline


def _create_base_pipeline(
    shard_urls: str,
    shuffle_buffer: int,
) -> "wds.DataPipeline":
    """
    创建基础 WebDataset 管道（读取、分发、shuffle）。

    Args:
        shard_urls: tar 分片 URL 模式。
        shuffle_buffer: shuffle 缓冲区大小。

    Returns:
        配置了基础阶段的 WebDataset Pipeline。
    """
    import webdataset as wds

    pipeline = wds.DataPipeline(
        wds.SimpleShardList(shard_urls),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(shuffle_buffer),
    )
    return pipeline


def _add_decoder_stage(pipeline: "wds.DataPipeline") -> "wds.DataPipeline":
    """
    为管道添加 TIFF 解码阶段。

    Args:
        pipeline: 基础 WebDataset 管道。

    Returns:
        添加了解码器的管道。
    """
    import webdataset as wds
    from .wds_decoder import decode_tiff_bytes

    def decode_sample(sample: dict) -> Optional[dict]:
        """解码单个样本中的 TIFF 图像。"""
        # 查找 tiff/tif 键
        image_key = _find_image_key(sample)
        if image_key is None:
            logger.warning(f"样本中未找到 TIFF 图像: {list(sample.keys())}")
            return None

        image_tensor = decode_tiff_bytes(sample[image_key])
        if image_tensor is None:
            return None  # 解码失败，跳过此样本

        return {"image": image_tensor, "__key__": sample.get("__key__", "")}

    return pipeline.map(decode_sample).select(lambda x: x is not None)


def _find_image_key(sample: dict) -> Optional[str]:
    """
    在样本字典中查找 TIFF 图像键。

    Args:
        sample: WebDataset 样本字典。

    Returns:
        找到的图像键，或 None。
    """
    tiff_extensions = ("tiff", "tif", "TIFF", "TIF")
    for key in sample:
        if any(key.endswith(ext) for ext in tiff_extensions):
            return key
    return None


def _add_transform_stage(
    pipeline: "wds.DataPipeline",
    transform: Optional[Callable],
) -> "wds.DataPipeline":
    """
    为管道添加图像变换阶段。

    Args:
        pipeline: 已添加解码器的管道。
        transform: DINOv3 Transform 函数。

    Returns:
        添加了变换的管道。
    """
    if transform is None:
        return pipeline

    def apply_transform(sample: dict) -> tuple:
        """应用变换并返回 (image, target) 元组。"""
        image = sample["image"]  # Shape: (C, H, W)
        transformed = transform(image)
        # DINOv3 transform 返回变换后的图像
        return transformed, ()  # 自监督无标签，target 为空元组

    return pipeline.map(apply_transform)


def is_webdataset(dataset) -> bool:
    """
    检查数据集是否为 WebDataset IterableDataset。

    Args:
        dataset: 待检查的数据集对象。

    Returns:
        如果是 IterableDataset 则返回 True。
    """
    return isinstance(dataset, torch.utils.data.IterableDataset)

