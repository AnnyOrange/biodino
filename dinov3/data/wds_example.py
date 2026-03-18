# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
WebDataset 使用示例模块。

展示如何将 WebDataset 与 DINOv3 集成使用。
"""

import logging
from typing import Optional

import torch

from .wds_pipeline import WdsConfig, build_wds_pipeline
from .loaders import make_data_loader

logger = logging.getLogger("dinov3")


def create_wds_dataloader_example(
    shard_pattern: str,
    transform: Optional[callable] = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """
    创建 WebDataset DataLoader 的示例函数。

    Args:
        shard_pattern: tar 分片路径模式，如 "/data/shards-{0000..0099}.tar"。
        transform: DINOv3 图像变换函数。
        batch_size: 批次大小。
        num_workers: 工作进程数。

    Returns:
        配置好的 DataLoader。

    Example:
        >>> from dinov3.data.augmentations import DataAugmentationDINO
        >>> transform = DataAugmentationDINO(...)
        >>> loader = create_wds_dataloader_example(
        ...     shard_pattern="/path/to/shards-{0000..0099}.tar",
        ...     transform=transform,
        ...     batch_size=32,
        ... )
        >>> for batch in loader:
        ...     images, targets = batch
        ...     # 训练逻辑
    """
    # 1. 配置 WebDataset
    config = WdsConfig(
        shard_urls=shard_pattern,
        shuffle_buffer=1000,
        num_workers=num_workers,
    )

    # 2. 构建管道
    pipeline = build_wds_pipeline(config, transform=transform)

    # 3. 创建 DataLoader（自动检测 WebDataset 并绕过 Sampler）
    dataloader = make_data_loader(
        dataset=pipeline,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,  # 会被 WebDataset 逻辑忽略
        drop_last=True,
    )

    return dataloader


# ============================================================================
# 使用说明
# ============================================================================
#
# 1. 数据准备：将 TIFF 图像打包为 WebDataset tar 格式
#    ```bash
#    # 使用 webdataset 工具打包
#    tar -cf shard-0000.tar image_0001.tiff image_0002.tiff ...
#    ```
#
# 2. 在训练脚本中使用：
#    ```python
#    from dinov3.data import WdsConfig, build_wds_pipeline, make_data_loader
#    from dinov3.data.augmentations import DataAugmentationDINO
#
#    # 创建 Transform
#    transform = DataAugmentationDINO(
#        global_crops_scale=(0.32, 1.0),
#        local_crops_scale=(0.05, 0.32),
#        local_crops_number=8,
#        global_crops_size=256,
#        local_crops_size=112,
#    )
#
#    # 构建 WebDataset 管道
#    config = WdsConfig(
#        shard_urls="/path/to/shards-{0000..0099}.tar",
#        shuffle_buffer=2000,
#    )
#    pipeline = build_wds_pipeline(config, transform=transform)
#
#    # 创建 DataLoader
#    loader = make_data_loader(
#        dataset=pipeline,
#        batch_size=32,
#        num_workers=8,
#    )
#
#    # 训练循环
#    for batch in loader:
#        images, targets = batch
#        # ... 训练逻辑
#    ```
#
# 3. 配置文件中使用（修改 train.dataset_path）：
#    将 dataset_path 设置为 "wds:/path/to/shards-{0000..0099}.tar"
#    前缀 "wds:" 表示使用 WebDataset 模式
# ============================================================================

