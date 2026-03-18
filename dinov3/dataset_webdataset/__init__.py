# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档模块。

"""
dataset_webdataset 包入口。

提供端到端的 100TB 医学影像数据归档流水线，
从 PostgreSQL 拉取元数据，经去重、张量重构、帧提取、
裁切过滤后打包为 WebDataset tar 分片。
"""

from .config import ImageMeta, PipelineConfig

# run_pipeline 延迟导入，避免包初始化时触发 dinov3.logging 遮蔽标准库 logging
__all__ = [
    "ImageMeta",
    "PipelineConfig",
    "run_pipeline",
]


def __getattr__(name: str):
    if name == "run_pipeline":
        from .pipeline import run_pipeline
        return run_pipeline
    raise AttributeError(f"module 'dataset_webdataset' has no attribute {name!r}")

