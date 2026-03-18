# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档 — TIFF 读取模块。

"""
TIFF 图像文件安全读取器（全量内存模式）。

内存充足（500GB+）场景下，所有文件统一使用 tifffile.imread
一次性全量加载到物理内存。NFS 对大块顺序读支持极好，
Linux 内核会触发底层网络预读（Read-Ahead），跑满网络带宽。

阈值 GIGANTIC_FILE_THRESH 已设为 100GB，实质上关闭了
memmap 和本地缓存分支，所有读取均走 imread。
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("dinov3.tiff_reader")


def read_tiff_safe(
    file_path: str,
    file_size_bytes: int = 0,
) -> Optional[np.ndarray]:
    """
    全量 imread 读取 TIFF 文件。

    Args:
        file_path: TIFF 文件物理路径。
        file_size_bytes: 文件字节数（保留参数兼容性，当前不使用）。

    Returns:
        读取到的 ndarray，失败返回 None（不中断流水线）。
    """
    try:
        import tifffile
    except ImportError:
        logger.error("tifffile 未安装: pip install tifffile")
        return None

    return _read_direct(file_path, tifffile)


def _read_direct(file_path: str, tifffile: object) -> Optional[np.ndarray]:
    """tifffile.imread 全量加载到物理内存。"""
    try:
        return tifffile.imread(file_path)
    except OSError as exc:
        logger.warning(f"TIFF 读取失败(OS) [{file_path}]: {exc}")
        return None
    except (ValueError, RuntimeError) as exc:
        logger.warning(f"TIFF 读取失败 [{file_path}]: {exc}")
        return None
