# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档 — Tar 分片写入模块。

"""
动态闭环 Tar 分片归档 (Dynamic Shard Writing)。

将合格的 NPY 张量流式写入 WebDataset tar 分片，
达到 ~3GB 时自动关闭当前分片并创建下一个。
支持主存储路径填满后自动回退到备用路径。
"""

import io
import logging
import os
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np

from .config import (
    FALLBACK_OUTPUT_DIR,
    MAX_SHARD_BYTES,
    PRIMARY_OUTPUT_DIR,
)

logger = logging.getLogger("dinov3")


class ShardWriter:
    """
    动态闭环 Tar 分片写入器。

    Attributes:
        output_dir: 当前输出目录。
        prefix: 分片文件名前缀。
        max_bytes: 单个分片的最大字节数。
    """

    def __init__(
        self,
        prefix: str,
        max_bytes: int = MAX_SHARD_BYTES,
    ) -> None:
        """
        初始化分片写入器。

        Args:
            prefix: 分片文件名前缀（如 "ori_4ch"）。
            max_bytes: 单个 tar 分片的最大字节数。
        """
        self._prefix = prefix
        self._max_bytes = max_bytes
        self._shard_idx: int = 0
        self._current_tar: Optional[tarfile.TarFile] = None
        self._current_size: int = 0
        self._total_samples: int = 0
        self._output_dir: Path = self._resolve_output_dir()

        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ShardWriter 初始化: dir={self._output_dir}")

    def write_npy(self, npy_name: str, array: np.ndarray) -> None:
        """
        将 ndarray 序列化为 NPY 并写入当前 tar 分片。

        Args:
            npy_name: NPY 文件名（含 .npy 后缀）。
            array: 待序列化的 ndarray（保留原始精度）。
        """
        npy_bytes = _serialize_npy(array)
        byte_size = len(npy_bytes)

        # 检查是否需要轮转到下一个分片
        if self._should_rotate(byte_size):
            self._rotate_shard()

        # 确保当前分片已打开
        if self._current_tar is None:
            self._open_new_shard()

        _write_to_tar(self._current_tar, npy_name, npy_bytes)
        self._current_size += byte_size
        self._total_samples += 1

    def close(self) -> None:
        """关闭当前打开的 tar 分片。"""
        if self._current_tar is not None:
            self._current_tar.close()
            logger.info(
                f"分片关闭: idx={self._shard_idx}, "
                f"size={self._current_size / 1e9:.2f}GB"
            )
            self._current_tar = None

        logger.info(f"归档完成: 总样本数={self._total_samples:,d}")

    def _should_rotate(self, pending_bytes: int) -> bool:
        """判断是否需要轮转分片。"""
        if self._current_tar is None:
            return False
        return (self._current_size + pending_bytes) > self._max_bytes

    def _rotate_shard(self) -> None:
        """关闭当前分片，递增索引。"""
        self.close()
        self._shard_idx += 1
        self._current_size = 0

    def _open_new_shard(self) -> None:
        """创建并打开新的 tar 分片文件。"""
        shard_path = self._build_shard_path()
        self._current_tar = tarfile.open(shard_path, "w")
        self._current_size = 0
        logger.info(f"新分片: {shard_path}")

    def _build_shard_path(self) -> str:
        """构建当前分片的完整文件路径。"""
        filename = f"{self._prefix}-{self._shard_idx:06d}.tar"
        return str(self._output_dir / filename)

    def _resolve_output_dir(self) -> Path:
        """解析输出目录，主路径满时回退到备用路径。"""
        primary = Path(PRIMARY_OUTPUT_DIR)
        if _has_disk_space(primary):
            return primary / "wds_shards"
        logger.warning(f"主路径空间不足: {primary}, 回退到备用路径")
        return Path(FALLBACK_OUTPUT_DIR) / "wds_shards"

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _serialize_npy(array: np.ndarray) -> bytes:
    """
    将 ndarray 序列化为 NPY 字节流（保留原始精度）。

    Args:
        array: 待序列化的 ndarray。

    Returns:
        NPY 格式的字节数据。
    """
    with io.BytesIO() as buffer:
        np.save(buffer, array)
        return buffer.getvalue()


def _write_to_tar(
    tar: tarfile.TarFile,
    name: str,
    data: bytes,
) -> None:
    """
    将字节数据写入 tar 归档。

    Args:
        tar: 打开的 TarFile 对象。
        name: 归档内的文件名。
        data: 文件字节内容。
    """
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    with io.BytesIO(data) as buffer:
        tar.addfile(info, buffer)


def _has_disk_space(path: Path, min_gb: float = 50.0) -> bool:
    """
    检查路径是否有足够磁盘空间。

    Args:
        path: 待检查的目录路径。
        min_gb: 最低可用空间 (GB)。

    Returns:
        True 表示空间充足。
    """
    try:
        stat = os.statvfs(str(path))
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        return free_gb > min_gb
    except OSError:
        return False

