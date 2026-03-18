"""In-memory tar assembly + atomic NFS push.

内存充足场景下，直接在内存中组装完整 tar 字节流，
然后一次性顺序写入 NFS（.tmp → rename），
不经过任何本地磁盘，避免撑爆小容量本地盘。
"""

import io
import logging
import os
import tarfile
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("dinov3")


def stage_tar_to_nfs(
    entries: List[Tuple[str, bytes]],
    tar_name: str,
    nfs_dir: str,
    local_stage_root: str = "",
) -> None:
    """在内存中组装 tar 后一次性写入 NFS。

    Args:
        entries: (文件名, npy字节) 列表。
        tar_name: 输出 tar 文件名。
        nfs_dir: NFS 输出目录。
        local_stage_root: 保留参数兼容性，当前不使用。
    """
    tar_bytes = _build_tar_in_memory(entries)
    _write_to_nfs_atomic(tar_bytes, tar_name, nfs_dir)
    del tar_bytes  # 立即释放内存中的 tar 副本


def _build_tar_in_memory(entries: List[Tuple[str, bytes]]) -> bytes:
    """在内存中组装完整 tar 字节流。"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in entries:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, fileobj=io.BytesIO(data))
    return buf.getvalue()


def _write_to_nfs_atomic(
    tar_bytes: bytes,
    tar_name: str,
    nfs_dir: str,
) -> None:
    """一次性顺序写入 NFS，.tmp → rename 保证原子性。"""
    nfs_root = Path(nfs_dir)
    nfs_root.mkdir(parents=True, exist_ok=True)
    tmp_path = nfs_root / f"{tar_name}.tmp"
    final_path = nfs_root / tar_name
    try:
        tmp_path.write_bytes(tar_bytes)
        os.rename(str(tmp_path), str(final_path))
    except OSError as exc:
        _remove_if_exists(str(tmp_path))
        logger.error(f"NFS 写入失败 [{final_path}]: {exc}")
        raise


def _remove_if_exists(path: str) -> None:
    """Best-effort remove file if exists."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
