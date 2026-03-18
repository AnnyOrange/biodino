# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档 — Ray Worker 模块。

"""
Ray Worker：处理一个 chunk 的完整样本流水线。

优化点：
1. dynamic + frame_idx 样本先按 file_path 分组；
2. 同一个 dynamic TIFF 只打开一次，连续读取多个目标 frame；
3. static / 非目标帧 dynamic 仍走原有单条处理逻辑。
"""

import gc
import io
import logging
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import ray

from .config import ImageMeta, RAY_LOCAL_TMP

logger = logging.getLogger("dinov3")


@ray.remote(num_gpus=0)
def process_chunk(
    meta_dicts: List[dict],
    chunk_id: int,
    worker_cfg: dict,
) -> Dict[str, object]:
    """
    处理单个数据块：全流程处理 → 直接写 tar 到 NFS。

    Args:
        meta_dicts: ImageMeta 序列化后的字典列表。
        chunk_id: 块唯一标识（决定 tar 文件名）。
        worker_cfg: {max_target_size, nfs_output_dir, shard_prefix}。

    Returns:
        统计信息字典 {chunk_id, samples, tar_name}。
    """
    metas = [ImageMeta(**d) for d in meta_dicts]
    max_target_size: int = worker_cfg["max_target_size"]

    npy_entries: List[Tuple[str, bytes]] = []

    # 1) dynamic 单帧样本：按文件分组，同一文件只打开一次
    dynamic_groups, single_metas = _group_metas_by_file(metas)

    for file_path, group in dynamic_groups.items():
        npy_entries.extend(_process_dynamic_file_group(file_path, group, max_target_size))

    # 2) 其他样本：保持原有单条处理
    for meta in single_metas:
        npy_entries.extend(_process_single_meta(meta, max_target_size))

    # 整块处理完毕后统一 GC 一次
    gc.collect()

    if not npy_entries:
        return {"chunk_id": chunk_id, "samples": 0, "tar_name": ""}

    tar_name = f"{worker_cfg['shard_prefix']}-{chunk_id:06d}.tar"
    _write_tar_staged(npy_entries, tar_name, worker_cfg["nfs_output_dir"])
    return {
        "chunk_id": chunk_id,
        "samples": len(npy_entries),
        "tar_name": tar_name,
    }


def _group_metas_by_file(
    metas: List[ImageMeta],
) -> Tuple[DefaultDict[str, List[ImageMeta]], List[ImageMeta]]:
    """
    把 metas 分成两类：
    1. dynamic + frame_idx 非空：按 file_path 分组
    2. 其他样本：继续单条处理
    """
    dynamic_groups: DefaultDict[str, List[ImageMeta]] = defaultdict(list)
    single_metas: List[ImageMeta] = []

    for meta in metas:
        if meta.is_dynamic and meta.frame_idx is not None:
            dynamic_groups[meta.file_path].append(meta)
        else:
            single_metas.append(meta)

    return dynamic_groups, single_metas


def _process_dynamic_file_group(
    file_path: str,
    metas: List[ImageMeta],
    max_target_size: int,
) -> List[Tuple[str, bytes]]:
    """
    同一个 dynamic TIFF 只打开一次，依次提取多个目标 frame。
    """
    from .lazy_frame_reader import read_dynamic_target_frame_from_tif

    if not metas:
        return []

    try:
        import tifffile
    except ImportError:
        logger.error("tifffile 未安装: pip install tifffile")
        return []

    # 按 frame_idx 排序，尽量让访问更连续
    metas = sorted(metas, key=lambda m: (-1 if m.frame_idx is None else m.frame_idx))

    entries: List[Tuple[str, bytes]] = []

    try:
        with tifffile.TiffFile(file_path) as tif:
            for meta in metas:
                frame = read_dynamic_target_frame_from_tif(tif, meta)
                if frame is None:
                    continue
                entries.extend(_collect_valid_patches(frame, meta, max_target_size))
                del frame
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        logger.warning(f"动态 TIFF 分组处理失败 [{file_path}]: {exc}")
        return []

    return entries


def _process_single_meta(
    meta: ImageMeta,
    max_target_size: int,
) -> List[Tuple[str, bytes]]:
    """单条 ImageMeta 全流程：读取→重构→帧提取→裁切→过滤→序列化。"""
    from .frame_extractor import extract_all_frames
    from .lazy_frame_reader import read_dynamic_target_frame
    from .tiff_reader import read_tiff_safe

    if meta.is_dynamic and meta.frame_idx is not None:
        frame = read_dynamic_target_frame(meta)
        if frame is None:
            return []
        return _collect_valid_patches(frame, meta, max_target_size)

    raw = read_tiff_safe(meta.file_path, meta.file_size_bytes)
    if raw is None:
        return []

    tensor = _reconstruct(raw, meta)
    del raw
    if tensor is None:
        return []

    frames = extract_all_frames(tensor, meta)
    entries: List[Tuple[str, bytes]] = []
    for frame in frames:
        entries.extend(_collect_valid_patches(frame, meta, max_target_size))
    del tensor, frames
    return entries


def _reconstruct(
    raw: np.ndarray,
    meta: ImageMeta,
) -> Optional[np.ndarray]:
    """根据数据来源路由到对应的张量重构策略。"""
    if meta.is_dynamic:
        from .tensor_dynamic import reconstruct_dynamic
        return reconstruct_dynamic(raw, meta)
    from .tensor_static import reconstruct_static
    return reconstruct_static(raw, meta)


def _collect_valid_patches(
    frame: np.ndarray,
    meta: ImageMeta,
    max_target_size: int,
) -> List[Tuple[str, bytes]]:
    """裁切 (C,H,W) 帧 → 条件质量过滤 → NPY 序列化。"""
    from .npy_namer import build_npy_name
    from .quality_filter import passes_quality_check
    from .spatial_slicer import needs_slicing, slice_wsi_patches

    is_sliced = needs_slicing(frame)
    entries: List[Tuple[str, bytes]] = []

    for patch in slice_wsi_patches(frame, max_target_size):
        if is_sliced and not passes_quality_check(patch.array):
            continue
        name = build_npy_name(meta, patch if is_sliced else None)
        entries.append((name, _serialize_npy(patch.array)))

    return entries


def _serialize_npy(array: np.ndarray) -> bytes:
    """将 ndarray 序列化为 NPY 字节流（保留原始精度）。"""
    with io.BytesIO() as buf:
        np.save(buf, array)
        return buf.getvalue()


def _write_tar_staged(
    entries: List[Tuple[str, bytes]],
    tar_name: str,
    nfs_dir: str,
) -> None:
    """本地封包后一次性推送至 NFS，避免 NFS 微写造成锁争用。"""
    from .worker_local_stage import stage_tar_to_nfs

    stage_tar_to_nfs(
        entries=entries,
        tar_name=tar_name,
        nfs_dir=nfs_dir,
        local_stage_root=RAY_LOCAL_TMP,
    )