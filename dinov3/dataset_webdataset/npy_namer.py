# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档 — 全息溯源命名模块。

"""
无 JSON 的全息溯源命名法 (JSON-Free Holographic Naming)。

文件名自身作为元信息容器，编码原图 ID、切块坐标和帧索引。
根据 meta.is_dynamic 属性自动路由到对应的命名策略。
"""

import logging
from typing import Optional

from .config import ImageMeta
from .spatial_slicer import PatchInfo

logger = logging.getLogger("dinov3")


def build_npy_name(
    meta: ImageMeta,
    patch: Optional[PatchInfo],
) -> str:
    """
    根据数据来源和裁切状态生成 NPY 文件名。

    由 meta.is_dynamic 自动路由，无需外部传参。

    Args:
        meta: 图像元数据（含 source_table, row_id, frame_idx）。
        patch: 裁切块信息，None 表示全图未裁切。

    Returns:
        完整的 NPY 文件名（含 .npy 后缀）。
    """
    if meta.is_dynamic:
        return _name_dynamic(meta, patch)
    return _name_static(meta, patch)


def _name_static(
    meta: ImageMeta,
    patch: Optional[PatchInfo],
) -> str:
    """
    静态图谱数据池命名 (original_images_all)。

    普通: [ID].npy
    裁切: [ID]_X[x]_Y[y]_H[h]_W[w].npy

    Args:
        meta: 图像元数据。
        patch: 裁切块信息。

    Returns:
        NPY 文件名。
    """
    base = str(meta.row_id)
    if patch is None:
        return f"{base}.npy"
    return _append_patch_suffix(base, patch) + ".npy"


def _name_dynamic(
    meta: ImageMeta,
    patch: Optional[PatchInfo],
) -> str:
    """
    动态序列数据池命名 (original_image_all_2p_parsed)。

    强制携带 2p_ 前缀。
    有帧: 2p_[ID]_frame[idx].npy 或 2p_[ID]_X.._frame[idx].npy
    无帧: 2p_[ID].npy 或 2p_[ID]_X.._.npy

    Args:
        meta: 图像元数据。
        patch: 裁切块信息。

    Returns:
        NPY 文件名。
    """
    base = f"2p_{meta.row_id}"

    if patch is not None:
        base = _append_patch_suffix(base, patch)

    if meta.frame_idx is not None:
        base = f"{base}_frame{meta.frame_idx}"

    return f"{base}.npy"


def _append_patch_suffix(base: str, patch: PatchInfo) -> str:
    """
    为基础名追加裁切块坐标后缀。

    Args:
        base: 基础文件名（不含后缀）。
        patch: 裁切块空间信息。

    Returns:
        追加坐标后的文件名。
    """
    return (
        f"{base}"
        f"_X{patch.x_start}"
        f"_Y{patch.y_start}"
        f"_H{patch.patch_h}"
        f"_W{patch.patch_w}"
    )
