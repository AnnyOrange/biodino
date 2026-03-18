# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档 — 动态序列张量重构模块。

"""
动态多维解析与 Fiji Hyperstack 解码 (original_image_all_2p_parsed)。

处理 2D~5D+ 任意维度 TIFF，利用 DB 元数据 (h, w, count, channel_count)
作为真值确定性重构为 (C, F, H, W) 或 (C, H, W)（单帧退化）。
"""

import logging
from dataclasses import replace
from itertools import permutations
from typing import List, Optional, Tuple

import numpy as np

from .config import ImageMeta

logger = logging.getLogger("dinov3")


def reconstruct_dynamic(
    raw_array: np.ndarray,
    meta: ImageMeta,
) -> Optional[np.ndarray]:
    """主入口：利用 DB 真值从任意维度原始张量重构为 (C,F,H,W)/(C,H,W)。"""
    arr = np.squeeze(raw_array)
    expected = meta.channel_count * meta.frame_count * meta.height * meta.width
    denom = meta.channel_count * meta.height * meta.width

    if denom <= 0:
        logger.warning(
            f"无效元数据: C*H*W={denom}, id={meta.row_id}"
        )
        return None

    if arr.size % denom != 0:
        logger.warning(
            f"元素总量不可整除 C*H*W: size={arr.size}, "
            f"C*H*W={meta.channel_count}*{meta.height}*{meta.width}={denom}, "
            f"id={meta.row_id}"
        )
        return None

    inferred_frames = arr.size // denom
    if inferred_frames <= 0:
        logger.warning(f"推断帧数无效: F={inferred_frames}, id={meta.row_id}")
        return None

    # 放宽策略：当 DB 的 frame_count 与实际不一致时，优先使用数据可推断的帧数。
    # 这样仅由 frame_idx 越界决定是否跳过，不因尾部多/少帧直接失败。
    runtime_meta = meta
    if inferred_frames != meta.frame_count:
        logger.warning(
            f"frame_count 与实际不一致: db={meta.frame_count}, actual={inferred_frames}, "
            f"size={arr.size}, expected={expected}, id={meta.row_id}"
        )
        runtime_meta = replace(meta, frame_count=inferred_frames)

    if runtime_meta.frame_count <= 1:
        return _handle_no_frame(arr, runtime_meta)
    return _dispatch_by_ndim(arr, runtime_meta)


def _handle_no_frame(
    arr: np.ndarray,
    meta: ImageMeta,
) -> Optional[np.ndarray]:
    """F≤1 退化：复用静态图谱轴对齐逻辑 → (C, H, W)。"""
    from .tensor_static import _align_axes_static
    return _align_axes_static(arr, meta.channel_count)


def _dispatch_by_ndim(
    arr: np.ndarray,
    meta: ImageMeta,
) -> Optional[np.ndarray]:
    """按 squeeze 后维度数分派重构策略 → (C, F, H, W)。"""
    if arr.ndim == 3:
        return _try_hyperstack_3d(arr, meta)
    if arr.ndim == 4:
        return _try_permutation_4d(arr, meta)
    if arr.ndim >= 5:
        return _try_collapse_nd(arr, meta)
    logger.warning(f"动态张量 ndim={arr.ndim} 无法处理, id={meta.row_id}")
    return None


def _try_hyperstack_3d(
    arr: np.ndarray,
    meta: ImageMeta,
) -> Optional[np.ndarray]:
    """3D Fiji Hyperstack: X=C*F 合并轴，搜索全部 6 种排列。"""
    C, F = meta.channel_count, meta.frame_count
    H, W = meta.height, meta.width
    target = (C * F, H, W)

    for perm in permutations(range(3)):
        if tuple(arr.shape[p] for p in perm) == target:
            return np.transpose(arr, perm).reshape(C, F, H, W)

    logger.warning(
        f"3D Hyperstack 还原失败: shape={arr.shape}, "
        f"target=(X={C * F}, H={H}, W={W}), id={meta.row_id}"
    )
    return None


def _try_permutation_4d(
    arr: np.ndarray,
    meta: ImageMeta,
) -> Optional[np.ndarray]:
    """4D 排列匹配：仅接受严格可判定的 (C, F, H, W) 置换。"""
    target = (meta.channel_count, meta.frame_count,
              meta.height, meta.width)

    for perm in permutations(range(4)):
        if tuple(arr.shape[p] for p in perm) == target:
            return np.transpose(arr, perm)

    logger.warning(
        f"4D 排列匹配失败: shape={arr.shape}, "
        f"target={target}, id={meta.row_id}"
    )
    return None


def _try_collapse_nd(
    arr: np.ndarray,
    meta: ImageMeta,
) -> Optional[np.ndarray]:
    """5D+ 轴识别：定位 H/W/C，剩余轴坍缩为 F（H/W 优先末尾两维）。"""
    C, F = meta.channel_count, meta.frame_count
    H, W = meta.height, meta.width
    hw = _find_spatial_axes(arr.shape, H, W)
    if hw is None:
        return _warn_nd("H/W", arr.shape, meta.row_id)
    h_ax, w_ax = hw
    remaining = [i for i in range(arr.ndim) if i not in (h_ax, w_ax)]
    c_ax = _find_axis_by_size(arr.shape, remaining, C)
    if c_ax is None:
        return _warn_nd("C", arr.shape, meta.row_id)
    f_axes = [i for i in remaining if i != c_ax]
    f_product = _axes_product(arr.shape, f_axes)
    if f_product != F:
        return _warn_nd(f"F({f_product}!={F})", arr.shape, meta.row_id)
    new_order = [c_ax] + f_axes + [h_ax, w_ax]
    return np.transpose(arr, new_order).reshape(C, F, H, W)


def _warn_nd(axis_name: str, shape: tuple, row_id: int) -> None:
    """5D+ 失败日志统一输出。"""
    logger.warning(f"5D+ 无法定位 {axis_name}: shape={shape}, id={row_id}")
    return None


def _find_spatial_axes(
    shape: tuple,
    target_h: int,
    target_w: int,
) -> Optional[Tuple[int, int]]:
    """在 shape 中定位 H/W 轴，优先末尾两维。"""
    n = len(shape)
    if shape[-2] == target_h and shape[-1] == target_w:
        return n - 2, n - 1
    if shape[-2] == target_w and shape[-1] == target_h:
        return n - 1, n - 2
    for i in range(n):
        for j in range(n):
            if i != j and shape[i] == target_h and shape[j] == target_w:
                return i, j
    return None


def _find_axis_by_size(
    shape: tuple,
    candidates: List[int],
    target: int,
) -> Optional[int]:
    """在候选轴中查找尺寸等于 target 的轴。"""
    for ax in candidates:
        if shape[ax] == target:
            return ax
    return None


def _axes_product(shape: tuple, axes: List[int]) -> int:
    """计算指定轴的尺寸乘积。"""
    result = 1
    for ax in axes:
        result *= shape[ax]
    return result
