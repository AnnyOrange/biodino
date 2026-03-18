# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档 — 静态图谱张量重构模块。

"""
静态图谱的维度安全清洗 (original_images_all)。

基于数据库元数据 (h, w, channel_count) 对读入的 TIFF 张量
执行确定性的维度检测、冗余维度剔除和轴向硬对齐。
"""

import logging
from typing import Optional, Tuple

import numpy as np

from .config import ImageMeta

logger = logging.getLogger("dinov3")


def reconstruct_static(
    raw_array: np.ndarray,
    meta: ImageMeta,
) -> Optional[np.ndarray]:
    """
    静态图谱张量的确定性重构。

    Args:
        raw_array: tifffile.imread 读入的原始 ndarray。
        meta: 数据库元数据。

    Returns:
        形状为 (C, H, W) 的 ndarray，失败返回 None。
    """
    squeezed = _squeeze_redundant(raw_array, meta.channel_count)
    if squeezed is None:
        return None
    return _align_axes_static(squeezed, meta.channel_count)


def _squeeze_redundant(
    array: np.ndarray,
    channel_count: int,
) -> Optional[np.ndarray]:
    """
    剔除冗余维度，保留 2D 或 3D 有效形状。

    Args:
        array: 可能含冗余维度的原始数组。
        channel_count: 期望通道数。

    Returns:
        维度精简后的 ndarray (2D 或 3D)。
    """
    arr = np.squeeze(array)

    if arr.ndim == 2:
        return arr  # Shape: (H, W)
    if arr.ndim == 3:
        return arr  # Shape: 可能是 (C,H,W) 或 (H,W,C)

    # ndim > 3: 尝试识别并去除与 channel_count 等值的冗余前导维度
    if arr.ndim > 3:
        shape_list = list(arr.shape)
        # 取最大的两个维度作为 H, W
        sorted_dims = sorted(range(len(shape_list)),
                             key=lambda i: shape_list[i], reverse=True)
        hw_axes = tuple(sorted(sorted_dims[:2]))
        remaining = [
            i for i in range(arr.ndim) if i not in hw_axes
        ]
        # 查找通道轴
        ch_axis = _find_channel_axis(shape_list, remaining, channel_count)
        if ch_axis is not None:
            axes_order = [ch_axis, *hw_axes]
            arr = _extract_3d(arr, axes_order)
            return arr  # Shape: (C, H, W)

    logger.warning(
        f"无法精简静态张量: shape={array.shape}, "
        f"channel_count={channel_count}"
    )
    return None


def _find_channel_axis(
    shape_list: list,
    candidate_axes: list,
    channel_count: int,
) -> Optional[int]:
    """
    在候选轴中查找通道轴。

    Args:
        shape_list: 数组各维度尺寸列表。
        candidate_axes: 候选轴索引列表。
        channel_count: 期望通道数。

    Returns:
        通道轴的索引，未找到返回 None。
    """
    for axis in candidate_axes:
        if shape_list[axis] == channel_count:
            return axis
    return None


def _extract_3d(
    array: np.ndarray,
    axes_order: list,
) -> np.ndarray:
    """
    从高维数组中提取 3D 切片并转置。

    Args:
        array: 高维 ndarray。
        axes_order: 目标轴顺序 [C_axis, H_axis, W_axis]。

    Returns:
        (C, H, W) 格式的 3D ndarray。
    """
    # 选取指定轴的第一个切片（对于多余的维度）
    slices = [0] * array.ndim
    for ax in axes_order:
        slices[ax] = slice(None)
    sub = array[tuple(slices)]
    return np.transpose(sub, [axes_order.index(a) for a in axes_order])


def _align_axes_static(
    array: np.ndarray,
    channel_count: int,
) -> Optional[np.ndarray]:
    """
    静态图谱轴向硬对齐。

    规则 A: (H, W) -> (1, H, W)
    规则 B: (C, H, W) 且 C==channel_count -> 直接放行
    规则 C: (H, W, C) -> np.transpose -> (C, H, W)

    Args:
        array: 2D 或 3D ndarray。
        channel_count: 期望通道数。

    Returns:
        (C, H, W) 格式的 ndarray，失败返回 None。
    """
    if array.ndim == 2:
        return np.expand_dims(array, axis=0)  # (H,W) -> (1,H,W)

    if array.ndim != 3:
        logger.warning(f"静态张量轴对齐失败: ndim={array.ndim}")
        return None

    # 规则 B: 首维等于 channel_count
    if array.shape[0] == channel_count:
        return array  # Shape: (C, H, W)

    # 规则 C: 末维等于 channel_count
    if array.shape[2] == channel_count:
        return np.transpose(array, (2, 0, 1))  # (H,W,C) -> (C,H,W)

    logger.warning(
        f"静态张量轴对齐: 通道维不匹配 shape={array.shape}, "
        f"expected_c={channel_count}"
    )
    return None

