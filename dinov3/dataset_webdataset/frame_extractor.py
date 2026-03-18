# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档 — 帧提取与通道保留模块。

"""
基于索引的精准帧提取 (Targeted Frame Extraction)。

核心原则：仅在时间/空间深度（Frame）上切片，坚决保留全部通道。

接收上游 tensor_static / tensor_dynamic 已标准化的张量
（原始 TIFF 可为 2D~5D+，但经重构后一律为下列两种格式）：

- (C, H, W)：静态或单帧退化，断言后直接作为单帧返回。
- (C, F, H, W)：多帧序列。frame_idx 指定时返回该帧，
  否则遍历全部 F 帧，每帧独立输出 (C, H, W)。
"""

import logging
from typing import List, Optional

import numpy as np

from .config import ImageMeta

logger = logging.getLogger("dinov3")


def extract_all_frames(
    array: np.ndarray,
    meta: ImageMeta,
) -> List[np.ndarray]:
    """
    提取所有目标帧（保留全部通道）。

    Args:
        array: 重构后的张量 (C, H, W) 或 (C, F, H, W)。
        meta: 含可选 frame_idx 的图像元数据。

    Returns:
        (channel_count, H, W) 帧列表，失败或无有效帧时返回空列表。
    """
    if array.ndim == 3:
        result = _pass_static(array, meta)
        return [result] if result is not None else []

    if array.ndim == 4:
        return _extract_4d_frames(array, meta)

    logger.warning(f"帧提取: 不支持的维度 ndim={array.ndim}, id={meta.row_id}")
    return []


def _extract_4d_frames(
    array: np.ndarray,
    meta: ImageMeta,
) -> List[np.ndarray]:
    """
    4D 序列帧提取：frame_idx 指定则取单帧，否则遍历全部帧。

    Args:
        array: (C, F, H, W) 格式的 ndarray。
        meta: 含可选 frame_idx 的元数据。

    Returns:
        (C, H, W) 帧列表。
    """
    if meta.frame_idx is not None:
        result = _slice_single_frame(array, meta)
        return [result] if result is not None else []
    return _slice_all_frames(array, meta)


def _slice_all_frames(
    array: np.ndarray,
    meta: ImageMeta,
) -> List[np.ndarray]:
    """
    遍历 (C, F, H, W) 中所有 F 帧，每帧输出断言通过的 (C, H, W)。

    Args:
        array: (C, F, H, W) 格式的 ndarray。
        meta: 图像元数据（用于通道数断言）。

    Returns:
        通过断言的帧列表（可能为空）。
    """
    frames: List[np.ndarray] = []
    for f_idx in range(array.shape[1]):
        frame = array[:, f_idx, :, :]
        validated = _assert_final_shape(frame, meta.channel_count)
        if validated is not None:
            frames.append(validated)
    return frames


def _pass_static(
    array: np.ndarray,
    meta: ImageMeta,
) -> Optional[np.ndarray]:
    """静态图谱直通（无需帧切片），执行出口断言。"""
    return _assert_final_shape(array, meta.channel_count)


def _slice_single_frame(
    array: np.ndarray,
    meta: ImageMeta,
) -> Optional[np.ndarray]:
    """
    从 (C, F, H, W) 中提取 frame_idx 指定的单帧。

    Args:
        array: (C, F, H, W) 格式的 ndarray。
        meta: 含 frame_idx 的元数据。

    Returns:
        (C, H, W) 格式的帧，越界时返回 None。
    """
    frame_count = array.shape[1]
    if meta.frame_idx >= frame_count:  # type: ignore[operator]
        logger.warning(
            f"frame_idx 越界: idx={meta.frame_idx}, "
            f"total={frame_count}, id={meta.row_id}"
        )
        return None
    frame = array[:, meta.frame_idx, :, :]
    return _assert_final_shape(frame, meta.channel_count)


def _assert_final_shape(
    array: np.ndarray,
    expected_channels: int,
) -> Optional[np.ndarray]:
    """
    训练出口断言：验证最终张量严格为 (C, H, W)。

    Args:
        array: 待验证的 ndarray。
        expected_channels: 期望的通道数。

    Returns:
        通过断言的 ndarray，失败返回 None。
    """
    if array.ndim != 3:
        logger.warning(f"出口断言失败: ndim={array.ndim}, 期望 3")
        return None
    if array.shape[0] != expected_channels:
        logger.warning(
            f"出口断言失败: C={array.shape[0]}, 期望 C={expected_channels}"
        )
        return None
    return array
