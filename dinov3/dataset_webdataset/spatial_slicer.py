# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档 — 条件动态等分切片模块。

"""
条件动态等分切片 (Conditional Dynamic Tiling)。

仅当图像满足 **长边 > TILING_LONG_THRESH 且 短边 > TILING_SHORT_THRESH**
时执行无重叠网格等分切片，残余像素并入末尾 patch。
不满足条件则将整张原图作为完整 Patch 输出。

DINOv3 原生支持不同比例输入（短边 > 1152 即可配合
RandomResizedCrop），因此无需固定窗口裁切。
"""

import logging
import math
from dataclasses import dataclass
from typing import Generator, Tuple

import numpy as np

from .config import MAX_TARGET_SIZE, TILING_LONG_THRESH, TILING_SHORT_THRESH

logger = logging.getLogger("dinov3")


@dataclass(frozen=True)
class PatchInfo:
    """单个裁切块的空间元信息。

    Attributes:
        x_start: 裁切起始 X 坐标。
        y_start: 裁切起始 Y 坐标。
        patch_h: 裁切块高度。
        patch_w: 裁切块宽度。
        array: 裁切出的 ndarray。
    """
    x_start: int
    y_start: int
    patch_h: int
    patch_w: int
    array: np.ndarray


def needs_slicing(array: np.ndarray) -> bool:
    """
    判断 (C, H, W) 张量是否满足切分触发条件。

    条件：长边 > 4096 **且** 短边 > 1024，否则直接输出整图。

    Args:
        array: (C, H, W) 格式的张量。

    Returns:
        True 表示需要执行动态等分切片。
    """
    _, img_h, img_w = array.shape
    return _should_tile(img_h, img_w)


def slice_wsi_patches(
    array: np.ndarray,
    max_target_size: int = MAX_TARGET_SIZE,
) -> Generator[PatchInfo, None, None]:
    """
    对 (C, H, W) 张量执行条件动态等分切片。

    满足切分条件时，将图像划分为无重叠网格；残余像素并入末尾 patch。
    不满足条件时，直接返回整张原图作为单个 Patch。

    Args:
        array: (C, H, W) 格式的输入张量。
        max_target_size: 期望的单片最大边长（默认 4000）。

    Yields:
        PatchInfo 对象，包含裁切块及其空间坐标。
    """
    _, img_h, img_w = array.shape  # Shape: (C, H, W)

    if not _should_tile(img_h, img_w):
        yield PatchInfo(
            x_start=0, y_start=0,
            patch_h=img_h, patch_w=img_w,
            array=array,
        )
        return

    yield from _generate_grid_patches(array, img_h, img_w, max_target_size)


# ============================================================================
# 内部辅助
# ============================================================================

def _should_tile(height: int, width: int) -> bool:
    """切分触发条件：长边 > 4096 且 短边 > 1024。"""
    long_edge = max(height, width)
    short_edge = min(height, width)
    return long_edge > TILING_LONG_THRESH and short_edge > TILING_SHORT_THRESH


def _compute_grid(length: int, max_size: int) -> Tuple[int, int]:
    """
    计算单轴网格数和每片基准尺寸。

    Args:
        length: 该轴的总长度。
        max_size: 期望的单片最大边长。

    Returns:
        (num_tiles, tile_size): 切片数量和每片基准像素数。
    """
    num_tiles = math.ceil(length / max_size)
    tile_size = length // num_tiles
    return num_tiles, tile_size


def _generate_grid_patches(
    array: np.ndarray,
    img_h: int,
    img_w: int,
    max_size: int,
) -> Generator[PatchInfo, None, None]:
    """
    按无重叠网格生成所有 patch，残余像素并入末行/末列。

    Args:
        array: (C, H, W) 张量。
        img_h: 图像高度。
        img_w: 图像宽度。
        max_size: 期望的单片最大边长。

    Yields:
        PatchInfo 对象。
    """
    ny, tile_h = _compute_grid(img_h, max_size)
    nx, tile_w = _compute_grid(img_w, max_size)

    for iy in range(ny):
        for ix in range(nx):
            y_start = iy * tile_h
            x_start = ix * tile_w
            # 末行/末列拼入残余像素，避免产生碎片
            y_end = img_h if iy == ny - 1 else y_start + tile_h
            x_end = img_w if ix == nx - 1 else x_start + tile_w

            patch = array[:, y_start:y_end, x_start:x_end]  # (C, h, w)
            yield PatchInfo(
                x_start=x_start,
                y_start=y_start,
                patch_h=y_end - y_start,
                patch_w=x_end - x_start,
                array=patch,
            )
