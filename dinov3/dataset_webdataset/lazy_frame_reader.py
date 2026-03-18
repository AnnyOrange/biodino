# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档 — 动态单帧读取模块。

"""
动态单帧读取 (Frame Reader)。

优化策略：
1. 对 dynamic + frame_idx 样本，优先按 TIFF pages 精准读取目标 frame；
2. 但对超大文件 / 超长序列，避免在 NFS 上走随机页访问；
3. 超大文件直接回退到顺序整读 + flatten_and_slice；
4. 支持“复用已打开的 tifffile.TiffFile 对象”，供 ray_worker 按文件分组批处理。

统一返回 (C, H, W)。
"""

import logging
from typing import Optional

import numpy as np

from .config import ImageMeta

logger = logging.getLogger("dinov3")

# -----------------------------
# 读取策略阈值（可按机器情况微调）
# -----------------------------
# 文件超过 2GB：优先整读，避免 NFS 上随机页 seek 太慢
PAGE_READ_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024

# 帧数超过 1000：优先整读
PAGE_READ_MAX_FRAME_COUNT = 1000

# 若目标帧接近末尾，随机 seek 往往更吃亏；这里给个可选阈值
# ratio > 0.8 时更倾向整读（可关闭）
TAIL_FRAME_RATIO_FOR_FULL_READ = 0.80


def read_dynamic_target_frame(meta: ImageMeta) -> Optional[np.ndarray]:
    """
    单条读取接口：
    - 小文件 / 中等序列：优先按页精准读取；
    - 超大文件 / 超长序列：优先顺序整读；
    - 失败时自动尝试另一种策略兜底。

    Returns:
        (C, H, W) 或 None
    """
    if meta.frame_idx is None:
        return None

    try:
        import tifffile
    except ImportError:
        logger.error("tifffile 未安装: pip install tifffile")
        return None

    prefer_full = _should_prefer_full_read(meta)

    # 先走首选策略
    if prefer_full:
        frame = _read_via_full_imread(meta, tifffile)
        if frame is not None:
            return frame

        frame = _read_via_pages_openclose(meta, tifffile)
        if frame is not None:
            return frame
    else:
        frame = _read_via_pages_openclose(meta, tifffile)
        if frame is not None:
            return frame

        frame = _read_via_full_imread(meta, tifffile)
        if frame is not None:
            return frame

    return None


def read_dynamic_target_frame_from_tif(tif, meta: ImageMeta) -> Optional[np.ndarray]:
    """
    基于已打开的 tifffile.TiffFile 对象读取目标 frame。
    这是给 ray_worker 分组处理时复用的核心函数。

    注意：
    - 这里只做“按页读取”；
    - 不做整读，因为外层已经决定复用打开的 tif 了。

    Returns:
        (C, H, W) 或 None
    """
    c = meta.channel_count
    h = meta.height
    w = meta.width
    fidx = meta.frame_idx

    if fidx is None:
        return None

    start = fidx * c
    end = start + c

    total_pages = len(tif.pages)
    if end > total_pages:
        logger.warning(
            f"page 越界: start={start}, end={end}, total_pages={total_pages}, id={meta.row_id}"
        )
        return None

    pages = []
    try:
        for i in range(start, end):
            arr = tif.pages[i].asarray()

            if arr.ndim != 2:
                logger.warning(
                    f"page 不是 2D，无法直接组成 frame: page_idx={i}, shape={arr.shape}, id={meta.row_id}"
                )
                return None

            if arr.shape == (h, w):
                pages.append(arr)
            elif arr.shape == (w, h):
                pages.append(arr.T)
            else:
                logger.warning(
                    f"page 尺寸不匹配: got={arr.shape}, expect=({h},{w}), id={meta.row_id}"
                )
                return None

        return np.stack(pages, axis=0)  # (C, H, W)

    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(f"按页读取失败 [{meta.file_path}]: {exc}")
        return None


# ============================================================
# 内部辅助函数
# ============================================================

def _should_prefer_full_read(meta: ImageMeta) -> bool:
    """
    判断该 dynamic 文件是否更适合顺序整读而不是按页随机读。
    """
    file_size = meta.file_size_bytes or 0
    frame_count = meta.frame_count or 0
    frame_idx = meta.frame_idx if meta.frame_idx is not None else 0

    if file_size >= PAGE_READ_MAX_FILE_BYTES:
        return True

    if frame_count >= PAGE_READ_MAX_FRAME_COUNT:
        return True

    if frame_count > 0 and frame_idx / max(frame_count, 1) >= TAIL_FRAME_RATIO_FOR_FULL_READ:
        return True

    return False


def _read_via_pages_openclose(meta: ImageMeta, tifffile) -> Optional[np.ndarray]:
    """
    打开文件后按页读取目标 frame。
    适合小/中等 dynamic TIFF。
    """
    try:
        with tifffile.TiffFile(meta.file_path) as tif:
            return read_dynamic_target_frame_from_tif(tif, meta)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        logger.warning(f"按页读取失败，回退其他策略 [{meta.file_path}]: {exc}")
        return None


def _read_via_full_imread(meta: ImageMeta, tifffile) -> Optional[np.ndarray]:
    """
    整文件顺序 imread 后，再 flatten + slice。
    对超大文件 / NFS 场景往往更稳。
    """
    try:
        full_array = tifffile.imread(meta.file_path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        logger.warning(f"整读 TIFF 失败 [{meta.file_path}]: {exc}")
        return None

    return _flatten_and_slice(full_array, meta)


def _flatten_and_slice(
    arr: np.ndarray,
    meta: ImageMeta,
) -> Optional[np.ndarray]:
    """
    找 H,W → 展平为 (N, H, W) → 按 frame_idx*C 切片 → (C, H, W)。
    作为整读后的切帧逻辑。
    """
    h, w, c = meta.height, meta.width, meta.channel_count
    fidx = meta.frame_idx

    pages = _reshape_to_pages(arr, h, w)
    if pages is None:
        logger.warning(
            f"无法展平为页序列: shape={arr.shape}, target_hw=({h},{w}), id={meta.row_id}"
        )
        return None

    total_pages = pages.shape[0]
    start = fidx * c
    end = start + c
    if end > total_pages:
        logger.warning(
            f"frame_idx 越界: start={start}, end={end}, total_pages={total_pages}, id={meta.row_id}"
        )
        return None

    return pages[start:end]


def _reshape_to_pages(arr: np.ndarray, target_h: int, target_w: int) -> Optional[np.ndarray]:
    """
    把任意包含 H/W 维度的数组展平成 (N, H, W)。
    """
    if arr.ndim < 2:
        return None

    hw_axes = _find_hw_axes(arr.shape, target_h, target_w)
    if hw_axes is None:
        return None

    h_ax, w_ax = hw_axes
    other_axes = [i for i in range(arr.ndim) if i not in (h_ax, w_ax)]
    perm = other_axes + [h_ax, w_ax]
    transposed = np.transpose(arr, perm)
    n_pages = transposed.size // (target_h * target_w)
    return transposed.reshape(n_pages, target_h, target_w)


def _find_hw_axes(shape: tuple, target_h: int, target_w: int) -> Optional[tuple]:
    """
    在 shape 中寻找对应 H/W 的两个轴。
    """
    ndim = len(shape)

    if ndim >= 2:
        if shape[-2] == target_h and shape[-1] == target_w:
            return (ndim - 2, ndim - 1)
        if shape[-2] == target_w and shape[-1] == target_h:
            return (ndim - 1, ndim - 2)

    for i in range(ndim):
        for j in range(ndim):
            if i != j and shape[i] == target_h and shape[j] == target_w:
                return (i, j)

    return None