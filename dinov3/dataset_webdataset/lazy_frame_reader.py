import logging
from typing import Optional, Tuple

import numpy as np

from .config import ImageMeta

logger = logging.getLogger("dinov3")


def read_dynamic_target_frame(meta: ImageMeta) -> Optional[np.ndarray]:
    frame, _ = read_dynamic_target_frame_with_strategy(meta)
    return frame


def read_dynamic_target_frame_with_strategy(
    meta: ImageMeta,
) -> Tuple[Optional[np.ndarray], str]:
    """
    返回 (frame, strategy)

    strategy:
        - memmap
        - key_range
        - page_loop
        - full_fallback
        - failed
    """
    if meta.frame_idx is None:
        return None, "failed"

    try:
        import tifffile
    except ImportError:
        logger.error("tifffile 未安装: pip install tifffile")
        return None, "failed"

    frame = _read_via_memmap(meta, tifffile)
    if frame is not None:
        return frame, "memmap"

    frame = _read_via_key_range(meta, tifffile)
    if frame is not None:
        return frame, "key_range"

    frame = _read_via_pages_openclose(meta, tifffile)
    if frame is not None:
        return frame, "page_loop"

    frame = _read_via_full_imread(meta, tifffile)
    if frame is not None:
        return frame, "full_fallback"

    return None, "failed"


def read_dynamic_target_frame_from_tif(tif, meta: ImageMeta) -> Optional[np.ndarray]:
    """
    给 ray_worker 按文件分组时复用。
    这里仍然只做 page-based 读取，因为外层已经持有打开的 tif。
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
                    f"page 不是 2D: page_idx={i}, shape={arr.shape}, id={meta.row_id}"
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

        return np.stack(pages, axis=0)

    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(f"按页读取失败 [{meta.file_path}]: {exc}")
        return None


def _read_via_memmap(meta: ImageMeta, tifffile) -> Optional[np.ndarray]:
    """
    最优先：如果 TIFF 可 memmap，则直接按页切。
    """
    try:
        arr = tifffile.memmap(meta.file_path)
    except Exception:
        return None

    return _flatten_and_slice(arr, meta)


def _read_via_key_range(meta: ImageMeta, tifffile) -> Optional[np.ndarray]:
    """
    次优先：让 tifffile 一次性读取连续 key 范围。
    通常比手工 for pages[i].asarray() 更快。
    """
    c = meta.channel_count
    fidx = meta.frame_idx
    start = fidx * c
    end = start + c

    try:
        arr = tifffile.imread(meta.file_path, key=range(start, end))
    except Exception:
        return None

    if arr is None:
        return None

    # 目标统一输出 (C, H, W)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim >= 3:
        # 如果 tifffile 直接给出 (C,H,W) 则保持
        # 如果有更复杂形状，则退回统一 flatten 逻辑
        if not (arr.shape[-2] == meta.height and arr.shape[-1] == meta.width):
            return _flatten_and_slice(arr, meta)

    if arr.shape[-2:] == (meta.width, meta.height):
        arr = np.swapaxes(arr, -1, -2)

    if arr.shape[0] != meta.channel_count or arr.shape[-2:] != (meta.height, meta.width):
        return _flatten_and_slice(arr, meta)

    return arr


def _read_via_pages_openclose(meta: ImageMeta, tifffile) -> Optional[np.ndarray]:
    try:
        with tifffile.TiffFile(meta.file_path) as tif:
            return read_dynamic_target_frame_from_tif(tif, meta)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return None


def _read_via_full_imread(meta: ImageMeta, tifffile) -> Optional[np.ndarray]:
    try:
        arr = tifffile.imread(meta.file_path)
    except Exception:
        return None

    return _flatten_and_slice(arr, meta)


def _flatten_and_slice(arr: np.ndarray, meta: ImageMeta) -> Optional[np.ndarray]:
    h, w, c = meta.height, meta.width, meta.channel_count
    fidx = meta.frame_idx

    pages = _reshape_to_pages(arr, h, w)
    if pages is None:
        logger.warning(
            f"无法展平为页序列: shape={arr.shape}, target_hw=({h},{w}), id={meta.row_id}"
        )
        return None

    start = fidx * c
    end = start + c
    if end > pages.shape[0]:
        logger.warning(
            f"frame_idx 越界: start={start}, end={end}, total_pages={pages.shape[0]}, id={meta.row_id}"
        )
        return None

    return pages[start:end]


def _reshape_to_pages(arr: np.ndarray, target_h: int, target_w: int) -> Optional[np.ndarray]:
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