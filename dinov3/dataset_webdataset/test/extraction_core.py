"""Core extraction helpers for test_extraction CLI."""

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import numpy as np

from dinov3.dataset_webdataset.config import ImageMeta
from dinov3.dataset_webdataset.frame_extractor import extract_all_frames
# from dinov3.dataset_webdataset.lazy_frame_reader import read_dynamic_target_frame
from dinov3.dataset_webdataset.lazy_frame_reader import (
    read_dynamic_target_frame_with_strategy,
)
from dinov3.dataset_webdataset.npy_namer import build_npy_name
from dinov3.dataset_webdataset.quality_filter import passes_quality_check
from dinov3.dataset_webdataset.spatial_slicer import needs_slicing, slice_wsi_patches
from dinov3.dataset_webdataset.tiff_reader import read_tiff_safe

logger = logging.getLogger("dinov3")


def process_single_meta(
    meta: ImageMeta,
    max_target_size: int,
    output_dir: Path,
    skip_quality: bool,
) -> int:
    """Process one ImageMeta and save tiffs for visual verification."""
    _log_meta_header(meta)

    # if meta.is_dynamic and meta.frame_idx is not None:
    #     t0 = time.perf_counter()
    #     frame = read_dynamic_target_frame(meta)
    #     t_read = time.perf_counter()

    #     logger.info(f"    ⏱ [dynamic读帧] {t_read - t0:.2f}s")

    #     if frame is None:
    #         logger.warning("  ❌ 单帧读取失败，跳过")
    #         return 0

    #     return _save_frames_timed([frame], meta, max_target_size, output_dir, skip_quality)
    if meta.is_dynamic and meta.frame_idx is not None:
        t0 = time.perf_counter()
        frame, strategy = read_dynamic_target_frame_with_strategy(meta)
        t_read = time.perf_counter()

        logger.info(f"    ⏱ [dynamic读帧: {strategy}] {t_read - t0:.2f}s")

        if frame is None:
            logger.warning("  ❌ 单帧读取失败，跳过")
            return 0

        return _save_frames_timed([frame], meta, max_target_size, output_dir, skip_quality)
    t0 = time.perf_counter()
    raw = read_tiff_safe(meta.file_path, meta.file_size_bytes)
    t_read = time.perf_counter()
    logger.info(f"    ⏱ [imread] {t_read - t0:.2f}s")
    if raw is None:
        logger.warning("  ❌ TIFF 读取失败，跳过")
        return 0

    logger.info(f"  📖 原始数据: shape={raw.shape}, dtype={raw.dtype}")

    tensor = reconstruct(raw, meta)
    del raw
    t_recon = time.perf_counter()
    logger.info(f"    ⏱ [重构] {t_recon - t_read:.2f}s")
    if tensor is None:
        logger.warning("  ❌ 张量重构失败，跳过")
        return 0

    logger.info(f"  🔧 重构后: shape={tensor.shape}")

    frames = extract_all_frames(tensor, meta)
    del tensor
    t_extract = time.perf_counter()
    logger.info(f"    ⏱ [帧提取] {t_extract - t_recon:.2f}s")
    if not frames:
        logger.warning("  ❌ 帧提取结果为空，跳过")
        return 0

    logger.info(f"  🎞️  提取到 {len(frames)} 帧, 每帧 shape={frames[0].shape}")
    return _save_frames_timed(frames, meta, max_target_size, output_dir, skip_quality)


def _save_frames_timed(
    frames: List[np.ndarray],
    meta: ImageMeta,
    max_target_size: int,
    output_dir: Path,
    skip_quality: bool,
) -> int:
    """Save with per-stage timing."""
    t0 = time.perf_counter()
    saved, rejected = 0, 0
    t_quality_total, t_write_total = 0.0, 0.0

    for f_idx, frame in enumerate(frames):
        is_sliced = needs_slicing(frame)
        naming_meta = (
            replace(meta, frame_idx=f_idx)
            if len(frames) > 1 and meta.frame_idx is None
            else meta
        )

        for patch in slice_wsi_patches(frame, max_target_size):
            if is_sliced and not skip_quality:
                t_q = time.perf_counter()
                ok = passes_quality_check(patch.array)
                t_quality_total += time.perf_counter() - t_q
                if not ok:
                    rejected += 1
                    continue

            npy_name = build_npy_name(naming_meta, patch if is_sliced else None)

            t_w = time.perf_counter()
            _save_tiff(patch.array, output_dir / npy_name.replace(".npy", ".tiff"))
            t_write_total += time.perf_counter() - t_w

            saved += 1

    del frames
    t_done = time.perf_counter()

    logger.info(
        f"    ⏱ [切片+遍历] 0.00s  "
        f"[质量过滤] {t_quality_total:.2f}s  "
        f"[写TIFF] {t_write_total:.2f}s  "
        f"[总后处理] {t_done - t0:.2f}s"
    )
    logger.info(
        f"  ✅ 完成: 保存 {saved} 个 TIFF"
        + (f", 质量过滤拒绝 {rejected} 个" if rejected else "")
    )
    return saved


def reconstruct(raw: np.ndarray, meta: ImageMeta) -> Optional[np.ndarray]:
    """Route to static/dynamic tensor reconstruction."""
    if meta.is_dynamic:
        from dinov3.dataset_webdataset.tensor_dynamic import reconstruct_dynamic
        return reconstruct_dynamic(raw, meta)
    from dinov3.dataset_webdataset.tensor_static import reconstruct_static
    return reconstruct_static(raw, meta)


def _save_tiff(array: np.ndarray, output_path: Path) -> None:
    """Write (C, H, W) array as ImageJ-compatible TIFF."""
    import tifffile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output_path), array, imagej=True)


def _log_meta_header(meta: ImageMeta) -> None:
    """Print extraction header for one sample."""
    source_label = "dynamic" if meta.is_dynamic else "static"
    logger.info(
        f"\n{'=' * 70}\n"
        f"  ID: {meta.row_id}  |  类型: {source_label}  |  "
        f"通道: {meta.channel_count}  |  帧数: {meta.frame_count}  |  "
        f"目标帧: {meta.frame_idx}\n"
        f"  路径: {meta.file_path}\n"
        f"{'=' * 70}"
    )