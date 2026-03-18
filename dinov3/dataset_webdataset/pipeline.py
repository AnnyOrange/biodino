# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档 — 主流水线编排模块。

"""
端到端数据归档流水线 (End-to-End Archiving Pipeline)。

编排完整处理流程：多表联合拉取 → 白名单拦截 → 跨源宏观 shuffle →
张量重构 → 帧提取 → 滑窗裁切 → 质量过滤 → NPY 命名 → Tar 归档。
"""

import logging
from typing import List, Optional, Set

import numpy as np

from .config import ImageMeta, PipelineConfig
from .db_client import fetch_all_tables
from .dedup_index import build_whitelist, filter_and_shuffle
from .frame_extractor import extract_target_frame
from .npy_namer import build_npy_name
from .quality_filter import passes_quality_check
from .shard_writer import ShardWriter
from .spatial_slicer import needs_slicing, slice_wsi_patches
from .tiff_reader import read_tiff_safe

logger = logging.getLogger("dinov3")


def run_pipeline(config: PipelineConfig) -> None:
    """
    执行端到端数据归档流水线。

    Args:
        config: 流水线配置。
    """
    tables_str = " + ".join(config.table_names)
    logger.info(f"流水线启动: tables=[{tables_str}], ch={config.channel_count}")

    whitelist = build_whitelist(config)
    all_metas = fetch_all_tables(config)

    # 多表联合 shuffle，保障 i.i.d. 原则
    valid_metas = filter_and_shuffle(all_metas, whitelist)

    shard_prefix = f"mixed_{config.channel_count}ch"
    with ShardWriter(prefix=shard_prefix) as writer:
        _process_meta_list(valid_metas, config, writer)

    logger.info("流水线完成")


def _process_meta_list(
    metas: List[ImageMeta],
    config: PipelineConfig,
    writer: ShardWriter,
) -> None:
    """
    遍历处理已 shuffle 的元数据列表。

    Args:
        metas: 已过滤并打乱的 ImageMeta 列表。
        config: 流水线配置。
        writer: Tar 分片写入器。
    """
    for idx, meta in enumerate(metas):
        _process_single_image(meta, config, writer)

        if (idx + 1) % 1000 == 0:
            logger.info(f"进度: {idx + 1:,d} / {len(metas):,d}")


def _process_single_image(
    meta: ImageMeta,
    config: PipelineConfig,
    writer: ShardWriter,
) -> None:
    """
    处理单张图像：读取 → 重构 → 帧提取 → 裁切 → 过滤 → 归档。

    Args:
        meta: 图像元数据（含 source_table 来源标记）。
        config: 流水线配置。
        writer: Tar 分片写入器。
    """
    if meta.is_dynamic and meta.frame_idx is not None:
        from .lazy_frame_reader import read_dynamic_target_frame
        frame = read_dynamic_target_frame(meta)
        if frame is None:
            return
        _slice_filter_archive(frame, meta, config, writer)
        return

    raw_array = read_tiff_safe(meta.file_path)
    if raw_array is None:
        return

    reconstructed = _reconstruct_tensor(raw_array, meta)
    if reconstructed is None:
        return

    frame_slice = extract_target_frame(reconstructed, meta)
    if frame_slice is None:
        return

    _slice_filter_archive(frame_slice, meta, config, writer)


def _reconstruct_tensor(
    raw_array: np.ndarray,
    meta: ImageMeta,
) -> Optional[np.ndarray]:
    """
    根据样本来源自动选择张量重构策略。

    由 meta.is_dynamic 属性路由到对应的重构逻辑，
    无需依赖全局 config 判断。

    Args:
        raw_array: 原始 TIFF ndarray。
        meta: 含来源标记的图像元数据。

    Returns:
        重构后的 ndarray，失败返回 None。
    """
    if meta.is_dynamic:
        from .tensor_dynamic import reconstruct_dynamic
        return reconstruct_dynamic(raw_array, meta)

    from .tensor_static import reconstruct_static
    return reconstruct_static(raw_array, meta)


def _slice_filter_archive(
    array: np.ndarray,
    meta: ImageMeta,
    config: PipelineConfig,
    writer: ShardWriter,
) -> None:
    """
    执行裁切 → 条件质量过滤 → 命名 → 归档。

    Args:
        array: (C, H, W) 帧切片张量。
        meta: 含来源标记的图像元数据。
        config: 流水线配置。
        writer: Tar 分片写入器。
    """
    is_sliced = needs_slicing(array)

    for patch_info in slice_wsi_patches(array, config.max_target_size):
        # 仅对切分后的 patch 过滤；未切分整图直接写入。
        if is_sliced and not passes_quality_check(patch_info.array):
            continue

        patch_arg = patch_info if is_sliced else None
        npy_name = build_npy_name(meta, patch_arg)
        writer.write_npy(npy_name, patch_info.array)
