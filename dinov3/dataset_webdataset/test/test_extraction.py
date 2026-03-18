#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visual extraction test CLI (whitelist-driven precise DB fetch)."""

import argparse
import logging
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import psycopg2

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dinov3.dataset_webdataset import config as config_mod
from dinov3.dataset_webdataset.config import (
    ALL_TABLES,
    DB_CONFIG,
    PipelineConfig,
    TABLE_DYNAMIC,
    TABLE_STATIC,
    ImageMeta,
)
from dinov3.dataset_webdataset.dedup_index import build_whitelist
from dinov3.dataset_webdataset.dedup_types import Whitelist
from dinov3.dataset_webdataset.test.extraction_core import process_single_meta

logger = logging.getLogger("dinov3")

TEST_INDEX_ROOT = "/mnt/deepcad_nfs/ssl-data-curation/dedupIndex_100t_test"
TEST_OUTPUT_DIR = "/mnt/deepcad_nfs/ssl-data-curation/dedupIndex_100t_test/output"

# 精确查库时，file_path 批大小
QUERY_CHUNK_SIZE = 1000


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行测试 pipeline，输出 TIFF 供 Fiji 视觉验证（白名单精确查库版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python test_extraction.py --channels 1\n"
            "  python test_extraction.py --channels 4 --skip-quality-filter\n"
            "  python test_extraction.py --channels 1 --table original_image_all_2p_parsed\n"
        ),
    )
    parser.add_argument(
        "--table", type=str, default="all",
        choices=["all", TABLE_STATIC, TABLE_DYNAMIC],
        help="测试哪个表 (默认 all)",
    )
    parser.add_argument(
        "--channels", type=int, required=True,
        help="目标通道数",
    )
    parser.add_argument(
        "--max-target-size", type=int, default=4000,
        help="动态等分切片最大边长 (默认 4000)",
    )
    parser.add_argument(
        "--index-root", type=str, default=TEST_INDEX_ROOT,
        help=f"测试索引根目录 (默认 {TEST_INDEX_ROOT})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=TEST_OUTPUT_DIR,
        help=f"TIFF 输出目录 (默认 {TEST_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-quality-filter", action="store_true",
        help="跳过质量过滤（保留所有 patch）",
    )
    parser.add_argument(
        "--max-items", type=int, default=0,
        help="最多处理 N 条 (0=不限制，方便 debug)",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="不打乱白名单结果，便于严格复现实验顺序",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    original_root = config_mod.DEDUP_INDEX_ROOT
    config_mod.DEDUP_INDEX_ROOT = args.index_root
    logger.info(f"索引根目录已切换: {args.index_root}")

    table_names = list(ALL_TABLES) if args.table == "all" else [args.table]
    config = PipelineConfig(
        table_names=table_names,
        channel_count=args.channels,
        max_target_size=args.max_target_size,
    )
    logger.info(f"配置: {config}")

    t0 = time.perf_counter()
    whitelist = build_whitelist(config)
    t_wl = time.perf_counter()
    logger.info(f"⏱ 白名单构建: {t_wl - t0:.1f}s")

    logger.info("正在基于白名单精确查库...")
    valid_metas = fetch_metas_by_whitelist(config, whitelist, shuffle=not args.no_shuffle)
    t_db = time.perf_counter()
    logger.info(f"⏱ 精确查库+展开: {t_db - t_wl:.1f}s ({len(valid_metas):,d} 条)")

    if args.max_items > 0 and len(valid_metas) > args.max_items:
        valid_metas = valid_metas[: args.max_items]
        logger.info(f"限制处理数量: 前 {args.max_items} 条")

    if not valid_metas:
        logger.warning("白名单精确查库后无匹配数据，请检查测试索引和数据库内容")
        config_mod.DEDUP_INDEX_ROOT = original_root
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    total_saved = 0
    total_failed = 0
    t_loop_start = time.perf_counter()

    for idx, meta in enumerate(valid_metas):
        t_item = time.perf_counter()
        logger.info(f"\n--- 进度: {idx + 1} / {len(valid_metas)} ---")

        saved = process_single_meta(
            meta,
            config.max_target_size,
            output_dir,
            skip_quality=args.skip_quality_filter,
        )

        elapsed = time.perf_counter() - t_item
        size_mb = meta.file_size_bytes / (1024 * 1024) if meta.file_size_bytes else 0
        logger.info(f"⏱ 本条耗时: {elapsed:.2f}s | 文件大小: {size_mb:.1f}MB | 保存: {saved} 个")

        if saved > 0:
            total_saved += saved
        else:
            total_failed += 1

    t_loop_end = time.perf_counter()

    config_mod.DEDUP_INDEX_ROOT = original_root
    total_time = t_loop_end - t0
    avg_time = (t_loop_end - t_loop_start) / max(len(valid_metas), 1)

    logger.info(
        f"\n{'=' * 70}\n"
        f"  全部完成!\n"
        f"  处理数据: {len(valid_metas)} 条\n"
        f"  成功保存: {total_saved} 个 TIFF\n"
        f"  失败/跳过: {total_failed} 条\n"
        f"  总耗时: {total_time:.1f}s | 处理耗时: {t_loop_end - t_loop_start:.1f}s | 平均: {avg_time:.2f}s/条\n"
        f"  输出目录: {output_dir}\n"
        f"{'=' * 70}"
    )


# ============================================================
# Whitelist-driven precise DB fetch
# ============================================================

def fetch_metas_by_whitelist(
    config: PipelineConfig,
    whitelist: Whitelist,
    shuffle: bool = True,
) -> List[ImageMeta]:
    """
    不再全表拉取，而是仅对白名单出现的 file_path 精确查库。

    返回值已经是最终待处理的 valid_metas：
    - path_only: 返回原始 meta
    - frame_map: 对 dynamic 展开为 replace(meta, frame_idx=...)
    """
    all_valid: List[ImageMeta] = []

    # 白名单中所有涉及的路径
    whitelist_paths = set(whitelist.path_set) | set(whitelist.frame_map.keys())
    logger.info(
        f"白名单精确查库: 路径总数 {len(whitelist_paths):,d} "
        f"(整文件 {len(whitelist.path_set):,d}, 帧级文件 {len(whitelist.frame_map):,d})"
    )

    if not whitelist_paths:
        return []

    with psycopg2.connect(**DB_CONFIG) as conn:
        for table_name in config.table_names:
            metas = _fetch_table_by_paths(
                conn=conn,
                table_name=table_name,
                channel_count=config.channel_count,
                paths=sorted(whitelist_paths),
            )
            logger.info(f"  {table_name}: 精确命中 {len(metas):,d} 条")

            expanded = _expand_metas_by_whitelist(metas, whitelist)
            logger.info(f"  {table_name}: 白名单展开后 {len(expanded):,d} 条")
            all_valid.extend(expanded)

    if shuffle:
        random.shuffle(all_valid)
        logger.info(f"白名单精确查库: 最终 {len(all_valid):,d} 条 (已打乱)")
    else:
        logger.info(f"白名单精确查库: 最终 {len(all_valid):,d} 条 (未打乱)")

    return all_valid


def _fetch_table_by_paths(
    conn,
    table_name: str,
    channel_count: int,
    paths: Sequence[str],
) -> List[ImageMeta]:
    """
    对单表按 file_path 精确查询，只返回白名单涉及的路径。
    """
    if not paths:
        return []

    is_static = table_name == TABLE_STATIC
    query = _build_exact_query(table_name, is_static)

    result: List[ImageMeta] = []

    for chunk in _chunked(paths, QUERY_CHUNK_SIZE):
        with conn.cursor() as cur:
            cur.execute(query, (channel_count, list(chunk)))
            rows = cur.fetchall()

        if is_static:
            result.extend(_static_rows_to_metas(rows))
        else:
            result.extend(_dynamic_rows_to_metas(rows, table_name))

    return result


def _build_exact_query(table_name: str, is_static: bool) -> str:
    if is_static:
        return f"""
            SELECT id, file_path, image_shape, channel_count, file_size_bytes
            FROM {table_name}
            WHERE channel_count = %s
              AND file_path = ANY(%s)
        """
    return f"""
        SELECT id, file_path, h, w, count, channel_count, file_size_bytes
        FROM {table_name}
        WHERE channel_count = %s
          AND file_path = ANY(%s)
    """


def _expand_metas_by_whitelist(
    metas: List[ImageMeta],
    whitelist: Whitelist,
) -> List[ImageMeta]:
    """
    把精确查到的 meta 根据 whitelist 展开为最终待处理样本。
    """
    valid: List[ImageMeta] = []

    for meta in metas:
        path = meta.file_path

        # 整文件白名单
        if path in whitelist.path_set:
            valid.append(meta)

        # 帧级白名单：仅对 dynamic 展开
        frame_indices = whitelist.frame_map.get(path)
        if frame_indices:
            if meta.is_dynamic:
                for fidx in sorted(frame_indices):
                    valid.append(replace(meta, frame_idx=fidx))
            else:
                logger.warning(
                    f"静态图命中了帧级白名单，已忽略 frame 索引: id={meta.row_id}, path={path}"
                )

    return valid


# ============================================================
# Row -> ImageMeta
# ============================================================

def _static_rows_to_metas(rows: List[tuple]) -> List[ImageMeta]:
    metas: List[ImageMeta] = []
    for row in rows:
        meta = _static_row_to_meta(row)
        if meta is not None:
            metas.append(meta)
    return metas


def _dynamic_rows_to_metas(rows: List[tuple], table_name: str) -> List[ImageMeta]:
    return [_dynamic_row_to_meta(row, table_name) for row in rows]


def _static_row_to_meta(row: tuple) -> ImageMeta | None:
    row_id, file_path, shape_str, channel_count, file_size = row
    try:
        height, width = _parse_image_shape(shape_str, channel_count)
    except (ValueError, IndexError) as exc:
        logger.warning(f"image_shape 解析失败 id={row_id}: {exc}")
        return None

    return ImageMeta(
        row_id=row_id,
        file_path=file_path,
        height=height,
        width=width,
        frame_count=1,
        channel_count=channel_count,
        source_table=TABLE_STATIC,
        file_size_bytes=file_size or 0,
    )


def _dynamic_row_to_meta(row: tuple, table_name: str) -> ImageMeta:
    return ImageMeta(
        row_id=row[0],
        file_path=row[1],
        height=row[2],
        width=row[3],
        frame_count=row[4],
        channel_count=row[5],
        source_table=table_name,
        file_size_bytes=row[6] or 0,
    )


def _parse_image_shape(shape_str: str, channel_count: int) -> Tuple[int, int]:
    """
    从 image_shape 字符串解析 (H, W)。
    支持 (H,W) / (C,H,W) / (H,W,C)；歧义时取最大两个维度。
    """
    dims = [int(x) for x in shape_str.strip().strip("()").split(",") if x.strip()]
    if len(dims) == 2:
        return dims[0], dims[1]
    if len(dims) >= 3:
        if dims[0] == channel_count:
            return dims[1], dims[2]   # (C,H,W)
        if dims[-1] == channel_count:
            return dims[0], dims[1]   # (H,W,C)
    top2 = sorted(dims, reverse=True)
    return top2[0], top2[1]


# ============================================================
# Utils
# ============================================================

def _chunked(seq: Sequence[str], chunk_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i + chunk_size]


if __name__ == "__main__":
    main()