# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档 — 数据库交互模块。

"""
PostgreSQL 数据库客户端（分表路由）。

两张表 Schema 不同：
- original_images_all          : id, file_path, image_shape, channel_count
- original_image_all_2p_parsed : id, file_path, h, w, count, channel_count, ...
"""

import logging
from contextlib import contextmanager
from typing import Generator, Iterator, List, Optional, Tuple

import psycopg2

from .config import DB_CONFIG, DB_FETCH_BATCH, TABLE_STATIC, ImageMeta, PipelineConfig

logger = logging.getLogger("dinov3")

_QUERY_STATIC = (
    "SELECT id, file_path, image_shape, channel_count, file_size_bytes "
    "FROM {table} WHERE channel_count = %s"
)
_QUERY_DYNAMIC = (
    "SELECT id, file_path, h, w, count, channel_count, file_size_bytes "
    "FROM {table} WHERE channel_count = %s"
)


@contextmanager
def connect_db() -> Generator:
    """PostgreSQL 连接上下文管理器，用完自动关闭。"""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def _parse_image_shape(shape_str: str, channel_count: int) -> Tuple[int, int]:
    """
    从 image_shape 字符串解析 (H, W)。

    支持 (H,W) / (C,H,W) / (H,W,C)；以 channel_count 辨别通道轴。
    歧义时取最大两个维度。
    """
    dims = [int(x) for x in shape_str.strip().strip("()").split(",")]
    if len(dims) == 2:
        return dims[0], dims[1]
    if len(dims) >= 3:
        if dims[0] == channel_count:
            return dims[1], dims[2]  # (C, H, W)
        if dims[-1] == channel_count:
            return dims[0], dims[1]  # (H, W, C)
    top2 = sorted(dims, reverse=True)
    return top2[0], top2[1]


def _static_row_to_meta(row: tuple) -> Optional[ImageMeta]:
    """静态表行 → ImageMeta；解析 image_shape，frame_count 固定为 1。"""
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
    """动态表行 (id, file_path, h, w, count, ch, file_size) → ImageMeta。"""
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


def _rows_to_metas(rows: list, is_static: bool, table_name: str) -> List[ImageMeta]:
    """按表类型批量转换数据库行为 ImageMeta 列表。"""
    result: List[ImageMeta] = []
    for r in rows:
        if is_static:
            meta = _static_row_to_meta(r)
            if meta is not None:
                result.append(meta)
        else:
            result.append(_dynamic_row_to_meta(r, table_name))
    return result


def _fetch_from_single_table(
    table_name: str, channel_count: int
) -> Iterator[List[ImageMeta]]:
    """
    从单表流式拉取，按表路由不同查询与行映射。

    Yields:
        每次 yield 一批 ImageMeta 列表。
    """
    is_static = table_name == TABLE_STATIC
    query = (_QUERY_STATIC if is_static else _QUERY_DYNAMIC).format(table=table_name)

    with connect_db() as conn:
        cursor_name = f"fetch_{table_name}_{channel_count}"
        with conn.cursor(name=cursor_name) as cursor:
            cursor.execute(query, (channel_count,))
            logger.info(f"拉取: {table_name}, ch={channel_count}")
            while True:
                rows = cursor.fetchmany(DB_FETCH_BATCH)
                if not rows:
                    break
                yield _rows_to_metas(rows, is_static, table_name)


def fetch_all_tables(config: PipelineConfig) -> List[ImageMeta]:
    """
    从配置中所有表拉取并合并为一个列表（调用方负责 shuffle）。

    Args:
        config: 含 table_names / channel_count 的流水线配置。

    Returns:
        合并后的完整 ImageMeta 列表。
    """
    all_metas: List[ImageMeta] = []
    for table_name in config.table_names:
        table_count = 0
        for batch in _fetch_from_single_table(table_name, config.channel_count):
            all_metas.extend(batch)
            table_count += len(batch)
        logger.info(f"  {table_name}: {table_count:,d} 条")
    logger.info(f"多表合并总量: {len(all_metas):,d} 条")
    return all_metas
