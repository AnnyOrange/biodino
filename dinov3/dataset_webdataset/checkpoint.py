# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档 — 断点续传检查点模块。

"""
断点续传检查点管理。

Master 每收到一个 Chunk 完成信号，就把该 Chunk 包含的数据复合键
追加到检查点文件。再次运行时加载检查点，自动跳过已处理数据。

检查点文件格式（Tab 分隔，向后兼容）：
- 无帧：original_images_all:67877069\\tmixed_1ch-000001.tar
- 有帧：original_image_all_2p_parsed:2250552:frame_108\\tmixed_1ch-000001.tar
- 旧格式（无 tar）：original_images_all:67877069
"""

import logging
from pathlib import Path
from typing import List, Optional, Set

from .config import ImageMeta

logger = logging.getLogger("dinov3")

_CKPT_TEMPLATE = "checkpoint_{ch}ch.txt"


def _ckpt_path(nfs_dir: str, channel_count: int) -> Path:
    """生成检查点文件路径。"""
    return Path(nfs_dir) / _CKPT_TEMPLATE.format(ch=channel_count)


def load_processed_keys(nfs_dir: str, channel_count: int) -> Set[str]:
    """
    加载已处理数据的复合键集合。

    兼容新旧两种格式：
    - 旧格式：source_table:row_id
    - 新格式：source_table:row_id\\ttar_name（只取 Tab 前的 key 部分）

    Args:
        nfs_dir: NFS 输出目录。
        channel_count: 通道数（决定检查点文件名）。

    Returns:
        已处理的复合键集合，文件不存在返回空集。
    """
    path = _ckpt_path(nfs_dir, channel_count)
    if not path.exists():
        return set()
    keys: Set[str] = set()
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # 兼容新格式：Tab 分隔时只取第一列（复合键）
            key = stripped.split("\t")[0]
            keys.add(key)
    logger.info(f"断点续传: 已加载 {len(keys):,d} 条已处理记录")
    return keys


def append_chunk_keys(
    nfs_dir: str,
    channel_count: int,
    meta_dicts: List[dict],
    tar_name: str = "",
) -> None:
    """
    将完成 Chunk 内的数据键追加到检查点文件。

    每行格式：复合键\\ttar文件名

    Args:
        nfs_dir: NFS 输出目录。
        channel_count: 通道数。
        meta_dicts: 已完成 Chunk 的 ImageMeta 序列化字典列表。
        tar_name: 该 Chunk 写入的 tar 文件名（用于后续去重溯源）。
    """
    path = _ckpt_path(nfs_dir, channel_count)
    with open(path, "a") as f:
        for d in meta_dicts:
            key = _dict_to_key(d)
            if tar_name:
                f.write(f"{key}\t{tar_name}\n")
            else:
                f.write(f"{key}\n")


def filter_unprocessed(
    metas: List[ImageMeta],
    processed: Set[str],
) -> List[ImageMeta]:
    """
    过滤掉已处理的 ImageMeta，返回待处理列表。

    Args:
        metas: 白名单过滤后的完整 ImageMeta 列表。
        processed: 已处理的复合键集合。

    Returns:
        仅包含未处理数据的 ImageMeta 列表。
    """
    if not processed:
        return metas
    before = len(metas)
    remaining = [m for m in metas if _meta_to_key(m) not in processed]
    skipped = before - len(remaining)
    logger.info(
        f"断点续传: 跳过 {skipped:,d} 已处理, "
        f"剩余 {len(remaining):,d} 待处理"
    )
    return remaining


def _meta_to_key(meta: ImageMeta) -> str:
    """
    ImageMeta → 复合唯一键。

    格式（向后兼容）：
    - 无帧：source_table:row_id
    - 有帧：source_table:row_id:frame_N
    """
    key = f"{meta.source_table}:{meta.row_id}"
    if meta.frame_idx is not None:
        key += f":frame_{meta.frame_idx}"
    return key


def _dict_to_key(d: dict) -> str:
    """
    序列化字典 → 复合唯一键。

    格式与 _meta_to_key 保持一致。
    """
    key = f"{d['source_table']}:{d['row_id']}"
    frame_idx = d.get('frame_idx')
    if frame_idx is not None:
        key += f":frame_{frame_idx}"
    return key
