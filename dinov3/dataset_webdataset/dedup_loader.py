"""Whitelist loading and parsing for dedup index."""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from . import config as _cfg
from .config import TABLE_STATIC, PipelineConfig
from .dedup_types import Whitelist

logger = logging.getLogger("dinov3")
_FRAME_SUFFIX_RE = re.compile(r"^(.+)#frame(\d+)$")


def build_whitelist(config: PipelineConfig) -> Whitelist:
    """Build merged whitelist from all configured tables."""
    whitelist = Whitelist()
    logger.info("开始构建联合白名单...")
    for table_name in config.table_names:
        before = len(whitelist)
        for index_dir in _resolve_index_dirs(table_name, config.channel_count):
            _load_dir_whitelist(index_dir, whitelist)
        logger.info(f"  {table_name}: {len(whitelist) - before:,d} 条")
    logger.info(
        f"联合白名单总量: {len(whitelist):,d} 条 "
        f"(整文件: {whitelist.path_only_count:,d}, "
        f"帧级: {whitelist.frame_entry_count:,d} 帧 / "
        f"{whitelist.frame_file_count:,d} 文件)"
    )
    return whitelist


def _resolve_index_dirs(table_name: str, channel_count: int) -> List[Path]:
    """Route table+channel to existing index directories."""
    root = Path(_cfg.DEDUP_INDEX_ROOT)
    ch_suffix = f"{channel_count}ch"
    dirs = [root / "ori" / ch_suffix] if table_name == TABLE_STATIC else [root / "slfmandhighna" / ch_suffix]
    if table_name != TABLE_STATIC and channel_count == 1:
        dirs.append(root / "slfmandhighna" / "slfm")
    existing = [d for d in dirs if d.exists()]
    for d in existing:
        logger.info(f"  路由: {table_name} -> {d}")
    if not existing:
        logger.warning(f"  无可用索引目录: {dirs}")
    return existing


def _load_dir_whitelist(index_dir: Path, whitelist: Whitelist) -> None:
    """Load all txt files under one index directory."""
    txt_files = sorted(index_dir.glob("*.txt"))
    if not txt_files:
        logger.warning(f"索引目录无 TXT 文件: {index_dir}")
        return
    for txt_path in txt_files:
        _load_single_txt(txt_path, whitelist)


def _load_single_txt(txt_path: Path, whitelist: Whitelist) -> int:
    """Load one txt whitelist file into Whitelist."""
    count = 0
    try:
        with open(txt_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parsed = _parse_whitelist_line(line)
                if parsed is None:
                    continue
                path, frame_idx = parsed
                whitelist.add_frame(path, frame_idx) if frame_idx is not None else whitelist.add_path(path)
                count += 1
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(f"白名单文件读取失败 {txt_path}: {exc}")
    logger.info(f"    已加载 {count:,d} 条: {txt_path.name}")
    return count


def _parse_whitelist_line(line: str) -> Optional[Tuple[str, Optional[int]]]:
    """Parse one whitelist line supporting path, #frameN, and path<TAB>idx."""
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split("\t")
    if len(parts) >= 2:
        path = parts[0].strip()
        try:
            return (path, int(parts[1].strip()))
        except ValueError:
            return (path, None)
    match = _FRAME_SUFFIX_RE.match(stripped)
    if match:
        return (match.group(1), int(match.group(2)))
    return (stripped, None)


