"""Whitelist filtering and frame-level expansion."""

import logging
import random
from dataclasses import replace
from typing import List

from .config import ImageMeta
from .dedup_types import Whitelist

logger = logging.getLogger("dinov3")


def filter_and_shuffle(metas: List[ImageMeta], whitelist: Whitelist) -> List[ImageMeta]:
    """Filter metas by whitelist and expand frame-level entries."""
    valid: List[ImageMeta] = []
    for meta in metas:
        if meta.file_path in whitelist.path_set:
            valid.append(meta)
            continue
        if meta.file_path not in whitelist.frame_map:
            continue
        for frame_idx in sorted(whitelist.frame_map[meta.file_path]):
            valid.append(replace(meta, frame_idx=frame_idx))
    random.shuffle(valid)
    logger.info(f"白名单过滤: {len(metas):,d} -> {len(valid):,d} (已打乱)")
    return valid


