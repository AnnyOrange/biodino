"""NeurIPS 2022 Multimodal CellSeg loader.

The local mirror stores split CSVs with image/mask paths relative to the
``neurips22_cellseg`` root.  Masks are instance-label images; the linear probe
uses a binary foreground mask like the other semantic segmentation benchmarks.
Very large WSI tiles are skipped here because the frozen-feature pipeline works
on fixed-size resized images and loading those whole-slide files can dominate or
exhaust memory in quick benchmark sweeps.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from dinov3.utils.bio_io import read_bio_image_as_numpy

from .base import BioSegDataset

logger = logging.getLogger(__name__)


class MultimodalCellSegDataset(BioSegDataset):
    """RGB/gray microscopy cell segmentation images with instance-label masks."""

    def load_image(self, path: str) -> np.ndarray:
        return read_bio_image_as_numpy(path, target_channels=3, normalize=False)

    def load_mask(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Cannot read mask: {path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return (mask > 0).astype(np.int64)


def _image_num_pixels(path: Path) -> int | None:
    try:
        from PIL import Image

        with Image.open(path) as pil_img:
            width, height = pil_img.size
        return int(width) * int(height)
    except Exception:
        return None


def get_multimodal_cellseg_paths(
    data_root: str,
    split: str = "train",
    max_pixels: int = 50_000_000,
) -> Tuple[List[str], List[str]]:
    """Read image/mask pairs from the prepared split CSVs.

    Args:
        data_root: ``.../Multimodal_CellSeg/neurips22_cellseg``.
        split: ``train``, ``val`` or ``test``.  ``test`` maps to the local
            source-heldout split.
        max_pixels: skip huge WSI files above this pixel count.
    """

    root = Path(data_root)
    split_file = {
        "train": root / "splits" / "train.csv",
        "val": root / "splits" / "val.csv",
        "test": root / "splits" / "test_source_heldout.csv",
    }.get(split)
    if split_file is None:
        raise ValueError(f"Unsupported Multimodal_CellSeg split: {split}")
    if not split_file.exists():
        raise FileNotFoundError(f"Multimodal_CellSeg split file not found: {split_file}")

    img_paths: List[str] = []
    mask_paths: List[str] = []
    skipped_missing = 0
    skipped_large = 0
    with split_file.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if str(row.get("has_mask", "1")) not in {"1", "True", "true"}:
                continue
            image_rel = row.get("image_path") or ""
            mask_rel = row.get("mask_path") or ""
            if not image_rel or not mask_rel:
                continue
            image_path = root / image_rel
            mask_path = root / mask_rel
            if not image_path.exists() or not mask_path.exists():
                skipped_missing += 1
                continue
            if row.get("source_dataset") == "WSI" or "WSI" in image_rel or "WSI" in mask_rel:
                skipped_large += 1
                continue
            n_pixels = _image_num_pixels(image_path)
            if n_pixels is None or n_pixels > max_pixels:
                skipped_large += 1
                continue
            img_paths.append(str(image_path))
            mask_paths.append(str(mask_path))

    logger.info(
        "[Multimodal_CellSeg %s] Found %d pairs (skipped_missing=%d, skipped_large=%d)",
        split,
        len(img_paths),
        skipped_missing,
        skipped_large,
    )
    if not img_paths:
        raise ValueError(f"No Multimodal_CellSeg pairs found for split={split} under {root}")
    return img_paths, mask_paths
