"""
Cellpose dataset loader for bio-segmentation evaluation.

Dataset format:
    <root>/
        train/train/
            <name>_img.png     (16-bit RGB: Blue=empty, Green=cytoplasm, Red=nucleus)
            <name>_masks.png
        test/test/
            ...

Usage:
    from dinov3.eval.bio_segmentation.datasets.cellpose import CellposeDataset, get_cellpose_paths
    img_paths, mask_paths = get_cellpose_paths('/data/Cellpose', split='train')
    dataset = CellposeDataset(img_paths, mask_paths, mode='hybrid')
"""

import logging
import os
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np

from .base import BioSegDataset

logger = logging.getLogger(__name__)


class CellposeDataset(BioSegDataset):
    """
    16-bit RGB PNG cell segmentation images from the Cellpose benchmark.

    Each image is (H, W, 3) uint16; channels are per-cell-component stains.
    cv2.IMREAD_UNCHANGED is used to preserve bit depth; BGR->RGB is applied.
    Masks are uint16/uint8 PNG; any nonzero pixel is treated as foreground.
    """

    def load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_mask(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Cannot read mask: {path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return (mask > 0).astype(np.int64)


def get_cellpose_paths(data_root: str, split: str = 'train') -> Tuple[List[str], List[str]]:
    """
    Discover Cellpose image/mask path pairs.

    The Cellpose dataset uses a doubled directory layout:
        <root>/<split>/<split>/<name>_img.png
        <root>/<split>/<split>/<name>_masks.png

    Falls back to <root>/<split>/ if the doubled layout is not found.

    Args:
        data_root: path to the Cellpose root directory.
        split: 'train' or 'test'.

    Returns:
        (img_paths, mask_paths) - sorted, matched pairs.
    """
    split_dir = os.path.join(data_root, split, split)
    if not os.path.exists(split_dir):
        split_dir = os.path.join(data_root, split)
    if not os.path.exists(split_dir):
        raise ValueError(f"Cellpose split directory not found: {split_dir}")

    img_paths = sorted(glob(os.path.join(split_dir, '*_img.png')))
    mask_paths = sorted(glob(os.path.join(split_dir, '*_masks.png')))

    logger.info(f"[Cellpose {split}] Found {len(img_paths)} images, {len(mask_paths)} masks")
    if len(img_paths) != len(mask_paths):
        raise ValueError(
            f"Mismatch: {len(img_paths)} images vs {len(mask_paths)} masks in {split_dir}"
        )
    return img_paths, mask_paths
