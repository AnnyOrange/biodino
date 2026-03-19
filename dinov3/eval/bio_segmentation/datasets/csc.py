"""
CSC (Cell Segmentation Challenge) dataset loader for bio-segmentation evaluation.

Dataset directory layout:
    <root>/
        Training-labeled/Training-labeled/images/   + labels/
        Tuning/Tuning/images/                       + labels/
        Testing/Testing/Public/images/              + labels/
        Testing/Testing/Hidden/images/              + osilab_seg/osilab_seg/

Supported image formats: tif, tiff, png, bmp, jpg.
Masks are instance-segmentation TIFF files (*_label.tiff); any nonzero ID = foreground.

Usage:
    from dinov3.eval.bio_segmentation.datasets.csc import CSCDataset, get_csc_paths
    img_paths, mask_paths = get_csc_paths('/data/CSC', split='train')
    dataset = CSCDataset(img_paths, mask_paths, mode='hybrid')
"""

import logging
import os
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
import tifffile

from .base import BioSegDataset

logger = logging.getLogger(__name__)


# ============================================================================
# File I/O helpers
# ============================================================================

def load_image_multi_format(img_path: str) -> np.ndarray:
    """
    Load an image from tif/tiff/png/bmp/jpg, preserving bit depth.

    Returns an (H, W) or (H, W, C) array in RGB channel order.
    """
    ext = os.path.splitext(img_path)[1].lower()
    if ext in ('.tif', '.tiff'):
        img = tifffile.imread(img_path)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    return img


def load_mask_tiff(mask_path: str) -> np.ndarray:
    """Load a TIFF instance mask (uint16); returns the raw array."""
    mask = tifffile.imread(mask_path)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    return mask


def _get_base_name(fname: str) -> str:
    """Strip dataset-specific suffixes and extensions to get a comparable base name."""
    base = fname.rsplit('_label', 1)[0]
    for ext in ('.bmp', '.png', '.tif', '.tiff', '.jpg', '.jpeg'):
        base = base.replace(ext, '')
    return base


# ============================================================================
# Dataset class
# ============================================================================

class CSCDataset(BioSegDataset):
    """
    Multi-format cell segmentation dataset from the Cell Segmentation Challenge.

    Images may be 8-bit or 16-bit, single- or multi-channel (gray, RGB, RGBA, TIFF).
    Masks are instance-segmentation TIFF files; converted to binary (0/1) on load.
    """

    def load_image(self, path: str) -> np.ndarray:
        return load_image_multi_format(path)

    def load_mask(self, path: str) -> np.ndarray:
        instance_mask = load_mask_tiff(path)
        return (instance_mask > 0).astype(np.int64)


# ============================================================================
# Path discovery
# ============================================================================

def get_csc_paths(data_root: str, split: str = 'train') -> Tuple[List[str], List[str]]:
    """
    Discover CSC image/mask path pairs for the requested split.

    Args:
        data_root: root directory of the CSC dataset.
        split: one of 'train', 'tune', 'test' / 'test_public', 'test_hidden'.

    Returns:
        (img_paths, mask_paths) - sorted, matched pairs.
    """
    _split_dirs = {
        'train':       ('Training-labeled', 'Training-labeled'),
        'tune':        ('Tuning', 'Tuning'),
        'test':        ('Testing', 'Testing', 'Public'),
        'test_public': ('Testing', 'Testing', 'Public'),
        'test_hidden': ('Testing', 'Testing', 'Hidden'),
    }
    if split not in _split_dirs:
        raise ValueError(
            f"Unknown split '{split}'. Choose from: {list(_split_dirs)}"
        )

    parts = _split_dirs[split]
    split_root = os.path.join(data_root, *parts)

    img_dir = os.path.join(split_root, 'images')

    if split == 'test_hidden':
        label_dir = os.path.join(split_root, 'osilab_seg', 'osilab_seg')
    else:
        label_dir = os.path.join(split_root, 'labels')

    if not os.path.exists(img_dir):
        raise ValueError(f"Image directory not found: {img_dir}")
    if not os.path.exists(label_dir):
        raise ValueError(f"Label directory not found: {label_dir}")

    img_files: List[str] = []
    for pattern in ('*.tif', '*.tiff', '*.png', '*.bmp', '*.jpg', '*.jpeg',
                    '*.TIF', '*.TIFF', '*.PNG', '*.BMP', '*.JPG', '*.JPEG'):
        img_files.extend(glob(os.path.join(img_dir, pattern)))
    img_files = sorted(f for f in img_files if not os.path.basename(f).startswith('.'))

    label_files = sorted(
        glob(os.path.join(label_dir, '*_label.tiff')) +
        glob(os.path.join(label_dir, '*_label.tif'))
    )

    img_by_base = {_get_base_name(os.path.basename(f)): f for f in img_files}
    lbl_by_base = {_get_base_name(os.path.basename(f)): f for f in label_files}
    matched = sorted(set(img_by_base) & set(lbl_by_base))

    img_paths = [img_by_base[b] for b in matched]
    mask_paths = [lbl_by_base[b] for b in matched]

    logger.info(
        f"[CSC {split}] {len(img_files)} images, {len(label_files)} labels, "
        f"{len(img_paths)} matched pairs"
    )
    return img_paths, mask_paths
