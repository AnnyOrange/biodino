"""
BBBC038 (Data Science Bowl 2018) dataset loader.

Dataset layout (after extraction):
    <root>/
        stage1_train/
            <sample_id>/
                images/   <sample_id>.png
                masks/    <inst_0>.png  <inst_1>.png  ...
        stage1_test/
            <sample_id>/
                images/   <sample_id>.png

Each training sample has an arbitrary number of per-instance binary PNG masks.
We merge them into a single integer instance map (0 = background, 1..N = instances).

Returns
-------
    img_tensor  : [3, H, W] float32 in [0, 1]
    mask_tensor : [H, W] int64, pixel values = instance IDs (0 = bg)

Usage:
    from dinov3.eval.bio_segmentation.datasets.bbbc038 import BBBC038Dataset, get_bbbc038_paths
    img_paths, mask_dirs = get_bbbc038_paths('/data1/xuzijing/dataset/bbbc038/extracted', 'train')
    dataset = BBBC038Dataset(img_paths, mask_dirs, size=(448, 448))
"""

import logging
import os
from glob import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np

import torch
from dinov3.utils.bio_io import read_bio_image_as_numpy
from dinov3.eval.bio_segmentation.preprocessing import apply_preprocessing
from .base import BioSegDataset

logger = logging.getLogger(__name__)


class BBBC038Dataset(BioSegDataset):
    """
    DSB-2018 / BBBC038 binary cell-instance segmentation dataset.

    img_paths  : list of paths to the single .png image per sample
    mask_paths : list of *directories* containing per-instance mask PNGs

    __getitem__ returns a 3-tuple:
        img_tensor  : [3, H, W] float32 in [0, 1]
        sem_tensor  : [H, W] int64  binary (0=bg, 1=cell)
        inst_tensor : [H, W] int64  instance IDs (0=bg, 1..N)
    """

    def load_image(self, path: str) -> np.ndarray:
        """Return (H, W, 3) float32 array in [0, 1]."""
        img = read_bio_image_as_numpy(path, target_channels=3, normalize=True)
        return img

    def _load_instance_map(self, mask_dir: str) -> np.ndarray:
        """
        Merge per-instance PNG files in *mask_dir* into an integer instance map.
        Returns (H, W) int64 with values 0 (background) .. N (instance IDs).
        """
        inst_files = sorted(
            glob(os.path.join(mask_dir, '*.png')) +
            glob(os.path.join(mask_dir, '*.PNG'))
        )
        if not inst_files:
            raise FileNotFoundError(f"No instance masks in: {mask_dir}")

        first = cv2.imread(inst_files[0], cv2.IMREAD_GRAYSCALE)
        if first is None:
            raise ValueError(f"Cannot read mask: {inst_files[0]}")
        instance_map = np.zeros(first.shape, dtype=np.int64)

        for inst_id, fpath in enumerate(inst_files, start=1):
            m = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            instance_map[m > 0] = inst_id

        return instance_map

    def load_mask(self, path: str) -> np.ndarray:
        """
        Return binary semantic mask (0=background, 1=cell).
        Satisfies the BioSegDataset base-class contract.
        For per-instance IDs use get_instance_map().
        """
        return (self._load_instance_map(path) > 0).astype(np.int64)

    def get_instance_map(self, idx: int) -> np.ndarray:
        """Return the raw (H, W) integer instance map at native resolution."""
        return self._load_instance_map(self.mask_paths[idx])

    def __getitem__(self, idx: int):
        """
        Override to return a 3-tuple (img, sem, inst) so that the feature
        extractor can cache both semantic and instance annotations.
        """
        img = self.load_image(self.img_paths[idx])

        # Ensure 3-channel
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        img = apply_preprocessing(img, mode=self.mode)

        inst_map = self._load_instance_map(self.mask_paths[idx])

        # Only resize when a fixed output size is requested (feature-extractor
        # caching mode).  For Mask2Former, size=None keeps native resolution so
        # that sliding-window evaluation sees the full-resolution image.
        if self.size is not None:
            h, w = self.size
            img      = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            inst_map = cv2.resize(inst_map.astype(np.float32), (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int64)

        sem_map = (inst_map > 0).astype(np.int64)

        if self.augment:
            if np.random.rand() > 0.5:
                img      = np.flip(img,      axis=1).copy()
                inst_map = np.flip(inst_map, axis=1).copy()
                sem_map  = np.flip(sem_map,  axis=1).copy()
            if np.random.rand() > 0.5:
                img      = np.flip(img,      axis=0).copy()
                inst_map = np.flip(inst_map, axis=0).copy()
                sem_map  = np.flip(sem_map,  axis=0).copy()

        img_t  = torch.from_numpy(img).permute(2, 0, 1).float()
        sem_t  = torch.from_numpy(sem_map).long()
        inst_t = torch.from_numpy(inst_map).long()
        return img_t, sem_t, inst_t


# ============================================================================
# Path discovery
# ============================================================================

def get_bbbc038_paths(
    data_root: str,
    split: str = 'train',
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Discover BBBC038 image / mask-directory pairs.

    BBBC038 has no official validation split.  When split='val' is requested
    a deterministic random subset (``val_ratio`` fraction of training data)
    is held out.  The remaining training samples are returned for split='train'.
    The held-out indices are cached to a file for reproducibility.

    Args:
        data_root : root directory (contains ``stage1_train``, ``stage1_test``).
        split     : 'train', 'val', or 'test'.
        val_ratio : fraction of training samples used for validation.
        seed      : RNG seed.

    Returns:
        (img_paths, mask_dirs) - mask_dirs are directories with per-instance masks.
    """
    need_train = split in ('train', 'val')
    raw_split  = 'train' if need_train else 'test'

    split_map = {
        'train': 'stage1_train',
        'test':  'stage1_test',
    }
    split_dir = os.path.join(data_root, split_map[raw_split])
    if not os.path.isdir(split_dir):
        candidates = glob(os.path.join(data_root, '*', split_map[raw_split]))
        if candidates:
            split_dir = candidates[0]
        else:
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Collect all train samples
    all_imgs, all_masks = [], []
    for sample_id in sorted(os.listdir(split_dir)):
        sample_dir = os.path.join(split_dir, sample_id)
        if not os.path.isdir(sample_dir):
            continue
        img_dir  = os.path.join(sample_dir, 'images')
        mask_dir = os.path.join(sample_dir, 'masks')

        imgs = glob(os.path.join(img_dir, '*.png')) + glob(os.path.join(img_dir, '*.tif'))
        if not imgs:
            continue
        if raw_split == 'train' and not os.path.isdir(mask_dir):
            continue

        all_imgs.append(sorted(imgs)[0])
        all_masks.append(mask_dir if raw_split == 'train' else '')

    if split == 'test':
        logger.info(f"[BBBC038 test] {len(all_imgs)} samples in {split_dir}")
        return all_imgs, all_masks

    # Train / val split
    total = len(all_imgs)
    idx_file = os.path.join(data_root, 'bbbc038_val_indices.npy')
    if os.path.exists(idx_file):
        val_idx = set(np.load(idx_file).tolist())
    else:
        rng     = np.random.default_rng(seed)
        perm    = rng.permutation(total)
        n_val   = max(1, int(total * val_ratio))
        val_idx = set(perm[:n_val].tolist())
        np.save(idx_file, np.array(sorted(val_idx)))

    if split == 'val':
        sel = [i for i in range(total) if i in val_idx]
    else:  # 'train'
        sel = [i for i in range(total) if i not in val_idx]

    img_paths  = [all_imgs[i]  for i in sel]
    mask_dirs  = [all_masks[i] for i in sel]
    logger.info(f"[BBBC038 {split}] {len(img_paths)}/{total} samples")
    return img_paths, mask_dirs
