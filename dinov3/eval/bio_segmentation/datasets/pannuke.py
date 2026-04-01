"""
PanNuke dataset loader.

PanNuke provides H&E images and multi-class nucleus instance annotations.
Data is split into 3 folds; each fold is a directory containing:
    Fold <k>/
        images/
            fold<k>/
                images.npy   [N, 256, 256, 3]  float64  RGB  (0-255 range)
        masks/
            fold<k>/
                masks.npy    [N, 256, 256, 6]  float64
                             channels 0-4 : instance maps for 5 cell types
                             channel 5   : background (1 where no cell)
                types.npy    [N]  strings, tissue type per image (optional)

Cell-type mapping (channels 0-4):
    0 Neoplastic
    1 Inflammatory
    2 Connective / Soft Tissue
    3 Dead
    4 Epithelial

Usage:
    from dinov3.eval.bio_segmentation.datasets.pannuke import PanNukeDataset, get_pannuke_paths
    fold_dirs = get_pannuke_paths('/data1/xuzijing/dataset/pannuke/extracted', folds=[1,2,3])
    dataset = PanNukeDataset(fold_dirs, split_folds=[1,2], size=(256,256))
"""

import logging
import os
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

NUM_CLASSES = 6   # 0=bg, 1-5 = Neoplastic/Inflammatory/Connective/Dead/Epithelial
CLASS_NAMES = ['background', 'neoplastic', 'inflammatory', 'connective', 'dead', 'epithelial']


def _load_fold(fold_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images.npy and masks.npy from a single fold directory.

    Searches recursively for the files.
    Returns:
        images : [N, 256, 256, 3] float32 in [0, 1]
        masks  : [N, 256, 256, 6] float32 (raw instance id arrays)
    """
    import glob as _glob

    img_npy  = _glob.glob(os.path.join(fold_dir, '**', 'images.npy'), recursive=True)
    mask_npy = _glob.glob(os.path.join(fold_dir, '**', 'masks.npy'),  recursive=True)

    if not img_npy:
        raise FileNotFoundError(f"images.npy not found under {fold_dir}")
    if not mask_npy:
        raise FileNotFoundError(f"masks.npy not found under {fold_dir}")

    images = np.load(img_npy[0]).astype(np.float32)  # [N, 256, 256, 3]
    masks  = np.load(mask_npy[0]).astype(np.float32) # [N, 256, 256, 6]

    # Normalise images to [0, 1] if they appear to be in [0, 255]
    if images.max() > 1.5:
        images = images / 255.0

    return images, masks


class PanNukeDataset(Dataset):
    """
    PanNuke multi-class nucleus instance segmentation dataset.

    Returns per sample:
        img_tensor      : [3, H, W] float32 in [0, 1]
        semantic_tensor : [H, W] int64  class IDs  (0=bg, 1-5=cell types)
        instance_tensor : [H, W] int64  instance IDs (0=bg, 1..N unique across classes)
    """

    def __init__(
        self,
        fold_dirs: Dict[int, str],
        split_folds: Optional[List[int]] = None,
        size: Tuple[int, int] = (256, 256),
        augment: bool = False,
    ):
        """
        Args:
            fold_dirs   : dict {fold_number: fold_directory_path}
            split_folds : list of fold numbers to include (None = all folds)
            size        : output (H, W)
            augment     : random horizontal/vertical flips
        """
        if split_folds is None:
            split_folds = list(fold_dirs.keys())

        all_images, all_masks = [], []
        for fold_k in sorted(split_folds):
            if fold_k not in fold_dirs:
                raise ValueError(f"Fold {fold_k} not in fold_dirs")
            imgs, msks = _load_fold(fold_dirs[fold_k])
            all_images.append(imgs)
            all_masks.append(msks)
            logger.info(f"[PanNuke] Loaded fold {fold_k}: {len(imgs)} samples")

        self.images = np.concatenate(all_images, axis=0)  # [N_total, 256, 256, 3]
        self.masks  = np.concatenate(all_masks,  axis=0)  # [N_total, 256, 256, 6]
        self.size   = size
        self.augment = augment
        logger.info(f"[PanNuke] Total samples: {len(self.images)}")

    def __len__(self) -> int:
        return len(self.images)

    def _masks_to_semantic_instance(self, mask6: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert raw 6-channel mask to semantic + unified instance maps.

        mask6 : [H, W, 6] float32 – channels 0-4 are instance maps per class, ch5=bg
        Returns:
            semantic : [H, W] int64  (0=bg, 1-5=class)
            instance : [H, W] int64  (0=bg, unique IDs across all classes)
        """
        h, w = mask6.shape[:2]
        semantic = np.zeros((h, w), dtype=np.int64)
        instance = np.zeros((h, w), dtype=np.int64)
        next_inst_id = 1

        for cls_idx in range(5):          # channels 0-4
            ch = mask6[:, :, cls_idx]
            unique_ids = np.unique(ch)
            for uid in unique_ids:
                if uid == 0:
                    continue
                region = ch == uid
                semantic[region] = cls_idx + 1   # 1-5
                instance[region] = next_inst_id
                next_inst_id += 1

        return semantic, instance

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img   = self.images[idx].copy()              # [256, 256, 3]
        mask6 = self.masks[idx].copy()              # [256, 256, 6]

        sem, inst = self._masks_to_semantic_instance(mask6)

        if self.size is None:
            h, w = img.shape[:2]
        else:
            h, w = self.size
        if img.shape[:2] != (h, w):
            img  = cv2.resize(img,  (w, h), interpolation=cv2.INTER_LINEAR)
            sem  = cv2.resize(sem.astype(np.float32),  (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(np.int64)
            inst = cv2.resize(inst.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(np.int64)

        if self.augment:
            if np.random.rand() > 0.5:
                img  = np.flip(img,  axis=1).copy()
                sem  = np.flip(sem,  axis=1).copy()
                inst = np.flip(inst, axis=1).copy()
            if np.random.rand() > 0.5:
                img  = np.flip(img,  axis=0).copy()
                sem  = np.flip(sem,  axis=0).copy()
                inst = np.flip(inst, axis=0).copy()

        img_t  = torch.from_numpy(img).permute(2, 0, 1).float()
        sem_t  = torch.from_numpy(sem).long()
        inst_t = torch.from_numpy(inst).long()
        return img_t, sem_t, inst_t

    def get_instance_map(self, idx: int) -> np.ndarray:
        """Return unified instance map at native resolution."""
        _, inst = self._masks_to_semantic_instance(self.masks[idx])
        return inst

    def get_semantic_mask(self, idx: int) -> np.ndarray:
        """Return semantic class map at native resolution."""
        sem, _ = self._masks_to_semantic_instance(self.masks[idx])
        return sem


# ============================================================================
# Path discovery
# ============================================================================

def get_pannuke_paths(
    data_root: str,
    folds: Optional[List[int]] = None,
) -> Dict[int, str]:
    """
    Discover PanNuke fold directories.

    Args:
        data_root : root directory (contains 'Fold 1', 'Fold 2', 'Fold 3' sub-dirs,
                    or 'fold1', 'fold2', 'fold3').
        folds     : list of fold numbers to include (default: [1, 2, 3]).

    Returns:
        dict {fold_number: fold_directory_path}
    """
    if folds is None:
        folds = [1, 2, 3]

    fold_dirs: Dict[int, str] = {}
    for k in folds:
        candidates = [
            os.path.join(data_root, f'Fold {k}'),
            os.path.join(data_root, f'fold{k}'),
            os.path.join(data_root, f'fold_{k}'),
            os.path.join(data_root, f'Fold_{k}'),
        ]
        # Also search one level deeper
        for parent in [data_root] + [os.path.join(data_root, d) for d in os.listdir(data_root)
                                     if os.path.isdir(os.path.join(data_root, d))]:
            for tmpl in [f'Fold {k}', f'fold{k}', f'fold_{k}']:
                candidates.append(os.path.join(parent, tmpl))

        for c in candidates:
            if os.path.isdir(c):
                fold_dirs[k] = c
                logger.info(f"[PanNuke] Fold {k} → {c}")
                break
        else:
            logger.warning(f"[PanNuke] Fold {k} directory not found under {data_root}")

    return fold_dirs
