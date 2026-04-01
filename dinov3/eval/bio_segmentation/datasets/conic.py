"""
CoNIC (Colon Nuclei Identification and Counting) dataset loader.

Dataset stored as NumPy arrays (not individual image files):
    images.npy  : [N, 256, 256, 3]   uint8   RGB images (0-255)
    labels.npy  : [N, 256, 256, 2]   int32
                  channel 0 = instance map  (0 = background)
                  channel 1 = semantic class (0-6)
                      0  background
                      1  neutrophil
                      2  epithelial
                      3  lymphocyte
                      4  plasma cell
                      5  eosinophil
                      6  connective tissue

A companion ``counts.csv`` (unused here) gives per-sample cell-type counts.

Usage:
    from dinov3.eval.bio_segmentation.datasets.conic import CoNICDataset, get_conic_paths
    indices = get_conic_paths('/data1/xuzijing/dataset/conic/extracted', 'train')
    dataset = CoNICDataset(
        images_npy='/data1/xuzijing/dataset/conic/extracted/images.npy',
        labels_npy='/data1/xuzijing/dataset/conic/extracted/labels.npy',
        indices=indices,
    )
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

NUM_CLASSES = 7          # 0=bg, 1-6 = cell types
CLASS_NAMES = [
    'background', 'neutrophil', 'epithelial',
    'lymphocyte', 'plasma_cell', 'eosinophil', 'connective',
]


class CoNICDataset(Dataset):
    """
    CoNIC array-based dataset.

    Returns per sample:
        img_tensor      : [3, H, W] float32 in [0, 1]
        semantic_tensor : [H, W] int64, class IDs 0-6
        instance_tensor : [H, W] int64, instance IDs (0 = bg)
    """

    def __init__(
        self,
        images_npy: str,
        labels_npy: str,
        indices: Optional[List[int]] = None,
        size: Tuple[int, int] = (256, 256),
        augment: bool = False,
    ):
        """
        Args:
            images_npy : path to images.npy
            labels_npy : path to labels.npy
            indices    : sample indices to use (None = all)
            size       : output (H, W) - images are already 256×256
            augment    : random horizontal/vertical flips
        """
        logger.info(f"Loading CoNIC arrays from {images_npy} ...")
        self.images = np.load(images_npy, mmap_mode='r')   # [N, 256, 256, 3]
        self.labels = np.load(labels_npy, mmap_mode='r')   # [N, 256, 256, 2]

        if indices is None:
            indices = list(range(len(self.images)))
        self.indices = indices
        self.size    = size
        self.augment = augment

        logger.info(f"CoNIC: {len(self.indices)} samples, size={size}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        img  = self.images[i].copy().astype(np.float32) / 255.0   # [H, W, 3]
        inst = self.labels[i, :, :, 0].copy().astype(np.int64)    # [H, W]
        sem  = self.labels[i, :, :, 1].copy().astype(np.int64)    # [H, W]

        if self.size is None:
            h, w = img.shape[:2]
        else:
            h, w = self.size
        if img.shape[:2] != (h, w):
            img  = cv2.resize(img,  (w, h), interpolation=cv2.INTER_LINEAR)
            inst = cv2.resize(inst.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(np.int64)
            sem  = cv2.resize(sem.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(np.int64)

        if self.augment:
            if np.random.rand() > 0.5:
                img  = np.flip(img,  axis=1).copy()
                inst = np.flip(inst, axis=1).copy()
                sem  = np.flip(sem,  axis=1).copy()
            if np.random.rand() > 0.5:
                img  = np.flip(img,  axis=0).copy()
                inst = np.flip(inst, axis=0).copy()
                sem  = np.flip(sem,  axis=0).copy()

        img_t  = torch.from_numpy(img).permute(2, 0, 1).float()   # [3, H, W]
        sem_t  = torch.from_numpy(sem).long()                      # [H, W]
        inst_t = torch.from_numpy(inst).long()                     # [H, W]
        return img_t, sem_t, inst_t

    def get_semantic_mask(self, idx: int) -> np.ndarray:
        """Return (H, W) semantic class map (no resize)."""
        i = self.indices[idx]
        return self.labels[i, :, :, 1].copy().astype(np.int64)

    def get_instance_map(self, idx: int) -> np.ndarray:
        """Return (H, W) integer instance map (no resize)."""
        i = self.indices[idx]
        return self.labels[i, :, :, 0].copy().astype(np.int64)


# ============================================================================
# Index splitting
# ============================================================================

def get_conic_paths(
    data_root: str,
    split: str = 'train',
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed: int = 42,
) -> Tuple[str, str, List[int]]:
    """
    Return (images_npy_path, labels_npy_path, indices) for the requested split.

    Since CoNIC has no official train/val split file, we do a random 80/10/10 split.
    If an ``indices_<split>.npy`` file exists in data_root, we use it instead.

    Args:
        data_root   : directory containing images.npy and labels.npy
        split       : 'train', 'val', or 'test'
        train_ratio : fraction of samples for training (when auto-splitting)
        val_ratio   : fraction for validation (remaining → test)
        seed        : RNG seed for reproducibility

    Returns:
        (images_npy, labels_npy, indices)
    """
    import os, glob as _glob
    EXTS = ('*.npy',)

    # Locate images.npy / labels.npy (may be in a sub-directory after extraction)
    def _find(root, name):
        cands = sorted(_glob.glob(os.path.join(root, '**', name), recursive=True))
        return cands[0] if cands else None

    images_npy = _find(data_root, 'images.npy')
    labels_npy = _find(data_root, 'labels.npy')
    if images_npy is None or labels_npy is None:
        raise FileNotFoundError(
            f"Cannot find images.npy or labels.npy under {data_root}"
        )

    # Check for pre-saved index files
    idx_file = os.path.join(data_root, f'indices_{split}.npy')
    if os.path.exists(idx_file):
        indices = np.load(idx_file).tolist()
        logger.info(f"[CoNIC {split}] Loaded {len(indices)} indices from {idx_file}")
        return images_npy, labels_npy, indices

    # Auto-split
    total = len(np.load(images_npy, mmap_mode='r'))
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(total)

    n_train = int(total * train_ratio)
    n_val   = int(total * val_ratio)

    split_indices = {
        'train': perm[:n_train].tolist(),
        'val':   perm[n_train:n_train + n_val].tolist(),
        'test':  perm[n_train + n_val:].tolist(),
    }
    if split not in split_indices:
        raise ValueError(f"Unknown split '{split}'. Choose from 'train', 'val', 'test'.")

    indices = split_indices[split]
    # Save for reproducibility
    np.save(os.path.join(data_root, f'indices_{split}.npy'), np.array(indices))
    logger.info(f"[CoNIC {split}] {len(indices)}/{total} samples")
    return images_npy, labels_npy, indices
