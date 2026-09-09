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

import csv
import glob
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from dinov3.eval.bio_segmentation.constants import MICRO_RGB_MEAN, MICRO_RGB_STD

from .base import resize_image_and_masks

logger = logging.getLogger(__name__)

NUM_CLASSES = 7          # 0=bg, 1-6 = cell types
FORMAL_SPLIT_PROTOCOL = 'official-baseline-fold0-nested-v1'
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
        size: Optional[Tuple[int, int]] = (256, 256),
        resize_mode: str = "stretch",
        augment: bool = False,
        rgb_mean=MICRO_RGB_MEAN,
        rgb_std=MICRO_RGB_STD,
        do_normalize: bool = True,
    ):
        """
        Args:
            images_npy : path to images.npy
            labels_npy : path to labels.npy
            indices    : sample indices to use (None = all)
            size       : output (H, W) - images are already 256×256
            resize_mode: "stretch" for direct resize, "pad" for keep-aspect
                         long-side resize plus centered padding.
            augment    : random horizontal/vertical flips
            rgb_mean / rgb_std / do_normalize : fixed normalisation after scaling to [0, 1].
        """
        logger.info(f"Loading CoNIC arrays from {images_npy} ...")
        self.images = np.load(images_npy, mmap_mode='r')   # [N, 256, 256, 3]
        self.labels = np.load(labels_npy, mmap_mode='r')   # [N, 256, 256, 2]

        if indices is None:
            indices = list(range(len(self.images)))
        self.indices = indices
        self.size = size
        self.resize_mode = resize_mode
        self.augment = augment
        self.do_normalize = do_normalize
        self.rgb_mean = torch.tensor(rgb_mean, dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor(rgb_std, dtype=torch.float32).view(3, 1, 1)

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
            img, resized_masks, _ = resize_image_and_masks(
                img,
                [sem.astype(np.int64), inst.astype(np.int64)],
                (h, w),
                mode=self.resize_mode,
                mask_pad_values=[255, 0],
            )
            sem = resized_masks[0].astype(np.int64)
            inst = resized_masks[1].astype(np.int64)

        if self.augment:
            if np.random.rand() > 0.5:
                img  = np.flip(img,  axis=1).copy()
                inst = np.flip(inst, axis=1).copy()
                sem  = np.flip(sem,  axis=1).copy()
            if np.random.rand() > 0.5:
                img  = np.flip(img,  axis=0).copy()
                inst = np.flip(inst, axis=0).copy()
                sem  = np.flip(sem,  axis=0).copy()

        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        if self.do_normalize:
            img_t = (img_t - self.rgb_mean) / self.rgb_std
        sem_t = torch.from_numpy(sem).long()
        inst_t = torch.from_numpy(inst).long()
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
    split_protocol: str = 'legacy-random',
) -> Tuple[str, str, List[int]]:
    """
    Return (images_npy_path, labels_npy_path, indices) for the requested split.

    ``official-baseline-fold0-nested-v1`` reproduces the public CoNIC baseline's
    source-level, cohort-stratified 80/20 split (seed 5, fold 0), then makes a
    fixed 87.5/12.5 split inside the 80% development-train sources.  The result
    is approximately 70/10/20 and, critically, patches from one source image
    never cross train/val/test.  The challenge test labels remain hidden, so the
    20% partition is a public development holdout rather than the challenge test.

    ``legacy-random`` retains the historical random patch-level 80/10/10 path
    only for reading old experiments.  It must not be used for formal results.

    Args:
        data_root   : directory containing images.npy and labels.npy
        split       : 'train', 'val', or 'test'
        train_ratio : fraction of samples for training (when auto-splitting)
        val_ratio   : fraction for validation (remaining → test)
        seed        : RNG seed for reproducibility

    Returns:
        (images_npy, labels_npy, indices)
    """
    # Locate images.npy / labels.npy (may be in a sub-directory after extraction)
    def _find(root, name):
        cands = sorted(glob.glob(os.path.join(root, '**', name), recursive=True))
        return cands[0] if cands else None

    images_npy = _find(data_root, 'images.npy')
    labels_npy = _find(data_root, 'labels.npy')
    if images_npy is None or labels_npy is None:
        raise FileNotFoundError(
            f"Cannot find images.npy or labels.npy under {data_root}"
        )

    if split not in {'train', 'val', 'test'}:
        raise ValueError(f"Unknown split '{split}'. Choose from 'train', 'val', 'test'.")

    if split_protocol == FORMAL_SPLIT_PROTOCOL:
        patch_info_path = _find(data_root, 'patch_info.csv')
        if patch_info_path is None:
            raise FileNotFoundError(
                f"{FORMAL_SPLIT_PROTOCOL} requires patch_info.csv under {data_root}"
            )
        indices_by_split = _official_baseline_nested_indices(patch_info_path)
        indices = indices_by_split[split]
        logger.info(
            "[CoNIC %s] protocol=%s samples=%d patch_info=%s",
            split,
            split_protocol,
            len(indices),
            patch_info_path,
        )
        return images_npy, labels_npy, indices

    if split_protocol != 'legacy-random':
        raise ValueError(
            f"Unknown CoNIC split_protocol={split_protocol!r}; choices are "
            f"'legacy-random' and {FORMAL_SPLIT_PROTOCOL!r}."
        )

    # Check for pre-saved legacy random index files
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
    indices = split_indices[split]
    # Save for reproducibility
    np.save(os.path.join(data_root, f'indices_{split}.npy'), np.array(indices))
    logger.info(f"[CoNIC {split}] {len(indices)}/{total} samples")
    return images_npy, labels_npy, indices


def _official_baseline_nested_indices(patch_info_path: str) -> dict[str, List[int]]:
    """Return deterministic source-disjoint CoNIC development partitions.

    The outer split is an exact implementation of the official CoNIC baseline
    ``generate_split.py``: source id is the prefix before ``-``, cohort is the
    prefix before ``_``, StratifiedShuffleSplit has 10 folds, train_size=.8,
    test_size=.2 and random_state=5, and fold 0 is selected.  A nested split of
    the outer training sources creates validation data without touching the
    outer holdout.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    patch_names: List[str] = []
    with open(patch_info_path, newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Empty CoNIC patch_info.csv: {patch_info_path}")
        column = 'patch_info' if 'patch_info' in reader.fieldnames else reader.fieldnames[0]
        patch_names = [str(row[column]).strip() for row in reader if str(row[column]).strip()]
    if not patch_names:
        raise ValueError(f"No patch identifiers in {patch_info_path}")

    image_sources = np.asarray([name.split('-')[0] for name in patch_names])
    unique_sources = np.asarray(sorted(set(image_sources.tolist())))
    cohorts = np.asarray([source.split('_')[0] for source in unique_sources])

    outer = StratifiedShuffleSplit(
        n_splits=10,
        train_size=0.8,
        test_size=0.2,
        random_state=5,
    )
    outer_train_pos, outer_test_pos = next(outer.split(unique_sources, cohorts))
    outer_train_sources = unique_sources[outer_train_pos]
    test_sources = unique_sources[outer_test_pos]

    nested_cohorts = np.asarray([source.split('_')[0] for source in outer_train_sources])
    nested = StratifiedShuffleSplit(
        n_splits=1,
        train_size=0.875,
        test_size=0.125,
        random_state=5,
    )
    train_pos, val_pos = next(nested.split(outer_train_sources, nested_cohorts))
    train_sources = set(outer_train_sources[train_pos].tolist())
    val_sources = set(outer_train_sources[val_pos].tolist())
    test_sources_set = set(test_sources.tolist())

    if train_sources & val_sources or train_sources & test_sources_set or val_sources & test_sources_set:
        raise AssertionError("CoNIC source leakage detected while constructing formal split")

    result = {'train': [], 'val': [], 'test': []}
    for idx, source in enumerate(image_sources.tolist()):
        if source in train_sources:
            result['train'].append(idx)
        elif source in val_sources:
            result['val'].append(idx)
        elif source in test_sources_set:
            result['test'].append(idx)
        else:
            raise AssertionError(f"Unassigned CoNIC source: {source}")
    return result
