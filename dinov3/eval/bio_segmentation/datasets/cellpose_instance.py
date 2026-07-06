"""
Cellpose instance-map loader for the HoVerNet instance-segmentation track.

The existing ``CellposeDataset`` intentionally binarizes masks for semantic
segmentation probes. HoVerNet target generation needs the original instance IDs,
so this loader keeps ``*_masks.png`` as an instance map and returns
``(image, semantic_foreground, instance_map)``.
"""

from __future__ import annotations

import logging
import os
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dinov3.eval.bio_segmentation.constants import MICRO_RGB_MEAN, MICRO_RGB_STD
from dinov3.eval.bio_segmentation.datasets.base import resize_image_and_masks
from dinov3.utils.bio_io import _normalize_to_float32

logger = logging.getLogger(__name__)


def _split_dir(data_root: str, split: str) -> str | None:
    """Return the Cellpose doubled-layout split directory if it exists."""
    doubled = os.path.join(data_root, split, split)
    if os.path.exists(doubled):
        return doubled
    flat = os.path.join(data_root, split)
    if os.path.exists(flat):
        return flat
    return None


def _collect_pairs(data_root: str, splits: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for split in splits:
        split_dir = _split_dir(data_root, split)
        if split_dir is None:
            logger.warning("[Cellpose instance] split not found, skipping: %s", split)
            continue
        img_paths = sorted(glob(os.path.join(split_dir, "*_img.png")))
        for img_path in img_paths:
            mask_path = img_path.replace("_img.png", "_masks.png")
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
            else:
                logger.warning("[Cellpose instance] missing mask for %s", img_path)
    return sorted(pairs)


def get_cellpose_instance_paths(
    data_root: str,
    split: str = "train",
    val_fraction: float = 0.2,
    seed: int = 0,
    include_cyto2: bool = True,
) -> Tuple[List[str], List[str]]:
    """Discover Cellpose image/instance-mask pairs for instance segmentation.

    ``train`` and ``val`` are a deterministic split of the public training pool.
    By default that pool includes both ``train`` and ``train_cyto2`` when present,
    giving the decoder the largest public Cellpose supervision set. ``test`` uses
    the official ``test`` split only.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported Cellpose split: {split!r}")

    if split == "test":
        pairs = _collect_pairs(data_root, ["test"])
    else:
        source_splits = ["train", "train_cyto2"] if include_cyto2 else ["train"]
        pairs = _collect_pairs(data_root, source_splits)
        if len(pairs) > 1:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(pairs))
            val_count = max(1, int(round(len(pairs) * val_fraction)))
            val_idx = set(int(i) for i in order[:val_count])
            if split == "val":
                pairs = [p for i, p in enumerate(pairs) if i in val_idx]
            else:
                pairs = [p for i, p in enumerate(pairs) if i not in val_idx]

    if not pairs:
        raise ValueError(f"No Cellpose instance pairs found for split={split!r} under {data_root}")

    img_paths, mask_paths = zip(*pairs)
    logger.info("[Cellpose instance %s] Found %d image/mask pairs", split, len(pairs))
    return list(img_paths), list(mask_paths)


class CellposeInstanceDataset(Dataset):
    """Cellpose dataset preserving instance IDs for HoVerNet targets."""

    def __init__(
        self,
        img_paths: List[str],
        mask_paths: List[str],
        size: Tuple[int, int] | None = None,
        resize_mode: str = "stretch",
        augment: bool = False,
        rgb_mean=MICRO_RGB_MEAN,
        rgb_std=MICRO_RGB_STD,
        do_normalize: bool = True,
    ):
        if len(img_paths) != len(mask_paths):
            raise ValueError(f"Image count ({len(img_paths)}) != mask count ({len(mask_paths)})")
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.size = size
        self.resize_mode = resize_mode
        self.augment = augment
        self.do_normalize = do_normalize
        self.rgb_mean = torch.tensor(rgb_mean, dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor(rgb_std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.img_paths)

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def _load_instance_mask(path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Cannot read mask: {path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask.astype(np.int64)

    def __getitem__(self, idx: int):
        img = self._load_image(self.img_paths[idx])
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = _normalize_to_float32(img)

        inst = self._load_instance_mask(self.mask_paths[idx])
        sem = (inst > 0).astype(np.int64)

        if self.size is not None:
            img, resized_masks, _valid = resize_image_and_masks(
                img,
                [sem.astype(np.int64), inst.astype(np.int64)],
                self.size,
                mode=self.resize_mode,
                mask_pad_values=[255, 0],
            )
            sem = resized_masks[0].astype(np.int64)
            inst = resized_masks[1].astype(np.int64)

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
                sem = np.flip(sem, axis=1).copy()
                inst = np.flip(inst, axis=1).copy()
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=0).copy()
                sem = np.flip(sem, axis=0).copy()
                inst = np.flip(inst, axis=0).copy()

        img_tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        if self.do_normalize:
            img_tensor = (img_tensor - self.rgb_mean) / self.rgb_std
        return img_tensor, torch.from_numpy(sem).long(), torch.from_numpy(inst).long()
