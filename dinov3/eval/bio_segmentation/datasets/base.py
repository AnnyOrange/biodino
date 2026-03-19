"""
Abstract base class for biological segmentation datasets.

Subclasses only need to implement `load_image` and `load_mask`.
All common logic (resize, preprocess, augment, tensor conversion) lives here.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dinov3.eval.bio_segmentation.preprocessing import (
    apply_preprocessing,
    get_size_multiple_of_patch,
)

logger = logging.getLogger(__name__)


class BioSegDataset(Dataset, ABC):
    """
    Base dataset for binary foreground / background cell segmentation.

    Each sample returns:
        img_tensor:  [3, H, W] float32 in [0, 1].
        mask_tensor: [H, W] int64 with values 0 (background) or 1 (cell).
    """

    def __init__(
        self,
        img_paths: List[str],
        mask_paths: List[str],
        mode: str = 'hybrid',
        size: Optional[Tuple[int, int]] = None,
        patch_size: int = 16,
        augment: bool = False,
    ):
        """
        Args:
            img_paths:  list of image file paths.
            mask_paths: list of corresponding mask file paths.
            mode:       preprocessing mode ('minmax', 'percentile', 'hybrid').
            size:       fixed (H, W) output size.  If None, inferred from the
                        first image rounded to patch_size multiples.
            patch_size: ViT patch size used when auto-inferring size.
            augment:    enable random horizontal/vertical flips.
        """
        assert len(img_paths) == len(mask_paths), (
            f"Image count ({len(img_paths)}) != mask count ({len(mask_paths)})"
        )
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.mode = mode
        self.patch_size = patch_size
        self.augment = augment

        if size is not None:
            self.size = size
        else:
            sample = self.load_image(img_paths[0])
            h, w = sample.shape[:2]
            self.size = get_size_multiple_of_patch((h, w), patch_size)
            logger.info(
                f"Auto image size: original ({h}, {w}) -> adjusted to {self.size}"
            )

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_image(self, path: str) -> np.ndarray:
        """Load an image and return an (H, W, C) or (H, W) uint array."""

    @abstractmethod
    def load_mask(self, path: str) -> np.ndarray:
        """Load a mask and return an (H, W) binary int64 array (0/1)."""

    # ------------------------------------------------------------------
    # Common pipeline
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.load_image(self.img_paths[idx])

        # Ensure 3-channel RGB
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        # Normalize per channel
        img = apply_preprocessing(img, mode=self.mode)

        # Resize
        h, w = self.size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        mask = self.load_mask(self.mask_paths[idx])
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int64)

        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long()
        return img_tensor, mask_tensor
