"""
Abstract base class for biological segmentation datasets.

Subclasses only need to implement `load_image` and `load_mask`.
All common logic (resize, preprocess, augment, tensor conversion) lives here.

Size semantics
--------------
size = (H, W)  → resize every sample to this fixed size.
                 Use for feature-extractor caching (linear probe).
size = None    → return each sample at its native resolution.
                 Use for Mask2Former training (random crop via collate_fn)
                 and sliding-window evaluation.
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
        img_tensor  : [3, H, W] float32 in [0, 1]
        mask_tensor : [H, W] int64  0=background, 1=cell

    When size=None the spatial dimensions (H, W) are the image's native
    resolution (rounded to the nearest multiple of patch_size only for
    informational logging — no actual resize is performed).
    """

    def __init__(
        self,
        img_paths:  List[str],
        mask_paths: List[str],
        mode:       str = 'hybrid',
        size:       Optional[Tuple[int, int]] = None,
        patch_size: int = 16,
        augment:    bool = False,
    ):
        """
        Args:
            img_paths:  list of image file paths.
            mask_paths: list of corresponding mask file paths.
            mode:       channel-wise normalisation ('minmax', 'percentile', 'hybrid').
            size:       (H, W) to resize every sample to, or None to keep
                        native resolution.
            patch_size: ViT patch size (used only for logging).
            augment:    enable random horizontal/vertical flips.
        """
        assert len(img_paths) == len(mask_paths), (
            f"Image count ({len(img_paths)}) != mask count ({len(mask_paths)})"
        )
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.mode       = mode
        self.patch_size = patch_size
        self.augment    = augment
        self.size       = size   # None → no resize (native resolution)

        if size is None:
            logger.info(
                "size=None → images returned at native resolution "
                "(for Mask2Former; use size=(H,W) for feature-extractor caching)"
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

        # Normalise per channel
        img = apply_preprocessing(img, mode=self.mode)

        mask = self.load_mask(self.mask_paths[idx])

        # Resize only when a fixed output size is requested
        if self.size is not None:
            h, w = self.size
            img  = cv2.resize(img,  (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(np.int64)

        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img  = np.flip(img,  axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() > 0.5:
                img  = np.flip(img,  axis=0).copy()
                mask = np.flip(mask, axis=0).copy()

        img_tensor  = torch.from_numpy(img).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long()
        return img_tensor, mask_tensor
