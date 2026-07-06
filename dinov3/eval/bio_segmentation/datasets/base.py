"""
Abstract base class for biological segmentation datasets.

Subclasses only need to implement `load_image` and `load_mask`.
All common logic (resize, dtype→[0,1], fixed mean/std, augment, tensor conversion) lives here.

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
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dinov3.eval.bio_segmentation.constants import MICRO_RGB_MEAN, MICRO_RGB_STD
from dinov3.utils.bio_io import _normalize_to_float32

logger = logging.getLogger(__name__)


def resize_image_and_masks(
    img: np.ndarray,
    masks: Sequence[np.ndarray],
    size: Tuple[int, int],
    mode: str = "stretch",
    mask_pad_values: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Resize image/masks either by stretching or by aspect-preserving pad.

    Padding regions are returned in ``valid_mask`` so semantic masks can mark
    them as ignore_index=255 instead of rewarding extra background pixels.
    """
    if mode not in {"stretch", "pad"}:
        raise ValueError(f"Unknown resize mode: {mode!r}")

    target_h, target_w = size
    if mask_pad_values is None:
        mask_pad_values = [0] * len(masks)
    if len(mask_pad_values) != len(masks):
        raise ValueError("mask_pad_values length must match masks length")

    if mode == "stretch":
        out_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        out_masks = [
            cv2.resize(mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(mask.dtype)
            for mask in masks
        ]
        valid_mask = np.ones((target_h, target_w), dtype=bool)
        return out_img, out_masks, valid_mask

    src_h, src_w = img.shape[:2]
    scale = min(target_h / max(src_h, 1), target_w / max(src_w, 1))
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out_shape = (target_h, target_w) + (() if img.ndim == 2 else img.shape[2:])
    if img.ndim == 2:
        out_img = np.full(out_shape, np.median(img), dtype=img.dtype)
    else:
        # Median padding is less disruptive than black/white bars for ViT attention.
        out_img = np.empty(out_shape, dtype=img.dtype)
        out_img[...] = np.median(img.reshape(-1, img.shape[2]), axis=0)
    out_img[top : top + new_h, left : left + new_w, ...] = resized_img

    out_masks: List[np.ndarray] = []
    for mask, pad_value in zip(masks, mask_pad_values):
        resized_mask = cv2.resize(mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(mask.dtype)
        out_mask = np.full((target_h, target_w), pad_value, dtype=mask.dtype)
        out_mask[top : top + new_h, left : left + new_w] = resized_mask
        out_masks.append(out_mask)

    valid_mask = np.zeros((target_h, target_w), dtype=bool)
    valid_mask[top : top + new_h, left : left + new_w] = True
    return out_img, out_masks, valid_mask


class BioSegDataset(Dataset, ABC):
    """
    Base dataset for binary foreground / background cell segmentation.

    Each sample returns:
        img_tensor  : [3, H, W] float32 — [0, 1] then optionally (x - mean) / std
        mask_tensor : [H, W] int64  0=background, 1=cell

    When size=None the spatial dimensions (H, W) are the image's native
    resolution (rounded to the nearest multiple of patch_size only for
    informational logging — no actual resize is performed).
    """

    def __init__(
        self,
        img_paths: List[str],
        mask_paths: List[str],
        mode: Optional[str] = None,  # kept only for backward compatibility
        size: Optional[Tuple[int, int]] = None,
        resize_mode: str = "stretch",
        patch_size: int = 16,
        augment: bool = False,
        rgb_mean=MICRO_RGB_MEAN,
        rgb_std=MICRO_RGB_STD,
        do_normalize: bool = True,
    ):
        """
        Args:
            img_paths:  list of image file paths.
            mask_paths: list of corresponding mask file paths.
            mode:       deprecated; ignored (was channel-wise normalisation mode).
            size:       (H, W) to resize every sample to, or None to keep
                        native resolution.
            resize_mode: "stretch" for direct resize, "pad" for keep-aspect
                        long-side resize plus centered padding.
            patch_size: ViT patch size (used only for logging).
            augment:    enable random horizontal/vertical flips.
            rgb_mean:   per-channel mean for fixed normalisation (length 3).
            rgb_std:    per-channel std for fixed normalisation (length 3).
            do_normalize: if True, apply (img - mean) / std after [0, 1] scaling.
        """
        assert len(img_paths) == len(mask_paths), (
            f"Image count ({len(img_paths)}) != mask count ({len(mask_paths)})"
        )
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.augment = augment
        self.size = size  # None → no resize (native resolution)
        self.resize_mode = resize_mode

        self.mode = mode  # deprecated: kept only for backward compatibility
        self.do_normalize = do_normalize
        self.rgb_mean = torch.tensor(rgb_mean, dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor(rgb_std, dtype=torch.float32).view(3, 1, 1)

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

        # Normalize to float32 [0, 1] exactly once, based on dtype / value range.
        # uint8  -> /255
        # uint16 -> /65535
        # float  -> clip to [0,1]
        img = _normalize_to_float32(img)

        mask = self.load_mask(self.mask_paths[idx])

        # Resize only when a fixed output size is requested
        if self.size is not None:
            img, resized_masks, _valid = resize_image_and_masks(
                img,
                [mask.astype(np.int64)],
                self.size,
                mode=self.resize_mode,
                mask_pad_values=[255],
            )
            mask = resized_masks[0].astype(np.int64)

        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        if self.do_normalize:
            img_tensor = (img_tensor - self.rgb_mean) / self.rgb_std

        mask_tensor = torch.from_numpy(mask).long()
        return img_tensor, mask_tensor
