"""
Shared preprocessing utilities for biological microscopy images.

Supports 8-bit and 16-bit images, single-channel and multi-channel.
Three normalization modes are provided for comparison:
  - minmax:     global min-max normalization
  - percentile: clip at [0.3%, 99.7%] then normalize
  - hybrid:     clip at 99.9% then normalize (default)
"""

from typing import Tuple

import cv2
import numpy as np


def apply_preprocessing_single_channel(img: np.ndarray, mode: str = 'hybrid') -> np.ndarray:
    """
    Normalize a single-channel image to [0, 1].

    Args:
        img: 2-D numpy array (H, W), any bit depth.
        mode: 'minmax', 'percentile', or 'hybrid'.

    Returns:
        Float32 array in [0, 1].
    """
    img = img.astype(np.float32)
    if img.max() == 0:
        return img

    if mode == 'minmax':
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    elif mode == 'percentile':
        p_low, p_high = np.percentile(img, [0.3, 99.7])
        if p_high <= p_low:
            return img / (img.max() + 1e-8)
        img = np.clip(img, p_low, p_high)
        return (img - p_low) / (p_high - p_low + 1e-8)

    elif mode == 'hybrid':
        p_high = np.percentile(img, 99.9)
        if p_high == 0:
            return img
        img = np.clip(img, 0, p_high)
        return img / (p_high + 1e-8)

    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}. Choose from 'minmax', 'percentile', 'hybrid'.")


def apply_preprocessing(img: np.ndarray, mode: str = 'hybrid') -> np.ndarray:
    """
    Normalize a (H, W) or (H, W, C) image to [0, 1] per channel.

    Args:
        img: numpy array, shape (H, W) or (H, W, C), any bit depth.
        mode: normalization mode (see apply_preprocessing_single_channel).

    Returns:
        Float32 array with the same shape, values in [0, 1].
    """
    if img.ndim == 2:
        return apply_preprocessing_single_channel(img, mode)
    elif img.ndim == 3:
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            result[:, :, c] = apply_preprocessing_single_channel(img[:, :, c], mode)
        return result
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")


def get_size_multiple_of_patch(
    original_size: Tuple[int, int],
    patch_size: int = 16,
) -> Tuple[int, int]:
    """
    Round spatial dimensions down to the nearest multiple of patch_size.

    Args:
        original_size: (H, W).
        patch_size: ViT patch size.

    Returns:
        (H', W') both divisible by patch_size.
    """
    h, w = original_size
    return (h // patch_size) * patch_size, (w // patch_size) * patch_size


def resize_to_patch_multiple(img: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """
    Resize an image so that H and W are multiples of patch_size.

    Args:
        img: numpy array (H, W) or (H, W, C).
        patch_size: ViT patch size.

    Returns:
        Resized numpy array.
    """
    h, w = img.shape[:2]
    new_h, new_w = get_size_multiple_of_patch((h, w), patch_size)
    if new_h == h and new_w == w:
        return img
    return cv2.resize(img, (new_w, new_h))
