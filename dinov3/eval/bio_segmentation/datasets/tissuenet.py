"""
TissueNet dataset loader.

TissueNet stores data in NPZ files, one per split:
    tissuenet_v1.1_train.npz   /   tissuenet_v1.1_val.npz   /   tissuenet_v1.1_test.npz

Each NPZ contains:
    X : [N, H, W, 2]  float32  two-channel fluorescence images
            channel 0 = nuclear channel
            channel 1 = cytoplasm/whole-cell channel
    y : [N, H, W, 2]  int32   two-channel instance maps
            channel 0 = nuclear instance map
            channel 1 = whole-cell instance map

Since DINOv3 expects 3-channel RGB input, we compose a pseudo-RGB tensor:
    R = nuclear channel (normalised)
    G = whole-cell channel (normalised)
    B = nuclear channel (copy of R)

Usage:
    from dinov3.eval.bio_segmentation.datasets.tissuenet import TissueNetDataset, get_tissuenet_paths
    npz_path = get_tissuenet_paths('/data1/xuzijing/dataset/tissuenet/extracted', 'train')
    dataset = TissueNetDataset(npz_path, target='nuclear', size=(448, 448))
"""

import logging
import os
from glob import glob
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from dinov3.eval.bio_segmentation.constants import MICRO_RGB_MEAN, MICRO_RGB_STD
from dinov3.eval.bio_segmentation.preprocessing import apply_preprocessing_single_channel

from .base import resize_image_and_masks

logger = logging.getLogger(__name__)


class TissueNetDataset(Dataset):
    """
    TissueNet fluorescence cell segmentation dataset.

    Returns per sample:
        img_tensor  : [3, H, W] float32 in [0, 1]  (pseudo-RGB from 2-ch input)
        inst_tensor : [H, W] int64  instance IDs for the selected target (0 = bg)
    """

    def __init__(
        self,
        npz_path: str,
        target: str = 'nuclear',
        size: Optional[Tuple[int, int]] = (448, 448),
        resize_mode: str = "stretch",
        augment: bool = False,
        norm_mode: str = 'percentile',
        cache_preprocessed: bool = False,
        rgb_mean=MICRO_RGB_MEAN,
        rgb_std=MICRO_RGB_STD,
        do_normalize: bool = True,
        multichannel: bool = False,
    ):
        """
        Args:
            npz_path  : path to the .npz file (e.g. tissuenet_v1.1_train.npz).
            target    : 'nuclear' (y[..., 0]) or 'cell' (y[..., 1]).
            size      : output (H, W).
            resize_mode: "stretch" for direct resize, "pad" for keep-aspect
                         long-side resize plus centered padding.
            augment   : random horizontal/vertical flips.
            norm_mode : per-channel normalisation – 'minmax', 'percentile', or 'hybrid'.
            cache_preprocessed : eagerly build a pseudo-RGB cache in RAM for faster training.
            rgb_mean / rgb_std / do_normalize : fixed normalisation on tensors after pseudo-RGB.
            multichannel : if True, return the TRUE 2-channel tensor [2,H,W]
                         (ch0=nuclear, ch1=whole-cell), per-channel normalised, instead of
                         the pseudo-RGB [3,H,W]. Lets the dual-route stem's pool path activate
                         on real channels. Default False keeps the historical pseudo-RGB path
                         byte-for-byte. (cache_preprocessed is ignored in this mode.)
        """
        logger.info(f"Loading TissueNet from {npz_path} ...")
        data = np.load(npz_path, mmap_mode='r')
        self.X = data['X']    # [N, H, W, 2]
        self.y = data['y']    # [N, H, W, 2]

        self.target    = 0 if target == 'nuclear' else 1
        self.size      = size
        self.resize_mode = resize_mode
        self.augment   = augment
        self.norm_mode = norm_mode
        self.multichannel = multichannel
        self.cache_preprocessed = cache_preprocessed and not multichannel
        self.do_normalize = do_normalize
        self.rgb_mean = torch.tensor(rgb_mean, dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor(rgb_std, dtype=torch.float32).view(3, 1, 1)
        self.rgb_cache: Optional[np.ndarray] = None

        if self.cache_preprocessed:
            logger.info("TissueNet: building cached pseudo-RGB tensors in RAM...")
            self.rgb_cache = self._build_rgb_cache().astype(np.float16, copy=False)
            self.X = None

        logger.info(
            f"TissueNet: {len(self.y)} samples  target={target}  size={size}  "
            f"cache_preprocessed={self.cache_preprocessed}"
        )

    def __len__(self) -> int:
        return len(self.y)

    def _normalize_stack(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32, copy=False)

        if self.norm_mode == 'minmax':
            mins = arr.min(axis=(1, 2), keepdims=True)
            maxs = arr.max(axis=(1, 2), keepdims=True)
            denom = maxs - mins
            out = np.zeros_like(arr, dtype=np.float32)
            valid = denom > 0
            np.divide(arr - mins, denom + 1e-8, out=out, where=valid)
            return out

        if self.norm_mode == 'percentile':
            p_low, p_high = np.percentile(arr, [0.3, 99.7], axis=(1, 2))
            p_low = p_low[:, None, None]
            p_high = p_high[:, None, None]
            out = np.empty_like(arr, dtype=np.float32)
            fallback = arr / (arr.max(axis=(1, 2), keepdims=True) + 1e-8)
            valid = p_high > p_low
            clipped = np.clip(arr, p_low, p_high)
            np.divide(clipped - p_low, p_high - p_low + 1e-8, out=out, where=valid)
            out = np.where(valid, out, fallback)
            return out

        if self.norm_mode == 'hybrid':
            p_high = np.percentile(arr, 99.9, axis=(1, 2))[:, None, None]
            clipped = np.clip(arr, 0, p_high)
            out = np.zeros_like(arr, dtype=np.float32)
            valid = p_high > 0
            np.divide(clipped, p_high + 1e-8, out=out, where=valid)
            return out

        raise ValueError(f"Unknown preprocessing mode: {self.norm_mode}")

    def _build_rgb_cache(self) -> np.ndarray:
        ch0 = self._normalize_stack(self.X[:, :, :, 0])
        ch1 = self._normalize_stack(self.X[:, :, :, 1])
        return np.stack([ch0, ch1, ch0], axis=-1)

    def _make_rgb(self, x: np.ndarray) -> np.ndarray:
        """
        Convert (H, W, 2) fluorescence image to (H, W, 3) pseudo-RGB in [0, 1].
        """
        ch0 = apply_preprocessing_single_channel(x[:, :, 0].astype(np.float32),
                                                  mode=self.norm_mode)
        ch1 = apply_preprocessing_single_channel(x[:, :, 1].astype(np.float32),
                                                  mode=self.norm_mode)
        return np.stack([ch0, ch1, ch0], axis=-1)   # [H, W, 3]

    def _make_multichannel(self, x: np.ndarray) -> np.ndarray:
        """
        Convert (H, W, 2) fluorescence image to TRUE 2-channel [H, W, 2] in [0, 1]
        (ch0=nuclear, ch1=whole-cell), per-channel normalised — NO pseudo-RGB duplication.
        """
        ch0 = apply_preprocessing_single_channel(x[:, :, 0].astype(np.float32),
                                                  mode=self.norm_mode)
        ch1 = apply_preprocessing_single_channel(x[:, :, 1].astype(np.float32),
                                                  mode=self.norm_mode)
        return np.stack([ch0, ch1], axis=-1)   # [H, W, 2]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a 3-tuple:
            img_t  : [3, H, W] float32 in [0, 1]  (pseudo-RGB from 2-ch input)
            sem_t  : [H, W] int64  binary semantic (0=bg, 1=cell)
            inst_t : [H, W] int64  instance IDs   (0=bg, 1..N)
        """
        if self.multichannel:
            img = self._make_multichannel(self.X[idx])   # [H, W, 2] float32 — true channels
        elif self.rgb_cache is not None:
            img = self.rgb_cache[idx].astype(np.float32, copy=False)
        else:
            img = self._make_rgb(self.X[idx])        # [H, W, 3] float32
        inst = self.y[idx, :, :, self.target].astype(np.int64)  # [H, W]

        # Resize only when a fixed output size is requested
        valid_mask = None
        if self.size is not None:
            h, w = self.size
            if img.shape[:2] != (h, w):
                img, resized_masks, valid_mask = resize_image_and_masks(
                    img,
                    [inst.astype(np.int64)],
                    (h, w),
                    mode=self.resize_mode,
                    mask_pad_values=[0],
                )
                inst = resized_masks[0].astype(np.int64)

        sem = (inst > 0).astype(np.int64)
        if valid_mask is not None:
            sem[~valid_mask] = 255

        if self.augment:
            if np.random.rand() > 0.5:
                img  = np.flip(img,  axis=1).copy()
                inst = np.flip(inst, axis=1).copy()
                sem  = np.flip(sem,  axis=1).copy()
            if np.random.rand() > 0.5:
                img  = np.flip(img,  axis=0).copy()
                inst = np.flip(inst, axis=0).copy()
                sem  = np.flip(sem,  axis=0).copy()

        img_t = torch.from_numpy(img).permute(2, 0, 1).float()   # [C, H, W], C=2 (mc) or 3 (rgb)
        if self.do_normalize:
            if self.multichannel:
                # per-channel (x-mean)/std, cycling the 3-element stats to C channels,
                # mirroring training's _adapt_stats_to_channels (data/augmentations.py).
                c = img_t.shape[0]
                ci = torch.tensor([i % 3 for i in range(c)])
                mean = self.rgb_mean.view(-1)[ci].view(c, 1, 1)
                std = self.rgb_std.view(-1)[ci].view(c, 1, 1)
                img_t = (img_t - mean) / std
            else:
                img_t = (img_t - self.rgb_mean) / self.rgb_std
        sem_t = torch.from_numpy(sem).long()
        inst_t = torch.from_numpy(inst).long()
        return img_t, sem_t, inst_t

    def get_instance_map(self, idx: int) -> np.ndarray:
        """Return original-resolution instance map."""
        return self.y[idx, :, :, self.target].astype(np.int64)

    def get_binary_mask(self, idx: int) -> np.ndarray:
        """Binary foreground at native resolution."""
        return (self.get_instance_map(idx) > 0).astype(np.int64)


# ============================================================================
# Path discovery
# ============================================================================

def get_tissuenet_paths(
    data_root: str,
    split: str = 'train',
) -> str:
    """
    Find the .npz file for the requested split.

    Args:
        data_root : directory containing the TissueNet NPZ files.
        split     : 'train', 'val', or 'test'.

    Returns:
        Absolute path to the .npz file.
    """
    import glob as _glob

    # Common naming conventions
    patterns = [
        f'*{split}*.npz',
        f'*{split.capitalize()}*.npz',
        f'tissuenet*{split}*.npz',
    ]

    for pat in patterns:
        cands = sorted(_glob.glob(os.path.join(data_root, '**', pat), recursive=True))
        if cands:
            logger.info(f"[TissueNet {split}] Found: {cands[0]}")
            return cands[0]

    raise FileNotFoundError(
        f"Cannot find TissueNet '{split}' NPZ file under {data_root}. "
        f"Expected a file matching patterns like *{split}*.npz"
    )
