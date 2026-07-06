"""
HoVerNet post-processing: (NP, HV) → instance map (touching nuclei separated).

The horizontal/vertical maps have a sharp gradient at the border between two
touching nuclei. We turn |∇HV| into an "energy" ridge, subtract it from the
foreground to carve markers, and run a marker-controlled watershed. This is the
step connected-components in the linear-probe track cannot do.

Reference: Graham et al., "HoVer-Net" (MedIA 2019), post_proc.__proc_np_hv.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as nd_label
from skimage.segmentation import watershed


def _odd_ksize(requested: int, h: int, w: int) -> int:
    """Clamp a Sobel kernel size to be odd and smaller than the image."""
    limit = max(3, min(requested, (min(h, w) // 2) * 2 - 1))
    return limit if limit % 2 == 1 else limit - 1


def _remove_small_labels(labeled: np.ndarray, min_size: int) -> np.ndarray:
    """Zero out labeled components whose area is < min_size (label-stable).

    Reimplements skimage.remove_small_objects to avoid its churning `min_size`
    deprecation; operates on an integer-labeled array and preserves remaining IDs.
    """
    if labeled.max() == 0:
        return labeled
    counts = np.bincount(labeled.ravel())
    small = np.where(counts < min_size)[0]
    small = small[small != 0]
    if small.size:
        labeled = labeled.copy()
        labeled[np.isin(labeled, small)] = 0
    return labeled


def _norm01(x: np.ndarray) -> np.ndarray:
    return cv2.normalize(x, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def proc_np_hv(
    np_prob: np.ndarray,
    hv: np.ndarray,
    fg_thresh: float = 0.5,
    energy_thresh: float = 0.4,
    sobel_ksize: int = 21,
    min_size: int = 10,
) -> np.ndarray:
    """Split a foreground probability + HV map into an instance map.

    Args:
        np_prob: (H, W) foreground probability in [0, 1].
        hv:      (2, H, W) raw horizontal/vertical maps.
        fg_thresh: threshold on np_prob to get the binary blob.
        energy_thresh: threshold on the gradient energy to carve borders.
        sobel_ksize: Sobel kernel size for the gradient (clamped to image).
        min_size: remove instances/markers smaller than this.

    Returns:
        (H, W) int32 instance map (0 = background, 1..N = instances).
    """
    h, w = np_prob.shape
    ksize = _odd_ksize(sobel_ksize, h, w)

    blb = (np_prob >= fg_thresh).astype(np.int32)
    blb = nd_label(blb)[0]
    blb = _remove_small_labels(blb, min_size=min_size)
    blb[blb > 0] = 1

    if blb.sum() == 0:
        return np.zeros((h, w), dtype=np.int32)

    h_dir = _norm01(hv[0].astype(np.float32))
    v_dir = _norm01(hv[1].astype(np.float32))

    sobelh = cv2.Sobel(h_dir, cv2.CV_32F, 1, 0, ksize=ksize)
    sobelv = cv2.Sobel(v_dir, cv2.CV_32F, 0, 1, ksize=ksize)
    sobelh = 1.0 - _norm01(sobelh)
    sobelv = 1.0 - _norm01(sobelv)

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1.0 - blb)
    overall[overall < 0] = 0

    # Watershed landscape: low inside cells, high on borders.
    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist.astype(np.float32), (3, 3), 0)

    overall = (overall >= energy_thresh).astype(np.int32)
    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype(np.uint8)
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    marker = nd_label(marker)[0]
    marker = _remove_small_labels(marker, min_size=min_size)

    inst = watershed(dist, markers=marker, mask=blb)
    return inst.astype(np.int32)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=0, keepdims=True) + 1e-8)


def assign_instance_classes(inst: np.ndarray, tp_label: np.ndarray) -> np.ndarray:
    """Build a semantic map by majority-voting the type label inside each instance."""
    sem = np.zeros_like(inst, dtype=np.int32)
    for inst_id in np.unique(inst):
        if inst_id == 0:
            continue
        region = inst == inst_id
        votes = np.bincount(tp_label[region].astype(np.int64))
        votes[0] = 0  # never assign background as the instance type
        cls = int(votes.argmax()) if votes.any() else 0
        sem[region] = cls
    return sem


def postprocess(
    np_logits: np.ndarray,
    hv: np.ndarray,
    tp_logits: Optional[np.ndarray] = None,
    **proc_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full single-image post-processing.

    Args:
        np_logits: (2, H, W) NP logits.
        hv:        (2, H, W) raw HV maps.
        tp_logits: (C, H, W) type logits, or None for binary datasets.

    Returns:
        (inst_map (H,W) int32, sem_map (H,W) int32). For binary datasets
        sem_map is just (inst>0).
    """
    np_prob = _softmax_np(np.asarray(np_logits, dtype=np.float32))[1]
    inst = proc_np_hv(np_prob, np.asarray(hv, dtype=np.float32), **proc_kwargs)

    if tp_logits is not None:
        tp_label = np.asarray(tp_logits).argmax(axis=0)
        sem = assign_instance_classes(inst, tp_label)
    else:
        sem = (inst > 0).astype(np.int32)
    return inst, sem
