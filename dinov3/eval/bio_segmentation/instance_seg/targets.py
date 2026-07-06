"""
HoVerNet target generation.

Turns a ground-truth instance map (+ optional semantic class map) into the three
supervision signals the HoVerNet decoder predicts:

    NP  : nucleus-pixel binary map        int64  [H, W]   {0, 1} (ignore=255)
    HV  : horizontal/vertical distance    float32 [2, H, W] in [-1, 1]
    TP  : nucleus type (semantic class)   int64  [H, W]   {0..C-1} (ignore=255)

The HV map is the crux of why this beats connected-components: within each
instance, channel 0 encodes the signed normalized horizontal distance to the
instance centroid and channel 1 the vertical one. Touching nuclei therefore have
a sharp HV discontinuity at their shared border, which the watershed in
``postproc.py`` uses to split them.

Reference: Graham et al., "HoVer-Net" (MedIA 2019), gen_targets.py.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy import ndimage


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (rmin, rmax, cmin, cmax) inclusive-exclusive bbox of a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(rmax) + 1, int(cmin), int(cmax) + 1


def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
    """Compute the [2, H, W] horizontal/vertical distance map for an instance map.

    Args:
        inst_map: (H, W) int array, 0 = background, 1..N = instance IDs.

    Returns:
        hv: (2, H, W) float32. hv[0] = horizontal, hv[1] = vertical, each in [-1, 1]
            (0 on background).
    """
    h, w = inst_map.shape
    x_map = np.zeros((h, w), dtype=np.float32)
    y_map = np.zeros((h, w), dtype=np.float32)

    inst_ids = [i for i in np.unique(inst_map) if i != 0]
    for inst_id in inst_ids:
        inst = (inst_map == inst_id).astype(np.uint8)
        rmin, rmax, cmin, cmax = _bounding_box(inst)
        crop = inst[rmin:rmax, cmin:cmax]
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            continue

        # center of mass returns (y, x)
        com_y, com_x = ndimage.center_of_mass(crop)
        com_y, com_x = int(com_y + 0.5), int(com_x + 0.5)

        x_range = np.arange(crop.shape[1], dtype=np.float32) - com_x
        y_range = np.arange(crop.shape[0], dtype=np.float32) - com_y
        gx, gy = np.meshgrid(x_range, y_range)

        gx[crop == 0] = 0.0
        gy[crop == 0] = 0.0

        # Normalize the negative/positive sides independently so the map spans
        # [-1, 1] regardless of instance size.
        if (gx < 0).any():
            gx[gx < 0] /= -gx[gx < 0].min()
        if (gx > 0).any():
            gx[gx > 0] /= gx[gx > 0].max()
        if (gy < 0).any():
            gy[gy < 0] /= -gy[gy < 0].min()
        if (gy > 0).any():
            gy[gy > 0] /= gy[gy > 0].max()

        fg = crop > 0
        x_map[rmin:rmax, cmin:cmax][fg] = gx[fg]
        y_map[rmin:rmax, cmin:cmax][fg] = gy[fg]

    return np.stack([x_map, y_map], axis=0)


def make_targets(
    inst_map: np.ndarray,
    sem_map: np.ndarray | None = None,
    ignore_index: int = 255,
) -> Dict[str, torch.Tensor]:
    """Build NP / HV / TP targets for a single sample.

    Args:
        inst_map: (H, W) int instance map (0 = bg). Padding regions must be 0.
        sem_map:  (H, W) int semantic class map (0 = bg, 1.. = type), or None for
                  binary datasets. Padded pixels are expected to carry
                  ``ignore_index`` and are excluded from NP/TP losses.
        ignore_index: value marking padded / unlabeled pixels.

    Returns:
        dict with keys 'np' (int64 [H,W]), 'hv' (float32 [2,H,W]),
        'tp' (int64 [H,W]) — 'tp' is all-``ignore_index`` outside foreground if
        ``sem_map`` is None (i.e. TP supervision disabled for binary datasets).
    """
    inst_map = np.asarray(inst_map)
    h, w = inst_map.shape

    # Valid (non-padded) region: derived from the semantic map when available,
    # otherwise everything is valid.
    if sem_map is not None:
        sem_map = np.asarray(sem_map)
        valid = sem_map != ignore_index
    else:
        valid = np.ones((h, w), dtype=bool)

    np_map = (inst_map > 0).astype(np.int64)
    np_map[~valid] = ignore_index

    hv_map = gen_instance_hv_map(inst_map)

    if sem_map is not None:
        tp_map = sem_map.astype(np.int64).copy()
        tp_map[~valid] = ignore_index
    else:
        tp_map = np.full((h, w), ignore_index, dtype=np.int64)

    return {
        "np": torch.from_numpy(np_map),
        "hv": torch.from_numpy(hv_map),
        "tp": torch.from_numpy(tp_map),
    }
