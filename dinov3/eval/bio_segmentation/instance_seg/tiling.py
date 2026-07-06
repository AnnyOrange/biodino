"""
Sliding-window inference for large native-resolution images.

PanNuke/CoNIC are 256² (single forward), but MoNuSeg (1000²) and LiveCell
(~520×696) must be tiled. We run the model on overlapping crops and average the
dense outputs (NP/HV/TP) in overlap regions, then post-process the full map once
so instances that straddle a tile border are still split correctly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def forward_tta(model: torch.nn.Module, tile: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
    """4-way flip TTA for one tile [1,3,H,W]. HV is directional, so flipping the
    input requires negating the matching HV channel (h↔W-flip, v↔H-flip)."""
    acc_np = acc_hv = acc_tp = None
    n = 0
    for fh in (False, True):
        for fv in (False, True):
            t = tile
            if fh:
                t = torch.flip(t, dims=[3])
            if fv:
                t = torch.flip(t, dims=[2])
            o = model(t)
            npm, hv, tp = o["np"], o["hv"].clone(), o.get("tp")
            if fh:
                npm = torch.flip(npm, [3]); hv = torch.flip(hv, [3])
                tp = torch.flip(tp, [3]) if tp is not None else None
                hv[:, 0] = -hv[:, 0]
            if fv:
                npm = torch.flip(npm, [2]); hv = torch.flip(hv, [2])
                tp = torch.flip(tp, [2]) if tp is not None else None
                hv[:, 1] = -hv[:, 1]
            acc_np = npm.float() if acc_np is None else acc_np + npm.float()
            acc_hv = hv.float() if acc_hv is None else acc_hv + hv.float()
            if tp is not None:
                acc_tp = tp.float() if acc_tp is None else acc_tp + tp.float()
            n += 1
    out = {"np": acc_np / n, "hv": acc_hv / n, "tp": (acc_tp / n) if acc_tp is not None else None}
    return out


def _starts(length: int, crop: int, stride: int) -> List[int]:
    if length <= crop:
        return [0]
    starts = list(range(0, length - crop + 1, stride))
    if starts[-1] != length - crop:
        starts.append(length - crop)
    return starts


def _round_up(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


@torch.inference_mode()
def sliding_window_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    crop_size: int = 256,
    stride: int = 192,
    patch_size: int = 16,
    num_types: int = 0,
    tta: bool = False,
) -> Dict[str, Optional[np.ndarray]]:
    """Run a DINOHoVerNet over a single (possibly large) image.

    Args:
        model: returns {"np": [B,2,h,w], "hv": [B,2,h,w], "tp": [B,C,h,w]|None}.
        image: (3, H, W) normalized tensor on the model's device.
        crop_size: tile size (must be a multiple of patch_size).
        stride: tile stride (overlap = crop_size - stride).
        num_types: 0 for binary datasets, else number of type channels.

    Returns:
        dict of numpy arrays at the original (H, W): "np" [2,H,W], "hv" [2,H,W],
        and "tp" [C,H,W] (or None).
    """
    device = image.device
    _, H, W = image.shape

    # Pad to >= crop and a multiple of patch_size (reflect padding).
    ph = max(crop_size, _round_up(H, patch_size))
    pw = max(crop_size, _round_up(W, patch_size))
    img = F.pad(image.unsqueeze(0), (0, pw - W, 0, ph - H), mode="reflect").squeeze(0)

    np_acc = np.zeros((2, ph, pw), dtype=np.float32)
    hv_acc = np.zeros((2, ph, pw), dtype=np.float32)
    tp_acc = np.zeros((num_types, ph, pw), dtype=np.float32) if num_types else None
    count = np.zeros((ph, pw), dtype=np.float32)

    for y in _starts(ph, crop_size, stride):
        for x in _starts(pw, crop_size, stride):
            tile = img[:, y : y + crop_size, x : x + crop_size].unsqueeze(0)
            out = forward_tta(model, tile) if tta else model(tile)
            np_acc[:, y : y + crop_size, x : x + crop_size] += out["np"][0].float().cpu().numpy()
            hv_acc[:, y : y + crop_size, x : x + crop_size] += out["hv"][0].float().cpu().numpy()
            if tp_acc is not None and out.get("tp") is not None:
                tp_acc[:, y : y + crop_size, x : x + crop_size] += out["tp"][0].float().cpu().numpy()
            count[y : y + crop_size, x : x + crop_size] += 1.0

    count = np.maximum(count, 1e-6)
    np_acc /= count[None]
    hv_acc /= count[None]
    result: Dict[str, Optional[np.ndarray]] = {
        "np": np_acc[:, :H, :W],
        "hv": hv_acc[:, :H, :W],
        "tp": None,
    }
    if tp_acc is not None:
        tp_acc /= count[None]
        result["tp"] = tp_acc[:, :H, :W]
    return result
