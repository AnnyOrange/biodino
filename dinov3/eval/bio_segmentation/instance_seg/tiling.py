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


def _dihedral_matrix(rot_k: int, hflip: bool) -> torch.Tensor:
    """Coordinate transform matrix for torch.rot90(k) followed by horizontal flip."""
    mats = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        torch.tensor([[-1.0, 0.0], [0.0, -1.0]]),
        torch.tensor([[0.0, -1.0], [1.0, 0.0]]),
    ]
    mat = mats[rot_k % 4]
    if hflip:
        mat = torch.tensor([[-1.0, 0.0], [0.0, 1.0]]) @ mat
    return mat


def _apply_dihedral(x: torch.Tensor, rot_k: int, hflip: bool) -> torch.Tensor:
    y = torch.rot90(x, k=rot_k, dims=[2, 3]) if rot_k else x
    return torch.flip(y, dims=[3]) if hflip else y


def _invert_dihedral_map(x: torch.Tensor, rot_k: int, hflip: bool) -> torch.Tensor:
    y = torch.flip(x, dims=[3]) if hflip else x
    return torch.rot90(y, k=-rot_k, dims=[2, 3]) if rot_k else y


@torch.inference_mode()
def forward_dihedral_tta(model: torch.nn.Module, tile: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
    """8-way dihedral TTA for one square tile.

    Predictions are merged before watershed.  HV channels are vector components:
    channel 0 is horizontal/x and channel 1 is vertical/y.  After undoing the
    spatial transform, we multiply the vector by the inverse transform matrix.
    """
    acc_np = acc_hv = acc_tp = None
    n = 0
    for rot_k in range(4):
        for hflip in (False, True):
            t = _apply_dihedral(tile, rot_k, hflip)
            o = model(t)
            npm = _invert_dihedral_map(o["np"], rot_k, hflip)
            hv = _invert_dihedral_map(o["hv"], rot_k, hflip).clone()
            tp = _invert_dihedral_map(o["tp"], rot_k, hflip) if o.get("tp") is not None else None

            inv = _dihedral_matrix(rot_k, hflip).t().to(device=hv.device, dtype=hv.dtype)
            h = hv[:, 0].clone()
            v = hv[:, 1].clone()
            hv[:, 0] = inv[0, 0] * h + inv[0, 1] * v
            hv[:, 1] = inv[1, 0] * h + inv[1, 1] * v

            acc_np = npm.float() if acc_np is None else acc_np + npm.float()
            acc_hv = hv.float() if acc_hv is None else acc_hv + hv.float()
            if tp is not None:
                acc_tp = tp.float() if acc_tp is None else acc_tp + tp.float()
            n += 1
    return {"np": acc_np / n, "hv": acc_hv / n, "tp": (acc_tp / n) if acc_tp is not None else None}


def _starts(length: int, crop: int, stride: int) -> List[int]:
    if length <= crop:
        return [0]
    starts = list(range(0, length - crop + 1, stride))
    if starts[-1] != length - crop:
        starts.append(length - crop)
    return starts


def _round_up(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


def _blend_window(crop_size: int, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones((crop_size, crop_size), dtype=np.float32)
    if mode != "gaussian":
        raise ValueError(f"Unsupported blend mode: {mode}")
    coords = np.linspace(-1.0, 1.0, crop_size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    sigma = 0.45
    win = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
    return np.maximum(win, 1e-3)


@torch.inference_mode()
def sliding_window_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    crop_size: int = 256,
    stride: int = 192,
    patch_size: int = 16,
    num_types: int = 0,
    tta: bool = False,
    tta_mode: str = "flip4",
    blend_mode: str = "uniform",
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
    blend = _blend_window(crop_size, blend_mode)

    for y in _starts(ph, crop_size, stride):
        for x in _starts(pw, crop_size, stride):
            tile = img[:, y : y + crop_size, x : x + crop_size].unsqueeze(0)
            if not tta:
                out = model(tile)
            elif tta_mode == "flip4":
                out = forward_tta(model, tile)
            elif tta_mode == "dihedral8":
                out = forward_dihedral_tta(model, tile)
            else:
                raise ValueError(f"Unsupported TTA mode: {tta_mode}")
            np_acc[:, y : y + crop_size, x : x + crop_size] += out["np"][0].float().cpu().numpy() * blend[None]
            hv_acc[:, y : y + crop_size, x : x + crop_size] += out["hv"][0].float().cpu().numpy() * blend[None]
            if tp_acc is not None and out.get("tp") is not None:
                tp_acc[:, y : y + crop_size, x : x + crop_size] += out["tp"][0].float().cpu().numpy() * blend[None]
            count[y : y + crop_size, x : x + crop_size] += blend

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
