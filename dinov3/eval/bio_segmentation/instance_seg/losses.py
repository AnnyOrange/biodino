"""
HoVerNet composite loss.

    NP : cross-entropy + soft Dice          (nucleus-pixel branch)
    HV : MSE + MSGE (Sobel-gradient MSE)     (distance regression branch)
    TP : cross-entropy + soft Dice          (type branch, multi-class only)

MSGE is the gradient-domain MSE on the HV maps, computed only over foreground
pixels. It is what sharpens the HV discontinuity between touching nuclei so the
watershed in ``postproc.py`` can separate them.

The default NP objective is intentionally unchanged. ``focal_tversky`` is an
optional screening variant for small-object recall experiments.

Reference: Graham et al., "HoVer-Net" (MedIA 2019), loss formulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    np_ce: float = 1.0
    np_dice: float = 1.0
    hv_mse: float = 1.0
    hv_msge: float = 1.0
    tp_ce: float = 1.0
    tp_dice: float = 1.0


def _sobel_kernels(size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """HoVer-Net Sobel-like gradient kernels (kernel_h, kernel_v)."""
    assert size % 2 == 1
    rng = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    h, v = np.meshgrid(rng, rng)              # h varies along x, v along y
    denom = (h * h + v * v) + 1e-15
    return h / denom, v / denom


def _soft_dice(prob: torch.Tensor, onehot: torch.Tensor, valid: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """1 - mean soft Dice over classes. prob/onehot: [B,C,H,W]; valid: [B,1,H,W]."""
    prob = prob * valid
    onehot = onehot * valid
    inter = (prob * onehot).sum(dim=(0, 2, 3))
    denom = prob.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _masked_focal_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    gamma: float,
    ignore_index: int,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** gamma) * ce
    valid_2d = valid.squeeze(1).to(loss.dtype)
    return (loss * valid_2d).sum() / (valid_2d.sum() + 1e-8)


def _foreground_tversky(
    prob: torch.Tensor,
    onehot: torch.Tensor,
    valid: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    fg_prob = prob[:, 1:2] * valid
    fg_true = onehot[:, 1:2] * valid
    tp = (fg_prob * fg_true).sum()
    fp = (fg_prob * (1.0 - fg_true) * valid).sum()
    fn = ((1.0 - fg_prob) * fg_true).sum()
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - score


class HoVerNetLoss(nn.Module):
    def __init__(
        self,
        num_types: int = 0,
        ignore_index: int = 255,
        weights: Optional[LossWeights] = None,
        np_loss_mode: str = "ce_dice",
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        hv_auxiliary_weight: float = 0.0,
    ):
        super().__init__()
        if np_loss_mode not in {"ce_dice", "focal_tversky"}:
            raise ValueError(f"Unsupported np_loss_mode={np_loss_mode!r}")
        self.num_types = num_types
        self.ignore_index = ignore_index
        self.w = weights or LossWeights()
        self.np_loss_mode = np_loss_mode
        self.focal_gamma = float(focal_gamma)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.hv_auxiliary_weight = float(hv_auxiliary_weight)
        if self.hv_auxiliary_weight < 0:
            raise ValueError("hv_auxiliary_weight must be non-negative")

        kh, kv = _sobel_kernels(5)
        # [out=1, in=1, 5, 5]
        self.register_buffer("kernel_h", torch.from_numpy(kh)[None, None], persistent=False)
        self.register_buffer("kernel_v", torch.from_numpy(kv)[None, None], persistent=False)

    def _hv_gradient(self, hv: torch.Tensor) -> torch.Tensor:
        """Gradient of the HV maps: d(h)/dx and d(v)/dy. hv: [B,2,H,W] → [B,2,H,W]."""
        kh = self.kernel_h.to(device=hv.device, dtype=hv.dtype)
        kv = self.kernel_v.to(device=hv.device, dtype=hv.dtype)
        dh = F.conv2d(hv[:, 0:1], kh, padding=2)
        dv = F.conv2d(hv[:, 1:2], kv, padding=2)
        return torch.cat([dh, dv], dim=1)

    def forward(
        self,
        pred: Dict[str, Optional[torch.Tensor]],
        target: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        np_logits = pred["np"]                       # [B,2,H,W]
        hv_pred = pred["hv"]                          # [B,2,H,W]
        np_t = target["np"].long()                    # [B,H,W] {0,1,255}
        hv_t = target["hv"].to(hv_pred.dtype)         # [B,2,H,W]

        comps: Dict[str, float] = {}

        # ---- NP branch ----
        np_valid = (np_t != self.ignore_index).unsqueeze(1).float()      # [B,1,H,W]
        np_prob = F.softmax(np_logits, dim=1)
        np_onehot = F.one_hot(np_t.clamp(0, 1), num_classes=2).permute(0, 3, 1, 2).float()
        if self.np_loss_mode == "ce_dice":
            np_loss_a = F.cross_entropy(np_logits, np_t, ignore_index=self.ignore_index)
            np_loss_b = _soft_dice(np_prob, np_onehot, np_valid)
            comps["np_ce"] = float(np_loss_a.detach())
            comps["np_dice"] = float(np_loss_b.detach())
        else:
            np_loss_a = _masked_focal_ce(
                np_logits,
                np_t,
                np_valid,
                gamma=self.focal_gamma,
                ignore_index=self.ignore_index,
            )
            np_loss_b = _foreground_tversky(
                np_prob,
                np_onehot,
                np_valid,
                alpha=self.tversky_alpha,
                beta=self.tversky_beta,
            )
            comps["np_focal"] = float(np_loss_a.detach())
            comps["np_tversky"] = float(np_loss_b.detach())

        # ---- HV branch ----
        valid = (np_t != self.ignore_index).unsqueeze(1).float()         # [B,1,H,W]
        sq_err = ((hv_pred - hv_t) ** 2) * valid                          # [B,2,H,W]
        hv_mse = sq_err.sum() / (valid.sum() * 2.0 + 1e-8)

        focus = (np_t == 1).unsqueeze(1).float()                          # foreground only
        grad_err = (self._hv_gradient(hv_pred) - self._hv_gradient(hv_t)) ** 2
        hv_msge = (grad_err * focus).sum() / (focus.sum() * 2.0 + 1e-8)
        comps["hv_mse"] = float(hv_mse.detach())
        comps["hv_msge"] = float(hv_msge.detach())

        total = (
            self.w.np_ce * np_loss_a
            + self.w.np_dice * np_loss_b
            + self.w.hv_mse * hv_mse
            + self.w.hv_msge * hv_msge
        )

        # Optional independent auxiliary branch.  It uses the same target as
        # the primary HV branch but never changes the postprocessing input.
        hv_aux = pred.get("hv_aux")
        if hv_aux is not None:
            if hv_aux.shape != hv_t.shape:
                raise ValueError(f"hv_aux shape {tuple(hv_aux.shape)} != target {tuple(hv_t.shape)}")
            aux_sq_err = ((hv_aux - hv_t.to(hv_aux.dtype)) ** 2) * valid
            hv_aux_mse = aux_sq_err.sum() / (valid.sum() * 2.0 + 1e-8)
            aux_grad_err = (self._hv_gradient(hv_aux) - self._hv_gradient(hv_t.to(hv_aux.dtype))) ** 2
            hv_aux_msge = (aux_grad_err * focus).sum() / (focus.sum() * 2.0 + 1e-8)
            comps["hv_aux_mse"] = float(hv_aux_mse.detach())
            comps["hv_aux_msge"] = float(hv_aux_msge.detach())
            comps["hv_aux_weight"] = self.hv_auxiliary_weight
            total = total + self.hv_auxiliary_weight * (self.w.hv_mse * hv_aux_mse + self.w.hv_msge * hv_aux_msge)

        # ---- TP branch ----
        if self.num_types and pred.get("tp") is not None:
            tp_logits = pred["tp"]                                        # [B,C,H,W]
            tp_t = target["tp"].long()                                   # [B,H,W]
            tp_ce = F.cross_entropy(tp_logits, tp_t, ignore_index=self.ignore_index)
            tp_valid = (tp_t != self.ignore_index).unsqueeze(1).float()
            tp_prob = F.softmax(tp_logits, dim=1)
            tp_onehot = F.one_hot(tp_t.clamp(0, self.num_types - 1), self.num_types).permute(0, 3, 1, 2).float()
            tp_dice = _soft_dice(tp_prob, tp_onehot, tp_valid)
            total = total + self.w.tp_ce * tp_ce + self.w.tp_dice * tp_dice
            comps["tp_ce"] = float(tp_ce.detach())
            comps["tp_dice"] = float(tp_dice.detach())

        comps["total"] = float(total.detach())
        return total, comps
