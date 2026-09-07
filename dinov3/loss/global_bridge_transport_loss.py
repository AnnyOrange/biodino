"""Transport an offline bridge displacement onto the current frozen-anchor view."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


GLOBAL_BRIDGE_FEATURE_PROTOCOLS = ("final_cls", "nlb2_cls", "nlb2_avg")


def compose_global_bridge_readout(
    *,
    final_cls: Tensor,
    feature_protocol: str,
    penultimate_cls: Tensor | None = None,
    final_patches: Tensor | None = None,
) -> Tensor:
    """Compose the same token vector used by the frozen retrieval evaluator."""
    protocol = str(feature_protocol).lower()
    if protocol not in GLOBAL_BRIDGE_FEATURE_PROTOCOLS:
        raise ValueError(
            f"Unknown global bridge feature protocol {feature_protocol!r}; "
            f"expected one of {GLOBAL_BRIDGE_FEATURE_PROTOCOLS}"
        )
    if final_cls.ndim < 2:
        raise ValueError("final_cls must have a feature dimension")
    if protocol == "final_cls":
        return final_cls
    if penultimate_cls is None or penultimate_cls.shape != final_cls.shape:
        raise ValueError("nlb2 protocols require penultimate_cls with the final CLS shape")
    components = [penultimate_cls, final_cls]
    if protocol == "nlb2_avg":
        if (
            final_patches is None
            or final_patches.ndim != final_cls.ndim + 1
            or final_patches.shape[:-2] != final_cls.shape[:-1]
            or final_patches.shape[-1] != final_cls.shape[-1]
        ):
            raise ValueError(
                "nlb2_avg requires final_patches shaped like final_cls with one patch axis"
            )
        components.append(final_patches.mean(dim=-2))
    return torch.cat(components, dim=-1)


class GlobalBridgeTransportLoss(nn.Module):
    def __init__(self, eps: float = 1.0e-8) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

    def forward(
        self,
        *,
        student_features: Tensor,
        current_anchor_features: Tensor,
        bank_anchor_features: Tensor,
        bank_target_features: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        shape = student_features.shape
        if student_features.ndim != 2 or current_anchor_features.shape != shape:
            raise ValueError("Student and current anchor must share [samples, dimensions]")
        if bank_anchor_features.shape != shape or bank_target_features.shape != shape:
            raise ValueError("Offline bridge features must match the current feature shape")
        if shape[0] == 0:
            zero = student_features.sum() * 0.0
            return zero, {
                "gbt_loss": zero.detach(),
                "gbt_active_rows": zero.detach(),
                "gbt_valid_direction_fraction": zero.detach(),
                "gbt_target_angle": zero.detach(),
                "gbt_student_target_cosine": zero.detach(),
                "gbt_anchor_target_cosine": zero.detach(),
            }

        student = F.normalize(student_features.float(), dim=-1, eps=self.eps)
        current_anchor = F.normalize(
            current_anchor_features.detach().float(), dim=-1, eps=self.eps
        )
        bank_anchor = F.normalize(bank_anchor_features.detach().float(), dim=-1, eps=self.eps)
        bank_target = F.normalize(bank_target_features.detach().float(), dim=-1, eps=self.eps)

        bank_cosine = (bank_anchor * bank_target).sum(dim=-1).clamp(-1.0, 1.0)
        target_angle = torch.acos(bank_cosine)
        direction = bank_target - bank_cosine[:, None] * bank_anchor
        direction = F.normalize(direction, dim=-1, eps=self.eps)
        direction = direction - (direction * current_anchor).sum(dim=-1, keepdim=True) * current_anchor
        direction_norm = direction.norm(dim=-1)
        valid = direction_norm > self.eps
        direction = F.normalize(direction, dim=-1, eps=self.eps)
        transported_target = (
            torch.cos(target_angle)[:, None] * current_anchor
            + torch.sin(target_angle)[:, None] * direction
        )
        transported_target = torch.where(
            valid[:, None],
            F.normalize(transported_target, dim=-1, eps=self.eps),
            current_anchor,
        ).detach()
        cosine = (student * transported_target).sum(dim=-1)
        loss = (1.0 - cosine).mean()
        metrics = {
            "gbt_loss": loss.detach(),
            "gbt_active_rows": loss.new_tensor(float(shape[0])),
            "gbt_valid_direction_fraction": valid.float().mean().detach(),
            "gbt_target_angle": target_angle.mean().detach(),
            "gbt_student_target_cosine": cosine.mean().detach(),
            "gbt_anchor_target_cosine": torch.cos(target_angle).mean().detach(),
        }
        return loss, metrics
