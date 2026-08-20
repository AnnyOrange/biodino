# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Anchor-relative deflation of known acquisition nuisance directions.

The frozen official model defines a local feature-space nuisance orbit from
label-preserving imaging perturbations. This module penalizes only the portion
of a continued-training displacement that lies in that orbit; it deliberately
does not remove nuisance information from the whole foundation representation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AcquisitionOrbitDeflationLoss(nn.Module):
    """Measure adaptation energy in a detached local nuisance tangent basis."""

    def __init__(
        self,
        *,
        min_singular_value: float = 1.0e-4,
        relative_singular_value: float = 0.05,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if min_singular_value < 0:
            raise ValueError("min_singular_value must be non-negative")
        if relative_singular_value < 0:
            raise ValueError("relative_singular_value must be non-negative")
        self.min_singular_value = float(min_singular_value)
        self.relative_singular_value = float(relative_singular_value)
        self.eps = float(eps)

    def forward(
        self,
        *,
        current_features: Tensor,
        anchor_features: Tensor,
        perturbed_anchor_features: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Return orbit fraction, normal displacement, and detached diagnostics.

        Args:
            current_features: Adapted features with shape ``[B, D]``.
            anchor_features: Frozen official features with shape ``[B, D]``.
            perturbed_anchor_features: Frozen official features for ``M``
                physical acquisition perturbations, shape ``[B, M, D]``.
        """
        if current_features.ndim != 2:
            raise ValueError(f"Expected current_features [B, D], got {tuple(current_features.shape)}")
        if anchor_features.shape != current_features.shape:
            raise ValueError(
                "anchor_features must match current_features, got "
                f"{tuple(anchor_features.shape)} and {tuple(current_features.shape)}"
            )
        expected = (*current_features.shape[:1], None, current_features.shape[-1])
        if perturbed_anchor_features.ndim != 3 or (
            perturbed_anchor_features.shape[0] != current_features.shape[0]
            or perturbed_anchor_features.shape[-1] != current_features.shape[-1]
        ):
            raise ValueError(
                "Expected perturbed_anchor_features [B, M, D] compatible with current_features, got "
                f"{tuple(perturbed_anchor_features.shape)} for expected {expected}"
            )
        if perturbed_anchor_features.shape[1] == 0:
            raise ValueError("At least one nuisance perturbation is required")

        # The anchor and basis are calibration artifacts. Gradients must only
        # move the adapted model, never rotate its frozen nuisance orbit.
        anchor = anchor_features.detach().float()
        perturbations = perturbed_anchor_features.detach().float()
        tangent_rows = perturbations - anchor.unsqueeze(1)

        # Vh contains an orthonormal basis for the row space of the finite
        # difference tangents. Singular-value gating avoids arbitrary QR
        # directions when two physical perturbations have the same response.
        _, singular_values, basis = torch.linalg.svd(tangent_rows, full_matrices=False)
        largest = singular_values[:, :1]
        threshold = torch.maximum(
            singular_values.new_full((1,), self.min_singular_value),
            largest * self.relative_singular_value,
        )
        active = (singular_values > threshold).to(dtype=current_features.dtype)

        displacement = current_features.float() - anchor
        coefficients = torch.einsum("bd,bmd->bm", displacement, basis)
        projected = torch.einsum("bm,bmd->bd", coefficients * active, basis)
        normal_displacement = displacement - projected

        nuisance_energy = projected.square().sum(dim=-1)
        displacement_energy = displacement.square().sum(dim=-1)
        orbit_fraction = nuisance_energy / (displacement_energy + self.eps)
        # A zero displacement receives zero loss but is not made optimal by
        # this module alone; the caller must retain its main SSL objective and
        # a normal-displacement variance/invariance term.
        loss = orbit_fraction.mean()

        metrics = {
            "orbit_deflation_loss": loss.detach(),
            "orbit_nuisance_fraction": orbit_fraction.mean().detach(),
            "orbit_nuisance_rank": active.sum(dim=-1).mean().detach(),
            "orbit_displacement_rms": displacement.square().mean().sqrt().detach(),
            "orbit_normal_rms": normal_displacement.square().mean().sqrt().detach(),
        }
        return loss, normal_displacement.to(dtype=current_features.dtype), metrics
