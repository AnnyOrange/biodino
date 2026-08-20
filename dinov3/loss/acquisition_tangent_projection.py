# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Projected feature-space descent against frozen acquisition tangents.

The official anchor maps deterministic, label-preserving acquisition changes
to a per-image tangent subspace.  The forward pass below is exactly the
identity; only the gradient entering the adapted backbone is projected away
from that frozen nuisance subspace.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_tangent_inputs(
    *,
    anchor_features: Tensor,
    perturbed_anchor_features: Tensor,
) -> None:
    if anchor_features.ndim != 2:
        raise ValueError(f"Expected anchor_features [B, D], got {tuple(anchor_features.shape)}")
    if perturbed_anchor_features.ndim != 3:
        raise ValueError(
            "Expected perturbed_anchor_features [B, M, D], got "
            f"{tuple(perturbed_anchor_features.shape)}"
        )
    if (
        perturbed_anchor_features.shape[0] != anchor_features.shape[0]
        or perturbed_anchor_features.shape[-1] != anchor_features.shape[-1]
    ):
        raise ValueError(
            "perturbed_anchor_features must agree with anchor_features in B and D, got "
            f"{tuple(anchor_features.shape)} and {tuple(perturbed_anchor_features.shape)}"
        )
    if perturbed_anchor_features.shape[1] == 0:
        raise ValueError("At least one acquisition perturbation is required")


def build_acquisition_tangent_basis(
    *,
    anchor_features: Tensor,
    perturbed_anchor_features: Tensor,
    min_singular_value: float = 1.0e-4,
    relative_singular_value: float = 0.05,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Return a detached row-orthonormal nuisance basis and its active rows."""
    if min_singular_value < 0:
        raise ValueError("min_singular_value must be non-negative")
    if relative_singular_value < 0:
        raise ValueError("relative_singular_value must be non-negative")
    _validate_tangent_inputs(
        anchor_features=anchor_features,
        perturbed_anchor_features=perturbed_anchor_features,
    )

    anchor = anchor_features.detach().float()
    perturbations = perturbed_anchor_features.detach().float()
    tangent_rows = perturbations - anchor.unsqueeze(1)
    _, singular_values, basis = torch.linalg.svd(tangent_rows, full_matrices=False)
    largest = singular_values[:, :1]
    threshold = torch.maximum(
        singular_values.new_full((1,), float(min_singular_value)),
        largest * float(relative_singular_value),
    )
    active = singular_values > threshold
    metrics = {
        "acq_tangent_rank": active.sum(dim=-1).float().mean().detach(),
        "acq_tangent_singular_rms": singular_values.square().mean().sqrt().detach(),
    }
    return basis.detach(), active.detach(), metrics


def project_onto_acquisition_tangent(
    vectors: Tensor,
    *,
    tangent_basis: Tensor,
    active_rows: Tensor,
) -> Tensor:
    """Project ``[B, D]`` vectors onto a batched, rank-gated row basis."""
    if vectors.ndim != 2:
        raise ValueError(f"Expected vectors [B, D], got {tuple(vectors.shape)}")
    if tangent_basis.ndim != 3 or tangent_basis.shape[0] != vectors.shape[0] or tangent_basis.shape[-1] != vectors.shape[-1]:
        raise ValueError(
            "Expected tangent_basis [B, M, D] compatible with vectors, got "
            f"{tuple(tangent_basis.shape)} for {tuple(vectors.shape)}"
        )
    if active_rows.shape != tangent_basis.shape[:2]:
        raise ValueError(
            "active_rows must have shape [B, M], got "
            f"{tuple(active_rows.shape)} for basis {tuple(tangent_basis.shape)}"
        )

    work = vectors.float()
    basis = tangent_basis.detach().float()
    active = active_rows.detach().to(device=work.device, dtype=work.dtype)
    coefficients = torch.einsum("bd,bmd->bm", work, basis)
    projected = torch.einsum("bm,bmd->bd", coefficients * active, basis)
    return projected.to(dtype=vectors.dtype)


def acquisition_tangent_fraction(
    vectors: Tensor,
    *,
    tangent_basis: Tensor,
    active_rows: Tensor,
    eps: float = 1.0e-8,
) -> Tensor:
    """Return the mean fraction of vector energy in the frozen tangent."""
    projected = project_onto_acquisition_tangent(
        vectors,
        tangent_basis=tangent_basis,
        active_rows=active_rows,
    )
    nuisance_energy = projected.float().square().sum(dim=-1)
    total_energy = vectors.float().square().sum(dim=-1)
    return (nuisance_energy / (total_energy + float(eps))).mean()


def rank_matched_random_tangent_basis(tangent_basis: Tensor) -> Tensor:
    """Sample a row-orthonormal random control with the same available rank."""
    if tangent_basis.ndim != 3:
        raise ValueError(f"Expected tangent_basis [B, M, D], got {tuple(tangent_basis.shape)}")
    if tangent_basis.shape[-2] > tangent_basis.shape[-1]:
        raise ValueError("The number of tangent rows must not exceed the feature dimension")
    noise = torch.randn_like(tangent_basis.float())
    orthogonal_columns, _ = torch.linalg.qr(noise.transpose(-2, -1), mode="reduced")
    return orthogonal_columns.transpose(-2, -1).to(dtype=tangent_basis.dtype)


class _TangentGradientProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        features: Tensor,
        tangent_basis: Tensor,
        active_rows: Tensor,
        strength: float,
    ) -> Tensor:
        ctx.save_for_backward(tangent_basis.detach(), active_rows.detach())
        ctx.strength = float(strength)
        # A clone makes the forward identity explicit while giving autograd a
        # distinct tensor whose backward rule can be changed safely.
        return features.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, None, None]:
        tangent_basis, active_rows = ctx.saved_tensors
        projected = project_onto_acquisition_tangent(
            grad_output,
            tangent_basis=tangent_basis,
            active_rows=active_rows,
        )
        return grad_output - ctx.strength * projected, None, None, None


def apply_acquisition_tangent_gradient_projection(
    features: Tensor,
    *,
    tangent_basis: Tensor,
    active_rows: Tensor,
    strength: float = 1.0,
) -> Tensor:
    """Keep features unchanged while removing tangent gradient components.

    At ``strength=1`` this is the Euclidean projection of the incoming
    feature-space gradient onto the orthogonal complement of the local
    acquisition tangent.  Values in ``[0, 1]`` provide a controlled partial
    projection for ablations.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    if features.ndim != 2:
        raise ValueError(f"Expected features [B, D], got {tuple(features.shape)}")
    # Validate before entering autograd so configuration errors are immediate.
    project_onto_acquisition_tangent(
        features.detach(),
        tangent_basis=tangent_basis,
        active_rows=active_rows,
    )
    return _TangentGradientProjection.apply(features, tangent_basis, active_rows, float(strength))
