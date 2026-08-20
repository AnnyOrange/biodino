# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Pretraining-relative relational transport for cross-scale adaptation.

The loss deliberately does *not* match a large model to a smaller adapted
model.  It compares the change in within-batch geometry from each model's own
pretraining anchor.  A cached small-model delta is therefore sufficient at
training time; the scout itself need not live in GPU memory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def centered_cosine_kernel(features: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Return a centered cosine sample kernel for ``[batch, feature]`` inputs."""
    if features.ndim != 2:
        raise ValueError(f"Expected [batch, feature] inputs, got {tuple(features.shape)}")
    if features.shape[0] < 2:
        raise ValueError("A relational kernel requires at least two samples")
    normalized = F.normalize(features.float(), dim=-1, eps=eps)
    kernel = normalized @ normalized.transpose(0, 1)
    row_mean = kernel.mean(dim=1, keepdim=True)
    return kernel - row_mean - row_mean.transpose(0, 1) + kernel.mean()


def _double_center(kernel: Tensor) -> Tensor:
    """Center an externally supplied sample kernel or kernel displacement."""
    row_mean = kernel.mean(dim=1, keepdim=True)
    return kernel - row_mean - row_mean.transpose(0, 1) + kernel.mean()


def cross_view_stable_kernel_delta(
    first_delta: Tensor,
    second_delta: Tensor,
    *,
    relative_eigenvalue: float = 0.05,
    min_eigenvalue: float = 1.0e-6,
    eps: float = 1.0e-8,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Keep only relation-change components reproducible across two views.

    If two views contain independent perturbation noise around the same
    biological kernel displacement, the expected symmetrized cross-product is
    positive semidefinite on the shared signal subspace.  The positive spectral
    part therefore supplies a small, batch-local estimator of that stable
    subspace without asking the large model to copy a scout's final geometry.
    """
    if first_delta.ndim != 2 or first_delta.shape[0] != first_delta.shape[1]:
        raise ValueError(f"Expected square first_delta, got {tuple(first_delta.shape)}")
    if second_delta.shape != first_delta.shape:
        raise ValueError(
            "second_delta must match first_delta, got "
            f"{tuple(second_delta.shape)} and {tuple(first_delta.shape)}"
        )
    if relative_eigenvalue < 0 or min_eigenvalue < 0 or eps <= 0:
        raise ValueError("relative_eigenvalue and min_eigenvalue must be non-negative and eps positive")

    first = _double_center(first_delta.detach().float())
    second = _double_center(second_delta.detach().float())
    average = 0.5 * (first + second)
    # For symmetric deltas, this is the self-adjoint cross-view covariance
    # operator. Negative eigenspaces encode view disagreement and are removed.
    cross_operator = 0.5 * (first @ second + second @ first)
    eigenvalues, eigenvectors = torch.linalg.eigh(cross_operator)
    largest_positive = eigenvalues.clamp_min(0.0).amax()
    threshold = torch.maximum(
        largest_positive.new_tensor(float(min_eigenvalue)),
        largest_positive * float(relative_eigenvalue),
    )
    active = eigenvalues > threshold
    projector = (eigenvectors * active.to(dtype=eigenvectors.dtype)) @ eigenvectors.transpose(0, 1)
    stable = _double_center(projector @ average @ projector)

    first_flat = first.flatten()
    second_flat = second.flatten()
    cross_view_alignment = torch.dot(first_flat, second_flat) / (
        first_flat.norm().clamp_min(eps) * second_flat.norm().clamp_min(eps)
    )
    average_norm = average.norm()
    stable_norm = stable.norm()
    metrics = {
        "skdt_cross_view_alignment": cross_view_alignment.detach(),
        "skdt_stable_rank": active.sum().detach(),
        "skdt_stable_energy_ratio": (stable_norm / average_norm.clamp_min(eps)).detach(),
        "skdt_stable_target_norm": stable_norm.detach(),
        "skdt_stable_active": (stable_norm > eps).to(dtype=stable.dtype).detach(),
    }
    return stable, metrics


class ScoutKernelDeltaTransportLoss(nn.Module):
    """Transport only a scout's normalized, pretraining-relative kernel change.

    Let ``K_L`` be the current large-model kernel, ``K_L0`` its frozen
    pretraining kernel, and ``Delta_s`` a cached scout adaptation delta.  The
    objective is ``1 - cos(K_L - K_L0, Delta_s)``.  The delta direction is
    width-agnostic and leaves the magnitude of the large-model update free.

    At the exact pretrained point the directional term is damped by
    ``directional_damping``.
    This is a trust-region scale, not a numerical afterthought: it bounds the
    initial directional gradient by ``1 / eps`` while retaining an initial push
    toward the scout direction without matching its absolute displacement.
    """

    def __init__(
        self,
        *,
        eps: float = 1e-4,
        directional_damping: float | None = None,
        displacement_budget_ratio: float = 0.0,
        metric_prefix: str = "skdt",
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if directional_damping is not None and directional_damping <= 0:
            raise ValueError(
                f"directional_damping must be positive when set, got {directional_damping}"
            )
        if displacement_budget_ratio < 0:
            raise ValueError(
                "displacement_budget_ratio must be non-negative, got "
                f"{displacement_budget_ratio}"
            )
        self.eps = float(eps)
        self.directional_damping = float(directional_damping or eps)
        # A positive budget makes transport local to the pretraining anchor.
        # The gate is detached so the large model cannot evade it by changing
        # its displacement norm instead of improving the transported direction.
        self.displacement_budget_ratio = float(displacement_budget_ratio)
        self.metric_prefix = str(metric_prefix).strip("_")
        if not self.metric_prefix:
            raise ValueError("metric_prefix must be non-empty")

    def _metric_name(self, suffix: str) -> str:
        return f"{self.metric_prefix}_{suffix}"

    def forward(
        self,
        *,
        current_features: Tensor,
        anchor_features: Tensor,
        scout_delta: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if current_features.shape != anchor_features.shape:
            raise ValueError(
                "current_features and anchor_features must have the same shape, got "
                f"{tuple(current_features.shape)} and {tuple(anchor_features.shape)}"
            )
        current_kernel = centered_cosine_kernel(current_features, eps=self.eps)
        anchor_kernel = centered_cosine_kernel(anchor_features, eps=self.eps)
        current_delta = current_kernel - anchor_kernel

        if scout_delta.ndim != 2 or scout_delta.shape[0] != scout_delta.shape[1]:
            raise ValueError(f"Expected a square scout_delta, got {tuple(scout_delta.shape)}")
        if scout_delta.shape != current_delta.shape:
            raise ValueError(
                "scout_delta must match the current sample kernel, got "
                f"{tuple(scout_delta.shape)} and {tuple(current_delta.shape)}"
            )

        # The scout target is a frozen calibration artifact.  Centering again
        # makes the contract robust when it was accumulated in lower precision.
        target_delta = _double_center(scout_delta.detach().float())
        current_flat = current_delta.reshape(-1)
        target_flat = target_delta.reshape(-1)
        target_norm = target_flat.norm()
        current_norm = current_flat.norm()

        # A zero scout change carries no direction and must not regularize L.
        if float(target_norm.detach().item()) <= self.eps:
            zero = current_features.sum() * 0.0
            metrics = {
                self._metric_name("loss"): zero.detach(),
                self._metric_name("alignment"): zero.detach(),
                self._metric_name("current_delta_norm"): current_norm.detach(),
                self._metric_name("scout_delta_norm"): target_norm.detach(),
                self._metric_name("relative_displacement"): zero.detach(),
                self._metric_name("budget_gate"): zero.detach(),
                self._metric_name("active"): zero.detach(),
            }
            return zero, metrics

        alignment = torch.dot(current_flat, target_flat) / (
            current_norm.clamp_min(self.directional_damping) * target_norm.clamp_min(self.eps)
        )
        relative_displacement = current_norm / target_norm.clamp_min(self.eps)
        budget_gate = current_norm.new_ones(())
        if self.displacement_budget_ratio > 0:
            budget = current_norm.new_tensor(self.displacement_budget_ratio)
            # This is one inside the local ball and decays as rho / r outside
            # it. It therefore transfers the Scout's tangent direction near
            # the anchor without turning it into a global final-state target.
            budget_gate = (budget / relative_displacement.clamp_min(budget)).detach()
        loss = budget_gate * (1.0 - alignment)
        metrics = {
            self._metric_name("loss"): loss.detach(),
            self._metric_name("alignment"): alignment.detach(),
            self._metric_name("current_delta_norm"): current_norm.detach(),
            self._metric_name("scout_delta_norm"): target_norm.detach(),
            self._metric_name("relative_displacement"): relative_displacement.detach(),
            self._metric_name("budget_gate"): budget_gate.detach(),
            self._metric_name("active"): loss.new_ones(()).detach(),
        }
        return loss, metrics
