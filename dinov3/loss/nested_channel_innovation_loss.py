# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Losses for nested-channel conditional innovation learning.

The conditional predictor is optimized on detached full/subset features.  The
backbone therefore cannot reduce prediction error by deleting information from
the full-channel representation.  Backbone gradients only come from losses on
the residual left after conditional prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed.nn import functional as dist_nn


class ConditionalFeaturePredictor(nn.Module):
    """Residual MLP initialized as the identity conditional predictor."""

    def __init__(self, dim: int, hidden_dim: int = 0):
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim else 2 * int(dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        first, last = self.mlp[0], self.mlp[2]
        nn.init.trunc_normal_(first.weight, std=0.02)
        nn.init.zeros_(first.bias)
        # Zeroing the final layer keeps q(s) = s exactly while giving that
        # layer a direct gradient on the first update. Earlier layers begin to
        # learn as soon as the final layer leaves zero.
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: Tensor) -> Tensor:
        correction = self.mlp(self.norm(x))
        return x + correction


@dataclass(frozen=True)
class NestedChannelInnovationWeights:
    predictor: float = 1.0
    invariance: float = 1.0
    variance: float = 1.0
    orthogonality: float = 1.0


def _distributed_sum(value: Tensor) -> Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    return dist_nn.all_reduce(value, op=dist.ReduceOp.SUM)


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    numerator = _distributed_sum((values * weights).sum())
    denominator = _distributed_sum(weights.sum()).clamp_min(1.0)
    return numerator / denominator


def conditional_innovation_residual(
    *,
    full_features: Tensor,
    subset_features: Tensor,
    predictor: nn.Module,
    stop_gradient: bool,
) -> Tensor:
    """Return the residual after a detached conditional prediction.

    This mirrors the residual branch of :class:`NestedChannelInnovationLoss`.
    The predictor is trained only by its regression term; an innovation loss
    cannot be reduced by changing the conditional predictor itself.
    """
    if full_features.shape != subset_features.shape or full_features.ndim != 3:
        raise ValueError(
            "full_features and subset_features must have matching [V, B, D] shapes, got "
            f"{tuple(full_features.shape)} and {tuple(subset_features.shape)}"
        )
    subset_target = subset_features.detach() if stop_gradient else subset_features
    predicted_full = predictor(subset_target)
    return full_features.float() - predicted_full.detach().float()


def martingale_increment_orthogonality(
    *,
    upper_increment: Tensor,
    lower_increment: Tensor,
    active_samples: Tensor,
    eps: float = 1.0e-4,
    metric_prefix: str = "nci_martingale",
) -> tuple[Tensor, dict[str, Tensor]]:
    """Penalize correlation between adjacent nested conditional increments.

    For nested sigma-algebras S subset M subset F, exact conditional
    expectations yield orthogonal martingale differences.  This finite-batch
    normalized cross-covariance is the trainable counterpart of that identity.
    """
    if upper_increment.shape != lower_increment.shape or upper_increment.ndim != 3:
        raise ValueError(
            "upper_increment and lower_increment must have matching [V, B, D] shapes, got "
            f"{tuple(upper_increment.shape)} and {tuple(lower_increment.shape)}"
        )
    n_views, batch_size, dim = upper_increment.shape
    if active_samples.shape != (batch_size,):
        raise ValueError(
            f"active_samples must have shape {(batch_size,)}, got {tuple(active_samples.shape)}"
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    active = active_samples.to(device=upper_increment.device, dtype=torch.bool)
    weights = active.unsqueeze(0).expand(n_views, -1).reshape(-1).float()
    global_count = _distributed_sum(weights.sum().to(upper_increment)).clamp_min(0.0)
    if float(global_count.detach().item()) < 2.0:
        zero = upper_increment.sum() * 0.0 + lower_increment.sum() * 0.0
        metrics = {
            f"{metric_prefix}_cross_orthogonality": zero.detach(),
            f"{metric_prefix}_active_fraction": active.float().mean().detach(),
        }
        return zero, metrics

    upper_flat = upper_increment.reshape(-1, dim)
    lower_flat = lower_increment.reshape(-1, dim)
    weight_column = weights[:, None].to(upper_flat)
    denominator = (global_count - 1.0).clamp_min(1.0)

    upper_mean = _distributed_sum((upper_flat * weight_column).sum(dim=0)) / global_count
    lower_mean = _distributed_sum((lower_flat * weight_column).sum(dim=0)) / global_count
    upper_centered = upper_flat - upper_mean
    lower_centered = lower_flat - lower_mean
    upper_var = _distributed_sum((upper_centered.square() * weight_column).sum(dim=0)) / denominator
    lower_var = _distributed_sum((lower_centered.square() * weight_column).sum(dim=0)) / denominator
    cross_cov = _distributed_sum(
        (upper_centered * weight_column).transpose(0, 1) @ lower_centered
    ) / denominator
    cross_corr = cross_cov / torch.sqrt(
        (upper_var[:, None] * lower_var[None, :]).clamp_min(eps)
    )
    loss = cross_corr.square().mean()
    metrics = {
        f"{metric_prefix}_cross_orthogonality": loss.detach(),
        f"{metric_prefix}_active_fraction": active.float().mean().detach(),
    }
    return loss, metrics


class NestedChannelInnovationLoss(nn.Module):
    """Conditional prediction plus non-collapsed, orthogonal innovation.

    Inputs have shape ``[V, B, D]``. ``active_samples`` marks samples for
    which the subset actually omits at least one valid channel.
    """

    def __init__(
        self,
        *,
        min_std: float = 0.1,
        eps: float = 1e-4,
        stop_gradient: bool = True,
        metric_prefix: str = "nci",
        weights: NestedChannelInnovationWeights | None = None,
    ):
        super().__init__()
        if min_std < 0:
            raise ValueError(f"min_std must be non-negative, got {min_std}")
        self.min_std = float(min_std)
        self.eps = float(eps)
        self.stop_gradient = bool(stop_gradient)
        self.metric_prefix = str(metric_prefix).strip("_")
        if not self.metric_prefix:
            raise ValueError("metric_prefix must be non-empty")
        self.weights = weights or NestedChannelInnovationWeights()

    def _metric_name(self, suffix: str) -> str:
        return f"{self.metric_prefix}_{suffix}"

    def forward(
        self,
        *,
        full_features: Tensor,
        subset_features: Tensor,
        active_samples: Tensor,
        predictor: nn.Module,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if full_features.shape != subset_features.shape or full_features.ndim != 3:
            raise ValueError(
                "full_features and subset_features must have the same [V, B, D] shape, got "
                f"{tuple(full_features.shape)} and {tuple(subset_features.shape)}"
            )
        n_views, batch_size, dim = full_features.shape
        if n_views < 2:
            raise ValueError(f"NCI requires at least two global views, got {n_views}")
        if active_samples.shape != (batch_size,):
            raise ValueError(
                f"active_samples must have shape {(batch_size,)}, got {tuple(active_samples.shape)}"
            )

        active_samples = active_samples.to(device=full_features.device, dtype=torch.bool)
        view_weights = active_samples.unsqueeze(0).expand(n_views, -1)
        flat_weights = view_weights.reshape(-1).float()
        global_active_views = _distributed_sum(flat_weights.sum().to(full_features)).detach()
        if float(global_active_views.item()) == 0.0:
            zero = full_features.sum() * 0.0
            metrics = {
                self._metric_name(suffix): zero.detach()
                for suffix in (
                    "predictor_loss",
                    "invariance_loss",
                    "variance_loss",
                    "orthogonality_loss",
                    "residual_rms",
                    "residual_std",
                    "full_subset_cosine",
                    "positive_cosine",
                    "alignment_margin",
                    "predictor_r2",
                    "active_fraction",
                )
            }
            return zero, metrics

        # Both predictor inputs and targets are detached.  Prediction estimates
        # E[z_full | z_subset] but cannot make z_full easier to predict by
        # changing either representation.
        subset_target = subset_features.detach() if self.stop_gradient else subset_features
        full_target = full_features.detach() if self.stop_gradient else full_features
        predicted_full = predictor(subset_target)
        predictor_error = (predicted_full.float() - full_target.float()).square().mean(dim=-1)
        predictor_loss = _weighted_mean(predictor_error.reshape(-1), flat_weights)

        # Only these residual terms update the full-channel representation.
        predicted_detached = predicted_full.detach()
        residual = full_features.float() - predicted_detached.float()

        residual_normalized = F.normalize(residual, dim=-1, eps=self.eps)
        pair_losses = []
        pair_weights = []
        for first_view in range(n_views):
            for second_view in range(first_view + 1, n_views):
                pair_losses.append(
                    1.0 - (residual_normalized[first_view] * residual_normalized[second_view]).sum(dim=-1)
                )
                pair_weights.append(active_samples.float())
        invariance_loss = _weighted_mean(torch.cat(pair_losses), torch.cat(pair_weights))

        # Distributed weighted moments avoid bias when batches contain a mix of
        # low-channel (inactive) and true nested-channel samples.
        residual_flat = residual.reshape(-1, dim)
        common_flat = predicted_detached.float().reshape(-1, dim)
        weight_column = flat_weights[:, None].to(residual_flat)
        global_count = _distributed_sum(flat_weights.sum().to(residual_flat)).clamp_min(2.0)

        residual_sum = _distributed_sum((residual_flat * weight_column).sum(dim=0))
        common_sum = _distributed_sum((common_flat * weight_column).sum(dim=0))
        residual_mean = residual_sum / global_count
        common_mean = common_sum / global_count
        residual_centered = residual_flat - residual_mean
        common_centered = common_flat - common_mean

        residual_var = _distributed_sum(
            (residual_centered.square() * weight_column).sum(dim=0)
        ) / (global_count - 1.0)
        common_var = _distributed_sum(
            (common_centered.square() * weight_column).sum(dim=0)
        ) / (global_count - 1.0)
        residual_std = torch.sqrt(residual_var.clamp_min(self.eps))
        common_std = torch.sqrt(common_var.clamp_min(self.eps))
        variance_loss = F.relu(self.min_std - residual_std).mean()

        cross_cov = _distributed_sum(
            (common_centered * weight_column).transpose(0, 1) @ residual_centered
        ) / (global_count - 1.0)
        cross_corr = cross_cov / (common_std[:, None] * residual_std[None, :]).clamp_min(self.eps)
        # The mean avoids a D / N finite-batch amplification when feature width
        # is much larger than the number of active samples. It has the same
        # population zero and gradient direction as the Frobenius penalty.
        orthogonality_loss = cross_corr.square().mean()

        total = (
            self.weights.predictor * predictor_loss
            + self.weights.invariance * invariance_loss
            + self.weights.variance * variance_loss
            + self.weights.orthogonality * orthogonality_loss
        )

        with torch.no_grad():
            residual_rms = torch.sqrt(
                _weighted_mean(residual_flat.square().mean(dim=-1), flat_weights).clamp_min(0.0)
            )
            full_subset_cosine = _weighted_mean(
                F.cosine_similarity(full_features.float(), subset_features.float(), dim=-1).reshape(-1),
                flat_weights,
            )
            positive_cosine = 1.0 - invariance_loss.detach()
            if batch_size > 1:
                negative_cosine = _weighted_mean(
                    (residual_normalized[0] * residual_normalized[1].roll(1, dims=0)).sum(dim=-1),
                    active_samples.float(),
                )
            else:
                negative_cosine = positive_cosine.new_zeros(())
            full_for_metrics = full_features.detach().float()
            target_var = _weighted_mean(
                (full_for_metrics - full_for_metrics.mean(dim=(0, 1), keepdim=True))
                .square()
                .mean(dim=-1)
                .reshape(-1),
                flat_weights,
            ).clamp_min(self.eps)
            predictor_r2 = 1.0 - predictor_loss.detach() / target_var

        metrics = {
            self._metric_name("predictor_loss"): predictor_loss.detach(),
            self._metric_name("invariance_loss"): invariance_loss.detach(),
            self._metric_name("variance_loss"): variance_loss.detach(),
            self._metric_name("orthogonality_loss"): orthogonality_loss.detach(),
            self._metric_name("residual_rms"): residual_rms,
            self._metric_name("residual_std"): residual_std.mean().detach(),
            self._metric_name("full_subset_cosine"): full_subset_cosine,
            self._metric_name("positive_cosine"): positive_cosine,
            self._metric_name("alignment_margin"): positive_cosine - negative_cosine,
            self._metric_name("predictor_r2"): predictor_r2,
            self._metric_name("active_fraction"): active_samples.float().mean(),
        }
        return total, metrics
