# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Pure-SSL acquisition factorization with mature-topology preservation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed.nn import functional as dist_nn


INTERVENTION_FACTORIZATION_MODES = frozenset({"baseline", "true", "shuffled_context"})


class InterventionContextHead(nn.Module):
    """Encode a paired feature displacement and regress its intervention."""

    def __init__(
        self,
        dim: int,
        *,
        context_dim: int = 64,
        hidden_dim: int = 256,
        num_interventions: int = 3,
    ) -> None:
        super().__init__()
        if min(dim, context_dim, hidden_dim, num_interventions) <= 0:
            raise ValueError("Intervention context-head dimensions must be positive")
        self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, hidden_dim)
        self.embedding = nn.Linear(hidden_dim, context_dim)
        self.predictor = nn.Linear(context_dim, num_interventions)

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        for layer in (self.projection, self.embedding):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        # A closed predictor gives every balanced intervention the same initial
        # loss while allowing the predictor to learn on the first update.
        nn.init.zeros_(self.predictor.weight)
        nn.init.zeros_(self.predictor.bias)

    def forward(self, base_features: Tensor, intervention_features: Tensor) -> tuple[Tensor, Tensor]:
        if base_features.shape != intervention_features.shape or base_features.ndim != 2:
            raise ValueError(
                "Context-head inputs must have matching [B, D] shapes, got "
                f"{tuple(base_features.shape)} and {tuple(intervention_features.shape)}"
            )
        displacement = intervention_features - base_features
        hidden = F.gelu(self.projection(self.norm(displacement)))
        context = F.gelu(self.embedding(hidden))
        return context, self.predictor(context)


@dataclass(frozen=True)
class InterventionFactorizedTopologyWeights:
    invariance: float = 1.0
    context: float = 1.0
    decorrelation: float = 0.1
    sample_topology: float = 1.0
    patch_topology: float = 1.0


def make_balanced_intervention_assignments(
    batch_size: int,
    *,
    num_interventions: int,
    iteration: int,
    rank: int = 0,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return balanced true IDs and a deterministic correspondence shuffle."""
    if batch_size < 2:
        raise ValueError("At least two samples are required for a shuffled context control")
    if num_interventions < 2:
        raise ValueError("At least two interventions are required for factorization")
    if iteration < 0 or rank < 0:
        raise ValueError("iteration and rank must be non-negative")

    row_ids = torch.arange(batch_size, dtype=torch.long)
    intervention_ids = (row_ids + int(iteration) + int(rank)) % int(num_interventions)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(71_939 + 1_009 * int(iteration) + 9_176 * int(rank))

    # Select the deterministic permutation with the fewest accidental
    # same-class matches while retaining the exact intervention histogram.
    best_shuffled = None
    best_agreement = batch_size + 1
    for _ in range(8):
        candidate = intervention_ids[torch.randperm(batch_size, generator=generator)]
        agreement = int((candidate == intervention_ids).sum().item())
        if agreement < best_agreement:
            best_shuffled = candidate
            best_agreement = agreement
    if best_shuffled is None:
        raise RuntimeError("Failed to construct a shuffled intervention assignment")
    return intervention_ids.to(device=device), best_shuffled.to(device=device)


def value_and_gradient_matched_shuffled_mse(
    prediction: Tensor,
    *,
    true_target: Tensor,
    shuffled_target: Tensor,
    eps: float = 1.0e-12,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Shuffle targets while matching true loss value and output-gradient norm."""
    if prediction.shape != true_target.shape or prediction.shape != shuffled_target.shape:
        raise ValueError(
            "prediction, true_target, and shuffled_target must match, got "
            f"{tuple(prediction.shape)}, {tuple(true_target.shape)}, and {tuple(shuffled_target.shape)}"
        )
    if prediction.numel() == 0:
        raise ValueError("Matched shuffled regression requires a non-empty prediction")

    prediction_float = prediction.float()
    true_target = true_target.to(prediction_float)
    shuffled_target = shuffled_target.to(prediction_float)
    true_error = prediction_float - true_target
    shuffled_error = prediction_float - shuffled_target
    true_loss = true_error.square().mean()
    shuffled_loss = shuffled_error.square().mean()
    gradient_scale = (
        true_error.detach().square().sum().sqrt()
        / shuffled_error.detach().square().sum().sqrt().clamp_min(float(eps))
    )
    scaled_shuffled_loss = gradient_scale * shuffled_loss
    matched_loss = scaled_shuffled_loss + (true_loss - scaled_shuffled_loss).detach()
    metrics = {
        "ift_context_true_loss": true_loss.detach(),
        "ift_context_shuffled_raw_loss": shuffled_loss.detach(),
        "ift_context_gradient_scale": gradient_scale.detach(),
    }
    return matched_loss, metrics


def _distributed_sum(value: Tensor) -> Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    return dist_nn.all_reduce(value, op=dist.ReduceOp.SUM)


def _gather_rows(value: Tensor, *, with_grad: bool) -> Tensor:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    value = value.contiguous()
    if with_grad:
        return torch.cat(dist_nn.all_gather(value), dim=0)
    with torch.no_grad():
        gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, value)
    return torch.cat(gathered, dim=0)


class InterventionFactorizedTopologyLoss(nn.Module):
    """Separate acquisition context while preserving mature sample/patch geometry."""

    def __init__(
        self,
        *,
        num_interventions: int = 3,
        sample_topk: int = 8,
        patch_radius: int = 1,
        huber_beta: float = 0.05,
        eps: float = 1.0e-4,
        weights: InterventionFactorizedTopologyWeights | None = None,
    ) -> None:
        super().__init__()
        if num_interventions < 2:
            raise ValueError("num_interventions must be at least two")
        if sample_topk <= 0 or patch_radius <= 0:
            raise ValueError("sample_topk and patch_radius must be positive")
        if huber_beta <= 0 or eps <= 0:
            raise ValueError("huber_beta and eps must be positive")
        self.num_interventions = int(num_interventions)
        self.sample_topk = int(sample_topk)
        self.patch_radius = int(patch_radius)
        self.huber_beta = float(huber_beta)
        self.eps = float(eps)
        self.weights = weights or InterventionFactorizedTopologyWeights()
        self._edge_cache: dict[tuple[int, str, int | None], tuple[Tensor, Tensor]] = {}

    def _decorrelation_loss(self, concept: Tensor, context: Tensor) -> Tensor:
        concept = concept.float()
        context = context.float()
        count = _distributed_sum(concept.new_tensor(float(concept.shape[0]))).clamp_min(2.0)
        concept_mean = _distributed_sum(concept.sum(dim=0)) / count
        context_mean = _distributed_sum(context.sum(dim=0)) / count
        concept_centered = concept - concept_mean
        context_centered = context - context_mean
        denominator = (count - 1.0).clamp_min(1.0)
        concept_var = _distributed_sum(concept_centered.square().sum(dim=0)) / denominator
        context_var = _distributed_sum(context_centered.square().sum(dim=0)) / denominator
        cross_covariance = _distributed_sum(concept_centered.T @ context_centered) / denominator
        cross_correlation = cross_covariance / torch.sqrt(
            (concept_var[:, None] * context_var[None, :]).clamp_min(self.eps)
        )
        return cross_correlation.square().mean()

    def _sample_topology_loss(self, student: Tensor, anchor: Tensor) -> tuple[Tensor, Tensor]:
        student = _gather_rows(student.float(), with_grad=True)
        anchor = _gather_rows(anchor.detach().float(), with_grad=False)
        if student.shape != anchor.shape or student.ndim != 2:
            raise ValueError(
                "Student and anchor sample features must match [B, D], got "
                f"{tuple(student.shape)} and {tuple(anchor.shape)}"
            )
        if student.shape[0] < 2:
            zero = student.sum() * 0.0
            return zero, zero.detach()

        student_similarity = F.normalize(student, dim=-1) @ F.normalize(student, dim=-1).T
        anchor_similarity = F.normalize(anchor, dim=-1) @ F.normalize(anchor, dim=-1).T
        ranking_similarity = anchor_similarity.clone()
        ranking_similarity.fill_diagonal_(float("-inf"))
        topk = min(self.sample_topk, student.shape[0] - 1)
        neighbor_indices = ranking_similarity.topk(topk, dim=-1).indices
        student_neighbors = student_similarity.gather(1, neighbor_indices)
        anchor_neighbors = anchor_similarity.gather(1, neighbor_indices)
        error = F.smooth_l1_loss(
            student_neighbors,
            anchor_neighbors,
            beta=self.huber_beta,
            reduction="mean",
        )
        return error, (student_neighbors.detach() - anchor_neighbors).abs().mean()

    def _local_patch_edges(self, num_patches: int, device: torch.device) -> tuple[Tensor, Tensor]:
        key = (num_patches, device.type, device.index)
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached
        grid = int(num_patches**0.5)
        if grid * grid != num_patches:
            raise ValueError(f"Patch topology requires a square grid, got {num_patches} patches")
        yy, xx = torch.meshgrid(
            torch.arange(grid, device=device),
            torch.arange(grid, device=device),
            indexing="ij",
        )
        coordinates = torch.stack((yy.flatten(), xx.flatten()), dim=-1)
        delta = (coordinates[:, None] - coordinates[None, :]).abs()
        keep = (
            (delta[..., 0] <= self.patch_radius)
            & (delta[..., 1] <= self.patch_radius)
            & (delta.sum(dim=-1) > 0)
        )
        edges = keep.nonzero(as_tuple=True)
        self._edge_cache[key] = edges
        return edges

    def _patch_topology_loss(
        self,
        student: Tensor,
        anchor: Tensor,
        masks: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if student.shape != anchor.shape or student.ndim != 3:
            raise ValueError(
                "Student and anchor patches must match [B, P, D], got "
                f"{tuple(student.shape)} and {tuple(anchor.shape)}"
            )
        batch_size, num_patches, _ = student.shape
        if masks is not None and masks.shape != (batch_size, num_patches):
            raise ValueError(
                f"Patch masks must have shape {(batch_size, num_patches)}, got {tuple(masks.shape)}"
            )
        source, target = self._local_patch_edges(num_patches, student.device)
        student = F.normalize(student.float(), dim=-1)
        anchor = F.normalize(anchor.detach().float(), dim=-1)
        student_edges = (student[:, source] * student[:, target]).sum(dim=-1)
        anchor_edges = (anchor[:, source] * anchor[:, target]).sum(dim=-1)
        if masks is None:
            weights = torch.ones_like(student_edges)
        else:
            visible = ~masks.to(device=student.device, dtype=torch.bool)
            weights = (visible[:, source] & visible[:, target]).to(student_edges)
        edge_error = F.smooth_l1_loss(
            student_edges,
            anchor_edges,
            beta=self.huber_beta,
            reduction="none",
        )
        weight_sum = _distributed_sum(weights.sum()).clamp_min(1.0)
        loss = _distributed_sum((edge_error * weights).sum()) / weight_sum
        absolute_error = _distributed_sum(((student_edges - anchor_edges).abs() * weights).sum()) / weight_sum
        visible_fraction = _distributed_sum(weights.sum()) / _distributed_sum(
            weights.new_tensor(float(weights.numel()))
        ).clamp_min(1.0)
        return loss, absolute_error.detach(), visible_fraction.detach()

    def forward(
        self,
        *,
        base_features: Tensor,
        intervention_features: Tensor,
        anchor_features: Tensor,
        base_patches: Tensor,
        anchor_patches: Tensor,
        intervention_ids: Tensor,
        shuffled_intervention_ids: Tensor,
        context_head: InterventionContextHead,
        mode: str,
        patch_masks: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if mode not in INTERVENTION_FACTORIZATION_MODES:
            raise ValueError(f"Unknown intervention factorization mode {mode!r}")
        if base_features.shape != intervention_features.shape or base_features.shape != anchor_features.shape:
            raise ValueError("Base, intervention, and anchor features must have the same shape")
        if intervention_ids.shape != (base_features.shape[0],):
            raise ValueError("intervention_ids must contain one ID per sample")
        if shuffled_intervention_ids.shape != intervention_ids.shape:
            raise ValueError("shuffled_intervention_ids must match intervention_ids")

        context, prediction = context_head(base_features, intervention_features)
        true_target = F.one_hot(
            intervention_ids.to(dtype=torch.long),
            num_classes=self.num_interventions,
        ).to(prediction)
        shuffled_target = F.one_hot(
            shuffled_intervention_ids.to(dtype=torch.long),
            num_classes=self.num_interventions,
        ).to(prediction)
        true_context_loss = F.mse_loss(prediction.float(), true_target.float())
        context_metrics = {
            "ift_context_true_loss": true_context_loss.detach(),
            "ift_context_shuffled_raw_loss": F.mse_loss(
                prediction.float(), shuffled_target.float()
            ).detach(),
            "ift_context_gradient_scale": prediction.new_tensor(1.0),
        }
        if mode == "shuffled_context":
            context_loss, context_metrics = value_and_gradient_matched_shuffled_mse(
                prediction,
                true_target=true_target,
                shuffled_target=shuffled_target,
            )
        else:
            context_loss = true_context_loss

        base_normalized = F.normalize(base_features.float(), dim=-1, eps=self.eps)
        intervention_normalized = F.normalize(intervention_features.float(), dim=-1, eps=self.eps)
        invariance_loss = (1.0 - (base_normalized * intervention_normalized).sum(dim=-1)).mean()
        concept = 0.5 * (base_normalized + intervention_normalized)
        decorrelation_loss = self._decorrelation_loss(concept, context)
        sample_topology_loss, sample_topology_error = self._sample_topology_loss(
            base_features,
            anchor_features,
        )
        patch_topology_loss, patch_topology_error, visible_edge_fraction = self._patch_topology_loss(
            base_patches,
            anchor_patches,
            patch_masks,
        )

        raw_total = (
            self.weights.invariance * invariance_loss
            + self.weights.context * context_loss
            + self.weights.decorrelation * decorrelation_loss
            + self.weights.sample_topology * sample_topology_loss
            + self.weights.patch_topology * patch_topology_loss
        )
        optimization_loss = raw_total if mode != "baseline" else raw_total * 0.0
        predicted_ids = prediction.detach().argmax(dim=-1)
        metrics = {
            "ift_raw_total_loss": raw_total.detach(),
            "ift_invariance_loss": invariance_loss.detach(),
            "ift_positive_cosine": (1.0 - invariance_loss).detach(),
            "ift_context_loss": context_loss.detach(),
            "ift_context_accuracy": (predicted_ids == intervention_ids).float().mean(),
            "ift_context_optimized_target_accuracy": (
                predicted_ids
                == (shuffled_intervention_ids if mode == "shuffled_context" else intervention_ids)
            )
            .float()
            .mean(),
            "ift_shuffled_target_agreement": (
                shuffled_intervention_ids == intervention_ids
            ).float().mean(),
            "ift_decorrelation_loss": decorrelation_loss.detach(),
            "ift_sample_topology_loss": sample_topology_loss.detach(),
            "ift_sample_topology_abs_error": sample_topology_error,
            "ift_patch_topology_loss": patch_topology_loss.detach(),
            "ift_patch_topology_abs_error": patch_topology_error,
            "ift_patch_visible_edge_fraction": visible_edge_fraction,
            "ift_optimization_active": prediction.new_tensor(float(mode != "baseline")),
            "ift_shuffled_context": prediction.new_tensor(float(mode == "shuffled_context")),
            **context_metrics,
        }
        return optimization_loss, metrics
