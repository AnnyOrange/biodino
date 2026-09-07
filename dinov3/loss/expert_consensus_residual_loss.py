# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the terms of the DINOv3 License Agreement.

"""Residual neighborhood distillation from agreeing frozen experts.

The mature model remains the target everywhere except on mutual-neighbor
edges supported by multiple experts and missed by that mature anchor.  This
makes the auxiliary objective a local ranking correction instead of a global
attempt to make one embedding imitate incompatible cell and tissue models.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _cosine_similarity(features: Tensor, eps: float) -> Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected [samples, features], got {tuple(features.shape)}")
    normalized = F.normalize(features.float(), dim=-1, eps=eps)
    return normalized @ normalized.T


def _off_diagonal(size: int, device: torch.device) -> Tensor:
    return ~torch.eye(size, dtype=torch.bool, device=device)


def _mutual_topk(similarity: Tensor, topk: int, candidate_edges: Tensor) -> Tensor:
    k = min(int(topk), similarity.shape[0] - 1)
    indices = torch.topk(
        similarity.masked_fill(~candidate_edges, -torch.inf),
        k=k,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    directed = torch.zeros_like(candidate_edges)
    directed.scatter_(1, indices, True)
    # Rows with fewer than k eligible cross-domain candidates otherwise pick
    # masked -inf entries. Intersecting again makes those padding picks inert.
    return directed & directed.T & candidate_edges


class ExpertConsensusResidualLoss(nn.Module):
    """Correct anchor-missed neighbors while preserving its other relations.

    Each expert can have a different feature width. ``expert_weights`` routes
    experts per sample, for example pathology experts for tissue/H&E and cell
    experts for fluorescence. An edge receives a vote only when both endpoints
    have positive weight for that expert and are mutual top-k neighbors.

    The target starts as the mature anchor's row-wise similarity distribution.
    On consensus edges where expert similarity exceeds anchor similarity, a
    detached positive residual is added to the anchor logits. KL distillation
    then changes only the ranking evidence that the anchor is missing.
    """

    def __init__(
        self,
        *,
        neighborhood_topk: int = 5,
        min_experts: int = 2,
        temperature: float = 0.07,
        min_residual: float = 0.0,
        residual_strength: float = 1.0,
        eps: float = 1.0e-8,
        shuffled_control: bool = False,
    ) -> None:
        super().__init__()
        if neighborhood_topk <= 0:
            raise ValueError(f"neighborhood_topk must be positive, got {neighborhood_topk}")
        if min_experts <= 0:
            raise ValueError(f"min_experts must be positive, got {min_experts}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if min_residual < 0:
            raise ValueError(f"min_residual must be non-negative, got {min_residual}")
        if residual_strength <= 0:
            raise ValueError(f"residual_strength must be positive, got {residual_strength}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.neighborhood_topk = int(neighborhood_topk)
        self.min_experts = int(min_experts)
        self.temperature = float(temperature)
        self.min_residual = float(min_residual)
        self.residual_strength = float(residual_strength)
        self.eps = float(eps)
        self.shuffled_control = bool(shuffled_control)

    @staticmethod
    def _shuffle_expert(features: Tensor, expert_index: int) -> Tensor:
        if features.shape[0] < 2:
            return features
        offset = 1 + (expert_index % (features.shape[0] - 1))
        return features.roll(shifts=offset, dims=0)

    def forward(
        self,
        *,
        student_features: Tensor,
        anchor_features: Tensor,
        expert_features: Sequence[Tensor],
        expert_weights: Tensor | None = None,
        edge_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if student_features.ndim != 2 or anchor_features.shape != student_features.shape:
            raise ValueError(
                "student_features and anchor_features must share [samples, features] shape, got "
                f"{tuple(student_features.shape)} and {tuple(anchor_features.shape)}"
            )
        num_samples = student_features.shape[0]
        if num_samples < 2:
            raise ValueError("Expert consensus requires at least two samples")
        if not expert_features:
            raise ValueError("At least one expert feature tensor is required")
        if self.min_experts > len(expert_features):
            raise ValueError(
                f"min_experts={self.min_experts} exceeds {len(expert_features)} supplied experts"
            )
        for index, features in enumerate(expert_features):
            if features.ndim != 2 or features.shape[0] != num_samples:
                raise ValueError(
                    f"expert {index} must have {num_samples} rows, got {tuple(features.shape)}"
                )

        device = student_features.device
        if expert_weights is None:
            expert_weights = torch.ones(
                (len(expert_features), num_samples), device=device, dtype=torch.float32
            )
        else:
            expected = (len(expert_features), num_samples)
            if expert_weights.shape != expected:
                raise ValueError(
                    f"expert_weights must have shape {expected}, got {tuple(expert_weights.shape)}"
                )
            expert_weights = expert_weights.to(device=device, dtype=torch.float32)
            if bool((expert_weights < 0).any()):
                raise ValueError("expert_weights must be non-negative")

        off_diagonal = _off_diagonal(num_samples, device)
        if edge_mask is None:
            candidate_edges = off_diagonal
        else:
            if edge_mask.shape != (num_samples, num_samples):
                raise ValueError(
                    "edge_mask must have shape "
                    f"{(num_samples, num_samples)}, got {tuple(edge_mask.shape)}"
                )
            candidate_edges = edge_mask.to(device=device, dtype=torch.bool)
            if not torch.equal(candidate_edges, candidate_edges.T):
                raise ValueError("edge_mask must be symmetric")
            candidate_edges = candidate_edges & off_diagonal
        vote_count = torch.zeros((num_samples, num_samples), device=device, dtype=torch.long)
        similarity_sum = torch.zeros((num_samples, num_samples), device=device)
        similarity_weight = torch.zeros_like(similarity_sum)
        expert_mutual_edges = []

        for expert_index, raw_features in enumerate(expert_features):
            features = raw_features.detach().to(device=device)
            if self.shuffled_control:
                features = self._shuffle_expert(features, expert_index)
            similarity = _cosine_similarity(features, self.eps)
            mutual = _mutual_topk(similarity, self.neighborhood_topk, candidate_edges)
            sample_weight = expert_weights[expert_index]
            pair_weight = sample_weight[:, None] * sample_weight[None, :]
            active = mutual & (pair_weight > 0)
            vote_count += active.to(dtype=torch.long)
            similarity_sum += similarity * pair_weight * active
            similarity_weight += pair_weight * active
            expert_mutual_edges.append(active.float().sum())

        consensus = (vote_count >= self.min_experts) & candidate_edges
        consensus_similarity = similarity_sum / similarity_weight.clamp_min(self.eps)
        anchor_similarity = _cosine_similarity(anchor_features.detach(), self.eps)
        residual = (consensus_similarity - anchor_similarity - self.min_residual).clamp_min(0.0)
        selected = consensus & (residual > 0) & (similarity_weight > 0)
        active_rows = selected.any(dim=-1)

        if not bool(active_rows.any()):
            zero = student_features.sum() * 0.0
            metrics = {
                "ecr_loss": zero.detach(),
                "ecr_active_rows": zero.detach(),
                "ecr_candidate_edges": candidate_edges.float().sum().detach(),
                "ecr_consensus_edges": consensus.float().sum().detach(),
                "ecr_selected_edges": zero.detach(),
                "ecr_selected_fraction": zero.detach(),
                "ecr_residual_mean": zero.detach(),
                "ecr_expert_mutual_edges_mean": torch.stack(expert_mutual_edges).mean().detach(),
                "ecr_shuffled_control": zero.new_tensor(float(self.shuffled_control)),
            }
            return zero, metrics

        anchor_logits = anchor_similarity / self.temperature
        target_logits = anchor_logits + (
            self.residual_strength * residual / self.temperature
        ) * selected
        student_logits = _cosine_similarity(student_features, self.eps) / self.temperature
        mask_value = torch.finfo(student_logits.dtype).min
        target_logits = target_logits.masked_fill(~off_diagonal, mask_value)
        student_logits = student_logits.masked_fill(~off_diagonal, mask_value)

        target_probability = F.softmax(target_logits.detach(), dim=-1)
        student_log_probability = F.log_softmax(student_logits, dim=-1)
        row_kl = F.kl_div(
            student_log_probability,
            target_probability,
            reduction="none",
        ).sum(dim=-1)
        loss = row_kl[active_rows].mean()

        selected_edges = selected.float().sum()
        metrics = {
            "ecr_loss": loss.detach(),
            "ecr_active_rows": active_rows.float().sum().detach(),
            "ecr_candidate_edges": candidate_edges.float().sum().detach(),
            "ecr_consensus_edges": consensus.float().sum().detach(),
            "ecr_selected_edges": selected_edges.detach(),
            "ecr_selected_fraction": (selected_edges / off_diagonal.float().sum()).detach(),
            "ecr_residual_mean": residual[selected].mean().detach(),
            "ecr_expert_mutual_edges_mean": torch.stack(expert_mutual_edges).mean().detach(),
            "ecr_shuffled_control": loss.new_tensor(float(self.shuffled_control)),
        }
        return loss, metrics
