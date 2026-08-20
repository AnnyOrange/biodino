# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the terms of the DINOv3 License Agreement.

"""Conditional local morphology-graph anchoring for heterogeneous channels.

The full-channel EMA teacher supplies a dense patch-relation target.  A
predictor estimates which of those relations are already recoverable from a
strict channel subset.  Only the remaining, conditional-innovation edges are
used to anchor the full student graph.  Thus this is neither global Gram
anchoring nor raw channel reconstruction: the subset branch chooses *where*
the dense teacher needs protecting, but receives no backbone gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed.nn import functional as dist_nn


def _distributed_sum(value: Tensor) -> Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    return dist_nn.all_reduce(value, op=dist.ReduceOp.SUM)


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    numerator = _distributed_sum((values * weights).sum())
    denominator = _distributed_sum(weights.sum()).clamp_min(1.0)
    return numerator / denominator


@dataclass(frozen=True)
class ConditionalMorphologyGraphWeights:
    predictor: float = 1.0
    graph: float = 1.0


class ConditionalEdgeGraphPredictor(nn.Module):
    """Directly predict a full-channel local edge from a subset observation.

    This is a residual predictor for ``E[G_C[i,j] | T(S)]`` rather than a
    predictor for token features followed by a nonlinear cosine operation.
    Its zero-gated correction makes the initial prediction exactly the raw
    subset graph, which provides a conservative DINOv3-compatible start.
    """

    def __init__(self, dim: int, *, edge_dim: int = 64, hidden_dim: int = 0) -> None:
        super().__init__()
        if edge_dim <= 0:
            raise ValueError(f"edge_dim must be positive, got {edge_dim}")
        hidden_dim = int(hidden_dim) if hidden_dim else 2 * int(edge_dim)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, edge_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * edge_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        first, last = self.mlp[0], self.mlp[2]
        nn.init.trunc_normal_(first.weight, std=0.02)
        nn.init.zeros_(first.bias)
        # The correction head starts closed, so q(G_S) = G_S exactly. The
        # final layer nevertheless receives a gradient immediately.
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        subset_features: Tensor,
        source: Tensor,
        target: Tensor,
        subset_graph: Tensor,
    ) -> Tensor:
        if subset_features.ndim != 4:
            raise ValueError(
                "ConditionalEdgeGraphPredictor expects [V, B, P, D] features, got "
                f"{tuple(subset_features.shape)}"
            )
        if subset_graph.shape != (*subset_features.shape[:2], source.numel()):
            raise ValueError(
                "subset_graph must have [V, B, E] shape matching source edges, got "
                f"{tuple(subset_graph.shape)}"
            )
        projected = self.projection(self.norm(subset_features))
        source_features = F.normalize(projected[..., source, :], dim=-1)
        target_features = F.normalize(projected[..., target, :], dim=-1)
        pair_features = torch.cat(
            (
                source_features * target_features,
                (source_features - target_features).abs(),
                subset_graph.to(dtype=projected.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        correction = self.mlp(pair_features).squeeze(-1).float()
        base_graph = subset_graph.float()
        # Keep an edge prediction in the cosine range while preserving the
        # zero-correction identity: |delta| <= 1 - |G_S|.
        capacity = (1.0 - base_graph.abs()).clamp_min(0.0)
        return base_graph + capacity * torch.tanh(correction)


class ConditionalMorphologyGraphLoss(nn.Module):
    """Anchor local teacher patch relations selected by conditional innovation.

    All feature tensors have shape ``[V, B, P, D]``.  ``full_features`` is the
    masked, differentiable student stream; the teacher and subset streams are
    detached by default.  In ``feature`` mode the predictor learns
    ``q(z_S) ~= E[z_C | z_S]`` and then derives a graph. In ``edge`` mode it
    directly learns ``q(G_C | T(S))``, which is the quantity used by the
    innovation gate. The backbone only receives gradients from the weighted
    full-student to full-teacher graph discrepancy.
    """

    def __init__(
        self,
        *,
        local_radius: int = 2,
        min_innovation: float = 0.0,
        selection_fraction: float = 1.0,
        max_edge_weight: float = 4.0,
        huber_beta: float = 0.05,
        gate_mode: str = "conditional",
        predictor_mode: str = "feature",
        stop_gradient: bool = True,
        weights: ConditionalMorphologyGraphWeights | None = None,
    ) -> None:
        super().__init__()
        if local_radius <= 0:
            raise ValueError(f"local_radius must be positive, got {local_radius}")
        if min_innovation < 0:
            raise ValueError(f"min_innovation must be non-negative, got {min_innovation}")
        if not 0.0 < selection_fraction <= 1.0:
            raise ValueError(
                "selection_fraction must be in (0, 1], got "
                f"{selection_fraction}"
            )
        if max_edge_weight <= 0:
            raise ValueError(f"max_edge_weight must be positive, got {max_edge_weight}")
        if huber_beta <= 0:
            raise ValueError(f"huber_beta must be positive, got {huber_beta}")
        if gate_mode not in {"conditional", "uniform"}:
            raise ValueError(f"CMGI gate_mode must be 'conditional' or 'uniform', got {gate_mode!r}")
        if predictor_mode not in {"feature", "edge"}:
            raise ValueError(
                "CMGI predictor_mode must be 'feature' or 'edge', got "
                f"{predictor_mode!r}"
            )
        self.local_radius = int(local_radius)
        self.min_innovation = float(min_innovation)
        self.selection_fraction = float(selection_fraction)
        self.max_edge_weight = float(max_edge_weight)
        self.huber_beta = float(huber_beta)
        self.gate_mode = gate_mode
        self.predictor_mode = predictor_mode
        self.stop_gradient = bool(stop_gradient)
        self.weights = weights or ConditionalMorphologyGraphWeights()
        self._edge_cache: dict[tuple[int, str, int | None], tuple[Tensor, Tensor]] = {}

    def _local_edges(self, num_patches: int, device: torch.device) -> tuple[Tensor, Tensor]:
        key = (num_patches, device.type, device.index)
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached
        grid = int(num_patches**0.5)
        if grid * grid != num_patches:
            raise ValueError(f"CMGI requires a square patch grid, got {num_patches} patches")
        yy, xx = torch.meshgrid(torch.arange(grid, device=device), torch.arange(grid, device=device), indexing="ij")
        coordinates = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=1)
        dy = (coordinates[:, None, 0] - coordinates[None, :, 0]).abs()
        dx = (coordinates[:, None, 1] - coordinates[None, :, 1]).abs()
        keep = (dy <= self.local_radius) & (dx <= self.local_radius) & ((dy + dx) > 0)
        edges = keep.nonzero(as_tuple=True)
        self._edge_cache[key] = edges
        return edges

    @staticmethod
    def _edge_cosine(features: Tensor, source: Tensor, target: Tensor) -> Tensor:
        features = F.normalize(features.float(), dim=-1)
        return (features[..., source, :] * features[..., target, :]).sum(dim=-1)

    @staticmethod
    def _select_top_fraction(scores: Tensor, active_edges: Tensor, fraction: float) -> Tensor:
        """Select a fixed fraction of valid edges independently for each view.

        ``argsort`` is deliberate here: the number of valid edges changes with
        DINO masking, so a single global threshold would not make a
        density-matched control. Invalid edges are sent to the end of the
        ordering before the per-sample top fraction is retained.
        """
        if fraction >= 1.0:
            return active_edges
        ranked_scores = scores.masked_fill(~active_edges, float("-inf"))
        ordering = ranked_scores.argsort(dim=-1, descending=True)
        ranks = ordering.argsort(dim=-1)
        edge_counts = active_edges.sum(dim=-1)
        selected_counts = torch.ceil(edge_counts.float() * fraction).to(dtype=torch.long).clamp_min(1)
        return active_edges & (ranks < selected_counts.unsqueeze(-1))

    @staticmethod
    def _content_independent_edge_scores(
        source: Tensor,
        target: Tensor,
        *,
        n_views: int,
        batch_size: int,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return a deterministic pseudo-random edge ordering for the control.

        The control must match the conditional gate's density without reading
        image features or advancing the training RNG. An integer hash of the
        directed patch edge and its view/sample index gives each image a
        different stable ordering, avoiding a fixed spatial-edge bias.
        """
        edge_ids = (source.to(torch.int64) * 4099 + target.to(torch.int64) + 1).view(1, 1, -1)
        view_ids = torch.arange(n_views, device=source.device, dtype=torch.int64).view(-1, 1, 1)
        sample_ids = torch.arange(batch_size, device=source.device, dtype=torch.int64).view(1, -1, 1)
        keys = edge_ids + 8_191 * view_ids + 65_537 * sample_ids
        hashed = torch.remainder(keys * 1_103_515_245 + 12_345, 2_147_483_647)
        return hashed.to(dtype=dtype)

    @staticmethod
    def _validate_features(*features: Tensor) -> tuple[int, int, int, int]:
        reference_shape = features[0].shape
        if len(reference_shape) != 4:
            raise ValueError(f"CMGI features must have [V, B, P, D] shape, got {tuple(reference_shape)}")
        if any(feature.shape != reference_shape for feature in features[1:]):
            shapes = ", ".join(str(tuple(feature.shape)) for feature in features)
            raise ValueError(f"CMGI full/teacher/subset features must match, got {shapes}")
        return tuple(int(x) for x in reference_shape)

    def forward(
        self,
        *,
        full_features: Tensor,
        teacher_features: Tensor,
        subset_features: Tensor,
        active_samples: Tensor,
        predictor: nn.Module,
        masks: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        n_views, batch_size, num_patches, _ = self._validate_features(
            full_features, teacher_features, subset_features
        )
        if active_samples.shape != (batch_size,):
            raise ValueError(
                f"active_samples must have shape {(batch_size,)}, got {tuple(active_samples.shape)}"
            )
        if masks is not None and masks.shape != (n_views, batch_size, num_patches):
            raise ValueError(
                "masks must have [V, B, P] shape matching features, got "
                f"{tuple(masks.shape)} for {(n_views, batch_size, num_patches)}"
            )

        active = active_samples.to(device=full_features.device, dtype=torch.bool)
        active_views = active.unsqueeze(0).expand(n_views, -1)
        local_active_count = active_views.sum().to(full_features)
        global_active_count = _distributed_sum(local_active_count.detach())
        if float(global_active_count.item()) == 0.0:
            zero = full_features.sum() * 0.0
            metrics = {
                "cmgi_predictor_loss": zero.detach(),
                "cmgi_graph_loss": zero.detach(),
                "cmgi_teacher_innovation": zero.detach(),
                "cmgi_edge_gate_fraction": zero.detach(),
                "cmgi_active_fraction": zero.detach(),
                "cmgi_teacher_student_edge_error": zero.detach(),
                "cmgi_predictor_vs_subset_edge_delta": zero.detach(),
                "cmgi_predictor_mode_edge": zero.detach(),
            }
            return zero, metrics

        subset_input = subset_features.detach() if self.stop_gradient else subset_features
        teacher_target = teacher_features.detach() if self.stop_gradient else teacher_features
        source, target = self._local_edges(num_patches, full_features.device)
        student_graph = self._edge_cosine(full_features, source, target)
        teacher_graph = self._edge_cosine(teacher_target, source, target)
        subset_graph = self._edge_cosine(subset_input, source, target)

        if masks is None:
            visible_edges = torch.ones_like(student_graph, dtype=torch.bool)
        else:
            visible = ~masks.to(device=full_features.device, dtype=torch.bool)
            visible_edges = visible[..., source] & visible[..., target]
        active_edges = visible_edges & active_views.unsqueeze(-1)
        edge_count = active_edges.sum(dim=-1).clamp_min(1)

        if self.predictor_mode == "feature":
            predicted_teacher = predictor(subset_input)
            prediction_error = (
                F.normalize(predicted_teacher.float(), dim=-1)
                - F.normalize(teacher_target.float(), dim=-1)
            )
            prediction_error = prediction_error.square().mean(dim=(-1, -2))
            predictor_loss = _weighted_mean(prediction_error, active_views.float())
            predicted_graph = self._edge_cosine(predicted_teacher.detach(), source, target)
        else:
            predicted_graph = predictor(
                subset_input,
                source,
                target,
                subset_graph,
            )
            # Conditional expectation is the L2 projection of G_C onto the
            # subset-observable sigma algebra, so regress the graph itself.
            prediction_error = (predicted_graph - teacher_graph).square()
            predictor_loss = _weighted_mean(prediction_error, active_edges.float())

        # This is the conditional residual R(C|S) used solely as a detached
        # reliability weight. The backbone cannot lower it by erasing signal.
        innovation = (teacher_graph - predicted_graph).abs().detach()
        mean_innovation = (innovation * active_edges).sum(dim=-1) / edge_count
        if self.gate_mode == "uniform":
            # Control: match the conditional gate's selected-edge density but
            # do not read conditional innovation. At fraction=1 this is the
            # original dense uniform-teacher-graph control.
            uniform_scores = self._content_independent_edge_scores(
                source,
                target,
                n_views=n_views,
                batch_size=batch_size,
                dtype=innovation.dtype,
            )
            selected_edges = self._select_top_fraction(
                uniform_scores,
                active_edges,
                self.selection_fraction,
            )
            edge_weights = selected_edges.float()
        else:
            selected_edges = self._select_top_fraction(
                innovation,
                active_edges,
                self.selection_fraction,
            )
            edge_weights = innovation / mean_innovation.unsqueeze(-1).clamp_min(1.0e-6)
            edge_weights = edge_weights.clamp(max=self.max_edge_weight)
            edge_weights = edge_weights * selected_edges
            if self.min_innovation > 0:
                edge_weights = edge_weights * (innovation >= self.min_innovation)

        graph_error = F.smooth_l1_loss(
            student_graph,
            teacher_graph.detach(),
            beta=self.huber_beta,
            reduction="none",
        )
        graph_loss = _weighted_mean(graph_error, edge_weights)
        total_loss = self.weights.predictor * predictor_loss + self.weights.graph * graph_loss

        raw_edge_weights = active_edges.float()
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        metrics = {
            "cmgi_predictor_loss": predictor_loss.detach(),
            "cmgi_graph_loss": graph_loss.detach(),
            "cmgi_teacher_innovation": _weighted_mean(innovation, raw_edge_weights).detach(),
            "cmgi_edge_gate_fraction": _weighted_mean((edge_weights > 0).float(), raw_edge_weights).detach(),
            "cmgi_active_fraction": (global_active_count / float(n_views * batch_size * world_size)).detach(),
            "cmgi_teacher_student_edge_error": _weighted_mean(graph_error, raw_edge_weights).detach(),
            "cmgi_predictor_vs_subset_edge_delta": _weighted_mean(
                (teacher_graph - predicted_graph).abs() - (teacher_graph - subset_graph.detach()).abs(),
                raw_edge_weights,
            ).detach(),
            "cmgi_predictor_mode_edge": torch.tensor(
                float(self.predictor_mode == "edge"),
                device=full_features.device,
            ),
        }
        return total_loss, metrics
