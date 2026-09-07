# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Image-only intervention views and intervention-stable neighbor graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


CONSTRUCTION_VIEWS = ("identity", "gain", "offset", "psf")
HELDOUT_VIEW = "resolution"
ALL_VIEWS = (*CONSTRUCTION_VIEWS, HELDOUT_VIEW)


def make_intervention_views(images: Tensor) -> dict[str, Tensor]:
    """Return conservative acquisition views of unnormalized ``[0, 1]`` images."""
    if images.ndim != 4:
        raise ValueError(f"Expected images [B, C, H, W], got {tuple(images.shape)}")
    if not images.is_floating_point():
        raise TypeError("images must be floating point")
    if images.shape[-2] < 4 or images.shape[-1] < 4:
        raise ValueError("images must have spatial size at least 4 x 4")

    work = images.float().clamp(0.0, 1.0)
    mean = work.mean(dim=(-2, -1), keepdim=True)
    std = work.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
    gain = (mean + 1.10 * (work - mean)).clamp(0.0, 1.0)
    offset = (work + 0.05 * std).clamp(0.0, 1.0)
    blurred = F.avg_pool2d(
        F.pad(work, (1, 1, 1, 1), mode="reflect"),
        kernel_size=3,
        stride=1,
    )
    psf = (0.75 * work + 0.25 * blurred).clamp(0.0, 1.0)
    low_size = (
        max(1, int(round(work.shape[-2] * 0.75))),
        max(1, int(round(work.shape[-1] * 0.75))),
    )
    low = F.interpolate(work, size=low_size, mode="bilinear", align_corners=False, antialias=True)
    resolution = F.interpolate(
        low,
        size=work.shape[-2:],
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).clamp(0.0, 1.0)
    return {
        "identity": work.to(dtype=images.dtype),
        "gain": gain.to(dtype=images.dtype),
        "offset": offset.to(dtype=images.dtype),
        "psf": psf.to(dtype=images.dtype),
        "resolution": resolution.to(dtype=images.dtype),
    }


def normalize_rows(features: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a feature matrix [N, D], got {array.shape}")
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), eps)


def aligned_view_cosines(features: np.ndarray) -> np.ndarray:
    """Return identity-to-view cosine for a normalized ``[V, N, D]`` bank."""
    array = np.asarray(features)
    if array.ndim != 3 or array.shape[0] < 2:
        raise ValueError(f"Expected features [V, N, D] with V >= 2, got {array.shape}")
    identity = normalize_rows(array[0])
    return np.stack(
        [np.einsum("nd,nd->n", identity, normalize_rows(array[index])) for index in range(array.shape[0])]
    )


def _topk_without_aligned_self(
    scores: Tensor,
    *,
    row_start: int,
    k: int,
    gallery_valid: Tensor,
) -> Tensor:
    scores[:, ~gallery_valid] = -torch.inf
    row_ids = torch.arange(row_start, row_start + scores.shape[0], device=scores.device)
    scores[torch.arange(scores.shape[0], device=scores.device), row_ids] = -torch.inf
    return torch.topk(scores, k=k, dim=1, largest=True, sorted=True).indices


def cross_view_knn(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    *,
    k: int,
    query_valid: np.ndarray | None = None,
    gallery_valid: np.ndarray | None = None,
    device: str = "cpu",
    chunk_size: int = 512,
) -> np.ndarray:
    """Exact cosine kNN between aligned feature banks, excluding the same key."""
    query = normalize_rows(query_features)
    gallery = normalize_rows(gallery_features)
    if query.shape != gallery.shape:
        raise ValueError(f"Aligned query/gallery banks must match, got {query.shape} and {gallery.shape}")
    n_samples = query.shape[0]
    if not 0 < k < n_samples:
        raise ValueError(f"k must be in [1, N-1], got k={k}, N={n_samples}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    query_mask = np.ones(n_samples, dtype=bool) if query_valid is None else np.asarray(query_valid, dtype=bool)
    gallery_mask = (
        np.ones(n_samples, dtype=bool) if gallery_valid is None else np.asarray(gallery_valid, dtype=bool)
    )
    if query_mask.shape != (n_samples,) or gallery_mask.shape != (n_samples,):
        raise ValueError("valid masks must have shape [N]")
    if int(gallery_mask.sum()) <= k:
        raise ValueError("Not enough valid gallery samples for requested k")

    torch_device = torch.device(device)
    compute_dtype = torch.float16 if torch_device.type == "cuda" else torch.float32
    gallery_tensor = torch.as_tensor(gallery, device=torch_device, dtype=compute_dtype)
    gallery_valid_tensor = torch.as_tensor(gallery_mask, device=torch_device)
    neighbors = np.full((n_samples, k), -1, dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            query_tensor = torch.as_tensor(query[start:end], device=torch_device, dtype=compute_dtype)
            scores = query_tensor @ gallery_tensor.T
            indices = _topk_without_aligned_self(
                scores,
                row_start=start,
                k=k,
                gallery_valid=gallery_valid_tensor,
            ).cpu().numpy()
            valid_rows = query_mask[start:end]
            neighbors[start:end][valid_rows] = indices[valid_rows]
    return neighbors


def mutual_edge_codes(forward: np.ndarray, reverse: np.ndarray) -> np.ndarray:
    """Return directed ``src * N + dst`` codes for reciprocal kNN edges."""
    forward = np.asarray(forward, dtype=np.int64)
    reverse = np.asarray(reverse, dtype=np.int64)
    if forward.ndim != 2 or reverse.shape != forward.shape:
        raise ValueError("forward and reverse neighbor arrays must have the same [N, K] shape")
    n_samples = forward.shape[0]
    safe = np.clip(forward, 0, max(0, n_samples - 1))
    reverse_for_candidates = reverse[safe]
    rows = np.arange(n_samples, dtype=np.int64)[:, None, None]
    mutual = (reverse_for_candidates == rows).any(axis=2) & (forward >= 0)
    src = np.broadcast_to(np.arange(n_samples, dtype=np.int64)[:, None], forward.shape)[mutual]
    dst = forward[mutual]
    return src * np.int64(n_samples) + dst


def retain_top_edges(
    codes: np.ndarray,
    weights: np.ndarray,
    *,
    n_samples: int,
    max_neighbors: int,
    min_weight: float,
    tie_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Threshold and retain deterministic top-weight outgoing edges."""
    codes = np.asarray(codes, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    if codes.shape != weights.shape:
        raise ValueError("codes and weights must have identical shapes")
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive")
    keep = weights >= float(min_weight)
    src = codes[keep] // np.int64(n_samples)
    dst = codes[keep] % np.int64(n_samples)
    weights = weights[keep]
    identity = normalize_rows(tie_features)
    similarities = np.einsum("nd,nd->n", identity[src], identity[dst]) if len(src) else np.empty(0)

    selected: list[int] = []
    for row in np.unique(src):
        candidates = np.flatnonzero(src == row)
        order = np.lexsort((dst[candidates], -similarities[candidates], -weights[candidates]))
        selected.extend(candidates[order[:max_neighbors]].tolist())
    selected_array = np.asarray(selected, dtype=np.int64)
    return src[selected_array], dst[selected_array], weights[selected_array]


@dataclass(frozen=True)
class SparseRelationGraph:
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    n_samples: int

    @property
    def codes(self) -> np.ndarray:
        return self.src.astype(np.int64) * np.int64(self.n_samples) + self.dst.astype(np.int64)


def _graph_from_neighbor_pair(
    forward: np.ndarray,
    reverse: np.ndarray,
    *,
    tie_features: np.ndarray,
    max_neighbors: int,
) -> SparseRelationGraph:
    n_samples = forward.shape[0]
    codes = mutual_edge_codes(forward, reverse)
    src = codes // np.int64(n_samples)
    dst = codes % np.int64(n_samples)
    identity = normalize_rows(tie_features)
    weight = np.einsum("nd,nd->n", identity[src], identity[dst]).astype(np.float32)
    src, dst, weight = retain_top_edges(
        codes,
        weight,
        n_samples=n_samples,
        max_neighbors=max_neighbors,
        min_weight=-1.0,
        tie_features=tie_features,
    )
    return SparseRelationGraph(src, dst, weight, n_samples)


def _survival_graph(
    view_features: np.ndarray,
    *,
    view_valid: np.ndarray,
    k: int,
    max_neighbors: int,
    min_survival: float,
    device: str,
    chunk_size: int,
) -> SparseRelationGraph:
    n_views, n_samples, _ = view_features.shape
    neighbors: dict[tuple[int, int], np.ndarray] = {}
    for query_view in range(n_views):
        for gallery_view in range(n_views):
            neighbors[(query_view, gallery_view)] = cross_view_knn(
                view_features[query_view],
                view_features[gallery_view],
                k=k,
                query_valid=view_valid[query_view],
                gallery_valid=view_valid[gallery_view],
                device=device,
                chunk_size=chunk_size,
            )

    events = []
    for query_view in range(n_views):
        for gallery_view in range(n_views):
            events.append(
                mutual_edge_codes(
                    neighbors[(query_view, gallery_view)],
                    neighbors[(gallery_view, query_view)],
                )
            )
    all_codes = np.concatenate(events) if events else np.empty(0, dtype=np.int64)
    unique_codes, counts = np.unique(all_codes, return_counts=True)
    src = unique_codes // np.int64(n_samples)
    dst = unique_codes % np.int64(n_samples)
    valid_counts = view_valid[:, src].sum(axis=0) * view_valid[:, dst].sum(axis=0)
    survival = counts.astype(np.float32) / np.maximum(valid_counts, 1)
    src, dst, survival = retain_top_edges(
        unique_codes,
        survival,
        n_samples=n_samples,
        max_neighbors=max_neighbors,
        min_weight=min_survival,
        tie_features=view_features[0],
    )
    return SparseRelationGraph(src, dst, survival, n_samples)


def match_outdegree(
    candidate: SparseRelationGraph,
    target: SparseRelationGraph,
) -> SparseRelationGraph:
    """Keep the best candidate edges while matching every target out-degree."""
    if candidate.n_samples != target.n_samples:
        raise ValueError("candidate and target graphs must have the same number of samples")
    target_degree = np.bincount(target.src, minlength=target.n_samples)
    selected: list[int] = []
    for row, degree in enumerate(target_degree):
        if degree == 0:
            continue
        candidates = np.flatnonzero(candidate.src == row)
        if len(candidates) < degree:
            raise ValueError(
                f"Candidate graph has {len(candidates)} edges for row {row}, fewer than target {degree}"
            )
        order = np.lexsort((candidate.dst[candidates], -candidate.weight[candidates]))
        selected.extend(candidates[order[:degree]].tolist())
    index = np.asarray(selected, dtype=np.int64)
    return SparseRelationGraph(
        candidate.src[index],
        candidate.dst[index],
        candidate.weight[index],
        candidate.n_samples,
    )


def build_relation_graphs(
    features_by_view: np.ndarray,
    *,
    view_names: tuple[str, ...] = ALL_VIEWS,
    k: int = 20,
    max_neighbors: int = 5,
    min_survival: float = 0.50,
    min_safe_cosine: float = 0.85,
    shuffle_seed: int = 0,
    device: str = "cpu",
    chunk_size: int = 512,
) -> tuple[dict[str, SparseRelationGraph], dict[str, object]]:
    """Build label-free single, mean, ISRD, and correspondence controls."""
    features = np.asarray(features_by_view)
    if features.ndim != 3:
        raise ValueError(f"Expected features [V, N, D], got {features.shape}")
    if len(view_names) != features.shape[0] or tuple(view_names) != ALL_VIEWS:
        raise ValueError(f"Expected ordered views {ALL_VIEWS}, got {view_names}")
    if features.shape[1] <= k:
        raise ValueError("Feature bank must contain more than k samples")

    features = np.stack([normalize_rows(view) for view in features])
    cosines = aligned_view_cosines(features)
    construction = features[: len(CONSTRUCTION_VIEWS)]
    construction_cosines = cosines[: len(CONSTRUCTION_VIEWS)]
    construction_valid = construction_cosines >= float(min_safe_cosine)
    construction_valid[0] = True

    identity_knn = cross_view_knn(
        construction[0], construction[0], k=k, device=device, chunk_size=chunk_size
    )
    single = _graph_from_neighbor_pair(
        identity_knn,
        identity_knn,
        tie_features=construction[0],
        max_neighbors=max_neighbors,
    )

    valid_count = construction_valid.sum(axis=0).astype(np.float32)
    mean_features = (construction * construction_valid[:, :, None]).sum(axis=0)
    mean_features /= np.maximum(valid_count[:, None], 1.0)
    mean_features = normalize_rows(mean_features)
    mean_knn = cross_view_knn(mean_features, mean_features, k=k, device=device, chunk_size=chunk_size)
    mean_graph = _graph_from_neighbor_pair(
        mean_knn,
        mean_knn,
        tie_features=construction[0],
        max_neighbors=max_neighbors,
    )

    isrd = _survival_graph(
        construction,
        view_valid=construction_valid,
        k=k,
        max_neighbors=max_neighbors,
        min_survival=min_survival,
        device=device,
        chunk_size=chunk_size,
    )

    rng = np.random.default_rng(shuffle_seed)
    shuffled = construction.copy()
    shuffled_valid = construction_valid.copy()
    for view_index in range(1, len(CONSTRUCTION_VIEWS)):
        permutation = rng.permutation(features.shape[1])
        shuffled[view_index] = shuffled[view_index, permutation]
        shuffled_valid[view_index] = shuffled_valid[view_index, permutation]
    shuffled_candidates = _survival_graph(
        shuffled,
        view_valid=shuffled_valid,
        k=k,
        max_neighbors=max_neighbors,
        min_survival=0.0,
        device=device,
        chunk_size=chunk_size,
    )
    shuffled_graph = match_outdegree(shuffled_candidates, isrd)

    heldout_forward = cross_view_knn(
        features[0],
        features[-1],
        k=k,
        device=device,
        chunk_size=chunk_size,
    )
    heldout_reverse = cross_view_knn(
        features[-1],
        features[0],
        k=k,
        device=device,
        chunk_size=chunk_size,
    )
    heldout_codes = np.unique(mutual_edge_codes(heldout_forward, heldout_reverse))
    graphs = {
        "single": single,
        "mean": mean_graph,
        "isrd": isrd,
        "view_shuffled": shuffled_graph,
    }
    diagnostics: dict[str, object] = {
        "view_cosine_median": {
            name: float(np.median(cosines[index])) for index, name in enumerate(view_names)
        },
        "view_unsafe_fraction": {
            name: float(np.mean(cosines[index] < min_safe_cosine))
            for index, name in enumerate(view_names[: len(CONSTRUCTION_VIEWS)])
        },
        "heldout_codes": heldout_codes,
    }
    return graphs, diagnostics


def graph_metrics(
    graph: SparseRelationGraph,
    *,
    heldout_codes: np.ndarray | None = None,
) -> dict[str, float]:
    outdegree = np.bincount(graph.src, minlength=graph.n_samples)
    indegree = np.bincount(graph.dst, minlength=graph.n_samples)
    total = max(1, int(indegree.sum()))
    nonzero = indegree[indegree > 0].astype(np.float64)
    if len(nonzero):
        probabilities = nonzero / nonzero.sum()
        effective_destinations = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    else:
        effective_destinations = 0.0
    metrics = {
        "edges": float(len(graph.src)),
        "coverage": float(np.mean(outdegree > 0)),
        "mean_outdegree": float(outdegree.mean()),
        "max_indegree_fraction": float(indegree.max(initial=0) / total),
        "effective_destination_fraction": float(effective_destinations / graph.n_samples),
        "mean_weight": float(graph.weight.mean()) if len(graph.weight) else 0.0,
    }
    if heldout_codes is not None:
        metrics["heldout_survival"] = (
            float(np.isin(graph.codes, np.asarray(heldout_codes, dtype=np.int64)).mean())
            if len(graph.src)
            else 0.0
        )
    return metrics


def graph_label_precision(graph: SparseRelationGraph, labels: np.ndarray) -> float:
    """Evaluation-only edge precision; labels never enter graph construction."""
    labels = np.asarray(labels)
    if labels.shape != (graph.n_samples,):
        raise ValueError("labels must have shape [N]")
    if not len(graph.src):
        return float("nan")
    return float(np.mean(labels[graph.src] == labels[graph.dst]))


def graph_payload(graphs: Mapping[str, SparseRelationGraph]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for name, graph in graphs.items():
        payload[f"{name}_src"] = graph.src.astype(np.int64)
        payload[f"{name}_dst"] = graph.dst.astype(np.int64)
        payload[f"{name}_weight"] = graph.weight.astype(np.float32)
    return payload
