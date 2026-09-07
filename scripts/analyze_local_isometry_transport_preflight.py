#!/usr/bin/env python3
"""Preflight a local-isometry firewall around cross-domain bridge targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-bank", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument(
        "--axis",
        choices=("organism", "acquisition", "acquisition_family"),
        required=True,
    )
    parser.add_argument("--local-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--smoothing-lambdas",
        type=float,
        nargs="+",
        default=(0.25, 0.5, 1.0, 2.0, 4.0),
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--knn-block-size", type=int, default=1024)
    parser.add_argument("--min-retained-gain", type=float, default=0.5)
    parser.add_argument("--min-q90-reduction", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-targets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalize_tensor(features: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(eps)


def numeric_summary(values: np.ndarray | torch.Tensor) -> dict[str, float | int]:
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q25": 0.0,
            "median": 0.0,
            "q75": 0.0,
            "q90": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def geodesic_directions(
    anchors: torch.Tensor,
    targets: torch.Tensor,
    angles: torch.Tensor,
    *,
    eps: float = 1.0e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover unit tangent directions from angle-matched spherical targets."""
    anchors = normalize_tensor(anchors.float())
    targets = normalize_tensor(targets.float())
    tangent = targets - (targets * anchors).sum(dim=-1, keepdim=True) * anchors
    tangent_norm = tangent.norm(dim=-1, keepdim=True)
    valid = (angles.abs() > eps) & (tangent_norm[:, 0] > eps)
    directions = torch.zeros_like(tangent)
    directions[valid] = tangent[valid] / tangent_norm[valid]
    return directions, valid


def geodesic_targets(
    anchors: torch.Tensor,
    directions: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    anchors = normalize_tensor(anchors.float())
    directions = directions - (directions * anchors).sum(dim=-1, keepdim=True) * anchors
    direction_norm = directions.norm(dim=-1, keepdim=True)
    valid = direction_norm[:, 0] > 1.0e-7
    unit_direction = torch.zeros_like(directions)
    unit_direction[valid] = directions[valid] / direction_norm[valid]
    targets = torch.cos(angles)[:, None] * anchors + torch.sin(angles)[:, None] * unit_direction
    targets[~valid] = anchors[~valid]
    return normalize_tensor(targets)


def _known_axis_mask(labels: np.ndarray) -> np.ndarray:
    normalized = np.char.lower(np.char.strip(labels.astype(str)))
    unknown = np.isin(normalized, ("", "unknown", "unresolved", "none", "nan"))
    return ~unknown


@torch.no_grad()
def build_same_axis_graph(
    anchors: torch.Tensor,
    labels: np.ndarray,
    *,
    local_k: int,
    temperature: float,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build a symmetric, weighted mature-anchor kNN graph within each axis."""
    if local_k < 1:
        raise ValueError("local-k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(labels) != len(anchors):
        raise ValueError("Axis-label count differs from anchor count")

    device = anchors.device
    labels = labels.astype(str)
    known = _known_axis_mask(labels)
    source_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    group_sizes: dict[str, int] = {}

    for label in sorted(set(labels[known].tolist())):
        group_np = np.flatnonzero(known & (labels == label))
        group_sizes[label] = int(len(group_np))
        if len(group_np) < 2:
            continue
        group = torch.as_tensor(group_np, dtype=torch.long, device=device)
        group_features = anchors[group]
        neighbors_per_row = min(local_k, len(group_np) - 1)
        for start in range(0, len(group_np), block_size):
            stop = min(start + block_size, len(group_np))
            similarities = group_features[start:stop] @ group_features.T
            row = torch.arange(stop - start, device=device)
            similarities[row, torch.arange(start, stop, device=device)] = -torch.inf
            values, local_neighbors = similarities.topk(neighbors_per_row, dim=1)
            weights = torch.softmax(values / temperature, dim=1)
            sources = group[start:stop, None].expand_as(local_neighbors)
            source_parts.append(sources.reshape(-1))
            target_parts.append(group[local_neighbors.reshape(-1)])
            weight_parts.append(weights.reshape(-1))

    if not source_parts:
        raise ValueError("No same-axis group contains at least two active samples")
    source = torch.cat(source_parts)
    target = torch.cat(target_parts)
    weight = torch.cat(weight_parts)

    # Symmetrization makes the fixed-point update correspond to a graph-Laplacian
    # trust region rather than a one-way neighbor propagation heuristic.
    indices = torch.stack((torch.cat((source, target)), torch.cat((target, source))))
    values = torch.cat((weight, weight))
    adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        size=(len(anchors), len(anchors)),
        device=device,
    ).coalesce()
    degree = torch.zeros(len(anchors), device=device)
    degree.scatter_add_(0, adjacency.indices()[0], adjacency.values())
    unique_source, unique_target = adjacency.indices()
    unique_edges = unique_source < unique_target
    edge_source = unique_source[unique_edges]
    edge_target = unique_target[unique_edges]
    report = {
        "known_samples": int(known.sum()),
        "excluded_unknown_samples": int((~known).sum()),
        "axis_groups": group_sizes,
        "directed_knn_edges_before_symmetrization": int(len(source)),
        "undirected_edges_after_symmetrization": int(unique_edges.sum().item()),
        "degree": numeric_summary(degree),
    }
    return adjacency, degree, torch.stack((edge_source, edge_target)), report


@torch.no_grad()
def smooth_tangent_fields(
    anchors: torch.Tensor,
    raw_fields: torch.Tensor,
    adjacency: torch.Tensor,
    degree: torch.Tensor,
    confidence: torch.Tensor,
    *,
    smoothing_lambda: float,
    iterations: int,
) -> torch.Tensor:
    """Jacobi-solve confidence-weighted log-map graph regularization."""
    if smoothing_lambda < 0:
        raise ValueError("smoothing-lambda must be non-negative")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if raw_fields.ndim != 3 or raw_fields.shape[:2] != (2, len(anchors)):
        raise ValueError("raw-fields must have shape [2, samples, feature_dim]")

    sample_count, feature_dim = raw_fields.shape[1:]
    current = raw_fields.permute(1, 0, 2).reshape(sample_count, 2 * feature_dim)
    raw = current.clone()
    denominator = confidence + float(smoothing_lambda) * degree
    for _ in range(iterations):
        neighbors = torch.sparse.mm(adjacency, current)
        current = (confidence[:, None] * raw + float(smoothing_lambda) * neighbors) / denominator[:, None].clamp_min(
            1.0e-12
        )
        current = current.reshape(sample_count, 2, feature_dim)
        current -= (current * anchors[:, None, :]).sum(dim=-1, keepdim=True) * anchors[:, None, :]
        current = current.reshape(sample_count, 2 * feature_dim)

    return current.reshape(sample_count, 2, feature_dim).permute(1, 0, 2)


def local_kernel_distortion(
    anchors: torch.Tensor,
    targets: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    source, target = edges
    anchor_kernel = (anchors[source] * anchors[target]).sum(dim=-1)
    target_kernel = (targets[source] * targets[target]).sum(dim=-1)
    return (target_kernel - anchor_kernel).abs()


def retained_tangent_gain(
    raw_tangent: torch.Tensor,
    new_tangent: torch.Tensor,
    confidence: torch.Tensor,
) -> float:
    """Return the fraction of the raw first-order bridge progress retained."""
    numerator = confidence * (raw_tangent * new_tangent).sum(dim=-1)
    denominator = confidence * raw_tangent.square().sum(dim=-1)
    if float(denominator.sum()) <= 1.0e-12:
        return 0.0
    return float(numerator.sum() / denominator.sum())


def angle_match_control_to_true(smoothed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match shuffled output angles while retaining its smoothed directions."""
    if smoothed.ndim != 3 or smoothed.shape[0] != 2:
        raise ValueError("smoothed fields must have shape [2, samples, feature_dim]")
    operator_angles = smoothed.norm(dim=-1)
    control_norm = operator_angles[1, :, None]
    valid = control_norm[:, 0] > 1.0e-7
    control_direction = torch.zeros_like(smoothed[1])
    control_direction[valid] = smoothed[1, valid] / control_norm[valid]
    calibrated = smoothed.clone()
    calibrated[1] = control_direction * operator_angles[0, :, None]
    return calibrated, operator_angles


def confidence_weights(payload: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    raw = np.asarray(
        payload.get("residual_weight_sum", np.ones(len(payload["keys"]))),
        dtype=np.float32,
    )
    positive = raw[raw > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    normalized = np.clip(raw / max(scale, 1.0e-12), 0.25, 4.0)
    return torch.as_tensor(normalized, device=device)


def _q90(values: torch.Tensor) -> float:
    return float(torch.quantile(values.float(), 0.90).item()) if len(values) else 0.0


def evaluate_candidate(
    anchors: torch.Tensor,
    raw_fields: torch.Tensor,
    adjacency: torch.Tensor,
    degree: torch.Tensor,
    edges: torch.Tensor,
    confidence: torch.Tensor,
    *,
    smoothing_lambda: float,
    iterations: int,
    min_retained_gain: float,
    min_q90_reduction: float,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    operator_smoothed = smooth_tangent_fields(
        anchors,
        raw_fields,
        adjacency,
        degree,
        confidence,
        smoothing_lambda=smoothing_lambda,
        iterations=iterations,
    )
    smoothed, operator_angles = angle_match_control_to_true(operator_smoothed)
    raw_angles = raw_fields.norm(dim=-1)
    smooth_angles = smoothed.norm(dim=-1)
    raw_targets = torch.stack([geodesic_targets(anchors, raw_fields[index], raw_angles[index]) for index in range(2)])
    smooth_targets = torch.stack(
        [geodesic_targets(anchors, smoothed[index], smooth_angles[index]) for index in range(2)]
    )
    arms = ("true", "shuffled")
    metrics: dict[str, Any] = {"smoothing_lambda": float(smoothing_lambda)}
    for index, arm in enumerate(arms):
        raw_distortion = local_kernel_distortion(anchors, raw_targets[index], edges)
        smooth_distortion = local_kernel_distortion(anchors, smooth_targets[index], edges)
        raw_q90 = _q90(raw_distortion)
        smooth_q90 = _q90(smooth_distortion)
        reduction = 1.0 - smooth_q90 / max(raw_q90, 1.0e-12)
        metrics[arm] = {
            "retained_tangent_gain": retained_tangent_gain(raw_fields[index], smoothed[index], confidence),
            "raw_target_angle_radians": numeric_summary(raw_angles[index]),
            "operator_target_angle_radians": numeric_summary(operator_angles[index]),
            "smoothed_target_angle_radians": numeric_summary(smooth_angles[index]),
            "displacement_energy_ratio": float(
                (confidence * smoothed[index].square().sum(dim=-1)).sum()
                / (confidence * raw_fields[index].square().sum(dim=-1)).sum().clamp_min(1.0e-12)
            ),
            "raw_local_kernel_distortion": numeric_summary(raw_distortion),
            "smoothed_local_kernel_distortion": numeric_summary(smooth_distortion),
            "q90_distortion_reduction_fraction": float(reduction),
        }
    metrics["gate_pass"] = bool(
        metrics["true"]["retained_tangent_gain"] >= min_retained_gain
        and metrics["true"]["q90_distortion_reduction_fraction"] >= min_q90_reduction
    )
    metrics["causal_audit"] = {
        "smoothed_true_minus_shuffled_q90_distortion": float(
            metrics["true"]["smoothed_local_kernel_distortion"]["q90"]
            - metrics["shuffled"]["smoothed_local_kernel_distortion"]["q90"]
        ),
        "true_minus_shuffled_retained_gain": float(
            metrics["true"]["retained_tangent_gain"] - metrics["shuffled"]["retained_tangent_gain"]
        ),
        "max_abs_output_angle_mismatch": float((smooth_angles[0] - smooth_angles[1]).abs().max()),
    }
    return metrics, smoothed, smooth_targets


def save_target_bank(
    output: Path,
    source_payload: dict[str, np.ndarray],
    smooth_targets: torch.Tensor,
    *,
    axis: str,
    local_k: int,
    temperature: float,
    smoothing_lambda: float,
    iterations: int,
    source_target_bank: Path,
    gate_pass: bool,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite target bank: {output}")
    payload = dict(source_payload)
    payload["source_target_angle_radians"] = np.asarray(source_payload["target_angle_radians"])
    payload["target_angle_radians"] = (
        torch.acos(
            (normalize_tensor(torch.as_tensor(payload["anchor_features"]).float()) * smooth_targets[0].cpu())
            .sum(dim=-1)
            .clamp(-1.0, 1.0)
        )
        .numpy()
        .astype(np.float32)
    )
    payload["control_target_angle_radians"] = (
        torch.acos(
            (normalize_tensor(torch.as_tensor(payload["anchor_features"]).float()) * smooth_targets[1].cpu())
            .sum(dim=-1)
            .clamp(-1.0, 1.0)
        )
        .numpy()
        .astype(np.float32)
    )
    payload["target_features"] = smooth_targets[0].cpu().numpy().astype(np.float16)
    payload["control_target_features"] = smooth_targets[1].cpu().numpy().astype(np.float16)
    payload.update(
        {
            "firewall_axis": np.asarray(axis),
            "firewall_local_k": np.asarray(local_k, dtype=np.int32),
            "firewall_temperature": np.asarray(temperature, dtype=np.float32),
            "firewall_lambda": np.asarray(smoothing_lambda, dtype=np.float32),
            "firewall_iterations": np.asarray(iterations, dtype=np.int32),
            "firewall_gate_pass": np.asarray(gate_pass),
            "firewall_control_type": np.asarray("direction_shuffled_same_operator_per_sample_output_angle_matched"),
            "source_target_bank": np.asarray(str(source_target_bank)),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {args.output}")
    if args.local_k < 1 or args.knn_block_size < 1:
        raise ValueError("local-k and knn-block-size must be positive")
    if args.iterations < 1 or args.temperature <= 0:
        raise ValueError("iterations and temperature must be positive")
    if any(value < 0 for value in args.smoothing_lambdas):
        raise ValueError("smoothing-lambdas must be non-negative")

    device = resolve_device(args.device)
    with np.load(args.target_bank, allow_pickle=False) as archive:
        target_payload = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(args.graph, allow_pickle=False) as archive:
        graph_payload = {name: np.asarray(archive[name]) for name in archive.files}
    axis = "acquisition_family" if args.axis == "acquisition" else args.axis

    indices = target_payload["sample_indices"].astype(np.int64)
    graph_keys = graph_payload["keys"].astype(str)
    target_keys = target_payload["keys"].astype(str)
    if not np.array_equal(graph_keys[indices], target_keys):
        raise ValueError("Target-bank keys do not align with graph sample indices")
    if axis not in graph_payload:
        raise KeyError(f"Graph does not contain the requested axis: {axis}")

    anchors = normalize_tensor(torch.as_tensor(target_payload["anchor_features"], device=device).float())
    true_targets = torch.as_tensor(target_payload["target_features"], device=device).float()
    control_targets = torch.as_tensor(target_payload["control_target_features"], device=device).float()
    angles = torch.as_tensor(target_payload["target_angle_radians"], device=device).float()
    true_direction, true_valid = geodesic_directions(anchors, true_targets, angles)
    control_direction, control_valid = geodesic_directions(anchors, control_targets, angles)
    if not bool(torch.equal(true_valid, control_valid)):
        raise RuntimeError("True and shuffled banks have different valid-angle masks")
    raw_fields = torch.stack((true_direction, control_direction)) * angles[None, :, None]
    confidence = confidence_weights(target_payload, device)
    labels = graph_payload[axis].astype(str)[indices]

    adjacency, degree, edges, graph_report = build_same_axis_graph(
        anchors,
        labels,
        local_k=args.local_k,
        temperature=args.temperature,
        block_size=args.knn_block_size,
    )
    candidate_reports: list[dict[str, Any]] = []
    selected: tuple[dict[str, Any], torch.Tensor, torch.Tensor] | None = None
    closest: tuple[dict[str, Any], torch.Tensor, torch.Tensor] | None = None
    for smoothing_lambda in args.smoothing_lambdas:
        result = evaluate_candidate(
            anchors,
            raw_fields,
            adjacency,
            degree,
            edges,
            confidence,
            smoothing_lambda=smoothing_lambda,
            iterations=args.iterations,
            min_retained_gain=args.min_retained_gain,
            min_q90_reduction=args.min_q90_reduction,
        )
        candidate_reports.append(result[0])
        if (
            closest is None
            or result[0]["true"]["q90_distortion_reduction_fraction"]
            > closest[0]["true"]["q90_distortion_reduction_fraction"]
        ):
            closest = result
        if selected is None and result[0]["gate_pass"]:
            selected = result

    gate_pass = selected is not None
    chosen = selected if selected is not None else closest
    assert chosen is not None
    if args.output_targets is not None and gate_pass:
        save_target_bank(
            args.output_targets,
            target_payload,
            chosen[2],
            axis=axis,
            local_k=args.local_k,
            temperature=args.temperature,
            smoothing_lambda=chosen[0]["smoothing_lambda"],
            iterations=args.iterations,
            source_target_bank=args.target_bank,
            gate_pass=True,
        )

    report = {
        "method": "local_isometry_constrained_cross_domain_transport_preflight",
        "status": "gate_pass" if gate_pass else "gate_fail_no_training_authorized",
        "target_bank": str(args.target_bank),
        "graph": str(args.graph),
        "axis": axis,
        "device": str(device),
        "samples": int(len(anchors)),
        "feature_dim": int(anchors.shape[1]),
        "valid_displacements": int(true_valid.sum().item()),
        "local_k": args.local_k,
        "temperature": args.temperature,
        "iterations": args.iterations,
        "selection_rule": "smallest preregistered lambda passing both true-arm gates",
        "gates": {
            "min_retained_tangent_gain": args.min_retained_gain,
            "min_q90_distortion_reduction_fraction": args.min_q90_reduction,
            "uses_downstream_labels": False,
            "true_and_shuffled_use_identical_graph_operator": True,
            "shuffled_output_angles_matched_to_true_per_sample": True,
        },
        "local_graph": graph_report,
        "confidence_weight": numeric_summary(confidence),
        "candidates": candidate_reports,
        "gate_pass": gate_pass,
        "selected_lambda": (float(selected[0]["smoothing_lambda"]) if selected is not None else None),
        "closest_candidate_lambda": float(chosen[0]["smoothing_lambda"]),
        "output_targets": (str(args.output_targets) if args.output_targets is not None and gate_pass else None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
