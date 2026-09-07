#!/usr/bin/env python3
"""Measure mature-HS6 gaps on a full-bank expert bridge graph and build targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--anchor-bank", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--bridge-strength", type=float, default=0.25)
    parser.add_argument("--output-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalize(features: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    return features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), eps)


def numeric_summary(values: np.ndarray) -> dict[str, float]:
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
        "max": float(values.max()),
    }


def geodesic_targets(
    anchors: np.ndarray,
    prototypes: np.ndarray,
    *,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchors = normalize(anchors)
    prototypes = normalize(prototypes)
    cosine = np.clip(np.sum(anchors * prototypes, axis=-1), -1.0, 1.0)
    tangent = prototypes - cosine[:, None] * anchors
    tangent_norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    valid = tangent_norm[:, 0] > 1.0e-8
    tangent[valid] /= tangent_norm[valid]
    tangent[~valid] = 0
    angle = np.arccos(cosine) * float(strength)
    targets = np.cos(angle)[:, None] * anchors + np.sin(angle)[:, None] * tangent
    targets[~valid] = anchors[~valid]
    return normalize(targets), tangent, angle.astype(np.float32)


def shuffled_geodesic_targets(
    anchors: np.ndarray,
    tangent: np.ndarray,
    angle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(anchors) < 2:
        return anchors.copy(), np.arange(len(anchors), dtype=np.int32)
    permutation = np.roll(np.arange(len(anchors)), 1)
    shuffled_direction = tangent[permutation].copy()
    shuffled_direction -= (
        np.sum(shuffled_direction * anchors, axis=-1)[:, None] * anchors
    )
    direction_norm = np.linalg.norm(shuffled_direction, axis=-1, keepdims=True)
    valid = direction_norm[:, 0] > 1.0e-8
    shuffled_direction[valid] /= direction_norm[valid]
    shuffled_direction[~valid] = tangent[~valid]
    targets = (
        np.cos(angle)[:, None] * anchors
        + np.sin(angle)[:, None] * shuffled_direction
    )
    return normalize(targets), permutation.astype(np.int32)


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output_targets.exists():
        raise FileExistsError("Refusing to overwrite report or target bank")
    if args.margin < 0 or not 0 < args.bridge_strength <= 1:
        raise ValueError("margin must be non-negative and bridge-strength in (0, 1]")

    with np.load(args.graph, allow_pickle=False) as payload:
        graph = {name: np.asarray(payload[name]) for name in payload.files}
    with np.load(args.anchor_bank, allow_pickle=False) as payload:
        anchor_bank = {name: np.asarray(payload[name]) for name in payload.files}
    graph_keys = graph["keys"].astype(str)
    anchor_keys = anchor_bank["keys"].astype(str)
    if not np.array_equal(graph_keys, anchor_keys):
        raise ValueError("Bridge graph and mature-anchor bank keys differ")
    anchors = normalize(anchor_bank["features"])
    feature_protocol = str(
        np.asarray(anchor_bank.get("feature_protocol", np.asarray("final_cls"))).item()
    ).lower()
    input_normalization = str(
        np.asarray(anchor_bank.get("normalization_protocol", np.asarray("train"))).item()
    ).lower()
    transform_resize_crop = np.asarray(
        anchor_bank.get("transform_resize_crop", np.asarray([0, 0]))
    ).reshape(-1)
    observation_crop_size = int(transform_resize_crop[-1]) if len(transform_resize_crop) else 0
    sample_count = len(graph_keys)
    if anchors.shape[0] != sample_count:
        raise ValueError("Anchor feature count differs from graph keys")

    offsets = graph["offsets"].astype(np.int64)
    neighbors = graph["neighbor_indices"].astype(np.int64)
    confidence = graph["confidence"].astype(np.float32)
    if offsets.shape != (sample_count + 1,) or offsets[-1] != len(neighbors):
        raise ValueError("Invalid bridge CSR offsets")
    if confidence.shape != neighbors.shape:
        raise ValueError("Bridge confidence and neighbors differ")
    source = np.repeat(np.arange(sample_count, dtype=np.int64), np.diff(offsets))
    anchor_similarity = np.sum(anchors[source] * anchors[neighbors], axis=-1)
    residual = confidence - anchor_similarity - args.margin
    selected = residual > 0
    selected_source = source[selected]
    selected_neighbor = neighbors[selected]
    selected_weight = residual[selected].astype(np.float32)

    prototype_sum = np.zeros_like(anchors, dtype=np.float32)
    weight_sum = np.zeros(sample_count, dtype=np.float32)
    np.add.at(
        prototype_sum,
        selected_source,
        anchors[selected_neighbor] * selected_weight[:, None],
    )
    np.add.at(weight_sum, selected_source, selected_weight)
    active = np.flatnonzero(weight_sum > 0)
    prototypes = normalize(prototype_sum[active])
    true_targets, tangent, angle = geodesic_targets(
        anchors[active],
        prototypes,
        strength=args.bridge_strength,
    )
    control_targets, control_permutation = shuffled_geodesic_targets(
        anchors[active],
        tangent,
        angle,
    )
    true_cosine = np.sum(anchors[active] * true_targets, axis=-1)
    control_cosine = np.sum(anchors[active] * control_targets, axis=-1)
    if not np.allclose(true_cosine, control_cosine, atol=2.0e-4):
        raise RuntimeError("Shuffled control does not preserve per-sample angular displacement")

    selected_degree = np.bincount(selected_source, minlength=sample_count)
    args.output_targets.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_targets,
        keys=graph_keys[active],
        sample_indices=active.astype(np.int32),
        target_features=true_targets.astype(np.float16),
        control_target_features=control_targets.astype(np.float16),
        anchor_features=anchors[active].astype(np.float16),
        target_angle_radians=angle.astype(np.float32),
        selected_degree=selected_degree[active].astype(np.int32),
        residual_weight_sum=weight_sum[active].astype(np.float32),
        control_permutation=control_permutation,
        control_type=np.asarray("direction_shuffled_per_sample_angle_matched"),
        margin=np.asarray(args.margin, dtype=np.float32),
        bridge_strength=np.asarray(args.bridge_strength, dtype=np.float32),
        source_graph=np.asarray(str(args.graph)),
        source_anchor_bank=np.asarray(str(args.anchor_bank)),
        feature_protocol=np.asarray(feature_protocol),
        input_normalization=np.asarray(input_normalization),
        observation_crop_size=np.asarray(observation_crop_size, dtype=np.int32),
    )
    report = {
        "graph": str(args.graph),
        "anchor_bank": str(args.anchor_bank),
        "output_targets": str(args.output_targets),
        "samples": sample_count,
        "feature_dim": int(anchors.shape[1]),
        "feature_protocol": feature_protocol,
        "input_normalization": input_normalization,
        "observation_crop_size": observation_crop_size,
        "directed_edges": int(len(source)),
        "undirected_edges": int(np.sum(source < neighbors)),
        "margin": args.margin,
        "bridge_strength": args.bridge_strength,
        "expert_consensus_similarity": numeric_summary(confidence),
        "mature_anchor_similarity": numeric_summary(anchor_similarity),
        "raw_expert_minus_anchor": numeric_summary(confidence - anchor_similarity),
        "selected_directed_edges": int(selected.sum()),
        "selected_undirected_edges": int(np.sum(selected & (source < neighbors))),
        "selected_edge_fraction": float(np.mean(selected)),
        "active_samples": int(len(active)),
        "active_sample_fraction": float(len(active) / sample_count),
        "active_routed_fraction": float(len(active) / max(np.sum(np.diff(offsets) > 0), 1)),
        "selected_degree": numeric_summary(selected_degree[active]),
        "target_angle_radians": numeric_summary(angle),
        "target_cosine_to_anchor": numeric_summary(true_cosine),
        "control_cosine_to_anchor": numeric_summary(control_cosine),
        "control_max_abs_cosine_mismatch": float(
            np.max(np.abs(true_cosine - control_cosine))
        ),
        "gate_pass": bool(
            np.sum(selected & (source < neighbors)) >= 500
            and len(active) / sample_count >= 0.02
            and np.mean(confidence - anchor_similarity) > args.margin
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
