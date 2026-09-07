#!/usr/bin/env python3
"""Paired, class-stratified bootstrap for the HPA-Cyclops retrieval screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


ARMS = ("baseline", "true", "shuffled")
READOUTS = ("nlb2_avg", "nlb2_cls")
ROLES = ("human", "yeast")
METRICS = ("map_at_5", "recall_at_1", "mrr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def per_query_metrics(
    gallery_features: np.ndarray,
    gallery_labels: np.ndarray,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    *,
    device: str,
    chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    """Return the per-query contributions used by query_gallery_metrics."""
    gallery_labels = np.asarray(gallery_labels).reshape(-1)
    query_labels = np.asarray(query_labels).reshape(-1)
    if not set(np.unique(query_labels)).issubset(set(np.unique(gallery_labels))):
        raise ValueError("Query labels are absent from the gallery")

    torch_device = torch.device(device)
    gallery = torch.as_tensor(gallery_features, dtype=torch.float32, device=torch_device)
    gallery = torch.nn.functional.normalize(gallery, dim=1)
    gallery_y = torch.as_tensor(gallery_labels, device=torch_device)
    max_k = min(10, len(gallery_labels))
    class_counts = {
        label: int((gallery_labels == label).sum()) for label in np.unique(gallery_labels)
    }
    outputs = {metric: np.zeros(len(query_labels), dtype=np.float64) for metric in METRICS}

    for start in range(0, len(query_labels), chunk_size):
        end = min(start + chunk_size, len(query_labels))
        query = torch.as_tensor(query_features[start:end], dtype=torch.float32, device=torch_device)
        query = torch.nn.functional.normalize(query, dim=1)
        query_y = torch.as_tensor(query_labels[start:end], device=torch_device)
        top_idx = torch.topk(query @ gallery.T, k=max_k, dim=1, largest=True, sorted=True).indices
        relevant = (gallery_y[top_idx] == query_y[:, None]).cpu().numpy()
        for local_index, rel in enumerate(relevant):
            output_index = start + local_index
            positive_ranks = np.flatnonzero(rel)
            if len(positive_ranks):
                outputs["mrr"][output_index] = 1.0 / float(positive_ranks[0] + 1)
            outputs["recall_at_1"][output_index] = float(rel[0])
            rel_at_5 = rel[: min(5, max_k)]
            denominator = min(class_counts[query_labels[output_index]], len(rel_at_5))
            if denominator:
                precision = np.cumsum(rel_at_5) / np.arange(1, len(rel_at_5) + 1)
                outputs["map_at_5"][output_index] = float(
                    (precision * rel_at_5).sum() / denominator
                )
    return outputs


def class_macro(values: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean([values[labels == label].mean() for label in np.unique(labels)]))


def paired_stratified_bootstrap(
    direction_values: dict[str, dict[str, dict[str, np.ndarray]]],
    direction_labels: dict[str, np.ndarray],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float | bool]]]]:
    """Bootstrap the bidirectional class macro while preserving arm pairing."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    observed = {
        arm: {
            metric: float(
                np.mean(
                    [
                        class_macro(direction_values[direction][arm][metric], direction_labels[direction])
                        for direction in direction_values
                    ]
                )
            )
            for metric in METRICS
        }
        for arm in ARMS
    }
    bootstraps = {
        metric: np.zeros((bootstrap_samples, len(ARMS)), dtype=np.float64) for metric in METRICS
    }
    direction_weight = 1.0 / len(direction_values)
    for direction, arm_values in direction_values.items():
        labels = direction_labels[direction]
        labels_unique = np.unique(labels)
        stratum_weight = direction_weight / len(labels_unique)
        for label in labels_unique:
            positions = np.flatnonzero(labels == label)
            sampled = rng.integers(0, len(positions), size=(bootstrap_samples, len(positions)))
            for metric in METRICS:
                values = np.stack(
                    [arm_values[arm][metric][positions] for arm in ARMS], axis=1
                )
                bootstraps[metric] += values[sampled].mean(axis=1) * stratum_weight

    contrasts: dict[str, dict[str, dict[str, float | bool]]] = {}
    for metric in METRICS:
        contrasts[metric] = {}
        for reference in ("baseline", "shuffled"):
            delta = bootstraps[metric][:, ARMS.index("true")] - bootstraps[metric][
                :, ARMS.index(reference)
            ]
            lower, upper = np.quantile(delta, (0.025, 0.975))
            contrasts[metric][f"true_vs_{reference}"] = {
                "observed": observed["true"][metric] - observed[reference][metric],
                "bootstrap_mean": float(delta.mean()),
                "ci95_low": float(lower),
                "ci95_high": float(upper),
                "probability_positive": float(np.mean(delta > 0)),
                "one_sided_p": float((1 + np.count_nonzero(delta <= 0)) / (bootstrap_samples + 1)),
                "significant_positive_95": bool(lower > 0),
            }
    return observed, contrasts


def _load_features(input_root: Path, arm: str, readout: str, role: str) -> dict[str, np.ndarray]:
    result_path = input_root / arm / readout / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    feature_path = Path(result["feature_files"][role])
    if not feature_path.exists():
        matches = sorted((result_path.parent / "features").glob(f"*_{role}.npz"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Cannot resolve feature file for {arm}/{readout}/{role}")
        feature_path = matches[0]
    with np.load(feature_path) as payload:
        return {
            "features": payload["features"].astype(np.float32),
            "labels": payload["labels"].astype(np.int64),
            "paths": payload["paths"].astype(str),
        }


def _validate_alignment(features: dict[str, dict[str, np.ndarray]], readout: str, role: str) -> None:
    reference = features["baseline"]
    for arm in ARMS[1:]:
        if not np.array_equal(features[arm]["labels"], reference["labels"]):
            raise ValueError(f"Label order differs for {readout}/{role}/{arm}")
        if not np.array_equal(features[arm]["paths"], reference["paths"]):
            raise ValueError(f"Path order differs for {readout}/{role}/{arm}")


def main() -> None:
    args = parse_args()
    payload: dict = {
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "inference_scope": (
            "Paired resampling of fixed evaluation queries, stratified by species direction and class; "
            "does not include continue-training seed variance."
        ),
        "readouts": {},
    }
    for readout_index, readout in enumerate(READOUTS):
        role_features = {
            role: {arm: _load_features(args.input_root, arm, readout, role) for arm in ARMS}
            for role in ROLES
        }
        for role in ROLES:
            _validate_alignment(role_features[role], readout, role)

        directions = {
            "human_query_yeast_gallery": ("human", "yeast"),
            "yeast_query_human_gallery": ("yeast", "human"),
        }
        direction_values = {}
        direction_labels = {}
        for direction, (query_role, gallery_role) in directions.items():
            direction_labels[direction] = role_features[query_role]["baseline"]["labels"]
            direction_values[direction] = {
                arm: per_query_metrics(
                    role_features[gallery_role][arm]["features"],
                    role_features[gallery_role][arm]["labels"],
                    role_features[query_role][arm]["features"],
                    role_features[query_role][arm]["labels"],
                    device=args.device,
                    chunk_size=args.chunk_size,
                )
                for arm in ARMS
            }
        observed, contrasts = paired_stratified_bootstrap(
            direction_values,
            direction_labels,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + readout_index,
        )
        reported = {
            arm: json.loads(
                (args.input_root / arm / readout / "result.json").read_text(encoding="utf-8")
            )["bidirectional_macro"]
            for arm in ARMS
        }
        max_match_error = max(
            abs(observed[arm][metric] - reported[arm][metric])
            for arm in ARMS
            for metric in METRICS
        )
        if max_match_error > 1e-6:
            raise ValueError(f"Per-query reconstruction mismatch for {readout}: {max_match_error}")
        payload["readouts"][readout] = {
            "observed": observed,
            "reported_metric_max_abs_error": max_match_error,
            "contrasts": contrasts,
        }

    primary = [
        payload["readouts"][readout]["contrasts"]["map_at_5"][f"true_vs_{reference}"]
        for readout in READOUTS
        for reference in ("baseline", "shuffled")
    ]
    payload["gates"] = {
        "all_map_deltas_positive": all(value["observed"] > 0 for value in primary),
        "all_map_ci95_exclude_zero": all(value["significant_positive_95"] for value in primary),
        "replicate_training_seeds": all(value["observed"] > 0 for value in primary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
