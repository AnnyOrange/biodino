#!/usr/bin/env python3
"""Paired query bootstrap for the four-arm topology ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_cross_species_paired_bootstrap import METRICS, class_macro, per_query_metrics


ARMS = ("baseline", "sample", "patch", "joint")
READOUTS = ("nlb2_avg", "nlb2_cls")
ROLES = ("human", "yeast")
CONTRASTS = (
    ("sample", "baseline"),
    ("patch", "baseline"),
    ("joint", "baseline"),
    ("joint", "sample"),
    ("joint", "patch"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def load_features(root: Path, arm: str, readout: str, role: str) -> dict[str, np.ndarray]:
    result_path = root / arm / readout / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    feature_path = Path(result["feature_files"][role])
    if not feature_path.exists():
        matches = sorted((result_path.parent / "features").glob(f"*_{role}.npz"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Cannot resolve {arm}/{readout}/{role}")
        feature_path = matches[0]
    with np.load(feature_path) as values:
        return {
            "features": values["features"].astype(np.float32),
            "labels": values["labels"].astype(np.int64),
            "paths": values["paths"].astype(str),
        }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    payload = {
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "inference_scope": (
            "Paired fixed-query resampling stratified by direction and class; "
            "continuation-seed variance is not included."
        ),
        "readouts": {},
    }
    for readout in READOUTS:
        features = {
            role: {arm: load_features(args.input_root, arm, readout, role) for arm in ARMS}
            for role in ROLES
        }
        for role in ROLES:
            for arm in ARMS[1:]:
                if not np.array_equal(features[role][arm]["labels"], features[role]["baseline"]["labels"]):
                    raise ValueError(f"Label order differs for {readout}/{role}/{arm}")
                if not np.array_equal(features[role][arm]["paths"], features[role]["baseline"]["paths"]):
                    raise ValueError(f"Path order differs for {readout}/{role}/{arm}")

        directions = {
            "human_query_yeast_gallery": ("human", "yeast"),
            "yeast_query_human_gallery": ("yeast", "human"),
        }
        direction_values = {}
        direction_labels = {}
        for direction, (query_role, gallery_role) in directions.items():
            direction_labels[direction] = features[query_role]["baseline"]["labels"]
            direction_values[direction] = {
                arm: per_query_metrics(
                    features[gallery_role][arm]["features"],
                    features[gallery_role][arm]["labels"],
                    features[query_role][arm]["features"],
                    features[query_role][arm]["labels"],
                    device=args.device,
                    chunk_size=args.chunk_size,
                )
                for arm in ARMS
            }

        observed = {
            arm: {
                metric: float(
                    np.mean(
                        [
                            class_macro(direction_values[direction][arm][metric], direction_labels[direction])
                            for direction in directions
                        ]
                    )
                )
                for metric in METRICS
            }
            for arm in ARMS
        }
        bootstraps = {
            metric: np.zeros((args.bootstrap_samples, len(ARMS)), dtype=np.float64)
            for metric in METRICS
        }
        for direction in directions:
            labels = direction_labels[direction]
            unique_labels = np.unique(labels)
            weight = 1.0 / (len(directions) * len(unique_labels))
            for label in unique_labels:
                positions = np.flatnonzero(labels == label)
                sampled = rng.integers(0, len(positions), size=(args.bootstrap_samples, len(positions)))
                for metric in METRICS:
                    values = np.stack(
                        [direction_values[direction][arm][metric][positions] for arm in ARMS],
                        axis=1,
                    )
                    bootstraps[metric] += values[sampled].mean(axis=1) * weight

        contrasts = {}
        for metric in METRICS:
            contrasts[metric] = {}
            for candidate, reference in CONTRASTS:
                delta = bootstraps[metric][:, ARMS.index(candidate)] - bootstraps[metric][
                    :, ARMS.index(reference)
                ]
                low, high = np.quantile(delta, (0.025, 0.975))
                contrasts[metric][f"{candidate}_vs_{reference}"] = {
                    "observed": observed[candidate][metric] - observed[reference][metric],
                    "bootstrap_mean": float(delta.mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "probability_positive": float(np.mean(delta > 0)),
                    "significant_positive_95": bool(low > 0),
                }
        payload["readouts"][readout] = {"observed": observed, "contrasts": contrasts}

    joint = [
        payload["readouts"][readout]["contrasts"]["map_at_5"]["joint_vs_baseline"]
        for readout in READOUTS
    ]
    payload["gates"] = {
        "joint_map_positive_both": all(value["observed"] > 0 for value in joint),
        "joint_map_ci95_positive_both": all(value["significant_positive_95"] for value in joint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
