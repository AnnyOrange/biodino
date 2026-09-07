#!/usr/bin/env python3
"""Recompute HPA robust-location NMI over deterministic KMeans seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score


READOUTS = ("nlb2_avg", "nlb2_cls")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def feature_path(summary_path: Path) -> tuple[Path, float]:
    with summary_path.open(newline="", encoding="utf-8") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row.get("dataset") == "hpa-subcellular"
            and row.get("task") == "clustering"
            and row.get("aggregation") == "location"
            and "single-location-ge10" in row.get("protocol", "")
            and not row.get("error")
        ]
    if len(matches) != 1:
        raise ValueError(f"Expected one robust-location row in {summary_path}")
    return Path(matches[0]["feature_file"]), float(matches[0]["nmi"])


def load_robust_features(path: Path, robust_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        features = payload["features"].astype(np.float32)
        labels = payload["labels"].astype(np.int64)
    if len(features) != len(robust_mask):
        raise ValueError(f"Feature/manifest length mismatch for {path}")
    features = features[robust_mask]
    labels = labels[robust_mask]
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1.0e-12), labels


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    if args.seeds < 2:
        raise ValueError("At least two KMeans seeds are required")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    robust_mask = np.asarray([row.get("robust_ge10") == "1" for row in rows])
    if not robust_mask.any():
        raise ValueError("Manifest contains no robust_ge10 rows")

    payload: dict = {
        "seeds": list(range(args.seeds)),
        "scope": (
            "KMeans-initialization robustness on fixed frozen features; this does not "
            "include continuation-training seed variance."
        ),
        "readouts": {},
    }
    for readout in READOUTS:
        values_by_arm = {}
        reported_seed0 = {}
        for arm in args.arms:
            path, reported = feature_path(args.input_root / arm / readout / "summary.csv")
            features, labels = load_robust_features(path, robust_mask)
            values = []
            for seed in range(args.seeds):
                prediction = MiniBatchKMeans(
                    n_clusters=len(np.unique(labels)),
                    random_state=seed,
                    batch_size=2048,
                    n_init="auto",
                ).fit_predict(features)
                values.append(float(normalized_mutual_info_score(labels, prediction)))
            values_by_arm[arm] = values
            reported_seed0[arm] = {
                "reported": reported,
                "recomputed": values[0],
                "abs_error": abs(reported - values[0]),
            }

        baseline = np.asarray(values_by_arm["baseline"])
        deltas = {}
        for arm in args.arms:
            if arm == "baseline":
                continue
            delta = np.asarray(values_by_arm[arm]) - baseline
            deltas[f"{arm}_vs_baseline"] = {
                **summarize(delta.tolist()),
                "positive_seeds": int((delta > 0).sum()),
                "nonnegative_seeds": int((delta >= 0).sum()),
            }
        if "true" in values_by_arm and "shuffled" in values_by_arm:
            delta = np.asarray(values_by_arm["true"]) - np.asarray(values_by_arm["shuffled"])
            deltas["true_vs_shuffled"] = {
                **summarize(delta.tolist()),
                "positive_seeds": int((delta > 0).sum()),
                "nonnegative_seeds": int((delta >= 0).sum()),
            }
        payload["readouts"][readout] = {
            "reported_seed0_check": reported_seed0,
            "arms": {
                arm: {**summarize(values), "values": values}
                for arm, values in values_by_arm.items()
            },
            "paired_deltas": deltas,
        }

    max_reproduction_error = max(
        check["abs_error"]
        for result in payload["readouts"].values()
        for check in result["reported_seed0_check"].values()
    )
    payload["seed0_reproduction_max_abs_error"] = max_reproduction_error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if max_reproduction_error > 1.0e-8:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
