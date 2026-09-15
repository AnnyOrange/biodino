#!/usr/bin/env python3
"""Measure clustering robustness across KMeans seeds on cached features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


DEFAULT_DATASETS = (
    "lc25000",
    "nct-crc-he-100",
    "nct-crc-he-1k",
    "crc-val-he-7k",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=ROOT",
        help="Named evaluation root containing one cached feature file per dataset.",
    )
    parser.add_argument("--comparison", nargs=2, metavar=("CANDIDATE", "REFERENCE"))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_arms(values: list[str]) -> dict[str, Path]:
    arms: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Arm must have NAME=ROOT form: {value!r}")
        name, raw_root = value.split("=", 1)
        if not name or name in arms:
            raise ValueError(f"Arm name is empty or duplicated: {name!r}")
        root = Path(raw_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Arm root does not exist: {root}")
        arms[name] = root
    return arms


def find_feature(root: Path, dataset: str) -> Path:
    matches = sorted(
        path
        for path in root.rglob("*.npz")
        if path.parent.name == dataset
        and path.parent.parent.name == "features"
    )
    retrieval_matches = [path for path in matches if "bio_retrieval" in path.parts]
    if retrieval_matches:
        matches = retrieval_matches
    if len(matches) != 1:
        raise ValueError(
            f"Expected one feature file for {dataset} under {root}, found {len(matches)}: {matches}"
        )
    return matches[0]


def load_feature(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        labels = np.asarray(payload["labels"])
        paths = np.asarray(payload["paths"]).astype(str)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1.0e-12
    return features, labels, paths


def cluster_accuracy(labels: np.ndarray, prediction: np.ndarray) -> float:
    label_ids = np.unique(labels, return_inverse=True)[1]
    n_clusters = len(np.unique(label_ids))
    table = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    np.add.at(table, (label_ids, prediction), 1)
    rows, columns = linear_sum_assignment(-table)
    return float(table[rows, columns].sum() / len(label_ids))


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
        "seed0": float(values[0]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.seeds < 2:
        raise ValueError("At least two KMeans seeds are required")
    arms = parse_arms(args.arm)
    if args.comparison and any(name not in arms for name in args.comparison):
        raise ValueError(f"Comparison names must be defined arms: {args.comparison}")

    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    sources: dict[str, dict[str, str]] = {}
    for arm, root in arms.items():
        sources[arm] = {}
        for dataset in args.datasets:
            feature_path = find_feature(root, dataset)
            cache[(arm, dataset)] = load_feature(feature_path)
            sources[arm][dataset] = str(feature_path)

    reference_arm = next(iter(arms))
    for dataset in args.datasets:
        _, reference_labels, reference_paths = cache[(reference_arm, dataset)]
        for arm in arms:
            _, labels, paths = cache[(arm, dataset)]
            if not np.array_equal(labels, reference_labels) or not np.array_equal(paths, reference_paths):
                raise ValueError(f"Unpaired labels or paths for arm={arm}, dataset={dataset}")

    per_seed_rows: list[dict] = []
    values: dict[tuple[str, str, str], list[float]] = {}
    for arm in arms:
        for dataset in args.datasets:
            features, labels, _ = cache[(arm, dataset)]
            n_clusters = len(np.unique(labels))
            for seed in range(args.seeds):
                prediction = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=seed,
                    batch_size=2048,
                    n_init="auto",
                ).fit_predict(features)
                metrics = {
                    "nmi": float(normalized_mutual_info_score(labels, prediction)),
                    "ari": float(adjusted_rand_score(labels, prediction)),
                    "cluster_accuracy": cluster_accuracy(labels, prediction),
                }
                per_seed_rows.append({"arm": arm, "dataset": dataset, "seed": seed, **metrics})
                for metric, value in metrics.items():
                    values.setdefault((arm, dataset, metric), []).append(value)

    summary_rows: list[dict] = []
    for (arm, dataset, metric), raw_values in values.items():
        summary_rows.append(
            {"arm": arm, "dataset": dataset, "metric": metric, **summarize(np.asarray(raw_values))}
        )

    delta_rows: list[dict] = []
    if args.comparison:
        candidate, reference = args.comparison
        for metric in ("nmi", "ari", "cluster_accuracy"):
            dataset_deltas = []
            for dataset in args.datasets:
                candidate_values = np.asarray(values[(candidate, dataset, metric)])
                reference_values = np.asarray(values[(reference, dataset, metric)])
                deltas = candidate_values - reference_values
                dataset_deltas.append(deltas)
                delta_rows.append(
                    {
                        "candidate": candidate,
                        "reference": reference,
                        "dataset": dataset,
                        "metric": metric,
                        **summarize(deltas),
                        "positive_seeds": int(np.count_nonzero(deltas > 0)),
                        "nonnegative_seeds": int(np.count_nonzero(deltas >= 0)),
                    }
                )
            macro_deltas = np.stack(dataset_deltas).mean(axis=0)
            delta_rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "dataset": "macro",
                    "metric": metric,
                    **summarize(macro_deltas),
                    "positive_seeds": int(np.count_nonzero(macro_deltas > 0)),
                    "nonnegative_seeds": int(np.count_nonzero(macro_deltas >= 0)),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed.csv", per_seed_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    if delta_rows:
        write_csv(args.output_dir / "paired_deltas.csv", delta_rows)
    payload = {
        "scope": "KMeans initialization robustness on fixed frozen features",
        "seeds": list(range(args.seeds)),
        "sources": sources,
        "summary": summary_rows,
        "paired_deltas": delta_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
