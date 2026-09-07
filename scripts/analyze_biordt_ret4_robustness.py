#!/usr/bin/env python3
"""Robustness analysis for the matched BioRDT Ret4 screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
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
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--comparison",
        nargs=2,
        action="append",
        metavar=("CANDIDATE", "REFERENCE"),
        required=True,
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--kmeans-seeds", type=int, default=10)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--bootstrap-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=290830)
    parser.add_argument("--retrieval-chunk-size", type=int, default=256)
    return parser.parse_args()


def read_index(eval_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for summary in sorted(eval_root.glob("*/summary.csv")):
        with summary.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("error"):
                    raise RuntimeError(f"Failed evaluation row in {summary}: {row['error']}")
                key = (summary.parent.name, row["dataset"])
                if key in index:
                    raise ValueError(f"Duplicate evaluation row: {key}")
                index[key] = row
    return index


def stable_seed(base_seed: int, *parts: str) -> int:
    payload = "\0".join(parts).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
    return (base_seed + offset) % (2**32)


def per_query_ap_at_k(features: np.ndarray, labels: np.ndarray, k: int, chunk_size: int) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    n = len(y)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    max_k = min(k, max(1, n - 1))
    class_counts = {label: int(np.sum(y == label)) for label in np.unique(y)}
    values = np.zeros(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        similarities = x[start:end] @ x.T
        row_ids = np.arange(start, end)
        similarities[np.arange(end - start), row_ids] = -np.inf
        candidates = np.argpartition(-similarities, kth=max_k - 1, axis=1)[:, :max_k]
        scores = np.take_along_axis(similarities, candidates, axis=1)
        order = np.argsort(-scores, axis=1)
        neighbors = np.take_along_axis(candidates, order, axis=1)
        relevant = y[neighbors] == y[row_ids, None]
        precision_denominator = np.arange(1, max_k + 1)
        for local_index, rel in enumerate(relevant):
            global_index = start + local_index
            denominator = min(class_counts[y[global_index]] - 1, max_k)
            if denominator > 0:
                precisions = np.cumsum(rel) / precision_denominator
                values[global_index] = float(np.sum(precisions * rel) / denominator)
    return values


def bootstrap_means(
    values: np.ndarray,
    *,
    rounds: int,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty(rounds, dtype=np.float64)
    for start in range(0, rounds, batch_size):
        end = min(start + batch_size, rounds)
        indices = rng.integers(0, len(values), size=(end - start, len(values)))
        output[start:end] = values[indices].mean(axis=1)
    return output


def summarize_bootstrap(point: float, samples: np.ndarray) -> dict[str, float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    nonpositive = (np.count_nonzero(samples <= 0) + 1) / (len(samples) + 1)
    nonnegative = (np.count_nonzero(samples >= 0) + 1) / (len(samples) + 1)
    return {
        "delta": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_two_sided": float(min(1.0, 2.0 * min(nonpositive, nonnegative))),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    index = read_index(args.eval_root)
    models = sorted({model for pair in args.comparison for model in pair})
    for model in models:
        for dataset in args.datasets:
            if (model, dataset) not in index:
                raise KeyError(f"Missing result for model={model} dataset={dataset}")

    @lru_cache(maxsize=None)
    def load_features(model: str, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row = index[(model, dataset)]
        feature_path = Path(row["feature_file"])
        with np.load(feature_path) as cached:
            features = np.asarray(cached["features"], dtype=np.float32)
            labels = np.asarray(cached["labels"])
            paths = np.asarray(cached["paths"]).astype(str)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
        return features, labels, paths

    kmeans_rows: list[dict] = []
    kmeans_lookup: dict[tuple[str, str, int], tuple[float, float]] = {}
    for model in models:
        for dataset in args.datasets:
            features, labels, _ = load_features(model, dataset)
            n_clusters = len(np.unique(labels))
            for seed in range(args.kmeans_seeds):
                prediction = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=seed,
                    batch_size=2048,
                    n_init=1,
                ).fit_predict(features)
                nmi = float(normalized_mutual_info_score(labels, prediction))
                ari = float(adjusted_rand_score(labels, prediction))
                kmeans_lookup[(model, dataset, seed)] = (nmi, ari)
                kmeans_rows.append(
                    {"model": model, "dataset": dataset, "seed": seed, "nmi": nmi, "ari": ari}
                )

    retrieval_cache: dict[tuple[str, str], np.ndarray] = {}
    comparison_rows: list[dict] = []
    dataset_rows: list[dict] = []
    for candidate, reference in args.comparison:
        retrieval_samples = []
        retrieval_points = []
        for dataset in args.datasets:
            for model in (candidate, reference):
                key = (model, dataset)
                if key not in retrieval_cache:
                    features, labels, _ = load_features(model, dataset)
                    retrieval_cache[key] = per_query_ap_at_k(
                        features,
                        labels,
                        k=5,
                        chunk_size=args.retrieval_chunk_size,
                    )
                    expected = float(index[key]["map_at_5"])
                    if not np.isclose(retrieval_cache[key].mean(), expected, atol=2e-7):
                        raise AssertionError(
                            f"mAP@5 mismatch for {key}: computed={retrieval_cache[key].mean()} expected={expected}"
                        )
            _, candidate_labels, candidate_paths = load_features(candidate, dataset)
            _, reference_labels, reference_paths = load_features(reference, dataset)
            if not np.array_equal(candidate_paths, reference_paths) or not np.array_equal(
                candidate_labels, reference_labels
            ):
                raise ValueError(f"Unpaired feature rows for {candidate} vs {reference} on {dataset}")
            difference = retrieval_cache[(candidate, dataset)] - retrieval_cache[(reference, dataset)]
            rng = np.random.default_rng(stable_seed(args.seed, candidate, reference, dataset, "retrieval"))
            samples = bootstrap_means(
                difference,
                rounds=args.bootstrap_rounds,
                batch_size=args.bootstrap_batch,
                rng=rng,
            )
            summary = summarize_bootstrap(float(difference.mean()), samples)
            retrieval_samples.append(samples)
            retrieval_points.append(summary["delta"])
            dataset_rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": "map_at_5",
                    "dataset": dataset,
                    **summary,
                }
            )
        retrieval_macro_samples = np.stack(retrieval_samples).mean(axis=0)
        retrieval_summary = summarize_bootstrap(
            float(np.mean(retrieval_points)), retrieval_macro_samples
        )

        seed_differences = []
        for seed in range(args.kmeans_seeds):
            dataset_differences = [
                kmeans_lookup[(candidate, dataset, seed)][0]
                - kmeans_lookup[(reference, dataset, seed)][0]
                for dataset in args.datasets
            ]
            seed_differences.append(float(np.mean(dataset_differences)))
        seed_differences_array = np.asarray(seed_differences)
        rng = np.random.default_rng(stable_seed(args.seed, candidate, reference, "nmi"))
        nmi_samples = bootstrap_means(
            seed_differences_array,
            rounds=args.bootstrap_rounds,
            batch_size=args.bootstrap_batch,
            rng=rng,
        )
        nmi_summary = summarize_bootstrap(float(seed_differences_array.mean()), nmi_samples)
        for dataset in args.datasets:
            values = np.asarray(
                [
                    kmeans_lookup[(candidate, dataset, seed)][0]
                    - kmeans_lookup[(reference, dataset, seed)][0]
                    for seed in range(args.kmeans_seeds)
                ]
            )
            rng = np.random.default_rng(stable_seed(args.seed, candidate, reference, dataset, "nmi"))
            samples = bootstrap_means(
                values,
                rounds=args.bootstrap_rounds,
                batch_size=args.bootstrap_batch,
                rng=rng,
            )
            dataset_rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": "nmi",
                    "dataset": dataset,
                    **summarize_bootstrap(float(values.mean()), samples),
                }
            )
        comparison_rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "map5_delta": retrieval_summary["delta"],
                "map5_ci_low": retrieval_summary["ci_low"],
                "map5_ci_high": retrieval_summary["ci_high"],
                "map5_p": retrieval_summary["p_two_sided"],
                "nmi_delta": nmi_summary["delta"],
                "nmi_ci_low": nmi_summary["ci_low"],
                "nmi_ci_high": nmi_summary["ci_high"],
                "nmi_p": nmi_summary["p_two_sided"],
                "kmeans_seeds": args.kmeans_seeds,
                "training_seeds": 1,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "kmeans_seed_metrics.csv", kmeans_rows)
    write_csv(args.output_dir / "comparison_summary.csv", comparison_rows)
    write_csv(args.output_dir / "comparison_by_dataset.csv", dataset_rows)
    payload = {
        "eval_root": str(args.eval_root),
        "datasets": args.datasets,
        "kmeans_seeds": args.kmeans_seeds,
        "bootstrap_rounds": args.bootstrap_rounds,
        "comparisons": comparison_rows,
        "caveat": "Evaluation-resampling uncertainty only; every model has one training seed.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    report = [
        "# BioRDT Ret4 robustness",
        "",
        "Evaluation-resampling uncertainty only; every model has one training seed.",
        "",
        "| candidate | reference | delta mAP@5 (95% CI) | delta NMI (95% CI) |",
        "|---|---|---:|---:|",
    ]
    for row in comparison_rows:
        report.append(
            "| {candidate} | {reference} | {map5_delta:+.6f} [{map5_ci_low:+.6f}, {map5_ci_high:+.6f}] "
            "| {nmi_delta:+.6f} [{nmi_ci_low:+.6f}, {nmi_ci_high:+.6f}] |".format(**row)
        )
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
