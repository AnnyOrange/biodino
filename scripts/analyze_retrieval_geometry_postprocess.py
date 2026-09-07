#!/usr/bin/env python3
"""Diagnose whether frozen-feature anisotropy explains weak clustering.

All transforms are fit without labels. Metrics use the locked in-repo
retrieval/clustering implementation, so this is a diagnostic of representation
geometry rather than a new supervised probe.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from dinov3.eval.bio_frozen_eval.retrieval_clustering import clustering_metrics, retrieval_metrics


def l2_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1.0e-12)


def remove_top_components(features: np.ndarray, count: int, fit_rows: np.ndarray) -> np.ndarray:
    mean = fit_rows.mean(axis=0, keepdims=True)
    pca = PCA(n_components=count, svd_solver="randomized", random_state=0)
    pca.fit(fit_rows - mean)
    centered = features - mean
    projected = centered @ pca.components_.T
    return l2_normalize(centered - projected @ pca.components_)


def whiten(features: np.ndarray, dimensions: int, fit_rows: np.ndarray) -> np.ndarray:
    dimensions = min(dimensions, fit_rows.shape[0] - 1, fit_rows.shape[1])
    pca = PCA(n_components=dimensions, whiten=True, svd_solver="randomized", random_state=0)
    pca.fit(fit_rows)
    return l2_normalize(pca.transform(features))


def evaluate(features: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    return {
        **retrieval_metrics(features, labels),
        **clustering_metrics(features, labels, seed=seed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-fit-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for path in args.feature_files:
        pack = np.load(path, allow_pickle=False)
        features = np.asarray(pack["features"], dtype=np.float32)
        labels = np.asarray(pack["labels"])
        fit_indices = rng.choice(
            len(features), size=min(len(features), args.max_fit_samples), replace=False
        )
        fit_rows = features[fit_indices]
        variants = {
            "raw": l2_normalize(features),
            "center": l2_normalize(features - fit_rows.mean(axis=0, keepdims=True)),
        }
        for count in (1, 2, 4, 8):
            variants[f"remove_pc{count}"] = remove_top_components(features, count, fit_rows)
        for dimensions in (64, 128, 256):
            variants[f"pca_whiten{dimensions}"] = whiten(features, dimensions, fit_rows)

        for name, transformed in variants.items():
            metrics = evaluate(transformed, labels, args.seed)
            rows.append(
                {
                    "feature_file": str(path),
                    "dataset": path.parent.name,
                    "variant": name,
                    "n_samples": len(features),
                    "n_features": transformed.shape[1],
                    **metrics,
                }
            )
            print(
                f"{path.parent.name:18s} {name:16s} "
                f"mAP@5={metrics['map_at_5']:.6f} NMI={metrics['nmi']:.6f}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
