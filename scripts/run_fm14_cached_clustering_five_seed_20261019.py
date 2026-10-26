#!/usr/bin/env python3
"""Resume FM14 KMeans seeds from the existing, immutable feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dinov3.eval.bio_frozen_eval.retrieval_clustering import (
    ManifestImageDataset, clustering_metrics, remap_labels,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/02_eval_runs/fm14_retrieval_multiseed_20261016"
DATASETS = ("nct-crc-he-100", "nct-crc-he-1k", "crc-val-he-7k",
            "hpa-subcellular", "rxrx1-cross", "lc25000")
MODELS = ("dinov2", "mae", "siglip2", "pe", "bioclip", "cytoself",
          "jump_cp", "cytoimagenet", "uni", "conch", "phikon2",
          "virchow2", "gigapath", "hoptimus0")


def feature_pack(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return data["features"].astype(np.float32), data["labels"].copy()


def cluster_input(row: dict, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    features, labels = feature_pack(row["feature_file"])
    if dataset == "hpa-subcellular" and row["protocol"].endswith("ge10-34"):
        protocol = Path("/mnt/huawei_deepcad/benchmark/Retrieval_Clustering/protocols/v1/hpa_single_location_clustering.csv")
        data_root = Path("/mnt/huawei_deepcad/benchmark/Retrieval_Clustering/HPA_Subcellular")
        samples = ManifestImageDataset(data_root, protocol)
        mask = np.asarray([record["robust_ge10"] == "1" for record in samples.rows])
        if len(mask) != len(features):
            raise ValueError("HPA cache and protocol manifest have different lengths")
        features, labels = features[mask], labels[mask]
    elif dataset == "rxrx1-cross":
        query_path = row["feature_file"].replace("_gallery.npz", "_query.npz")
        if query_path == row["feature_file"]:
            raise ValueError("RxRx1 clustering row does not reference a gallery cache")
        query_features, query_labels = feature_pack(query_path)
        features = np.concatenate((features, query_features))
        labels = np.concatenate((labels, query_labels))
    if len(features) != row["n_samples"]:
        raise ValueError(f"Cache has {len(features)} rows; expected {row['n_samples']}")
    return features, remap_labels(labels)


def evaluate(payload: dict, dataset: str, seed: int) -> dict:
    result = json.loads(json.dumps(payload))
    rows = result.get("rows", [result])
    for row in rows:
        if row.get("task") != "clustering" and "cluster_accuracy" not in row:
            continue
        features, labels = cluster_input(row, dataset)
        row.update(clustering_metrics(features, labels, seed=seed))
    result["seed"] = seed
    result["multiseed_protocol"] = "fm14-v4-clustering-seed-repeat"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--verify-seed2", action="store_true")
    args = parser.parse_args()
    for dataset in args.datasets:
        cell = RUN / args.model / dataset
        reference = json.loads((cell / "seed2.json").read_text())
        seeds = (2,) if args.verify_seed2 else (3, 4)
        for seed in seeds:
            target = cell / f"seed{seed}.json"
            if target.is_file() and not args.verify_seed2:
                print(f"[skip] {args.model}/{dataset}/seed{seed}", flush=True)
                continue
            result = evaluate(reference, dataset, seed)
            if args.verify_seed2:
                old_rows = reference.get("rows", [reference])
                new_rows = result.get("rows", [result])
                for old, new in zip(old_rows, new_rows):
                    if "cluster_accuracy" in old:
                        for metric in ("cluster_accuracy", "ari", "nmi", "silhouette_cosine"):
                            if not np.isclose(old[metric], new[metric], atol=1e-6):
                                raise ValueError(f"Mismatch in {dataset}/{metric}: {old[metric]} vs {new[metric]}")
                print(f"[verified] {args.model}/{dataset}/seed2", flush=True)
                continue
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(target)
            print(f"[complete] {args.model}/{dataset}/seed{seed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
