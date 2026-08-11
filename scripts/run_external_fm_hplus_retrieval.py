#!/usr/bin/env python3
"""Evaluate external foundation models on the four H+ retrieval datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path("/mnt/huawei_deepcad/dinov3")
BENCHMARK_MODEL_ROOT = Path("/mnt/huawei_deepcad/benchmark_model")
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(BENCHMARK_MODEL_ROOT)]
sys.path.append("/mnt/huawei_deepcad/benchmark_model/_vendor/external_gapfill_py311")

from benchmark_eval.retrieval_clustering import (  # noqa: E402
    build_retrieval_dataset,
    clustering_metrics,
    retrieval_metrics,
)
from run_external_fm_linear_probe import ExternalEncoderAdapter  # noqa: E402


DATASETS = ("lc25000", "nct-crc-he-100", "nct-crc-he-1k", "crc-val-he-7k")
FIELDS = (
    "model", "dataset", "task", "n_samples", "n_classes", "recall_at_1",
    "recall_at_5", "recall_at_10", "map_at_1", "map_at_5", "map_at_10",
    "mrr", "cluster_accuracy", "ari", "nmi", "silhouette_cosine",
    "feature_file", "encoder_preprocess", "error",
)


def pil_collate(batch):
    images, labels, paths = zip(*batch)
    return list(images), np.asarray(labels), list(paths)


def extract(dataset, encoder, path: Path, batch_size: int, num_workers: int, model: str):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pil_collate,
        pin_memory=True,
    )
    features, labels, paths = [], [], []
    for index, (images, batch_labels, batch_paths) in enumerate(loader, 1):
        features.append(encoder.encode_images(images))
        labels.append(batch_labels)
        paths.extend(batch_paths)
        if index == 1 or index % 50 == 0 or index == len(loader):
            print(f"[features] {model}: {len(paths)}/{len(dataset)}", flush=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez(
        temporary,
        features=np.concatenate(features),
        labels=np.concatenate(labels),
        paths=np.asarray(paths),
        model=model,
    )
    temporary.replace(path)


def evaluate(path: Path, seed: int) -> dict[str, float | int]:
    pack = np.load(path, allow_pickle=False)
    features = pack["features"].astype(np.float32)
    labels = pack["labels"].astype(int)
    return {
        "n_samples": int(len(labels)),
        "n_classes": int(len(np.unique(labels))),
        **retrieval_metrics(features, labels),
        **clustering_metrics(features, labels, seed=seed),
    }


def write_summary(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDS} for row in rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.csv"
    existing = {}
    if summary_path.exists():
        with summary_path.open(newline="") as handle:
            existing = {
                row["dataset"]: row
                for row in csv.DictReader(handle)
                if row.get("dataset") and not row.get("error")
            }

    encoder = ExternalEncoderAdapter(args.model, "cuda", args.batch_size)
    rows: list[dict] = []
    for dataset_name in args.datasets:
        if dataset_name in existing:
            rows.append(existing[dataset_name])
            print(f"[skip] {args.model}/{dataset_name}", flush=True)
            continue
        feature_file = output / "features" / dataset_name / f"{args.model}.npz"
        try:
            dataset, classes = build_retrieval_dataset(dataset_name, max_samples=args.max_samples)
            print(f"[run] {args.model}/{dataset_name} n={len(dataset)} classes={len(classes)}", flush=True)
            extract(dataset, encoder, feature_file, args.batch_size, args.num_workers, args.model)
            row = {
                "model": args.model,
                "dataset": dataset_name,
                "task": "retrieval_clustering",
                "feature_file": str(feature_file),
                "encoder_preprocess": "model-native",
                **evaluate(feature_file, args.seed),
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                "model": args.model,
                "dataset": dataset_name,
                "task": "retrieval_clustering",
                "feature_file": str(feature_file),
                "encoder_preprocess": "model-native",
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        write_summary(summary_path, rows)
        (output / "last_result.json").write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(row, indent=2), flush=True)

    write_summary(summary_path, rows)
    return 1 if any(row.get("error") for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
