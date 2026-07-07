#!/usr/bin/env python3
"""Frozen-feature retrieval + clustering benchmark for a DINOv3 checkpoint.

In-repo port of `benchmark_model/run_dinov3_retrieval_clustering_benchmark.py`
(source of `benchmark_results_retrieval_clustering.md`). Features are the same
L2-normalised frozen features as the classification benchmark; metrics are
kNN retrieval (recall@k / mAP@k / MRR) and MiniBatchKMeans clustering aligned
with Hungarian matching (cluster_accuracy / ARI / NMI / silhouette).

Example::

    python -m dinov3.eval.bio_frozen_eval.run_retrieval_clustering \
        --checkpoint /path/to/ckpt/12299/checkpoint.pth \
        --train-config /path/to/ckpt/config.yaml \
        --benchmark-root /mnt/huawei_deepcad/benchmark \
        --datasets lc25000 nct-crc-he-1k crc-val-he-7k \
        --output-dir outputs/bio_eval/retrieval --model-name dinov3-12299
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .encoder import CHANNEL_POLICIES, Dinov3CkptEncoder, completed, extract_features, parse_autocast_dtype
from .retrieval_clustering import build_retrieval_dataset, clustering_metrics, retrieval_metrics

DATASET_CHOICES = ["lc25000", "nct-crc-he-100", "nct-crc-he-1k", "crc-val-he-7k"]

CSV_FIELDS = [
    "model", "dataset", "task", "n_samples", "n_classes",
    "recall_at_1", "recall_at_5", "recall_at_10",
    "map_at_1", "map_at_5", "map_at_10", "mrr",
    "cluster_accuracy", "ari", "nmi", "silhouette_cosine",
    "feature_file", "channel_policy", "channel_tta_samples", "checkpoint", "train_config", "error",
]


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="checkpoint.pth or a DCP checkpoint directory.")
    p.add_argument("--train-config", required=True)
    p.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    p.add_argument("--datasets", nargs="+", default=["lc25000", "nct-crc-he-1k", "crc-val-he-7k"], choices=DATASET_CHOICES)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="dinov3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--n-last-blocks", type=int, default=1)
    p.add_argument("--no-avgpool", action="store_true")
    p.add_argument("--autocast-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--channel-policy", default="auto", choices=CHANNEL_POLICIES)
    p.add_argument("--channel-tta-samples", type=int, default=8)
    p.add_argument("--channel-policy-seed", type=int, default=0)
    p.add_argument("--overwrite-features", action="store_true")
    p.add_argument("--overwrite-results", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_root = Path(args.output_dir)
    checkpoint = Path(args.checkpoint)
    train_config = Path(args.train_config)
    autocast_dtype = parse_autocast_dtype(args.autocast_dtype)
    model_name = args.model_name
    summary_path = out_root / "summary.csv"

    print(f"[ckpt] {model_name} checkpoint={checkpoint}", flush=True)
    encoder = Dinov3CkptEncoder(
        checkpoint=checkpoint,
        train_config=train_config,
        device=args.device,
        n_last_blocks=args.n_last_blocks,
        use_avgpool=not args.no_avgpool,
        autocast_dtype=autocast_dtype,
        channel_policy=args.channel_policy,
        channel_tta_samples=args.channel_tta_samples,
        channel_policy_seed=args.channel_policy_seed,
    )

    for dataset_name in args.datasets:
        if (
            completed(
                summary_path,
                dataset_name,
                model_name,
                channel_policy=args.channel_policy,
                channel_tta_samples=args.channel_tta_samples,
            )
            and not args.overwrite_results
        ):
            print(f"[skip] {model_name} {dataset_name} already in {summary_path}", flush=True)
            continue
        feature_stem = model_name
        if args.channel_policy != "auto":
            feature_stem += f"_cp{args.channel_policy}"
            if args.channel_policy == "sample3_tta":
                feature_stem += f"_tta{args.channel_tta_samples}"
        feature_file = out_root / "features" / dataset_name / f"{feature_stem}.npz"
        try:
            dataset, classes = build_retrieval_dataset(
                dataset_name, max_samples=args.max_samples, benchmark_root=args.benchmark_root
            )
            print(f"[run] model={model_name} dataset={dataset_name} n={len(dataset)} classes={len(classes)}", flush=True)
            features, labels = extract_features(
                dataset, encoder, feature_file, args.batch_size, args.num_workers,
                args.overwrite_features, model_name, save_features=True, save_paths=True,
            )
            features = features.astype(np.float32)
            labels = labels.astype(int)
            row = {
                "model": model_name,
                "dataset": dataset_name,
                "task": "retrieval_clustering",
                "feature_file": str(feature_file),
                "channel_policy": args.channel_policy,
                "channel_tta_samples": args.channel_tta_samples,
                "checkpoint": str(checkpoint),
                "train_config": str(train_config),
                "n_samples": int(len(labels)),
                "n_classes": int(len(np.unique(labels))),
                **retrieval_metrics(features, labels),
                **clustering_metrics(features, labels, seed=args.seed),
            }
        except Exception as exc:
            row = {
                "model": model_name,
                "dataset": dataset_name,
                "task": "retrieval_clustering",
                "feature_file": str(feature_file),
                "channel_policy": args.channel_policy,
                "channel_tta_samples": args.channel_tta_samples,
                "checkpoint": str(checkpoint),
                "train_config": str(train_config),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[error] {model_name} {dataset_name}: {row['error']}", flush=True)
        append_csv(summary_path, row)
        (out_root / "last_result.json").write_text(json.dumps(row, indent=2))
        print(json.dumps(row, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
