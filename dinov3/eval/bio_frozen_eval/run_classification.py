#!/usr/bin/env python3
"""Frozen-feature classification / regression / multilabel benchmark for a DINOv3 checkpoint.

This is the in-repo port of `benchmark_model/run_dinov3_ckpt_benchmark.py` — the
script that produced `benchmark_results_classification.md`,
`benchmark_results_regression.md` and `benchmark_results_multilabel_classification.md`.
Default `--probe-backend sklearn` reproduces those (StandardScaler +
LogisticRegression / Ridge, stratified/random 80/20 split, seed 0).

Example (one checkpoint, several datasets)::

    python -m dinov3.eval.bio_frozen_eval.run_classification \
        --checkpoint /path/to/ckpt/12299/checkpoint.pth \
        --train-config /path/to/ckpt/config.yaml \
        --benchmark-root /mnt/huawei_deepcad/benchmark \
        --datasets bloodmnist cyclops-protein-loc bbbc048-cellcycle midog25-atypical bbbc013 chestmnist \
        --output-dir outputs/bio_eval/frozen --model-name dinov3-12299
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .encoder import Dinov3CkptEncoder, completed, extract_features, parse_autocast_dtype
from .probes import (
    run_classification_probe,
    run_multilabel_classification_probe,
    run_regression_probe,
    run_torch_classification_probe,
    run_torch_multilabel_classification_probe,
)
from .registry import ALL_DATASETS, build_dataset

CSV_FIELDS = [
    "model", "dataset", "task", "n_train", "n_test",
    "accuracy", "balanced_accuracy", "macro_f1",
    "label_accuracy", "micro_f1", "macro_auc", "micro_auc",
    "macro_average_precision", "micro_average_precision",
    "mae", "r2", "spearman", "feature_file",
    "checkpoint", "train_config", "error",
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
    p.add_argument("--train-config", required=True, help="config.yaml matching the checkpoint architecture.")
    p.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    p.add_argument("--datasets", nargs="+", required=True, help=f"Subset of: {ALL_DATASETS}")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="dinov3", help="Value of the CSV 'model' column / feature filename stem.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--max-per-class", type=int)
    p.add_argument("--n-last-blocks", type=int, default=1)
    p.add_argument("--no-avgpool", action="store_true")
    p.add_argument("--autocast-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--overwrite-features", action="store_true")
    p.add_argument("--overwrite-results", action="store_true")
    p.add_argument("--no-save-features", action="store_true")
    p.add_argument("--save-paths", action="store_true")
    p.add_argument("--probe-backend", default="sklearn", choices=["sklearn", "torch"])
    p.add_argument("--torch-probe-epochs", type=int, default=20)
    p.add_argument("--torch-probe-batch-size", type=int, default=8192)
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
    )

    for dataset_name in args.datasets:
        if completed(summary_path, dataset_name, model_name) and not args.overwrite_results:
            print(f"[skip] {model_name} {dataset_name} already in {summary_path}", flush=True)
            continue
        print(f"[dataset] {dataset_name}", flush=True)
        feature_file = out_root / "features" / dataset_name / f"{model_name}.npz"
        try:
            dataset, task = build_dataset(
                dataset_name, "train", args.max_samples, args.max_per_class, benchmark_root=args.benchmark_root
            )
            features, labels = extract_features(
                dataset, encoder, feature_file, args.batch_size, args.num_workers,
                args.overwrite_features, model_name,
                save_features=not args.no_save_features, save_paths=args.save_paths,
            )
            if task == "classification":
                if args.probe_backend == "torch":
                    result = run_torch_classification_probe(
                        features, labels, args.train_fraction, args.seed, device=args.device,
                        epochs=args.torch_probe_epochs, batch_size=args.torch_probe_batch_size,
                    )
                else:
                    result = run_classification_probe(features, labels, args.train_fraction, args.seed)
            elif task == "multilabel_classification":
                if args.probe_backend == "torch":
                    result = run_torch_multilabel_classification_probe(
                        features, labels, args.train_fraction, args.seed, device=args.device,
                        epochs=args.torch_probe_epochs, batch_size=args.torch_probe_batch_size,
                    )
                else:
                    result = run_multilabel_classification_probe(features, labels, args.train_fraction, args.seed)
            else:
                result = run_regression_probe(features, labels, args.train_fraction, args.seed)
            row = {
                "model": model_name,
                "dataset": dataset_name,
                "feature_file": str(feature_file),
                "checkpoint": str(checkpoint),
                "train_config": str(train_config),
                **result.to_dict(),
            }
        except Exception as exc:
            row = {
                "model": model_name,
                "dataset": dataset_name,
                "task": "unknown",
                "feature_file": str(feature_file),
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
