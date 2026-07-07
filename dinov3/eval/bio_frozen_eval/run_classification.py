#!/usr/bin/env python3
"""Frozen-feature classification / regression / multilabel benchmark for a DINOv3 checkpoint.

This is the in-repo port of `benchmark_model/run_dinov3_ckpt_benchmark.py` — the
script that produced `benchmark_results_classification.md`,
`benchmark_results_regression.md` and `benchmark_results_multilabel_classification.md`.
This entry uses the canonical sklearn probes (StandardScaler +
LogisticRegression / Ridge) with dataset-appropriate held-out splits:
official test sets when available, committed leakage-safe group splits for
source-linked microscopy datasets, and the historical deterministic 80/20
split only for datasets without a better held-out protocol. The legacy torch
probe backend was removed to avoid producing non-comparable classification
numbers.

Example (one checkpoint, several datasets)::

    python -m dinov3.eval.bio_frozen_eval.run_classification \
        --checkpoint /path/to/ckpt/12299/checkpoint.pth \
        --train-config /path/to/ckpt/config.yaml \
        --benchmark-root /mnt/huawei_deepcad/benchmark \
        --datasets bloodmnist cyclops-protein-loc bbbc048-cellcycle midog25-atypical bbbc013 chestmnist \
        --output-dir outputs/bio_eval/frozen --model-name dinov3-12299

Use `--image-size 384` (optionally with `--resize-size`) for high-resolution
classification ablations. `--resolution-protocol best` uses the 2026-06-23
dualroute ep15all ablation-backed size table for the five tested classification
datasets and falls back to `--image-size` for everything else.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

from .encoder import CHANNEL_POLICIES, Dinov3CkptEncoder, completed, extract_features, parse_autocast_dtype
from .group_keys import GROUP_SPLIT_DATASETS
from .make_group_splits import group_split_indices
from .probes import (
    run_classification_probe,
    run_classification_probe_split,
    run_multilabel_classification_probe,
    run_multilabel_classification_probe_split,
    run_regression_probe,
    run_regression_probe_split,
)
from .registry import ALL_DATASETS, NATIVE_TEST_SPLIT_DATASETS, build_dataset


def _probe_split_for_task(task: str):
    """Explicit-split probe matching the dataset task."""
    if task == "multilabel_classification":
        return run_multilabel_classification_probe_split
    if task == "regression":
        return run_regression_probe_split
    return run_classification_probe_split


def split_protocol_for_dataset(dataset_name: str) -> str:
    """Return the published evaluation split protocol label for a dataset."""
    if dataset_name in NATIVE_TEST_SPLIT_DATASETS:
        return "official-test"
    if dataset_name in GROUP_SPLIT_DATASETS:
        return "group-split"
    return "internal-80-20"


# Best image sizes from
# outputs/classification_imgsize_ablation_dualroute_ep15all_20260623
# (dualroute ep15all, ckpt 15374, completed 2026-06-23). Only these five
# datasets were swept; all other datasets fall back to the manual --image-size.
BEST_IMAGE_SIZE_BY_DATASET = {
    "bloodmnist": 384,
    "bbbc048-cellcycle": 512,
    "cyclops-protein-loc": 224,
    "midog25-atypical": 384,
    "chestmnist": 512,
}


CSV_FIELDS = [
    "model", "dataset", "task", "split", "resolution_protocol", "n_train", "n_test",
    "accuracy", "balanced_accuracy", "macro_f1",
    "label_accuracy", "micro_f1", "macro_auc", "micro_auc",
    "macro_average_precision", "micro_average_precision",
    "mae", "r2", "spearman", "feature_file",
    "image_size", "resize_size", "channel_policy", "channel_tta_samples",
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


def resolve_resize_size(image_size: int, resize_size: int | None) -> int:
    if image_size <= 0:
        raise ValueError("--image-size must be positive")
    if image_size % 16 != 0:
        raise ValueError("--image-size must be divisible by 16 for ViT patch-size compatibility")
    if resize_size and resize_size > 0:
        if resize_size < image_size:
            raise ValueError("--resize-size must be >= --image-size")
        return int(resize_size)
    return int(round(256 * image_size / 224))


def resolve_image_size(dataset_name: str, protocol: str, manual_image_size: int) -> int:
    if protocol == "manual":
        return int(manual_image_size)
    if protocol == "best":
        return int(BEST_IMAGE_SIZE_BY_DATASET.get(dataset_name, manual_image_size))
    raise ValueError(f"Unknown --resolution-protocol {protocol!r}")


def feature_cache_stem(
    model_name: str,
    image_size: int,
    resize_size: int,
    channel_policy: str = "auto",
    channel_tta_samples: int = 8,
) -> str:
    parts = [model_name]
    if image_size != 224 or resize_size != 256:
        parts.append(f"s{image_size}_r{resize_size}")
    if channel_policy != "auto":
        parts.append(f"cp{channel_policy}")
        if channel_policy == "sample3_tta":
            parts.append(f"tta{channel_tta_samples}")
    return "_".join(parts)


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
    p.add_argument(
        "--channel-policy",
        default="auto",
        choices=CHANNEL_POLICIES,
        help=(
            "How tensor multichannel eval samples are fed to the backbone. "
            "auto = native for multichannel stems, first3 for RGB stems; "
            "native requires a multichannel stem; first3/compact3/zerofill3/mean3/sample3_tta "
            "collapse C,H,W tensors to RGB-compatible 3,H,W inputs."
        ),
    )
    p.add_argument(
        "--channel-tta-samples",
        type=int,
        default=8,
        help="Number of channel draws for --channel-policy sample3_tta.",
    )
    p.add_argument(
        "--channel-policy-seed",
        type=int,
        default=0,
        help="Seed for stochastic channel policies such as sample3_tta.",
    )
    p.add_argument(
        "--resolution-protocol",
        default="best",
        choices=["manual", "best"],
        help=(
            "manual uses --image-size for all datasets. best uses the "
            "dualroute_ep15all_20260623 ablation table for bloodmnist, "
            "bbbc048-cellcycle, cyclops-protein-loc, midog25-atypical, and "
            "chestmnist; other datasets fall back to --image-size."
        ),
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Manual/fallback final square crop size fed to the backbone.",
    )
    p.add_argument(
        "--resize-size",
        type=int,
        default=0,
        help="Pre-crop resize short-side size. 0 keeps the ImageNet eval ratio: round(256 * image_size / 224).",
    )
    p.add_argument("--overwrite-features", action="store_true")
    p.add_argument("--overwrite-results", action="store_true")
    p.add_argument("--no-save-features", action="store_true")
    p.add_argument("--save-paths", action="store_true")
    p.add_argument(
        "--probe-backend",
        default="sklearn",
        choices=["sklearn"],
        help="Deprecated compatibility flag; sklearn is the only supported backend.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_root = Path(args.output_dir)
    checkpoint = Path(args.checkpoint)
    train_config = Path(args.train_config)
    autocast_dtype = parse_autocast_dtype(args.autocast_dtype)
    model_name = args.model_name
    summary_path = out_root / "summary.csv"
    current_encoder_key = None
    current_encoder = None

    print(f"[ckpt] {model_name} checkpoint={checkpoint}", flush=True)

    def get_encoder(image_size: int, resize_size: int) -> Dinov3CkptEncoder:
        nonlocal current_encoder_key, current_encoder
        key = (int(image_size), int(resize_size))
        if current_encoder_key == key and current_encoder is not None:
            return current_encoder
        if current_encoder is not None:
            del current_encoder
            current_encoder = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        print(f"[encoder] image_size={image_size} resize_size={resize_size}", flush=True)
        current_encoder = Dinov3CkptEncoder(
            checkpoint=checkpoint,
            train_config=train_config,
            device=args.device,
            n_last_blocks=args.n_last_blocks,
            use_avgpool=not args.no_avgpool,
            autocast_dtype=autocast_dtype,
            image_size=image_size,
            resize_size=resize_size,
            channel_policy=args.channel_policy,
            channel_tta_samples=args.channel_tta_samples,
            channel_policy_seed=args.channel_policy_seed,
        )
        current_encoder_key = key
        return current_encoder

    for dataset_name in args.datasets:
        split_label = split_protocol_for_dataset(dataset_name)
        image_size = resolve_image_size(dataset_name, args.resolution_protocol, args.image_size)
        resize_size = resolve_resize_size(image_size, args.resize_size)
        if (
            completed(
                summary_path,
                dataset_name,
                model_name,
                image_size,
                resize_size,
                split_label,
                args.channel_policy,
                args.channel_tta_samples,
            )
            and not args.overwrite_results
        ):
            print(f"[skip] {model_name} {dataset_name} already in {summary_path}", flush=True)
            continue
        print(
            f"[dataset] {dataset_name} split={split_label} "
            f"resolution_protocol={args.resolution_protocol} "
            f"image_size={image_size} resize_size={resize_size}",
            flush=True,
        )
        stem = feature_cache_stem(
            model_name,
            image_size,
            resize_size,
            args.channel_policy,
            args.channel_tta_samples,
        )
        feature_file = out_root / "features" / dataset_name / f"{stem}.npz"
        try:
            encoder = get_encoder(image_size, resize_size)
            if dataset_name in NATIVE_TEST_SPLIT_DATASETS:
                # Datasets with a publication-standard train/test split: extract
                # features for each split and probe on the official held-out test set.
                train_ds, task = build_dataset(
                    dataset_name, "train", args.max_samples, args.max_per_class, benchmark_root=args.benchmark_root
                )
                test_ds, _ = build_dataset(
                    dataset_name, "test", args.max_samples, args.max_per_class, benchmark_root=args.benchmark_root
                )
                feat_dir = out_root / "features" / dataset_name
                train_ff = feat_dir / f"{stem}_train.npz"
                test_ff = feat_dir / f"{stem}_test.npz"
                x_train, y_train = extract_features(
                    train_ds, encoder, train_ff, args.batch_size, args.num_workers,
                    args.overwrite_features, model_name,
                    save_features=not args.no_save_features, save_paths=args.save_paths,
                )
                x_test, y_test = extract_features(
                    test_ds, encoder, test_ff, args.batch_size, args.num_workers,
                    args.overwrite_features, model_name,
                    save_features=not args.no_save_features, save_paths=args.save_paths,
                )
                result = _probe_split_for_task(task)(x_train, y_train, x_test, y_test)
                feature_file = train_ff
            elif dataset_name in GROUP_SPLIT_DATASETS:
                # No official test split: fixed, documented, leakage-safe group split
                # (splits/<dataset>.json). Extract whole-set features once, then probe
                # on the source-grouped train/test partition.
                dataset, task = build_dataset(
                    dataset_name, "train", args.max_samples, args.max_per_class, benchmark_root=args.benchmark_root
                )
                features, labels = extract_features(
                    dataset, encoder, feature_file, args.batch_size, args.num_workers,
                    args.overwrite_features, model_name,
                    save_features=not args.no_save_features, save_paths=args.save_paths,
                )
                train_idx, test_idx = group_split_indices(dataset_name, dataset, args.benchmark_root)
                result = _probe_split_for_task(task)(
                    features[train_idx], labels[train_idx], features[test_idx], labels[test_idx]
                )
            else:
                dataset, task = build_dataset(
                    dataset_name, "train", args.max_samples, args.max_per_class, benchmark_root=args.benchmark_root
                )
                features, labels = extract_features(
                    dataset, encoder, feature_file, args.batch_size, args.num_workers,
                    args.overwrite_features, model_name,
                    save_features=not args.no_save_features, save_paths=args.save_paths,
                )
                if task == "classification":
                    result = run_classification_probe(features, labels, args.train_fraction, args.seed)
                elif task == "multilabel_classification":
                    result = run_multilabel_classification_probe(features, labels, args.train_fraction, args.seed)
                else:
                    result = run_regression_probe(features, labels, args.train_fraction, args.seed)
            row = {
                "model": model_name,
                "dataset": dataset_name,
                "split": split_label,
                "resolution_protocol": args.resolution_protocol,
                "feature_file": str(feature_file),
                "image_size": image_size,
                "resize_size": resize_size,
                "channel_policy": args.channel_policy,
                "channel_tta_samples": args.channel_tta_samples,
                "checkpoint": str(checkpoint),
                "train_config": str(train_config),
                **result.to_dict(),
            }
        except Exception as exc:
            row = {
                "model": model_name,
                "dataset": dataset_name,
                "task": "unknown",
                "split": split_label,
                "resolution_protocol": args.resolution_protocol,
                "feature_file": str(feature_file),
                "image_size": image_size,
                "resize_size": resize_size,
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
