#!/usr/bin/env python3
"""Evaluate external frozen encoders with the canonical DINOv3 probe protocol.

The encoder and its published preprocessing are model-specific. Everything
downstream of feature extraction is shared with the DINOv3 benchmark: dataset
registry, official/group splits, split validation, and sklearn estimators.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DINOV3_ROOT = Path("/mnt/huawei_deepcad/dinov3")
BENCHMARK_MODEL_ROOT = Path("/mnt/huawei_deepcad/benchmark_model")
sys.path.insert(0, str(DINOV3_ROOT))
sys.path.insert(0, str(BENCHMARK_MODEL_ROOT))

from benchmark_eval.encoders import MODEL_REGISTRY, build_encoder  # noqa: E402
from dinov3.eval.bio_frozen_eval.encoder import extract_features  # noqa: E402
from dinov3.eval.bio_frozen_eval.group_keys import GROUP_SPLIT_DATASETS  # noqa: E402
from dinov3.eval.bio_frozen_eval.make_group_splits import group_split_indices  # noqa: E402
from dinov3.eval.bio_frozen_eval.probes import (  # noqa: E402
    run_bbbc013_compound_oof_probe,
    run_classification_probe,
    run_classification_probe_split,
    run_multilabel_classification_probe,
    run_multilabel_classification_probe_split,
    run_regression_probe,
    run_regression_probe_split,
)
from dinov3.eval.bio_frozen_eval.registry import (  # noqa: E402
    ALL_DATASETS,
    NATIVE_TEST_SPLIT_DATASETS,
    UNSUPPORTED_OFFICIAL_SPLIT_DATASETS,
    build_dataset,
)
from dinov3.eval.bio_frozen_eval.run_classification import (  # noqa: E402
    BBBC013_SPLIT_PROTOCOL,
    _validate_explicit_split_labels,
    capped_split_protocol_label,
    split_protocol_for_dataset,
)


CSV_FIELDS = [
    "model", "dataset", "task", "split", "probe", "encoder_preprocess",
    "channel_policy", "n_train", "n_test", "accuracy", "balanced_accuracy",
    "macro_f1", "label_accuracy", "micro_f1", "macro_auc", "micro_auc",
    "macro_average_precision", "micro_average_precision", "mae", "r2",
    "spearman", "wortmannin_mae", "wortmannin_r2", "wortmannin_spearman",
    "ly294002_mae", "ly294002_r2", "ly294002_spearman", "target_transform",
    "fold_protocol", "ridge_alpha", "n_compounds", "n_folds", "oof_samples",
    "feature_file", "model_kind", "model_path", "error",
]


class ExternalEncoderAdapter:
    """Expose the DINOv3 extractor interface for PIL-only external encoders."""

    def __init__(self, model_name: str, device: str, batch_size: int):
        self.encoder = build_encoder(model_name, device=device, batch_size=batch_size)

    @staticmethod
    def _tensor_first3_to_pil(image: torch.Tensor) -> Image.Image:
        if image.ndim != 3:
            raise ValueError(f"Expected C,H,W tensor, got {tuple(image.shape)}")
        image = image.detach().to(dtype=torch.float32, device="cpu").clamp(0.0, 1.0)
        channels = int(image.shape[0])
        if channels == 0:
            raise ValueError("Cannot encode a zero-channel image")
        if channels < 3:
            image = torch.cat([image, image[-1:].expand(3 - channels, -1, -1)], dim=0)
        else:
            image = image[:3]
        array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(array, mode="RGB")

    def encode_images(self, images: list) -> np.ndarray:
        rgb = [self._tensor_first3_to_pil(x) if torch.is_tensor(x) else x.convert("RGB") for x in images]
        # Match the DINOv3 cache precision; sklearn receives float32 after load.
        return self.encoder.encode_pil(rgb).astype(np.float16)


def _probe_split_for_task(task: str):
    if task == "multilabel_classification":
        return run_multilabel_classification_probe_split
    if task == "regression":
        return run_regression_probe_split
    return run_classification_probe_split


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def _completed(path: Path, model: str, dataset: str, split: str) -> bool:
    if not path.exists():
        return False
    with path.open(newline="") as handle:
        return any(
            row.get("model") == model
            and row.get("dataset") == dataset
            and row.get("split") == split
            and not row.get("error")
            for row in csv.DictReader(handle)
        )


def _split_label(args: argparse.Namespace, dataset_name: str) -> str:
    base = split_protocol_for_dataset(dataset_name)
    if args.max_samples is None and args.max_per_class is None:
        return base
    if dataset_name in UNSUPPORTED_OFFICIAL_SPLIT_DATASETS:
        return base
    subset_base = (
        "subset-official-test"
        if dataset_name in NATIVE_TEST_SPLIT_DATASETS
        else "subset-internal-80-20"
    )
    return capped_split_protocol_label(subset_base, args.max_samples, args.max_per_class)


def _extract(
    dataset,
    encoder: ExternalEncoderAdapter,
    feature_file: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    return extract_features(
        dataset,
        encoder,
        feature_file,
        args.batch_size,
        args.num_workers,
        args.overwrite_features,
        args.model,
        save_features=not args.no_save_features,
        save_paths=args.save_paths,
    )


def evaluate_dataset(
    args: argparse.Namespace,
    encoder: ExternalEncoderAdapter,
    dataset_name: str,
) -> dict:
    split = _split_label(args, dataset_name)
    out_root = Path(args.output_dir)
    feature_dir = (
        Path(args.feature_root) / dataset_name
        if args.feature_root
        else out_root / "features" / dataset_name
    )
    feature_file = feature_dir / f"{args.model}.npz"

    if dataset_name in UNSUPPORTED_OFFICIAL_SPLIT_DATASETS:
        raise ValueError(
            f"{dataset_name} is an open-set CHAMMI task and is not valid for the "
            "closed-set sklearn linear probe"
        )

    if dataset_name in NATIVE_TEST_SPLIT_DATASETS:
        train_ds, task = build_dataset(
            dataset_name, "train", args.max_samples, args.max_per_class,
            benchmark_root=args.benchmark_root,
        )
        test_ds, _ = build_dataset(
            dataset_name, "test", args.max_samples, args.max_per_class,
            benchmark_root=args.benchmark_root,
        )
        train_file = feature_dir / f"{args.model}_train.npz"
        test_file = feature_dir / f"{args.model}_test.npz"
        x_train, y_train = _extract(train_ds, encoder, train_file, args)
        x_test, y_test = _extract(test_ds, encoder, test_file, args)
        _validate_explicit_split_labels(task, y_train, y_test, dataset_name)
        result = _probe_split_for_task(task)(x_train, y_train, x_test, y_test)
        feature_file = train_file
    elif dataset_name == "bbbc013" and split == BBBC013_SPLIT_PROTOCOL:
        dataset, task = build_dataset(
            dataset_name, "train", args.max_samples, args.max_per_class,
            benchmark_root=args.benchmark_root,
        )
        features, labels = _extract(dataset, encoder, feature_file, args)
        paths = [str(sample.image_path) for sample in dataset.samples]
        result = run_bbbc013_compound_oof_probe(features, labels, paths)
    elif dataset_name in GROUP_SPLIT_DATASETS and split == "group-split":
        dataset, task = build_dataset(
            dataset_name, "train", args.max_samples, args.max_per_class,
            benchmark_root=args.benchmark_root,
        )
        features, labels = _extract(dataset, encoder, feature_file, args)
        train_idx, test_idx = group_split_indices(dataset_name, dataset, args.benchmark_root)
        result = _probe_split_for_task(task)(
            features[train_idx], labels[train_idx], features[test_idx], labels[test_idx]
        )
    else:
        dataset, task = build_dataset(
            dataset_name, "train", args.max_samples, args.max_per_class,
            benchmark_root=args.benchmark_root,
        )
        features, labels = _extract(dataset, encoder, feature_file, args)
        if task == "classification":
            result = run_classification_probe(features, labels, args.train_fraction, args.seed)
        elif task == "multilabel_classification":
            result = run_multilabel_classification_probe(features, labels, args.train_fraction, args.seed)
        else:
            result = run_regression_probe(features, labels, args.train_fraction, args.seed)

    spec = MODEL_REGISTRY[args.model]
    row = {
        "model": args.model,
        "dataset": dataset_name,
        "split": split,
        "probe": "StandardScaler+balanced-LogisticRegression"
        if result.task in {"classification", "multilabel_classification"}
        else "StandardScaler+Ridge(alpha=1)",
        "encoder_preprocess": "model-native",
        "channel_policy": "first3",
        "feature_file": str(feature_file),
        "model_kind": spec.kind,
        "model_path": str(spec.path),
        **result.to_dict(),
    }
    if dataset_name == "bbbc013" and split == BBBC013_SPLIT_PROTOCOL:
        row.update({"target_transform": "log1p", "fold_protocol": "leave-one-replicate-row-out"})
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-root", help="Optional shared feature-cache root across task runners")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--no-save-features", action="store_true")
    parser.add_argument("--save-paths", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.output_dir)
    summary_path = out_root / "summary.csv"
    pending = [
        name for name in args.datasets
        if args.overwrite_results or not _completed(summary_path, args.model, name, _split_label(args, name))
    ]
    if not pending:
        print(f"[complete] {args.model}: all requested datasets already have valid rows", flush=True)
        return 0

    print(
        f"[protocol] model={args.model} frozen=true preprocess=model-native "
        "channel_policy=first3 probe=canonical-sklearn",
        flush=True,
    )
    encoder = ExternalEncoderAdapter(args.model, args.device, args.batch_size)
    failed: list[str] = []
    for dataset_name in pending:
        print(f"[dataset] {args.model}/{dataset_name} split={_split_label(args, dataset_name)}", flush=True)
        try:
            row = evaluate_dataset(args, encoder, dataset_name)
        except Exception as exc:
            failed.append(dataset_name)
            spec = MODEL_REGISTRY[args.model]
            row = {
                "model": args.model,
                "dataset": dataset_name,
                "task": "unknown",
                "split": _split_label(args, dataset_name),
                "probe": "canonical-sklearn",
                "encoder_preprocess": "model-native",
                "channel_policy": "first3",
                "model_kind": spec.kind,
                "model_path": str(spec.path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[error] {args.model}/{dataset_name}: {row['error']}", flush=True)
        _append_csv(summary_path, row)
        (out_root / "last_result.json").write_text(json.dumps(row, indent=2))
        print(json.dumps(row, indent=2), flush=True)

    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if failed:
        print(f"[failed] {args.model}: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
