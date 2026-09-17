#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_eval.datasets import (
    BBBC005RegressionDataset,
    BBBC013RegressionDataset,
    CHAMMIRegressionDataset,
    CoNICCellCountRegressionDataset,
    CSVImageClassificationDataset,
    ImageFolderDataset,
    NPZClassificationDataset,
    NPZMultiLabelClassificationDataset,
    LIVECellCountRegressionDataset,
)
from benchmark_eval.encoders import MODEL_REGISTRY, extract_features
from benchmark_eval.probes import (
    run_classification_probe,
    run_multilabel_classification_probe,
    run_regression_probe,
    run_regression_probe_split,
)


BENCHMARK_ROOT = Path("/mnt/huawei_deepcad/benchmark")
NATIVE_REGRESSION_PROTOCOLS = {
    "allen-cell-volume": ("CHAMMI-Train-to-Task_one", "log1p(cell_volume)"),
    "conic-cell-count": (
        "source-image-grouped-CoNIC-10fold-v1",
        "raw_official_central_224px_total_cell_count",
    ),
    "livecell-cell-count": ("official-LIVECell-train-to-test", "raw_COCO_instance_count"),
}


def build_dataset(name: str, split: str, max_samples: int | None, max_per_class: int | None):
    if name == "cyclops-protein-loc":
        return ImageFolderDataset(BENCHMARK_ROOT / "Classification/cyclops-protein-loc", max_per_class=max_per_class), "classification"
    if name == "bbbc048-cellcycle":
        classes = ["Anaphase", "G1", "G2", "Metaphase", "Prophase", "S", "Telophase"]
        return ImageFolderDataset(BENCHMARK_ROOT / "Classification/BBBC048v1/CellCycle", max_per_class=max_per_class, class_names=classes), "classification"
    medmnist_names = {
        "bloodmnist", "pathmnist", "tissuemnist", "breastmnist",
        "organamnist", "organcmnist", "organsmnist", "dermamnist",
        "octmnist", "pneumoniamnist", "retinamnist", "chestmnist",
    }
    if name in medmnist_names:
        med_root = BENCHMARK_ROOT / "Classification/MedMNIST"
        med_path = med_root / f"{name}.npz"
        if med_path.exists():
            if name == "chestmnist":
                return NPZMultiLabelClassificationDataset(med_path, split=split, max_samples=max_samples), "multilabel_classification"
            return NPZClassificationDataset(med_path, split=split, max_samples=max_samples), "classification"
        if name == "bloodmnist":
            return NPZClassificationDataset(BENCHMARK_ROOT / "Classification/bloodmnist_64.npz?download=1", split=split, max_samples=max_samples), "classification"
    if name == "midog25-atypical":
        return CSVImageClassificationDataset(
            BENCHMARK_ROOT / "segmentation/MIDOG25_Atypical_Classification_Train_Set.csv",
            BENCHMARK_ROOT / "segmentation/MIDOG25_Binary_Classification_Train_Set",
            image_col="image_id",
            label_col="majority",
            label_map={"NMF": 0, "AMF": 1},
            max_samples=max_samples,
        ), "classification"
    if name == "bbbc013":
        return BBBC013RegressionDataset(BENCHMARK_ROOT / "Regression/BBBC013", max_samples=max_samples), "regression"
    if name == "bbbc005":
        return BBBC005RegressionDataset(BENCHMARK_ROOT / "Regression/BBBC005/extracted/BBBC005_v1_images", max_samples=max_samples), "regression"
    if name == "allen-cell-volume":
        split_name = "Train" if split == "train" else "Task_one"
        return CHAMMIRegressionDataset(
            BENCHMARK_ROOT / "Classification/CHAMMI",
            split_name=split_name,
            target_col="cell_volume",
            max_samples=max_samples,
        ), "regression"
    if name == "conic-cell-count":
        return CoNICCellCountRegressionDataset(
            BENCHMARK_ROOT / "Regression/CoNIC_Cell_Count",
            split=split,
            max_samples=max_samples,
        ), "regression"
    if name == "livecell-cell-count":
        return LIVECellCountRegressionDataset(
            BENCHMARK_ROOT / "Regression/LIVECell_Cell_Count",
            split=split,
            max_samples=max_samples,
        ), "regression"
    raise KeyError(f"Unknown dataset {name}")


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = [
        "model",
        "dataset",
        "task",
        "n_train",
        "n_test",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "label_accuracy",
        "micro_f1",
        "macro_auc",
        "micro_auc",
        "macro_average_precision",
        "micro_average_precision",
        "mae",
        "r2",
        "spearman",
        "target_transform",
        "split_protocol",
        "feature_file",
        "error",
    ]
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    from run_fm_rules_suite import main as rules_main
    return rules_main('classification')


def legacy_main() -> int:
    global BENCHMARK_ROOT
    parser = argparse.ArgumentParser(description="Frozen-feature benchmark runner")
    parser.add_argument("--models", nargs="+", default=["dinov2"], help=f"Model names: {sorted(MODEL_REGISTRY)}")
    parser.add_argument(
        "--dataset",
        default="cyclops-protein-loc",
        choices=[
            "cyclops-protein-loc", "bbbc048-cellcycle", "midog25-atypical", "bbbc013", "bbbc005",
            "allen-cell-volume", "conic-cell-count", "livecell-cell-count",
            "bloodmnist", "pathmnist", "tissuemnist", "breastmnist",
            "organamnist", "organcmnist", "organsmnist", "dermamnist",
            "octmnist", "pneumoniamnist", "retinamnist", "chestmnist",
        ],
    )
    parser.add_argument("--split", default="train", help="NPZ split for bloodmnist feature extraction")
    parser.add_argument("--benchmark-root", default=str(BENCHMARK_ROOT))
    parser.add_argument("--output-dir", default="benchmark_runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-max-iter", type=int, default=10000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--overwrite-features", action="store_true")
    args = parser.parse_args()

    BENCHMARK_ROOT = Path(args.benchmark_root)

    out_dir = Path(args.output_dir)
    native_regression_split = args.dataset in NATIVE_REGRESSION_PROTOCOLS
    dataset, task = build_dataset(
        args.dataset,
        "train" if native_regression_split else args.split,
        args.max_samples,
        args.max_per_class,
    )
    test_dataset = None
    if native_regression_split:
        test_dataset, _ = build_dataset(
            args.dataset,
            "test",
            args.max_samples,
            args.max_per_class,
        )
        split_protocol, _target_transform = NATIVE_REGRESSION_PROTOCOLS[args.dataset]
        print(
            f"[dataset] {args.dataset}: train={len(dataset)} test={len(test_dataset)} "
            f"task={task} split={split_protocol}"
        )
    else:
        print(f"[dataset] {args.dataset}: n={len(dataset)} task={task}")

    all_rows = []
    for model in args.models:
        feature_file = out_dir / "features" / args.dataset / f"{model}.npz"
        print(f"[model] {model}")
        try:
            if native_regression_split:
                train_file = feature_file.with_name(f"{model}_train.npz")
                test_file = feature_file.with_name(f"{model}_test.npz")
                extract_features(
                    dataset,
                    model,
                    train_file,
                    device=args.device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    overwrite=args.overwrite_features,
                )
                extract_features(
                    test_dataset,
                    model,
                    test_file,
                    device=args.device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    overwrite=args.overwrite_features,
                )
                train_pack = np.load(train_file, allow_pickle=True)
                test_pack = np.load(test_file, allow_pickle=True)
                result = run_regression_probe_split(
                    train_pack["features"],
                    train_pack["labels"],
                    test_pack["features"],
                    test_pack["labels"],
                )
                feature_file = train_file
            else:
                extract_features(
                    dataset,
                    model,
                    feature_file,
                    device=args.device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    overwrite=args.overwrite_features,
                )
                pack = np.load(feature_file, allow_pickle=True)
                features = pack["features"]
                labels = pack["labels"]
            if not native_regression_split and task == "classification":
                result = run_classification_probe(features, labels, args.train_fraction, args.seed, args.probe_max_iter)
            elif not native_regression_split and task == "multilabel_classification":
                result = run_multilabel_classification_probe(features, labels, args.train_fraction, args.seed, args.probe_max_iter)
            elif not native_regression_split:
                result = run_regression_probe(features, labels, args.train_fraction, args.seed)
            row = {"model": model, "dataset": args.dataset, "feature_file": str(feature_file), **result.to_dict()}
            if native_regression_split:
                split_protocol, target_transform = NATIVE_REGRESSION_PROTOCOLS[args.dataset]
                row.update(
                    {
                        "target_transform": target_transform,
                        "split_protocol": split_protocol,
                    }
                )
        except Exception as e:
            row = {
                "model": model,
                "dataset": args.dataset,
                "task": task,
                "feature_file": str(feature_file),
                "error": f"{type(e).__name__}: {e}",
            }
            print(f"[error] {model}: {row['error']}")
        all_rows.append(row)
        append_csv(out_dir / "summary.csv", row)
        (out_dir / "last_result.json").write_text(json.dumps(row, indent=2))
        print(json.dumps(row, indent=2))

    (out_dir / f"{args.dataset}_results.json").write_text(json.dumps(all_rows, indent=2))
    return int(any(row.get("error") for row in all_rows))


if __name__ == "__main__":
    raise SystemExit(main())
