#!/usr/bin/env python3
"""Re-evaluate cached HS6-L features with readout, kNN, and C-grid controls.

The source cache stores [final-block CLS, final-block mean-patch].  This script
splits that vector without another backbone pass, L2-normalizes every requested
readout, selects logistic-regression C on an inner validation split, refits on
the full source training split, and evaluates the untouched source test split.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dinov3.eval.bio_frozen_eval.datasets import random_indices, stratified_indices
from dinov3.eval.bio_frozen_eval.group_keys import GROUP_SPLIT_DATASETS
from dinov3.eval.bio_frozen_eval.make_group_splits import group_split_indices
from dinov3.eval.bio_frozen_eval.registry import build_dataset


REPO = Path("/mnt/huawei_deepcad/dinov3")
DEFAULT_EVAL_ROOT = REPO / "outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908"
DEFAULT_OUTPUT = REPO / "plot/fig2/readout_probe_results"
READOUTS = ("cls", "mean_patch", "cls_mean_patch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark-root", type=Path, default=Path("/mnt/huawei_deepcad/benchmark"))
    parser.add_argument("--checkpoints", type=int, nargs="+", default=(20007, 28791))
    parser.add_argument("--c-grid", type=float, nargs="+", default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--inner-train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--knn-query-chunk", type=int, default=256)
    parser.add_argument("--linear-jobs", type=int, default=4)
    parser.add_argument("--readout-jobs", type=int, default=3)
    parser.add_argument("--datasets", nargs="*", default=())
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_result_paths(eval_root: Path, checkpoint: int) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for lane in sorted((eval_root / f"point_{checkpoint}").glob("classification_*")):
        for path in sorted(lane.glob("**/last_result.json")):
            payload = json.loads(path.read_text())
            if payload.get("error"):
                continue
            dataset = str(payload["dataset"])
            if dataset in paths:
                raise ValueError(f"Duplicate dataset at ck{checkpoint}: {dataset}")
            paths[dataset] = path
    if len(paths) != 25:
        raise ValueError(f"Expected 25 datasets at ck{checkpoint}, found {len(paths)}")
    return paths


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return payload["features"].astype(np.float32), np.asarray(payload["labels"])


def official_test_path(train_path: Path) -> Path:
    name = train_path.name
    if not name.endswith("_train.npz"):
        raise ValueError(f"Official train cache does not end in _train.npz: {train_path}")
    path = train_path.with_name(name[: -len("_train.npz")] + "_test.npz")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_split(result_path: Path, benchmark_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    result = json.loads(result_path.read_text())
    dataset_name = str(result["dataset"])
    split = str(result["split"])
    feature_file = Path(result["feature_file"])
    if split == "official-test":
        x_train, y_train = load_npz(feature_file)
        test_file = official_test_path(feature_file)
        x_test, y_test = load_npz(test_file)
    else:
        features, labels = load_npz(feature_file)
        if split == "group-split":
            if dataset_name not in GROUP_SPLIT_DATASETS:
                raise ValueError(f"Unexpected group-split dataset: {dataset_name}")
            dataset, _ = build_dataset(dataset_name, "train", None, None, benchmark_root=benchmark_root)
            train_idx, test_idx = group_split_indices(dataset_name, dataset, benchmark_root)
        elif split == "internal-80-20":
            train_idx, test_idx = stratified_indices(labels.astype(int), float(result["train_fraction"]), int(result["seed"]))
        else:
            raise ValueError(f"Unsupported source split for {dataset_name}: {split}")
        x_train, y_train = features[train_idx], labels[train_idx]
        x_test, y_test = features[test_idx], labels[test_idx]
    metadata = {
        "dataset": dataset_name,
        "task": str(result["task"]),
        "split": split,
        "feature_file": str(feature_file),
        "source_result": str(result_path),
        "source_score": float(result.get("macro_auc", result.get("balanced_accuracy"))),
    }
    return x_train, y_train, x_test, y_test, metadata


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, 1.0e-12)


def select_readout(features: np.ndarray, readout: str) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] % 2:
        raise ValueError(f"Expected even-dimensional [CLS, mean-patch] cache, got {features.shape}")
    half = features.shape[1] // 2
    if readout == "cls":
        selected = features[:, :half]
    elif readout == "mean_patch":
        selected = features[:, half:]
    elif readout == "cls_mean_patch":
        selected = features
    else:
        raise ValueError(readout)
    return l2_normalize(selected).astype(np.float32, copy=False)


def metric(task: str, targets: np.ndarray, scores_or_predictions: np.ndarray, *, probabilities: bool) -> float:
    if task == "multilabel_classification":
        probabilities_array = np.asarray(scores_or_predictions)
        return float(roc_auc_score(targets.astype(int), probabilities_array, average="macro"))
    predictions = np.asarray(scores_or_predictions).astype(int)
    return float(balanced_accuracy_score(targets.astype(int), predictions))


def inner_indices(task: str, labels: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if task == "multilabel_classification":
        return random_indices(len(labels), fraction, seed)
    return stratified_indices(labels.astype(int), fraction, seed)


def estimator(task: str, c_value: float):
    base = LogisticRegression(C=c_value, max_iter=10_000, class_weight="balanced", n_jobs=1)
    model = OneVsRestClassifier(base) if task == "multilabel_classification" else base
    return make_pipeline(StandardScaler(), model)


def fit_and_score(model, task: str, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray) -> float:
    model.fit(x_train, y_train.astype(int))
    if task == "multilabel_classification":
        output = model.predict_proba(x_eval)
        return metric(task, y_eval, output, probabilities=True)
    output = model.predict(x_eval)
    return metric(task, y_eval, output, probabilities=False)


def tuned_linear_probe(
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_grid: tuple[float, ...],
    inner_fraction: float,
    seed: int,
    linear_jobs: int,
) -> tuple[float, float, dict[str, float]]:
    inner_train, inner_val = inner_indices(task, y_train, inner_fraction, seed)
    inner_x_train = x_train[inner_train]
    inner_y_train = y_train[inner_train]
    inner_x_val = x_train[inner_val]
    inner_y_val = y_train[inner_val]
    scores = Parallel(n_jobs=linear_jobs, prefer="threads")(
        delayed(fit_and_score)(
            estimator(task, c_value), task,
            inner_x_train, inner_y_train, inner_x_val, inner_y_val,
        )
        for c_value in c_grid
    )
    validation = {str(c_value): score for c_value, score in zip(c_grid, scores, strict=True)}
    best_c = max(c_grid, key=lambda value: (validation[str(value)], -value))
    test_score = fit_and_score(estimator(task, best_c), task, x_train, y_train, x_test, y_test)
    return float(best_c), test_score, validation


def evaluate_linear_readout(
    readout: str,
    metadata: dict[str, Any],
    checkpoint: int,
    x_train_full: np.ndarray,
    y_train: np.ndarray,
    x_test_full: np.ndarray,
    y_test: np.ndarray,
    c_grid: tuple[float, ...],
    inner_fraction: float,
    seed: int,
    linear_jobs: int,
) -> dict[str, Any]:
    print(f"[linear] ck{checkpoint} {metadata['dataset']} {readout}", flush=True)
    x_train = select_readout(x_train_full, readout)
    x_test = select_readout(x_test_full, readout)
    best_c, test_score, validation = tuned_linear_probe(
        metadata["task"], x_train, y_train, x_test, y_test,
        c_grid, inner_fraction, seed, linear_jobs,
    )
    return {
        "checkpoint": checkpoint,
        "dataset": metadata["dataset"],
        "task": metadata["task"],
        "split": metadata["split"],
        "readout": readout,
        "probe": "tuned_linear",
        "score": test_score,
        "best_c": best_c,
        "inner_validation_scores": json.dumps(validation, sort_keys=True),
        "source_score": metadata["source_score"],
        "n_train": len(y_train),
        "n_test": len(y_test),
        "seed": seed,
    }


@torch.inference_mode()
def exact_weighted_knn(
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    k: int,
    temperature: float,
    device: torch.device,
    query_chunk: int,
) -> float:
    # fp16 is used only for similarity search; voting and metrics are float32.
    train = torch.from_numpy(x_train).to(device=device, dtype=torch.float16).T.contiguous()
    train_labels = torch.from_numpy(np.asarray(y_train)).to(device)
    outputs = []
    for start in range(0, len(x_test), query_chunk):
        query = torch.from_numpy(x_test[start : start + query_chunk]).to(device=device, dtype=torch.float16)
        similarities = query @ train
        top_similarity, top_index = similarities.topk(min(k, len(x_train)), dim=1)
        weights = (top_similarity.float() / temperature).softmax(dim=1)
        neighbor_labels = train_labels[top_index]
        if task == "multilabel_classification":
            probabilities = (neighbor_labels.float() * weights.unsqueeze(-1)).sum(dim=1)
            outputs.append(probabilities.cpu().numpy())
        else:
            classes = int(np.asarray(y_train).max()) + 1
            votes = torch.zeros(len(query), classes, device=device, dtype=torch.float32)
            votes.scatter_add_(1, neighbor_labels.long(), weights)
            outputs.append(votes.argmax(dim=1).cpu().numpy())
        del query, similarities, top_similarity, top_index, weights, neighbor_labels
    output = np.concatenate(outputs, axis=0)
    del train, train_labels
    torch.cuda.empty_cache()
    return metric(task, y_test, output, probabilities=task == "multilabel_classification")


def main() -> int:
    args = parse_args()
    args.eval_root = args.eval_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    c_grid = tuple(sorted(set(float(value) for value in args.c_grid)))
    device = torch.device(args.device)
    all_rows: list[dict[str, Any]] = []
    for checkpoint in args.checkpoints:
        result_paths = read_result_paths(args.eval_root, checkpoint)
        datasets = sorted(args.datasets or result_paths)
        for dataset_name in datasets:
            result_file = args.output / f"ck{checkpoint}_{dataset_name}.json"
            if result_file.exists():
                payload = json.loads(result_file.read_text())
                all_rows.extend(payload["rows"])
                print(f"[skip] ck{checkpoint} {dataset_name}", flush=True)
                continue
            print(f"[load] ck{checkpoint} {dataset_name}", flush=True)
            x_train_full, y_train, x_test_full, y_test, metadata = load_split(
                result_paths[dataset_name], args.benchmark_root
            )
            rows = Parallel(n_jobs=args.readout_jobs, prefer="threads")(
                delayed(evaluate_linear_readout)(
                    readout, metadata, checkpoint, x_train_full, y_train, x_test_full, y_test,
                    c_grid, args.inner_train_fraction, args.seed, args.linear_jobs,
                )
                for readout in READOUTS
            )
            concat_train = select_readout(x_train_full, "cls_mean_patch")
            concat_test = select_readout(x_test_full, "cls_mean_patch")
            concat_row = next(row for row in rows if row["readout"] == "cls_mean_patch")
            print(f"[knn] ck{checkpoint} {dataset_name}", flush=True)
            knn_score = exact_weighted_knn(
                metadata["task"], concat_train, y_train, concat_test, y_test,
                args.knn_k, args.knn_temperature, device, args.knn_query_chunk,
            )
            rows.append(
                {
                    **concat_row,
                    "probe": "knn",
                    "score": knn_score,
                    "best_c": "",
                    "inner_validation_scores": "",
                }
            )
            payload = {
                "status": "VALID_COMPLETE",
                "checkpoint": checkpoint,
                "dataset": dataset_name,
                "feature_layout": "[final-block CLS, final-block mean-patch]",
                "feature_normalization": "per-readout row-wise L2",
                "linear_protocol": {
                    "estimator": "StandardScaler + balanced LogisticRegression",
                    "c_grid": c_grid,
                    "selection": f"inner {args.inner_train_fraction:.2f}/{1-args.inner_train_fraction:.2f} split, seed {args.seed}",
                    "test_policy": "select on inner validation, refit full source train, score source test once",
                },
                "knn_protocol": {"distance": "cosine", "k": args.knn_k, "temperature": args.knn_temperature},
                "rows": rows,
            }
            atomic_json(result_file, payload)
            all_rows.extend(rows)
            atomic_csv(args.output / "summary.csv", all_rows)
            del x_train_full, x_test_full, concat_train, concat_test
            gc.collect()
    atomic_csv(args.output / "summary.csv", all_rows)
    print(json.dumps({"status": "VALID_COMPLETE", "rows": len(all_rows), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
