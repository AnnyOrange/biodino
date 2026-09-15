#!/usr/bin/env python3
"""Audit NCT-to-CRC linear readouts from retained frozen features.

Regularization is selected only on a stratified split of NCT train. The CRC
test labels are never used for model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as pack:
        return np.asarray(pack["features"], dtype=np.float32), np.asarray(pack["labels"]).reshape(-1)


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    c_value: float,
) -> np.ndarray:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    clf = LogisticRegression(C=c_value, max_iter=10_000, class_weight="balanced")
    clf.fit(x_train_scaled, y_train)
    return clf.predict(x_test_scaled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c-grid", default="0.0001,0.001,0.01,0.1,1,10,100")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "tuned_c_result.json"
    if result_path.is_file():
        print(f"skip existing {result_path}", flush=True)
        return

    x_train, y_train = load_xy(args.train)
    x_test, y_test = load_xy(args.test)
    indices = np.arange(len(y_train))
    fit_idx, val_idx = train_test_split(
        indices,
        test_size=args.validation_fraction,
        random_state=args.seed,
        stratify=y_train,
    )
    c_grid = [float(value) for value in args.c_grid.split(",")]

    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x_train[fit_idx])
    x_val = scaler.transform(x_train[val_idx])
    validation = []
    best: tuple[float, float] | None = None
    for c_value in c_grid:
        clf = LogisticRegression(C=c_value, max_iter=10_000, class_weight="balanced")
        clf.fit(x_fit, y_train[fit_idx])
        score = metrics(y_train[val_idx], clf.predict(x_val))
        validation.append({"C": c_value, **score})
        candidate = (score["macro_f1"], -c_value)
        if best is None or candidate > best:
            best = candidate
        print(f"C={c_value:g} val_macro_f1={score['macro_f1']:.6f}", flush=True)

    assert best is not None
    best_c = -best[1]
    pred = fit_predict(x_train, y_train, x_test, best_c)
    test_metrics = metrics(y_test, pred)
    labels = np.unique(np.concatenate((y_train, y_test)))
    matrix = confusion_matrix(y_test, pred, labels=labels)
    np.savez_compressed(
        args.output / "tuned_c_predictions.npz",
        y_true=y_test,
        y_pred=pred,
        labels=labels,
        confusion_matrix=matrix,
    )
    payload = {
        "protocol": "StandardScaler+balanced-LogisticRegression",
        "selection": "stratified NCT train validation only",
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "n_train": int(len(y_train)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(y_test)),
        "c_grid": c_grid,
        "validation": validation,
        "selected_C": best_c,
        "test": test_metrics,
        "labels": labels.tolist(),
        "confusion_matrix": matrix.tolist(),
        "train_features": str(args.train),
        "test_features": str(args.test),
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_C": best_c, **test_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
