# Frozen-feature probes (sklearn only).
#
# Vendored verbatim from `benchmark_model/benchmark_eval/probes.py` — the exact
# probe implementations that produced the reported `benchmark_results_*.md`
# numbers. The sklearn variants are the canonical ("uncontaminated") probes.
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .datasets import random_indices, stratified_indices


@dataclass
class ProbeResult:
    task: str
    n_train: int
    n_test: int
    metrics: dict[str, float]

    def to_dict(self):
        d = asdict(self)
        d.update(self.metrics)
        d.pop("metrics")
        return d


BBBC013_COMPOUND_ROWS = {
    "wortmannin": ("A", "B", "C", "D"),
    "ly294002": ("E", "F", "G", "H"),
}


def run_bbbc013_compound_oof_probe(
    features: np.ndarray,
    targets: np.ndarray,
    sample_paths: list[str | Path],
    alpha: float = 1.0,
) -> ProbeResult:
    """Evaluate BBBC013 per compound with leave-one-replicate-row-out folds.

    BBBC013 contains four replicated dose curves for each of two compounds.
    Combining their raw doses is not meaningful because the compounds have
    different dose-response curves. This protocol predicts log1p dose within
    each compound and concatenates four held-out-row predictions per compound.
    """
    features = np.asarray(features)
    targets = np.asarray(targets, dtype=float)
    if len(features) != len(targets) or len(targets) != len(sample_paths):
        raise ValueError(
            "BBBC013 features, targets, and sample paths must have equal lengths "
            f"({len(features)}, {len(targets)}, {len(sample_paths)})"
        )
    if np.any(targets < 0):
        raise ValueError("BBBC013 dose targets must be non-negative for log1p")

    rows: list[str] = []
    for path in sample_paths:
        match = re.search(r"Channel\d+-\d+-([A-H])-\d+\.BMP$", Path(path).name, re.IGNORECASE)
        if match is None:
            raise ValueError(f"Cannot parse BBBC013 replicate row from {Path(path).name!r}")
        rows.append(match.group(1).upper())
    rows_array = np.asarray(rows)
    log_targets = np.log1p(targets)

    compound_metrics: dict[str, dict[str, float]] = {}
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    train_samples_per_fold: set[int] = set()

    for compound, compound_rows in BBBC013_COMPOUND_ROWS.items():
        compound_mask = np.isin(rows_array, compound_rows)
        if int(compound_mask.sum()) != 48:
            raise ValueError(
                f"BBBC013 {compound} must contain 48 wells, found {int(compound_mask.sum())}"
            )
        predictions = np.full(len(targets), np.nan, dtype=float)
        for held_out_row in compound_rows:
            test_mask = rows_array == held_out_row
            train_mask = compound_mask & ~test_mask
            if int(test_mask.sum()) != 12 or int(train_mask.sum()) != 36:
                raise ValueError(
                    f"BBBC013 {compound} row {held_out_row}: expected 36 train/12 test, "
                    f"found {int(train_mask.sum())}/{int(test_mask.sum())}"
                )
            reg = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            reg.fit(features[train_mask], log_targets[train_mask])
            predictions[test_mask] = reg.predict(features[test_mask])
            train_samples_per_fold.add(int(train_mask.sum()))

        compound_predictions = predictions[compound_mask]
        compound_targets = log_targets[compound_mask]
        if not np.isfinite(compound_predictions).all():
            raise RuntimeError(f"BBBC013 {compound} OOF predictions are incomplete")
        rho = spearmanr(compound_targets, compound_predictions).correlation
        compound_metrics[compound] = {
            "r2": float(r2_score(compound_targets, compound_predictions)),
            "spearman": float(rho) if rho == rho else float("nan"),
            "mae": float(mean_absolute_error(compound_targets, compound_predictions)),
        }
        all_predictions.append(compound_predictions)
        all_targets.append(compound_targets)

    metric_names = ("r2", "spearman", "mae")
    macro = {
        metric: float(np.mean([compound_metrics[name][metric] for name in BBBC013_COMPOUND_ROWS]))
        for metric in metric_names
    }
    metrics = dict(macro)
    for compound, values in compound_metrics.items():
        for metric, value in values.items():
            metrics[f"{compound}_{metric}"] = value
    metrics.update(
        {
            "ridge_alpha": float(alpha),
            "n_compounds": float(len(BBBC013_COMPOUND_ROWS)),
            "n_folds": float(sum(len(rows) for rows in BBBC013_COMPOUND_ROWS.values())),
            "oof_samples": float(sum(len(values) for values in all_targets)),
        }
    )
    return ProbeResult(
        task="regression",
        n_train=min(train_samples_per_fold),
        n_test=sum(len(values) for values in all_predictions),
        metrics=metrics,
    )


def run_classification_probe(
    features: np.ndarray,
    labels: np.ndarray,
    train_fraction: float = 0.8,
    seed: int = 0,
    max_iter: int = 10000,
) -> ProbeResult:
    train_idx, test_idx = stratified_indices(labels.astype(int), train_fraction, seed)
    x_train, y_train = features[train_idx], labels[train_idx].astype(int)
    x_test, y_test = features[test_idx], labels[test_idx].astype(int)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, class_weight="balanced", n_jobs=1),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return ProbeResult(
        task="classification",
        n_train=len(train_idx),
        n_test=len(test_idx),
        metrics={
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
        },
    )


def run_classification_probe_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 10000,
) -> ProbeResult:
    """Linear probe with an explicit (native) train/test split.

    Same StandardScaler + balanced LogisticRegression estimator and metrics as
    :func:`run_classification_probe`; used for datasets that ship their own
    train/test split (e.g. NCT-CRC-HE: train on NCT-CRC-HE-100K, test on the
    different-patient CRC-VAL-HE-7K) instead of an internal stratified split.
    """
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, class_weight="balanced", n_jobs=1),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return ProbeResult(
        task="classification",
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        metrics={
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
        },
    )


def _safe_multilabel_score(fn, y_true: np.ndarray, y_score: np.ndarray, average: str) -> float:
    try:
        value = fn(y_true, y_score, average=average)
        return float(value) if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def run_multilabel_classification_probe(
    features: np.ndarray,
    labels: np.ndarray,
    train_fraction: float = 0.8,
    seed: int = 0,
    max_iter: int = 10000,
) -> ProbeResult:
    labels = np.asarray(labels).astype(int)
    train_idx, test_idx = random_indices(len(labels), train_fraction, seed)
    x_train, y_train = features[train_idx], labels[train_idx]
    x_test, y_test = features[test_idx], labels[test_idx]
    clf = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(LogisticRegression(max_iter=max_iter, class_weight="balanced", n_jobs=1)),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test)
    return ProbeResult(
        task="multilabel_classification",
        n_train=len(train_idx),
        n_test=len(test_idx),
        metrics={
            "accuracy": float(accuracy_score(y_test, pred)),
            "label_accuracy": float((y_test == pred).mean()),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_test, pred, average="micro", zero_division=0)),
            "macro_auc": _safe_multilabel_score(roc_auc_score, y_test, prob, "macro"),
            "micro_auc": _safe_multilabel_score(roc_auc_score, y_test, prob, "micro"),
            "macro_average_precision": _safe_multilabel_score(average_precision_score, y_test, prob, "macro"),
            "micro_average_precision": _safe_multilabel_score(average_precision_score, y_test, prob, "micro"),
        },
    )


def run_regression_probe(
    features: np.ndarray,
    targets: np.ndarray,
    train_fraction: float = 0.8,
    seed: int = 0,
    alpha: float = 1.0,
) -> ProbeResult:
    train_idx, test_idx = random_indices(len(targets), train_fraction, seed)
    x_train, y_train = features[train_idx], targets[train_idx].astype(float)
    x_test, y_test = features[test_idx], targets[test_idx].astype(float)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    reg.fit(x_train, y_train)
    pred = reg.predict(x_test)
    rho = spearmanr(y_test, pred).correlation
    return ProbeResult(
        task="regression",
        n_train=len(train_idx),
        n_test=len(test_idx),
        metrics={
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
            "spearman": float(rho) if rho == rho else float("nan"),
        },
    )


def run_multilabel_classification_probe_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 10000,
) -> ProbeResult:
    """Multi-label probe with an explicit (native) train/test split.

    Same StandardScaler + OneVsRest balanced LogisticRegression estimator and
    metrics as :func:`run_multilabel_classification_probe`; used for datasets
    that ship their own train/test split (e.g. MedMNIST chestmnist official
    train vs test) instead of an internal random split.
    """
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)
    clf = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(LogisticRegression(max_iter=max_iter, class_weight="balanced", n_jobs=1)),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test)
    return ProbeResult(
        task="multilabel_classification",
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        metrics={
            "accuracy": float(accuracy_score(y_test, pred)),
            "label_accuracy": float((y_test == pred).mean()),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_test, pred, average="micro", zero_division=0)),
            "macro_auc": _safe_multilabel_score(roc_auc_score, y_test, prob, "macro"),
            "micro_auc": _safe_multilabel_score(roc_auc_score, y_test, prob, "micro"),
            "macro_average_precision": _safe_multilabel_score(average_precision_score, y_test, prob, "macro"),
            "micro_average_precision": _safe_multilabel_score(average_precision_score, y_test, prob, "micro"),
        },
    )


def run_regression_probe_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
) -> ProbeResult:
    """Ridge regression probe with an explicit train/test split (e.g. a fixed
    group-aware split for BBBC013/BBBC005). Same estimator/metrics as
    :func:`run_regression_probe`."""
    y_train = np.asarray(y_train).astype(float)
    y_test = np.asarray(y_test).astype(float)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    reg.fit(x_train, y_train)
    pred = reg.predict(x_test)
    rho = spearmanr(y_test, pred).correlation
    return ProbeResult(
        task="regression",
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        metrics={
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
            "spearman": float(rho) if rho == rho else float("nan"),
        },
    )
