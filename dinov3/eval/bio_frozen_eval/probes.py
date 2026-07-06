# Frozen-feature probes (sklearn only).
#
# Vendored verbatim from `benchmark_model/benchmark_eval/probes.py` — the exact
# probe implementations that produced the reported `benchmark_results_*.md`
# numbers. The sklearn variants are the canonical ("uncontaminated") probes.
from __future__ import annotations

from dataclasses import asdict, dataclass

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
