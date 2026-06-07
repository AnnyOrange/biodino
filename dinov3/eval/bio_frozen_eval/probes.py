# Frozen-feature probes (sklearn + torch).
#
# Vendored verbatim from `benchmark_model/benchmark_eval/probes.py` — the exact
# probe implementations that produced the reported `benchmark_results_*.md`
# numbers. The sklearn variants are the canonical ("uncontaminated") probes;
# the torch variants are kept only for parity / ablation via --probe-backend.
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


def run_torch_classification_probe(
    features: np.ndarray,
    labels: np.ndarray,
    train_fraction: float = 0.8,
    seed: int = 0,
    device: str = "cuda",
    epochs: int = 20,
    batch_size: int = 8192,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
) -> ProbeResult:
    import torch

    labels = labels.reshape(-1).astype(int)
    train_idx, test_idx = stratified_indices(labels, train_fraction, seed)
    x_train_np, y_train_np = features[train_idx].astype(np.float32), labels[train_idx]
    x_test_np, y_test = features[test_idx].astype(np.float32), labels[test_idx]

    dev = torch.device(device)
    x_train = torch.from_numpy(x_train_np).to(dev)
    y_train = torch.from_numpy(y_train_np).long().to(dev)
    x_test = torch.from_numpy(x_test_np).to(dev)
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    n_classes = int(labels.max()) + 1
    model = torch.nn.Linear(x_train.shape[1], n_classes).to(dev)
    counts = torch.bincount(y_train, minlength=n_classes).float().clamp_min(1)
    weights = counts.sum() / (n_classes * counts)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(x_train.shape[0], generator=gen, device=dev)
        for start in range(0, x_train.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()

    with torch.inference_mode():
        pred = model(x_test).argmax(dim=1).cpu().numpy()
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


def run_torch_multilabel_classification_probe(
    features: np.ndarray,
    labels: np.ndarray,
    train_fraction: float = 0.8,
    seed: int = 0,
    device: str = "cuda",
    epochs: int = 20,
    batch_size: int = 8192,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
) -> ProbeResult:
    import torch

    labels = np.asarray(labels).astype(np.float32)
    train_idx, test_idx = random_indices(len(labels), train_fraction, seed)
    x_train_np, y_train_np = features[train_idx].astype(np.float32), labels[train_idx]
    x_test_np, y_test = features[test_idx].astype(np.float32), labels[test_idx]

    dev = torch.device(device)
    x_train = torch.from_numpy(x_train_np).to(dev)
    y_train = torch.from_numpy(y_train_np).to(dev)
    x_test = torch.from_numpy(x_test_np).to(dev)
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    model = torch.nn.Linear(x_train.shape[1], y_train.shape[1]).to(dev)
    positives = y_train.sum(dim=0)
    negatives = y_train.shape[0] - positives
    pos_weight = (negatives / positives.clamp_min(1)).clamp(max=100)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(x_train.shape[0], generator=gen, device=dev)
        for start in range(0, x_train.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()

    with torch.inference_mode():
        prob = torch.sigmoid(model(x_test)).cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    y_test_i = y_test.astype(int)
    return ProbeResult(
        task="multilabel_classification",
        n_train=len(train_idx),
        n_test=len(test_idx),
        metrics={
            "accuracy": float(accuracy_score(y_test_i, pred)),
            "label_accuracy": float((y_test_i == pred).mean()),
            "macro_f1": float(f1_score(y_test_i, pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_test_i, pred, average="micro", zero_division=0)),
            "macro_auc": _safe_multilabel_score(roc_auc_score, y_test_i, prob, "macro"),
            "micro_auc": _safe_multilabel_score(roc_auc_score, y_test_i, prob, "micro"),
            "macro_average_precision": _safe_multilabel_score(average_precision_score, y_test_i, prob, "macro"),
            "micro_average_precision": _safe_multilabel_score(average_precision_score, y_test_i, prob, "micro"),
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
