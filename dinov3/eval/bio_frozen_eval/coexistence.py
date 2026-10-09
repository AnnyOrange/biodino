"""Label-safe, identity-matched operations for frozen checkpoint coexistence.

This module prepares features only. Task probes and dense heads must follow
their own locked evaluation protocols; running this file is not a v4 score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureBank:
    features: np.ndarray
    labels: np.ndarray
    paths: np.ndarray


def load_feature_bank(path: str | Path) -> FeatureBank:
    """Load one split; sample identities and labels are mandatory, not inferred."""
    with np.load(path, allow_pickle=False) as data:
        missing = {"features", "labels", "paths"} - set(data.files)
        if missing:
            raise ValueError(f"Feature bank {path} lacks {sorted(missing)}")
        bank = FeatureBank(
            features=np.asarray(data["features"], dtype=np.float32),
            labels=np.asarray(data["labels"]),
            paths=np.asarray(data["paths"]),
        )
    if bank.features.ndim != 2 or bank.features.shape[1] < 1:
        raise ValueError(f"Expected a nonempty [N,D] bank: {path}")
    if len(bank.features) != len(bank.labels) or len(bank.features) != len(bank.paths):
        raise ValueError(f"Feature/label/path lengths differ: {path}")
    if not np.isfinite(bank.features).all():
        raise ValueError(f"Non-finite frozen features: {path}")
    return bank


def verify_same_samples(*banks: FeatureBank) -> None:
    """Fail closed on ordered image/sample and label differences across teachers."""
    if len(banks) < 2:
        raise ValueError("Need at least two teacher feature banks")
    reference = banks[0]
    for i, bank in enumerate(banks[1:], 1):
        if bank.features.shape != reference.features.shape:
            raise ValueError(f"Teacher {i} has incompatible feature shape")
        if not np.array_equal(bank.paths, reference.paths):
            raise ValueError(f"Teacher {i} has different ordered sample paths")
        if not np.array_equal(bank.labels, reference.labels):
            raise ValueError(f"Teacher {i} has different ordered labels")


def unit_rows(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected finite [N,D] features")
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Zero-length frozen feature")
    return array / norm


def balanced_concat(first: FeatureBank, second: FeatureBank) -> FeatureBank:
    """Per-branch L2, equal-energy concat, and final L2 for cosine readouts."""
    verify_same_samples(first, second)
    features = np.concatenate((unit_rows(first.features), unit_rows(second.features)), axis=1)
    return FeatureBank(unit_rows(features), first.labels, first.paths)


def fit_train_only_pca(
    train: FeatureBank, evaluation: FeatureBank, *, dimension: int, seed: int = 0
) -> tuple[FeatureBank, FeatureBank]:
    """Reduce a concatenated feature to one-checkpoint dimension, without test fit.

    Neither labels nor evaluation samples are used to fit PCA. For an
    unsupervised retrieval test set, `train` must be an independently locked
    non-evaluation calibration bank; it cannot be the query/gallery features.
    """
    from sklearn.decomposition import PCA

    if dimension <= 0 or train.features.shape[1] != evaluation.features.shape[1]:
        raise ValueError("PCA dimension or train/evaluation feature dimension mismatch")
    # Centering bounds the fitted rank by N - 1, so N == d still leaves one
    # zero-variance output direction and is not a valid d-dimensional control.
    if len(train.features) <= dimension or train.features.shape[1] <= dimension:
        raise ValueError(
            "PCA_UNAVAILABLE_INSUFFICIENT_TRAIN_RANK: require train N > d and concat D > d"
        )
    if not np.isfinite(train.features).all() or not np.isfinite(evaluation.features).all():
        raise ValueError("PCA requires finite train and evaluation features")
    pca = PCA(n_components=dimension, svd_solver="randomized", random_state=seed)
    train_projected = unit_rows(pca.fit_transform(train.features))
    eval_projected = unit_rows(pca.transform(evaluation.features))
    return (
        FeatureBank(train_projected, train.labels, train.paths),
        FeatureBank(eval_projected, evaluation.labels, evaluation.paths),
    )
