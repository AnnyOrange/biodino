"""No GPU, split identities or downstream test labels are used to select PCA."""

from __future__ import annotations

import numpy as np
import pytest

from dinov3.eval.bio_frozen_eval.coexistence import (
    FeatureBank,
    balanced_concat,
    fit_train_only_pca,
    verify_same_samples,
)


def bank(rows, paths):
    return FeatureBank(np.asarray(rows, dtype=np.float32), np.arange(len(paths)), np.asarray(paths))


def test_balanced_concat_preserves_identity_and_unit_norm():
    e = bank([[3, 0, 0], [0, 2, 0]], ["a", "b"])
    l = bank([[0, 4, 0], [0, 0, 5]], ["a", "b"])
    merged = balanced_concat(e, l)
    assert merged.features.shape == (2, 6)
    np.testing.assert_allclose(np.linalg.norm(merged.features, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(merged.features[:, :3], axis=1), 2**-.5, atol=1e-6)


def test_order_and_labels_must_match():
    e = bank([[1, 0], [0, 1]], ["a", "b"])
    with pytest.raises(ValueError, match="ordered sample paths"):
        verify_same_samples(e, bank([[1, 0], [0, 1]], ["b", "a"]))
    bad_labels = FeatureBank(e.features, np.array([1, 0]), e.paths)
    with pytest.raises(ValueError, match="ordered labels"):
        verify_same_samples(e, bad_labels)


def test_pca_rejects_insufficient_training_samples():
    features = bank([[1, 0, 0, 0], [0, 1, 0, 0]], ["a", "b"])
    with pytest.raises(ValueError, match="PCA_UNAVAILABLE_INSUFFICIENT_TRAIN_RANK"):
        fit_train_only_pca(features, features, dimension=3)


def test_pca_rejects_n_equal_to_output_dimension_after_centering():
    features = bank([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], ["a", "b", "c"])
    with pytest.raises(ValueError, match="PCA_UNAVAILABLE_INSUFFICIENT_TRAIN_RANK"):
        fit_train_only_pca(features, features, dimension=3)


def test_pca_does_not_fit_evaluation_samples_or_labels():
    train = bank([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], ["a", "b", "c", "d"])
    evaluation = bank([[2, 0, 0, 1]], ["test"])
    tr1, te1 = fit_train_only_pca(train, evaluation, dimension=2)
    changed_test = FeatureBank(evaluation.features, np.array([999]), evaluation.paths)
    tr2, te2 = fit_train_only_pca(train, changed_test, dimension=2)
    np.testing.assert_allclose(tr1.features, tr2.features)
    np.testing.assert_allclose(te1.features, te2.features)
