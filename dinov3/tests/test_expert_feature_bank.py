import numpy as np
import pytest
import torch

from dinov3.data.expert_feature_bank import (
    ExpertFeatureBank,
    build_cross_domain_edge_mask,
    stable_metadata_codes,
)


def _write_bank(path, keys, features, reliability=None, **metadata):
    values = {
        "keys": np.asarray(keys),
        "features": np.asarray(features, dtype=np.float16),
    }
    if reliability is not None:
        values["reliability"] = np.asarray(reliability, dtype=np.float32)
    values.update({name: np.asarray(items) for name, items in metadata.items()})
    np.savez(path, **values)


def test_lookup_keeps_only_intersection_in_batch_order(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_bank(first, ["a", "b", "c"], [[1, 0], [2, 0], [3, 0]], [1, 0.5, 1])
    _write_bank(second, ["c", "a"], [[30, 0, 0], [10, 0, 0]])
    bank = ExpertFeatureBank([first, second])
    batch = bank.lookup(["missing", "c", "a", "b"], device="cpu")
    assert batch.sample_indices.tolist() == [1, 2]
    assert batch.features[0][:, 0].tolist() == [3, 1]
    assert batch.features[1][:, 0].tolist() == [30, 10]
    assert torch.allclose(batch.weights, torch.tensor([[1.0, 1.0], [1.0, 1.0]]))


def test_empty_lookup_has_consistent_shapes(tmp_path):
    path = tmp_path / "expert.npz"
    _write_bank(path, ["a"], [[1, 2]])
    batch = ExpertFeatureBank([path]).lookup(["b"], device="cpu")
    assert batch.size == 0
    assert batch.features == ()
    assert batch.weights.shape == (1, 0)


def test_duplicate_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.npz"
    _write_bank(path, ["a", "a"], [[1], [2]])
    with pytest.raises(ValueError, match="duplicate"):
        ExpertFeatureBank([path])


def test_cross_domain_edge_mask_excludes_unknown_metadata(tmp_path):
    path = tmp_path / "expert.npz"
    _write_bank(
        path,
        ["a", "b", "c", "d"],
        np.eye(4),
        organism=["human", "mouse", "unresolved", "human"],
        acquisition_family=["fluorescence", "fluorescence", "electron", "electron"],
    )
    batch = ExpertFeatureBank([path]).lookup(["a", "b", "c", "d"], device="cpu")
    mask = build_cross_domain_edge_mask(
        batch,
        scope="cross_organism_or_acquisition",
        device="cpu",
    )
    assert mask[0, 1]
    assert mask[0, 3]
    assert mask[1, 3]
    assert mask[2, 0]
    assert not mask[2, 3]
    assert torch.equal(mask, mask.T)


def test_metadata_disagreement_between_banks_is_rejected(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_bank(first, ["a"], [[1]], organism=["human"])
    _write_bank(second, ["a"], [[2]], organism=["mouse"])
    bank = ExpertFeatureBank([first, second])
    with pytest.raises(ValueError, match="disagree"):
        bank.lookup(["a"], device="cpu")


def test_stable_metadata_codes_preserve_equality_and_unknowns():
    codes = stable_metadata_codes(
        ["Homo sapiens", " homo sapiens ", "Mus musculus", "unresolved", ""],
        device="cpu",
    )
    assert codes[0] == codes[1]
    assert codes[0] != codes[2]
    assert codes[3:].tolist() == [0, 0]
