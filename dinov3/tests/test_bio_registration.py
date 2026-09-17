import numpy as np
import pytest

from dinov3.eval.bio_registration import (apply_affine, fit_correspondences,
    match_descriptors, patch_coordinates, register_descriptors,
    registration_metrics, voxel_coordinates)
from dinov3.eval.bio_registration.datasets import RegistrationPair, grouped_partition


def test_native_patch_coordinate_frame():
    np.testing.assert_allclose(patch_coordinates((2, 2), (40, 80)),
        [[20, 10], [60, 10], [20, 30], [60, 30]])


def test_anisotropic_volume_xyz_calibration():
    np.testing.assert_allclose(voxel_coordinates([[1, 2, 3]], [.02, .03, .1], [4, 5, 6]),
                               [[4.07, 5.075, 6.15]])
    with pytest.raises(ValueError):
        voxel_coordinates([[1, 2, 3]], [.02, 0, .1])


@pytest.mark.parametrize("dim", [2, 3])
def test_actual_robust_affine_and_metric(dim):
    rng = np.random.default_rng(12)
    source = rng.normal(size=(100, dim)) * 10
    matrix = np.eye(dim, dim + 1)
    matrix[:, :dim] *= 1.2
    matrix[:, dim] = np.arange(dim) + 4
    target = apply_affine(source, matrix)
    corrupted = target.copy()
    corrupted[:15] = rng.normal(size=(15, dim)) * 100
    fitted = fit_correspondences(source, corrupted, threshold=.1, seed=0)
    assert fitted.success and fitted.inliers >= 85
    np.testing.assert_allclose(fitted.matrix, matrix, atol=1e-4)
    scores = registration_metrics(source, target, apply_affine(source, fitted.matrix), 100)
    assert scores["median_rtre"] < 1e-6 and scores["robustness"] == 1


def test_descriptor_matching_drives_coordinates_not_landmark_oracle():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(30, 16))
    source = rng.normal(size=(30, 2)) * 10
    permutation = rng.permutation(30)
    target = (source + [4, 7])[permutation]
    result = register_descriptors(features, features[permutation], source, target, threshold=.01)
    assert result.success
    np.testing.assert_allclose(apply_affine(source, result.matrix), source + [4, 7], atol=1e-5)


def test_failed_registration_scored_as_identity_all_landmarks():
    result = fit_correspondences(np.empty((0, 2)), np.empty((0, 2)))
    assert not result.success
    scores = registration_metrics([[0, 0], [1, 1]], [[3, 4], [4, 5]],
                                  apply_affine([[0, 0], [1, 1]], result.matrix), 10)
    assert scores["landmarks"] == 2 and scores["median_rtre"] == .5
    with pytest.raises(ValueError):
        registration_metrics([[0, 0]], [[0, 0], [1, 1]], [[0, 0]], 10)


def test_zero_and_duplicate_descriptors_filtered():
    assert len(match_descriptors(np.zeros((2, 4)), np.ones((3, 4)))) == 0
    assert len(match_descriptors(np.ones((2, 4)), np.ones((3, 4)))) == 0


def test_all_stains_of_tissue_stay_together():
    pairs = [RegistrationPair(str(i), f"tissue{i // 3}", None, None, None, None, "training")
             for i in range(12)]
    mapping = grouped_partition(pairs, 0, exclude_development_groups=["tissue0"])
    assert mapping == grouped_partition(pairs, 0, exclude_development_groups=["tissue0"])
    for group in {p.group for p in pairs}:
        assert len({mapping[p.pair_id] for p in pairs if p.group == group}) == 1
    assert mapping["0"] == "evaluation"


def test_true_3d_descriptor_correspondence_in_physical_units():
    from dinov3.eval.bio_registration.volume import VolumeGeometry, evaluate_volume_correspondence
    rng = np.random.default_rng(9)
    indices = rng.integers(0, 30, size=(100, 3)).astype(float)
    source_geometry = VolumeGeometry((40, 40, 40), (.05, .1, .2))
    target_geometry = VolumeGeometry((40, 40, 40), (.05, .1, .2), (2, 3, 4))
    features = rng.normal(size=(100, 16))
    source = source_geometry.coordinates(indices[:10])
    target = target_geometry.coordinates(indices[:10])
    result = evaluate_volume_correspondence(features, features, indices, indices,
        source_geometry, target_geometry, source, target, threshold_um=.01)
    assert result["success"] and result["metrics"]["coordinate_unit"] == "micrometers"
    assert result["metrics"]["median_tre"] < 1e-5


def test_acrobat_2022_annotator_averaging_then_percentile():
    from dinov3.eval.bio_registration.acrobat import acrobat_pair_score, acrobat_2022_aggregate
    prediction = [[0, 0], [0, 0]]
    targets = [[[3, 4], [0, 10]], [[0, 10], [0, 20]]]
    assert acrobat_pair_score(prediction, targets) == pytest.approx(14.25)
    assert acrobat_2022_aggregate([10, 20, 40]) == 20
    with pytest.raises(ValueError):
        acrobat_pair_score(prediction, targets[:1])


def test_exact_torch_and_scipy_matching_agree():
    rng = np.random.default_rng(9)
    source = rng.normal(size=(25, 16))
    target = source[rng.permutation(25)] + rng.normal(size=(25, 16)) * .05
    np.testing.assert_array_equal(match_descriptors(source, target, backend="torch", device="cpu"),
                                  match_descriptors(source, target))


def test_protocol_signature_binds_selected_layers_environment_and_probe():
    from dinov3.eval.bio_registration.sweep import protocol_signature
    frozen = {key: "fixed" for key in ["dataset", "code_commit", "registry_hash", "manifest_hash",
        "model_family", "environment_sha256", "selected", "selected_zero_based_layers", "seed",
        "aggregation", "metric", "preprocessing", "evaluator", "selection_checkpoint_sha256",
        "selection_train_config_sha256"]}
    original = protocol_signature(frozen)
    for key in ["environment_sha256", "selected_zero_based_layers", "selected", "preprocessing"]:
        assert protocol_signature(frozen | {key: "changed"}) != original


def test_manifest_binds_file_contents_and_each_shared_file_hashed_once(tmp_path, monkeypatch):
    from dinov3.eval.bio_registration.sweep import bind_manifest, canonical_hash
    from dinov3.eval.bio_frozen_eval import candidate_datasets
    calls = []
    original = candidate_datasets.file_digest
    def record(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(candidate_datasets, "file_digest", record)
    paths = [tmp_path / "source.jpg", tmp_path / "target.jpg", tmp_path / "source.csv", tmp_path / "target.csv"]
    for path in paths:
        path.write_text("original")
    pair = RegistrationPair("p0", "tissue0", *paths, "training")
    first = bind_manifest([pair, pair], tmp_path, {"p0": "development"})
    assert len(calls) == 4
    paths[0].write_text("changed")
    second = bind_manifest([pair, pair], tmp_path, {"p0": "development"})
    assert canonical_hash(first) != canonical_hash(second)
