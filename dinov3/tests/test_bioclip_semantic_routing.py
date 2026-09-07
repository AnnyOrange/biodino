import numpy as np

from scripts.calibrate_bioclip_semantic_routing import (
    accepted_mask,
    aggregate_by_source,
    canonical_acquisition,
    canonical_organism,
    fit_class_thresholds,
    grouped_truth,
    stratified_split,
)


def test_canonical_metadata() -> None:
    assert canonical_organism("Homo sapiens | Influenza A virus") == "human"
    assert canonical_organism("Tribolium castaneum") == "insect"
    assert canonical_organism("mixed sample") == ""
    assert canonical_acquisition("confocal_microscopy") == "fluorescence_microscopy"
    assert canonical_acquisition("unresolved") == ""


def test_source_grouping_and_stratified_split() -> None:
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    source_ids = np.asarray([7, 7, 9])
    unique, inverse, grouped = aggregate_by_source(features, source_ids)
    assert unique.tolist() == [7, 9]
    np.testing.assert_allclose(np.linalg.norm(grouped, axis=1), 1.0)
    truth = grouped_truth(
        np.asarray(["Homo sapiens", "Homo sapiens", "Mus musculus"]),
        inverse,
        canonical_organism,
    )
    assert truth.tolist() == ["human", "mouse"]

    labels = np.asarray(["human"] * 4 + ["mouse"] * 2 + [""])
    calibration, holdout = stratified_split(labels, 0.5, seed=3)
    assert set(calibration).isdisjoint(set(holdout))
    assert set(labels[calibration]) == {"human", "mouse"}
    assert set(labels[holdout]) == {"human", "mouse"}


def test_class_specific_abstention_thresholds() -> None:
    labels = ["human", "mouse"]
    predicted = np.asarray([0, 0, 0, 1, 1, 1])
    margins = np.asarray([0.9, 0.8, 0.1, 0.9, 0.8, 0.1], dtype=np.float32)
    truth = np.asarray(["human", "human", "mouse", "mouse", "human", "human"])
    thresholds = fit_class_thresholds(
        predicted,
        margins,
        truth,
        np.arange(6),
        labels,
        target_precision=1.0,
        min_accepted=2,
    )
    accepted = accepted_mask(predicted, margins, labels, thresholds)
    assert accepted.tolist() == [True, True, False, False, False, False]
