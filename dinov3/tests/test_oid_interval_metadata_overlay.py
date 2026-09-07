import numpy as np

from scripts.build_oid_interval_metadata_overlay import (
    alternating_cv_predictions,
    fit_class_distance_thresholds,
    nearest_predictions,
    propagate_task,
)


def test_nearest_predictions_and_tie_break_left() -> None:
    labels, distance = nearest_predictions(
        np.asarray([9, 15, 21]),
        np.asarray([10, 20]),
        np.asarray(["a", "b"]),
    )
    assert labels.tolist() == ["a", "a", "b"]
    assert distance.tolist() == [1, 5, 1]


def test_alternating_cv_and_class_thresholds_on_contiguous_blocks() -> None:
    source_ids = np.asarray([10, 11, 12, 100, 101, 102])
    truth = np.asarray(["a", "a", "a", "b", "b", "b"])
    indices, predicted, distance = alternating_cv_predictions(source_ids, truth)
    assert np.array_equal(predicted, truth[indices])
    thresholds = fit_class_distance_thresholds(
        truth[indices],
        predicted,
        distance,
        target_precision=1.0,
        min_accepted=2,
        max_distance=10,
    )
    assert thresholds == {"a": 1, "b": 1}
    routed, routed_distance, accepted = propagate_task(
        np.asarray([9, 13, 99, 103, 50]), source_ids, truth, thresholds
    )
    assert routed.tolist() == ["a", "a", "b", "b", ""]
    assert accepted.tolist() == [True, True, True, True, False]
    assert routed_distance[-1] > 2


def test_wrong_distant_predictions_are_abstained() -> None:
    truth = np.asarray(["a", "a", "b", "b"])
    predicted = np.asarray(["a", "a", "a", "b"])
    distance = np.asarray([1, 2, 100, 1])
    thresholds = fit_class_distance_thresholds(
        truth,
        predicted,
        distance,
        target_precision=1.0,
        min_accepted=2,
        max_distance=1000,
    )
    assert thresholds["a"] == 2
    assert thresholds["b"] is None
