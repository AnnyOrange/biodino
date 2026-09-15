import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

from dinov3.eval.bio_tracking.ctc_metrics_requested import (
    calculate_requested_metrics,
    create_edge_mapping_indexed,
)


def _fixture():
    labels = [
        np.asarray([1, 2]),
        np.asarray([1, 3, 4]),
        np.asarray([3, 4, 5, 6]),
        np.asarray([5, 6]),
    ]
    tracks = np.asarray(
        [
            [1, 0, 1, 0],
            [2, 0, 0, 0],
            [3, 1, 2, 1],
            [4, 1, 2, 1],
            [5, 2, 3, 3],
            [6, 2, 3, 4],
        ]
    )
    return labels, tracks


def test_indexed_edge_mapping_matches_pinned_implementation():
    vendor = Path(__file__).parents[2] / "outputs/02_eval_runtime/py-ctcmetrics"
    sys.path.insert(0, str(vendor))
    try:
        from ctc_metrics.utils.representations import create_edge_mapping
    finally:
        sys.path.pop(0)

    labels, tracks = _fixture()
    cumulative_indices = np.cumsum([0] + [len(frame) for frame in labels])
    vertex_true_positive = np.asarray(
        [True, False, True, True, False, True, False, True, True, False, True]
    )

    expected = create_edge_mapping(
        tracks,
        labels,
        vertex_true_positive,
        cumulative_indices,
    )
    actual = create_edge_mapping_indexed(
        tracks,
        labels,
        vertex_true_positive,
        cumulative_indices,
    )
    np.testing.assert_array_equal(actual, expected)


def test_requested_metrics_match_pinned_calculate_metrics():
    vendor = Path(__file__).parents[2] / "outputs/02_eval_runtime/py-ctcmetrics"
    sys.path.insert(0, str(vendor))
    try:
        from ctc_metrics.scripts.evaluate import calculate_metrics
    finally:
        sys.path.pop(0)

    labels, tracks = _fixture()
    trajectory = {
        "labels_ref": labels,
        "labels_comp": labels,
        "mapped_ref": labels,
        "mapped_comp": labels,
    }
    segmentation = {
        "labels_ref": labels,
        "ious": [np.ones(len(frame)) for frame in labels],
    }
    expected = calculate_metrics(
        tracks,
        tracks,
        deepcopy(trajectory),
        deepcopy(segmentation),
        metrics=["Valid", "DET", "SEG", "TRA"],
        is_valid=True,
    )
    actual, _ = calculate_requested_metrics(
        tracks,
        tracks,
        deepcopy(trajectory),
        deepcopy(segmentation),
        True,
    )

    for key in ("Valid", "DET", "SEG", "TRA", "AOGM", "AOGM_0"):
        assert actual[key] == expected[key]
    for key in ("NS", "FN", "FP", "ED", "EA", "EC"):
        assert actual[f"AOGM_{key}"] == expected[f"AOGM_{key}"]
