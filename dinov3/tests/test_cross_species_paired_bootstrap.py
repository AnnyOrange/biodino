import numpy as np

from scripts.analyze_cross_species_paired_bootstrap import (
    ARMS,
    METRICS,
    class_macro,
    paired_stratified_bootstrap,
    per_query_metrics,
)


def test_per_query_metrics_matches_hand_computation():
    gallery_x = np.eye(3, dtype=np.float32)
    gallery_y = np.array([0, 1, 0])
    query_x = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    query_y = np.array([0, 1])

    metrics = per_query_metrics(gallery_x, gallery_y, query_x, query_y, device="cpu")

    np.testing.assert_allclose(metrics["recall_at_1"], [1.0, 1.0])
    np.testing.assert_allclose(metrics["mrr"], [1.0, 1.0])
    np.testing.assert_allclose(metrics["map_at_5"], [5.0 / 6.0, 1.0])


def test_class_macro_balances_classes():
    values = np.array([1.0, 1.0, 0.0])
    labels = np.array([0, 0, 1])
    assert class_macro(values, labels) == 0.5


def test_paired_bootstrap_detects_uniform_true_improvement():
    labels = np.array([0, 0, 1, 1])
    direction_values = {}
    direction_labels = {}
    for direction in ("forward", "reverse"):
        direction_labels[direction] = labels
        direction_values[direction] = {}
        for arm in ARMS:
            level = {"baseline": 0.4, "true": 0.6, "shuffled": 0.3}[arm]
            direction_values[direction][arm] = {
                metric: np.full(len(labels), level, dtype=np.float64) for metric in METRICS
            }

    observed, contrasts = paired_stratified_bootstrap(
        direction_values, direction_labels, bootstrap_samples=100, seed=7
    )

    assert observed["true"]["map_at_5"] == 0.6
    assert contrasts["map_at_5"]["true_vs_baseline"]["significant_positive_95"]
    assert contrasts["map_at_5"]["true_vs_shuffled"]["significant_positive_95"]
