import numpy as np
import torch

from dinov3.eval.bio_frozen_eval.intervention_stable_relations import (
    ALL_VIEWS,
    build_relation_graphs,
    cross_view_knn,
    graph_label_precision,
    graph_metrics,
    make_intervention_views,
    mutual_edge_codes,
)


def test_intervention_views_are_aligned_bounded_and_nontrivial():
    torch.manual_seed(31)
    images = torch.rand(3, 3, 24, 20)
    views = make_intervention_views(images)

    assert tuple(views) == ALL_VIEWS
    for view in views.values():
        assert view.shape == images.shape
        assert view.min().item() >= 0.0
        assert view.max().item() <= 1.0
    assert torch.equal(views["identity"], images)
    assert not torch.equal(views["gain"], images)
    assert not torch.equal(views["resolution"], images)


def test_cross_view_knn_excludes_aligned_self_and_honors_validity():
    features = np.eye(6, dtype=np.float32)
    valid = np.array([True, True, True, True, True, False])
    neighbors = cross_view_knn(
        features,
        features,
        k=2,
        query_valid=valid,
        gallery_valid=valid,
    )

    assert np.all(neighbors[np.arange(5)] != np.arange(5)[:, None])
    assert not np.any(neighbors[:5] == 5)
    assert np.all(neighbors[5] == -1)


def test_mutual_edge_codes_only_retains_reciprocal_neighbors():
    forward = np.array([[1, 2], [0, 2], [1, 3], [2, 1]])
    reverse = np.array([[1, 3], [0, 2], [3, 1], [2, 1]])
    codes = set(mutual_edge_codes(forward, reverse).tolist())

    assert 0 * 4 + 1 in codes
    assert 1 * 4 + 0 in codes
    assert 0 * 4 + 2 not in codes


def test_survival_graph_recovers_stable_clusters_and_shuffling_weakens_it():
    rng = np.random.default_rng(37)
    n_clusters = 6
    samples_per_cluster = 8
    dim = 16
    labels = np.repeat(np.arange(n_clusters), samples_per_cluster)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    base = centers[labels] + 0.04 * rng.normal(size=(len(labels), dim))
    views = [base]
    for _ in range(len(ALL_VIEWS) - 1):
        views.append(base + 0.03 * rng.normal(size=base.shape))
    bank = np.stack(views).astype(np.float32)

    graphs, diagnostics = build_relation_graphs(
        bank,
        k=5,
        max_neighbors=3,
        min_survival=0.50,
        shuffle_seed=41,
        chunk_size=16,
    )
    heldout = diagnostics["heldout_codes"]
    proposed = graph_metrics(graphs["isrd"], heldout_codes=heldout)
    shuffled = graph_metrics(graphs["view_shuffled"], heldout_codes=heldout)

    assert proposed["coverage"] > 0.8
    assert proposed["heldout_survival"] > shuffled["heldout_survival"]
    assert graph_label_precision(graphs["isrd"], labels) > 0.95


def test_label_precision_is_evaluation_only_and_checks_alignment():
    rng = np.random.default_rng(43)
    features = rng.normal(size=(len(ALL_VIEWS), 24, 8)).astype(np.float32)
    graphs, _ = build_relation_graphs(
        features,
        k=3,
        max_neighbors=2,
        min_survival=0.0,
        min_safe_cosine=-1.0,
        chunk_size=8,
    )
    try:
        graph_label_precision(graphs["single"], np.arange(23))
    except ValueError as error:
        assert "shape [N]" in str(error)
    else:
        raise AssertionError("Expected misaligned labels to fail")
