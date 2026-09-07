import numpy as np

from scripts.analyze_expert_consensus_domain_enrichment import (
    edge_attribute_stats,
    mutual_topk,
)


def test_mutual_topk_returns_symmetric_edges_without_diagonal() -> None:
    features = np.asarray([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]], dtype=np.float32)
    edges = mutual_topk(features, topk=1)
    assert np.array_equal(edges, edges.T)
    assert not np.diag(edges).any()
    assert edges[0, 1]
    assert not edges[0, 2]


def test_edge_attribute_stats_excludes_unknown_labels() -> None:
    edges = np.ones((4, 4), dtype=np.bool_)
    np.fill_diagonal(edges, False)
    stats = edge_attribute_stats(edges, np.asarray(["human", "mouse", "human", "unresolved"]))
    assert stats["eligible_edges"] == 3
    assert stats["cross_edges"] == 2
    assert stats["cross_fraction"] == 2 / 3
