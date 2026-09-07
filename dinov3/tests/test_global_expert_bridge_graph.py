import numpy as np
import torch

from scripts.build_global_cross_domain_knn import candidate_mask, encode_metadata
from scripts.merge_global_expert_bridge_graph import (
    build_graph_payload,
    consensus_codes,
    graph_stats,
    mutual_payload,
)


def _cache() -> dict[str, np.ndarray]:
    return {
        "keys": np.asarray(["a", "b", "c", "d"]),
        "key_digest": np.asarray("digest"),
        "scope": np.asarray("cross_organism"),
        "query_indices": np.arange(4, dtype=np.int32),
        "neighbor_indices": np.asarray([[2], [3], [0], [1]], dtype=np.int32),
        "similarities": np.asarray([[0.9], [0.8], [0.9], [0.8]], dtype=np.float16),
        "pair_weights": np.ones((4, 1), dtype=np.float16),
        "organism": np.asarray(["human", "human", "mouse", "mouse"]),
        "acquisition_family": np.asarray(["fluorescence"] * 4),
    }


def test_encode_metadata_reserves_zero_for_unknown() -> None:
    encoded = encode_metadata(np.asarray(["Human", "", "unknown", "mouse", "human"]))
    assert encoded[0] == encoded[4] != 0
    assert encoded[1] == encoded[2] == 0
    assert encoded[3] != encoded[0]


def test_candidate_mask_crosses_known_organisms_only() -> None:
    organism = torch.tensor([1, 1, 2, 0])
    acquisition = torch.zeros(4, dtype=torch.long)
    mask = candidate_mask(
        organism,
        acquisition,
        organism,
        acquisition,
        scope="cross_organism",
    )
    assert mask.tolist() == [
        [False, False, True, False],
        [False, False, True, False],
        [True, True, False, False],
        [False, False, False, False],
    ]


def test_mutual_consensus_builds_symmetric_csr() -> None:
    first = _cache()
    second = _cache()
    first_payload = mutual_payload(first, topk=1, sample_count=4)
    second_payload = mutual_payload(second, topk=1, sample_count=4)
    consensus = consensus_codes([first_payload, second_payload], min_experts=2)
    assert consensus.tolist() == [2, 7, 8, 13]

    stats = graph_stats(
        consensus,
        sample_count=4,
        organism=first["organism"],
        acquisition=first["acquisition_family"],
    )
    assert stats["undirected_edges"] == 2
    assert stats["active_samples"] == 4
    assert stats["undirected_cross_organism"] == 2

    graph = build_graph_payload(
        [first, second],
        [first_payload, second_payload],
        consensus,
        topk=1,
        min_experts=2,
    )
    assert graph["offsets"].tolist() == [0, 1, 2, 3, 4]
    assert graph["neighbor_indices"].tolist() == [2, 3, 0, 1]
    assert np.allclose(graph["confidence"], [0.9, 0.8, 0.9, 0.8], atol=1e-3)
    assert graph["expert_votes"].tolist() == [2, 2, 2, 2]
