#!/usr/bin/env python3
"""Merge full-bank expert KNN caches into a cross-domain consensus bridge graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true-caches", nargs="+", type=Path, required=True)
    parser.add_argument("--shuffled-caches", nargs="+", type=Path)
    parser.add_argument("--topks", nargs="+", type=int, default=(5, 10, 20, 50, 100))
    parser.add_argument("--min-experts", type=int, default=2)
    parser.add_argument("--graph-topk", type=int, default=20)
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        cache = {name: np.asarray(payload[name]) for name in payload.files}
    required = {
        "keys",
        "key_digest",
        "scope",
        "shuffled_control",
        "query_indices",
        "neighbor_indices",
        "similarities",
        "pair_weights",
        "organism",
        "acquisition_family",
    }
    missing = required.difference(cache)
    if missing:
        raise ValueError(f"Cache {path} is missing arrays: {sorted(missing)}")
    cache["path"] = np.asarray(str(path))
    return cache


def validate_caches(caches: list[dict[str, np.ndarray]], *, shuffled: bool) -> None:
    if not caches:
        raise ValueError("At least one expert cache is required")
    reference_keys = caches[0]["keys"].astype(str)
    reference_digest = str(caches[0]["key_digest"].item())
    reference_scope = str(caches[0]["scope"].item())
    reference_organism = caches[0]["organism"].astype(str)
    reference_acquisition = caches[0]["acquisition_family"].astype(str)
    for cache in caches:
        if not np.array_equal(cache["keys"].astype(str), reference_keys):
            raise ValueError(f"Cache keys disagree: {cache['path'].item()}")
        if str(cache["key_digest"].item()) != reference_digest:
            raise ValueError(f"Cache key digests disagree: {cache['path'].item()}")
        if str(cache["scope"].item()) != reference_scope:
            raise ValueError(f"Cache scopes disagree: {cache['path'].item()}")
        if not np.array_equal(cache["organism"].astype(str), reference_organism):
            raise ValueError(f"Cache organism metadata disagree: {cache['path'].item()}")
        if not np.array_equal(
            cache["acquisition_family"].astype(str), reference_acquisition
        ):
            raise ValueError(f"Cache acquisition metadata disagree: {cache['path'].item()}")
        if bool(cache["shuffled_control"].item()) != shuffled:
            raise ValueError(
                f"Cache shuffled flag disagrees with its arm: {cache['path'].item()}"
            )
        shape = cache["neighbor_indices"].shape
        if cache["similarities"].shape != shape or cache["pair_weights"].shape != shape:
            raise ValueError(f"Neighbor payload shapes disagree: {cache['path'].item()}")
        if shape[0] != len(cache["query_indices"]):
            raise ValueError(f"Query and neighbor rows disagree: {cache['path'].item()}")


def directed_payload(
    cache: dict[str, np.ndarray],
    *,
    topk: int,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = cache["neighbor_indices"].shape[1]
    if topk <= 0 or topk > width:
        raise ValueError(f"topk={topk} is outside cache width [1, {width}]")
    source = np.repeat(cache["query_indices"].astype(np.int64), topk)
    target = cache["neighbor_indices"][:, :topk].reshape(-1).astype(np.int64)
    similarity = cache["similarities"][:, :topk].reshape(-1).astype(np.float32)
    pair_weight = cache["pair_weights"][:, :topk].reshape(-1).astype(np.float32)
    valid = (target >= 0) & np.isfinite(similarity) & (pair_weight > 0)
    source = source[valid]
    target = target[valid]
    codes = source * sample_count + target
    if len(np.unique(codes)) != len(codes):
        raise ValueError("A KNN cache contains duplicate directed edges")
    order = np.argsort(codes)
    return codes[order], similarity[valid][order], pair_weight[valid][order]


def mutual_payload(
    cache: dict[str, np.ndarray],
    *,
    topk: int,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes, similarity, pair_weight = directed_payload(
        cache,
        topk=topk,
        sample_count=sample_count,
    )
    source, target = np.divmod(codes, sample_count)
    reverse_codes = target * sample_count + source
    mutual_codes = np.intersect1d(codes, reverse_codes, assume_unique=True)
    positions = np.searchsorted(codes, mutual_codes)
    return mutual_codes, similarity[positions], pair_weight[positions]


def consensus_codes(
    expert_payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    min_experts: int,
) -> np.ndarray:
    combined = np.concatenate([payload[0] for payload in expert_payloads])
    codes, counts = np.unique(combined, return_counts=True)
    return codes[counts >= min_experts]


def array_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "nonzero_fraction": 0.0}
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "nonzero_fraction": float(np.mean(values > 0)),
    }


def graph_stats(
    codes: np.ndarray,
    *,
    sample_count: int,
    organism: np.ndarray,
    acquisition: np.ndarray,
) -> dict:
    source, target = np.divmod(codes, sample_count)
    degree = np.bincount(source, minlength=sample_count)
    undirected = source < target
    left = source[undirected]
    right = target[undirected]
    normalized_organism = np.char.lower(np.char.strip(organism.astype(str)))
    normalized_acquisition = np.char.lower(np.char.strip(acquisition.astype(str)))
    unknown = np.asarray(["", "unknown", "unresolved", "none", "nan", "n/a", "na"])
    organism_known = ~np.isin(normalized_organism, unknown)
    acquisition_known = ~np.isin(normalized_acquisition, unknown)
    cross_organism = (
        organism_known[left]
        & organism_known[right]
        & (normalized_organism[left] != normalized_organism[right])
    )
    cross_acquisition = (
        acquisition_known[left]
        & acquisition_known[right]
        & (normalized_acquisition[left] != normalized_acquisition[right])
    )
    return {
        "directed_edges": int(len(codes)),
        "undirected_edges": int(undirected.sum()),
        "active_samples": int(np.sum(degree > 0)),
        "active_sample_fraction": float(np.mean(degree > 0)),
        "degree": array_summary(degree),
        "undirected_cross_organism": int(cross_organism.sum()),
        "undirected_cross_acquisition": int(cross_acquisition.sum()),
        "undirected_cross_either": int((cross_organism | cross_acquisition).sum()),
        "undirected_cross_both": int((cross_organism & cross_acquisition).sum()),
    }


def arm_at_topk(
    caches: list[dict[str, np.ndarray]],
    *,
    topk: int,
    min_experts: int,
    sample_count: int,
    organism: np.ndarray,
    acquisition: np.ndarray,
) -> tuple[dict, list[tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray]:
    payloads = [
        mutual_payload(cache, topk=topk, sample_count=sample_count)
        for cache in caches
    ]
    consensus = consensus_codes(payloads, min_experts=min_experts)
    stats = graph_stats(
        consensus,
        sample_count=sample_count,
        organism=organism,
        acquisition=acquisition,
    )
    stats["expert_mutual_directed_edges"] = [int(len(payload[0])) for payload in payloads]
    return stats, payloads, consensus


def build_graph_payload(
    caches: list[dict[str, np.ndarray]],
    expert_payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    consensus: np.ndarray,
    *,
    topk: int,
    min_experts: int,
) -> dict[str, np.ndarray]:
    keys = caches[0]["keys"].astype(str)
    sample_count = len(keys)
    similarity_sum = np.zeros(len(consensus), dtype=np.float64)
    weight_sum = np.zeros(len(consensus), dtype=np.float64)
    votes = np.zeros(len(consensus), dtype=np.uint8)
    for codes, similarities, pair_weights in expert_payloads:
        positions = np.searchsorted(codes, consensus)
        safe_positions = np.minimum(positions, max(len(codes) - 1, 0))
        present = (positions < len(codes))
        if len(codes):
            present &= codes[safe_positions] == consensus
        else:
            present[:] = False
        selected = safe_positions[present]
        similarity_sum[present] += similarities[selected] * pair_weights[selected]
        weight_sum[present] += pair_weights[selected]
        votes[present] += 1
    confidence = (similarity_sum / np.maximum(weight_sum, 1.0e-12)).astype(np.float32)
    source, target = np.divmod(consensus, sample_count)
    degree = np.bincount(source, minlength=sample_count)
    offsets = np.zeros(sample_count + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(degree)
    active = np.flatnonzero(degree > 0)
    control_source_permutation = np.arange(sample_count, dtype=np.int32)
    if len(active) > 1:
        control_source_permutation[active] = np.roll(active, 1).astype(np.int32)
    return {
        "keys": keys,
        "key_digest": caches[0]["key_digest"],
        "scope": caches[0]["scope"],
        "topk": np.asarray(topk, dtype=np.int64),
        "min_experts": np.asarray(min_experts, dtype=np.int64),
        "offsets": offsets,
        "neighbor_indices": target.astype(np.int32),
        "confidence": confidence,
        "expert_votes": votes,
        "degree": degree.astype(np.int32),
        "control_source_permutation": control_source_permutation,
        "organism": caches[0]["organism"].astype(str),
        "acquisition_family": caches[0]["acquisition_family"].astype(str),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.graph_output.exists():
        raise FileExistsError("Refusing to overwrite report or graph output")
    if args.min_experts <= 0 or args.min_experts > len(args.true_caches):
        raise ValueError("min-experts must be in [1, number of true caches]")
    if args.graph_topk not in args.topks:
        raise ValueError("graph-topk must be included in topks")
    if any(topk <= 0 for topk in args.topks):
        raise ValueError("All topks must be positive")

    true_caches = [load_cache(path) for path in args.true_caches]
    validate_caches(true_caches, shuffled=False)
    shuffled_caches = None
    if args.shuffled_caches:
        if len(args.shuffled_caches) != len(args.true_caches):
            raise ValueError("true-caches and shuffled-caches must have equal length")
        shuffled_caches = [load_cache(path) for path in args.shuffled_caches]
        validate_caches(shuffled_caches, shuffled=True)
        if not np.array_equal(
            shuffled_caches[0]["keys"].astype(str), true_caches[0]["keys"].astype(str)
        ):
            raise ValueError("True and shuffled cache keys disagree")

    keys = true_caches[0]["keys"].astype(str)
    sample_count = len(keys)
    organism = true_caches[0]["organism"].astype(str)
    acquisition = true_caches[0]["acquisition_family"].astype(str)
    results = {"true": {}, "shuffled": {}}
    graph_payloads = None
    graph_consensus = None
    for topk in sorted(set(args.topks)):
        stats, payloads, consensus = arm_at_topk(
            true_caches,
            topk=topk,
            min_experts=args.min_experts,
            sample_count=sample_count,
            organism=organism,
            acquisition=acquisition,
        )
        results["true"][str(topk)] = stats
        if topk == args.graph_topk:
            graph_payloads = payloads
            graph_consensus = consensus
        if shuffled_caches is not None:
            shuffled_stats, _, _ = arm_at_topk(
                shuffled_caches,
                topk=topk,
                min_experts=args.min_experts,
                sample_count=sample_count,
                organism=organism,
                acquisition=acquisition,
            )
            results["shuffled"][str(topk)] = shuffled_stats

    assert graph_payloads is not None and graph_consensus is not None
    graph = build_graph_payload(
        true_caches,
        graph_payloads,
        graph_consensus,
        topk=args.graph_topk,
        min_experts=args.min_experts,
    )
    args.graph_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.graph_output, **graph)
    selected_true = results["true"][str(args.graph_topk)]
    selected_shuffled = results["shuffled"].get(str(args.graph_topk))
    gate_pass = bool(
        selected_true["active_sample_fraction"] >= 0.05
        and selected_true["undirected_edges"] >= 100
        and (
            selected_shuffled is None
            or selected_true["undirected_edges"] > selected_shuffled["undirected_edges"]
        )
    )
    report = {
        "true_caches": [str(path) for path in args.true_caches],
        "shuffled_caches": [str(path) for path in args.shuffled_caches or []],
        "samples": sample_count,
        "scope": str(true_caches[0]["scope"].item()),
        "topks": sorted(set(args.topks)),
        "min_experts": args.min_experts,
        "graph_topk": args.graph_topk,
        "graph_output": str(args.graph_output),
        "results": results,
        "gate_pass": gate_pass,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
