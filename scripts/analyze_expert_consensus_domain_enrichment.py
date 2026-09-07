#!/usr/bin/env python3
"""Audit whether frozen-expert consensus edges bridge recovered bio domains."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from scripts.calibrate_bioclip_semantic_routing import (
        aggregate_by_source,
        canonical_acquisition,
        canonical_organism,
        grouped_truth,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ first on sys.path.
    from calibrate_bioclip_semantic_routing import (
        aggregate_by_source,
        canonical_acquisition,
        canonical_organism,
        grouped_truth,
    )


UNKNOWN = {"", "unknown", "unresolved", "none", "nan", "n/a", "na"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", nargs=2, type=Path, required=True)
    parser.add_argument("--truth-overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, np.finfo(np.float32).eps)


def mutual_topk(features: np.ndarray, topk: int) -> np.ndarray:
    features = normalize_rows(features)
    size = len(features)
    if size < 2:
        raise ValueError("At least two rows are required")
    k = min(int(topk), size - 1)
    if k <= 0:
        raise ValueError("topk must be positive")
    similarity = features @ features.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    directed = np.zeros((size, size), dtype=np.bool_)
    directed[np.arange(size)[:, None], neighbors] = True
    return directed & directed.T


def edge_attribute_stats(edges: np.ndarray, values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values).astype(str)
    upper = np.triu(np.asarray(edges, dtype=np.bool_), k=1)
    known = np.asarray([value.strip().lower() not in UNKNOWN for value in values])
    eligible = upper & known[:, None] & known[None, :]
    different = eligible & (values[:, None] != values[None, :])
    eligible_count = int(eligible.sum())
    different_count = int(different.sum())
    class_pairs = Counter()
    for left, right in np.argwhere(eligible):
        pair = tuple(sorted((str(values[left]), str(values[right]))))
        class_pairs[" <> ".join(pair)] += 1
    return {
        "eligible_edges": eligible_count,
        "cross_edges": different_count,
        "cross_fraction": different_count / eligible_count if eligible_count else 0.0,
        "class_pairs": dict(class_pairs.most_common()),
    }


def topology_report(
    edges: np.ndarray,
    *,
    organism: np.ndarray,
    acquisition: np.ndarray,
    domain: np.ndarray,
) -> dict[str, object]:
    return {
        "undirected_edges": int(np.triu(edges, k=1).sum()),
        "organism": edge_attribute_stats(edges, organism),
        "acquisition": edge_attribute_stats(edges, acquisition),
        "source_domain": edge_attribute_stats(edges, domain),
    }


def add_enrichment(report: dict[str, object], baseline: dict[str, object]) -> None:
    for attribute in ("organism", "acquisition", "source_domain"):
        observed = float(report[attribute]["cross_fraction"])
        expected = float(baseline[attribute]["cross_fraction"])
        report[attribute]["cross_fraction_baseline"] = expected
        report[attribute]["cross_enrichment_ratio"] = (
            observed / expected if expected > 0 else None
        )


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be positive")

    bank_keys = None
    grouped_features = []
    source_ids = None
    inverse = None
    with np.load(args.truth_overlay, allow_pickle=False) as overlay:
        truth_keys = np.asarray(overlay["keys"]).astype(str)
        truth_source_ids = np.asarray(overlay["source_id"], dtype=np.int64)
        recovered = np.asarray(overlay["recovered"], dtype=np.bool_)
        raw_organism = np.asarray(overlay["organism"]).astype(str)
        raw_acquisition = np.asarray(overlay["acquisition_family"]).astype(str)
        raw_domain = np.asarray(overlay["domain"]).astype(str)
    for path in args.banks:
        with np.load(path, allow_pickle=False) as bank:
            keys = np.asarray(bank["keys"]).astype(str)
            features = np.asarray(bank["features"], dtype=np.float32)
        if bank_keys is None:
            bank_keys = keys
        elif not np.array_equal(bank_keys, keys):
            raise ValueError(f"Expert bank keys disagree: {path}")
        if not np.array_equal(keys, truth_keys):
            raise ValueError(f"Expert bank and truth overlay keys disagree: {path}")
        group_ids, group_inverse, means = aggregate_by_source(features, truth_source_ids)
        if source_ids is None:
            source_ids, inverse = group_ids, group_inverse
        elif not np.array_equal(source_ids, group_ids):
            raise AssertionError("Source grouping differs between expert banks")
        grouped_features.append(means)
    assert source_ids is not None and inverse is not None

    grouped_recovered = np.zeros(len(source_ids), dtype=np.bool_)
    np.logical_or.at(grouped_recovered, inverse, recovered)
    organism = grouped_truth(raw_organism, inverse, canonical_organism)
    acquisition = grouped_truth(raw_acquisition, inverse, canonical_acquisition)
    domain = grouped_truth(raw_domain, inverse, lambda value: value if value != "unresolved" else "")
    selected = grouped_recovered & ((organism != "") | (acquisition != "") | (domain != ""))
    organism = organism[selected]
    acquisition = acquisition[selected]
    domain = domain[selected]
    first = grouped_features[0][selected]
    second = grouped_features[1][selected]

    all_edges = ~np.eye(len(first), dtype=np.bool_)
    first_mutual = mutual_topk(first, args.topk)
    second_mutual = mutual_topk(second, args.topk)
    true_consensus = first_mutual & second_mutual
    # Match the training control: expert 0 rolls by one row and expert 1 by two.
    shuffled_first = mutual_topk(np.roll(first, shift=1, axis=0), args.topk)
    shuffled_second = mutual_topk(np.roll(second, shift=2, axis=0), args.topk)
    shuffled_consensus = shuffled_first & shuffled_second

    baseline = topology_report(
        all_edges,
        organism=organism,
        acquisition=acquisition,
        domain=domain,
    )
    reports = {
        "expert_0_mutual": topology_report(
            first_mutual, organism=organism, acquisition=acquisition, domain=domain
        ),
        "expert_1_mutual": topology_report(
            second_mutual, organism=organism, acquisition=acquisition, domain=domain
        ),
        "true_consensus": topology_report(
            true_consensus, organism=organism, acquisition=acquisition, domain=domain
        ),
        "shuffled_consensus": topology_report(
            shuffled_consensus, organism=organism, acquisition=acquisition, domain=domain
        ),
    }
    for report in reports.values():
        add_enrichment(report, baseline)
    payload = {
        "banks": [str(path) for path in args.banks],
        "truth_overlay": str(args.truth_overlay),
        "topk": args.topk,
        "recovered_source_images": int(selected.sum()),
        "organism_counts": dict(Counter(value for value in organism if value)),
        "acquisition_counts": dict(Counter(value for value in acquisition if value)),
        "domain_counts": dict(Counter(value for value in domain if value)),
        "all_pairs_baseline": baseline,
        **reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
