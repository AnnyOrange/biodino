#!/usr/bin/env python3
"""Estimate routed expert-consensus activity before expensive continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from dinov3.data.expert_feature_bank import ExpertFeatureBank, build_cross_domain_edge_mask
    from dinov3.loss.expert_consensus_residual_loss import _cosine_similarity, _mutual_topk
except ModuleNotFoundError:  # Portable expert-bank audit outside the training checkout.
    from expert_feature_bank import ExpertFeatureBank, build_cross_domain_edge_mask
    from expert_consensus_residual_loss import _cosine_similarity, _mutual_topk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", nargs="+", type=Path, required=True)
    parser.add_argument("--scope", default="cross_organism_or_acquisition")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--min-experts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enforce-gates", action="store_true")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "nonzero_fraction": float(np.mean(array > 0)),
    }


def batch_activity(
    bank: ExpertFeatureBank,
    keys: list[str],
    *,
    scope: str,
    topk: int,
    min_experts: int,
    shuffled: bool,
) -> dict[str, float]:
    batch = bank.lookup(keys, device="cpu")
    candidate = build_cross_domain_edge_mask(batch, scope=scope, device="cpu")
    vote_count = torch.zeros_like(candidate, dtype=torch.long)
    mutual_counts = []
    for expert_index, raw_features in enumerate(batch.features):
        features = raw_features
        if shuffled and len(features) > 1:
            offset = 1 + (expert_index % (len(features) - 1))
            features = features.roll(shifts=offset, dims=0)
        similarity = _cosine_similarity(features, 1.0e-8)
        mutual = _mutual_topk(similarity, topk, candidate)
        pair_weight = batch.weights[expert_index, :, None] * batch.weights[expert_index, None, :]
        active = mutual & (pair_weight > 0)
        vote_count += active.long()
        mutual_counts.append(float(active.sum().item()))
    consensus = (vote_count >= min_experts) & candidate
    return {
        "bank_samples": float(batch.size),
        "candidate_edges": float(candidate.sum().item()),
        "consensus_edges": float(consensus.sum().item()),
        "expert_mutual_edges_mean": float(np.mean(mutual_counts)),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 2 or args.batches <= 0 or args.topk <= 0:
        raise ValueError("batch-size >= 2, batches > 0, and topk > 0 are required")
    if args.min_experts <= 0 or args.min_experts > len(args.banks):
        raise ValueError("min-experts must be in [1, number of banks]")
    bank = ExpertFeatureBank(args.banks)
    reference_keys = bank.banks[0].keys.astype(str)
    for expert in bank.banks[1:]:
        if not np.array_equal(reference_keys, expert.keys.astype(str)):
            raise ValueError(f"Expert bank keys disagree: {expert.path}")
    if len(reference_keys) < args.batch_size:
        raise ValueError("Expert bank has fewer rows than the requested batch")

    rng = np.random.default_rng(args.seed)
    results = {"true": [], "shuffled": []}
    for _ in range(args.batches):
        indices = rng.choice(len(reference_keys), size=args.batch_size, replace=False)
        keys = reference_keys[indices].tolist()
        results["true"].append(
            batch_activity(
                bank,
                keys,
                scope=args.scope,
                topk=args.topk,
                min_experts=args.min_experts,
                shuffled=False,
            )
        )
        results["shuffled"].append(
            batch_activity(
                bank,
                keys,
                scope=args.scope,
                topk=args.topk,
                min_experts=args.min_experts,
                shuffled=True,
            )
        )

    activity = {}
    for arm, rows in results.items():
        activity[arm] = {
            metric: summarize([row[metric] for row in rows])
            for metric in rows[0]
        }
    metadata = bank.banks[0].metadata
    known_organism = np.asarray([bool(value.strip()) for value in metadata["organism"]])
    known_acquisition = np.asarray(
        [value.strip().lower() not in {"", "unknown", "unresolved"} for value in metadata["acquisition_family"]]
    )
    gate_pass = bool(
        activity["true"]["candidate_edges"]["nonzero_fraction"] >= 0.95
        and activity["true"]["consensus_edges"]["nonzero_fraction"] >= 0.8
        and activity["true"]["consensus_edges"]["mean"]
        > activity["shuffled"]["consensus_edges"]["mean"]
    )
    payload = {
        "banks": [str(path) for path in args.banks],
        "samples": int(len(reference_keys)),
        "batch_size": args.batch_size,
        "batches": args.batches,
        "scope": args.scope,
        "topk": args.topk,
        "min_experts": args.min_experts,
        "known_organism_fraction": float(known_organism.mean()),
        "known_acquisition_fraction": float(known_acquisition.mean()),
        "activity": activity,
        "gate_pass": gate_pass,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if args.enforce_gates and not gate_pass:
        raise SystemExit("Expert-consensus bank preflight failed")


if __name__ == "__main__":
    main()
