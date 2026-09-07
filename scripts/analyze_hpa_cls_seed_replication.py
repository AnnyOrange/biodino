#!/usr/bin/env python3
"""Summarize a preregistered HPA CLS endpoint across training seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


ARMS = ("baseline", "true", "shuffled")
ENDPOINTS = {
    "hpa_retrieval": {
        "task": "retrieval",
        "aggregation": "global",
        "protocol_contains": "same-gene-query-gallery",
        "metric": "map_at_5",
    },
    "hpa_clustering_ge10": {
        "task": "clustering",
        "aggregation": "location",
        "protocol_contains": "single-location-ge10",
        "metric": "nmi",
    },
}


def parse_seed_root(value: str) -> tuple[int, Path]:
    seed_text, separator, root_text = value.partition("=")
    if not separator or not seed_text or not root_text:
        raise argparse.ArgumentTypeError("expected SEED=ROOT")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid seed {seed_text!r}") from exc
    return seed, Path(root_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-root",
        action="append",
        type=parse_seed_root,
        required=True,
        help="Seed and evaluation root in SEED=ROOT form; repeat for each seed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clustering-min-delta", type=float, default=0.01)
    return parser.parse_args()


def load_endpoints(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values: dict[str, float] = {}
    for name, spec in ENDPOINTS.items():
        matches = [
            row
            for row in rows
            if row.get("dataset") == "hpa-subcellular"
            and row.get("task") == spec["task"]
            and row.get("aggregation") == spec["aggregation"]
            and spec["protocol_contains"] in row.get("protocol", "")
            and not row.get("error")
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {name} row in {path}, found {len(matches)}")
        values[name] = float(matches[0][spec["metric"]])
    return values


def summarize_seed(root: Path) -> dict:
    results = {arm: load_endpoints(root / arm / "nlb2_cls" / "summary.csv") for arm in ARMS}
    deltas = {
        endpoint: {
            "metric": spec["metric"],
            "true_vs_baseline": results["true"][endpoint] - results["baseline"][endpoint],
            "true_vs_shuffled": results["true"][endpoint] - results["shuffled"][endpoint],
        }
        for endpoint, spec in ENDPOINTS.items()
    }
    return {"results": results, "deltas": deltas}


def build_payload(seed_roots: list[tuple[int, Path]], clustering_min_delta: float) -> dict:
    if len(seed_roots) < 2:
        raise ValueError("At least two seed roots are required for a replication report")
    seeds = [seed for seed, _ in seed_roots]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seed roots must be unique, got {seeds}")
    per_seed = {str(seed): summarize_seed(root) for seed, root in seed_roots}
    aggregate = {}
    for endpoint in ENDPOINTS:
        aggregate[endpoint] = {}
        for contrast in ("true_vs_baseline", "true_vs_shuffled"):
            values = [entry["deltas"][endpoint][contrast] for entry in per_seed.values()]
            aggregate[endpoint][contrast] = {
                "values": values,
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "all_positive": all(value > 0 for value in values),
            }

    clustering = aggregate["hpa_clustering_ge10"]
    both_endpoints_beat_baseline = all(
        aggregate[endpoint]["true_vs_baseline"]["all_positive"] for endpoint in ENDPOINTS
    )
    both_endpoints_beat_shuffled = all(
        aggregate[endpoint]["true_vs_shuffled"]["all_positive"] for endpoint in ENDPOINTS
    )
    clustering_reaches_magnitude = all(
        value >= clustering_min_delta
        for contrast in ("true_vs_baseline", "true_vs_shuffled")
        for value in clustering[contrast]["values"]
    )
    gates = {
        "all_seeds_both_endpoints_beat_baseline": both_endpoints_beat_baseline,
        "all_seeds_both_endpoints_beat_shuffled": both_endpoints_beat_shuffled,
        "all_seeds_clustering_reaches_min_delta": clustering_reaches_magnitude,
    }
    gates["replication_pass"] = all(gates.values())
    return {
        "readout": "nlb2_cls",
        "training_seeds": seeds,
        "clustering_min_delta": clustering_min_delta,
        "inference_scope": (
            "Replication across continue-training seeds with deterministic frozen evaluation; "
            "does not establish cross-species or cross-acquisition generalization."
        ),
        "seeds": per_seed,
        "aggregate": aggregate,
        "gates": gates,
    }


def main() -> None:
    args = parse_args()
    payload = build_payload(args.seed_root, args.clustering_min_delta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["gates"], sort_keys=True))


if __name__ == "__main__":
    main()
