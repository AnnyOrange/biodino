#!/usr/bin/env python3
"""Summarize acquisition-focused HPA and RxRx1 causal evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ARMS = ("baseline", "single", "consensus", "shuffled")
READOUTS = ("nlb2_avg", "nlb2_cls")
ENDPOINTS = {
    "hpa_retrieval": {
        "dataset": "hpa-subcellular",
        "task": "retrieval",
        "aggregation": "global",
        "protocol_contains": "same-gene-query-gallery",
        "metrics": ("map_at_5", "recall_at_1"),
    },
    "hpa_clustering_ge10": {
        "dataset": "hpa-subcellular",
        "task": "clustering",
        "aggregation": "location",
        "protocol_contains": "single-location-ge10",
        "metrics": ("nmi", "ari"),
    },
    "rxrx1_retrieval_global": {
        "dataset": "rxrx1-cross",
        "task": "retrieval",
        "aggregation": "global",
        "protocol_contains": "official-cross-experiment-core",
        "metrics": ("map_at_5", "recall_at_1"),
    },
    "rxrx1_retrieval_macro_cell": {
        "dataset": "rxrx1-cross",
        "task": "retrieval",
        "aggregation": "macro-cell-type",
        "protocol_contains": "official-cross-experiment-core",
        "metrics": ("map_at_5", "recall_at_1"),
    },
    "rxrx1_clustering": {
        "dataset": "rxrx1-cross",
        "task": "clustering",
        "aggregation": "global-perturbation",
        "protocol_contains": "official-cross-experiment-core",
        "metrics": ("nmi", "ari"),
    },
}
PRIMARY_ENDPOINTS = {
    "hpa_retrieval": "map_at_5",
    "hpa_clustering_ge10": "nmi",
    "rxrx1_retrieval_macro_cell": "map_at_5",
    "rxrx1_clustering": "nmi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Record absent endpoints instead of failing, for example after a dataset integrity error.",
    )
    return parser.parse_args()


def load_endpoint(
    rows: list[dict[str, str]],
    name: str,
    *,
    allow_missing: bool,
) -> dict[str, float] | None:
    spec = ENDPOINTS[name]
    matches = [
        row
        for row in rows
        if row.get("dataset") == spec["dataset"]
        and row.get("task") == spec["task"]
        and row.get("aggregation") == spec["aggregation"]
        and spec["protocol_contains"] in row.get("protocol", "")
        and not row.get("error")
    ]
    if not matches and allow_missing:
        return None
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} row, found {len(matches)}")
    row = matches[0]
    return {metric: float(row[metric]) for metric in spec["metrics"]}


def load_summary(
    path: Path,
    *,
    allow_missing: bool,
) -> dict[str, dict[str, float] | None]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        name: load_endpoint(rows, name, allow_missing=allow_missing)
        for name in ENDPOINTS
    }


def main() -> None:
    args = parse_args()
    results = {
        readout: {
            arm: load_summary(
                args.input_root / arm / readout / "summary.csv",
                allow_missing=args.allow_missing,
            )
            for arm in ARMS
        }
        for readout in READOUTS
    }
    deltas = {}
    directions = {}
    missing_primary = []
    for readout in READOUTS:
        deltas[readout] = {}
        directions[readout] = {}
        for endpoint, metric in PRIMARY_ENDPOINTS.items():
            arm_endpoints = {
                arm: results[readout][arm][endpoint]
                for arm in ("baseline", "consensus", "shuffled")
            }
            if any(value is None for value in arm_endpoints.values()):
                missing_primary.append({"readout": readout, "endpoint": endpoint})
                continue
            value = arm_endpoints["consensus"][metric]
            baseline = arm_endpoints["baseline"][metric]
            shuffled = arm_endpoints["shuffled"][metric]
            deltas[readout][endpoint] = {
                "metric": metric,
                "consensus_vs_baseline": value - baseline,
                "consensus_vs_shuffled": value - shuffled,
            }
            directions[readout][endpoint] = {
                "beats_baseline": value > baseline,
                "beats_shuffled": value > shuffled,
            }

    primary = [
        comparison
        for readout_values in directions.values()
        for comparison in readout_values.values()
    ]
    payload = {
        "results": results,
        "primary_deltas": deltas,
        "primary_directions": directions,
        "missing_primary": missing_primary,
        "direction_counts": {
            "comparisons": len(primary),
            "beats_baseline": sum(item["beats_baseline"] for item in primary),
            "beats_shuffled": sum(item["beats_shuffled"] for item in primary),
            "beats_both": sum(
                item["beats_baseline"] and item["beats_shuffled"] for item in primary
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["direction_counts"], sort_keys=True))
    for readout in READOUTS:
        print(readout)
        for endpoint, metric in PRIMARY_ENDPOINTS.items():
            values = results[readout]
            if any(values[arm][endpoint] is None for arm in ARMS):
                print(f"  {endpoint:28s} unavailable")
                continue
            print(
                f"  {endpoint:28s} {metric:8s} "
                f"baseline={values['baseline'][endpoint][metric]:.6f} "
                f"single={values['single'][endpoint][metric]:.6f} "
                f"consensus={values['consensus'][endpoint][metric]:.6f} "
                f"shuffled={values['shuffled'][endpoint][metric]:.6f}"
            )


if __name__ == "__main__":
    main()
