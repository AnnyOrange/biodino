#!/usr/bin/env python3
"""Summarize the acquisition-focused HPA global-bridge control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ARMS = ("baseline", "true", "shuffled")
READOUTS = ("nlb2_avg", "nlb2_cls")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = {}
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


def main() -> None:
    args = parse_args()
    results = {
        readout: {
            arm: load_summary(args.input_root / arm / readout / "summary.csv")
            for arm in ARMS
        }
        for readout in READOUTS
    }
    deltas = {
        readout: {
            endpoint: {
                "metric": ENDPOINTS[endpoint]["metric"],
                "true_vs_baseline": values["true"][endpoint] - values["baseline"][endpoint],
                "true_vs_shuffled": values["true"][endpoint] - values["shuffled"][endpoint],
            }
            for endpoint in ENDPOINTS
        }
        for readout, values in results.items()
    }
    comparisons = [
        value
        for readout_values in deltas.values()
        for value in readout_values.values()
    ]
    payload = {
        "results": results,
        "deltas": deltas,
        "direction_counts": {
            "comparisons": len(comparisons),
            "beats_baseline": sum(value["true_vs_baseline"] > 0 for value in comparisons),
            "beats_shuffled": sum(value["true_vs_shuffled"] > 0 for value in comparisons),
            "beats_both": sum(
                value["true_vs_baseline"] > 0 and value["true_vs_shuffled"] > 0
                for value in comparisons
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["direction_counts"], sort_keys=True))
    for readout in READOUTS:
        print(readout)
        for endpoint, spec in ENDPOINTS.items():
            print(
                f"  {endpoint:22s} {spec['metric']:8s} "
                + " ".join(
                    f"{arm}={results[readout][arm][endpoint]:.6f}" for arm in ARMS
                )
            )


if __name__ == "__main__":
    main()
