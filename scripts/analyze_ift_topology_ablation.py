#!/usr/bin/env python3
"""Aggregate generic, HPA, or cross-species topology ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ARMS = ("baseline", "sample", "patch", "joint")
READOUTS = ("nlb2_avg", "nlb2_cls")
GENERIC_DATASETS = ("lc25000", "nct-crc-he-100", "nct-crc-he-1k", "crc-val-he-7k")
GENERIC_METRICS = ("map_at_5", "recall_at_1", "nmi", "ari")
HPA_ENDPOINTS = {
    "hpa_retrieval": ("retrieval", "global", "same-gene-query-gallery", "map_at_5"),
    "hpa_clustering_ge10": ("clustering", "location", "single-location-ge10", "nmi"),
}
CROSS_METRICS = ("map_at_5", "recall_at_1", "mrr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("generic", "hpa", "cross"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_generic(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("dataset") in GENERIC_DATASETS and not row.get("error")
        ]
    if {row["dataset"] for row in rows} != set(GENERIC_DATASETS):
        raise ValueError(f"Incomplete generic panel in {path}")
    return {
        "macro": {
            metric: sum(float(row[metric]) for row in rows) / len(rows)
            for metric in GENERIC_METRICS
        },
        "datasets": {
            row["dataset"]: {metric: float(row[metric]) for metric in GENERIC_METRICS}
            for row in rows
        },
    }


def load_hpa(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = {}
    for endpoint, (task, aggregation, protocol, metric) in HPA_ENDPOINTS.items():
        matches = [
            row
            for row in rows
            if row.get("dataset") == "hpa-subcellular"
            and row.get("task") == task
            and row.get("aggregation") == aggregation
            and protocol in row.get("protocol", "")
            and not row.get("error")
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {endpoint} row in {path}, found {len(matches)}")
        values[endpoint] = float(matches[0][metric])
    return values


def analyze_generic(root: Path) -> dict:
    results = {
        readout: {
            arm: load_generic(root / arm / readout / "summary.csv") for arm in ARMS
        }
        for readout in READOUTS
    }
    deltas = {
        readout: {
            arm: {
                metric: values[arm]["macro"][metric] - values["baseline"]["macro"][metric]
                for metric in GENERIC_METRICS
            }
            for arm in ARMS[1:]
        }
        for readout, values in results.items()
    }
    avg_map = deltas["nlb2_avg"]["joint"]["map_at_5"]
    cls_nmi = deltas["nlb2_cls"]["joint"]["nmi"]
    return {
        "mode": "generic",
        "results": results,
        "deltas_vs_baseline": deltas,
        "gates": {
            "joint_retrieval_plus_0p3pp": avg_map >= 0.003,
            "joint_clustering_plus_1pp": cls_nmi >= 0.01,
            "joint_beats_singles_primary": all(
                results[readout]["joint"]["macro"][metric]
                > max(results[readout][arm]["macro"][metric] for arm in ("sample", "patch"))
                for readout, metric in (("nlb2_avg", "map_at_5"), ("nlb2_cls", "nmi"))
            ),
        },
    }


def analyze_hpa(root: Path) -> dict:
    results = {
        readout: {
            arm: load_hpa(root / arm / readout / "summary.csv") for arm in ARMS
        }
        for readout in READOUTS
    }
    deltas = {
        readout: {
            arm: {
                endpoint: values[arm][endpoint] - values["baseline"][endpoint]
                for endpoint in HPA_ENDPOINTS
            }
            for arm in ARMS[1:]
        }
        for readout, values in results.items()
    }
    return {
        "mode": "hpa",
        "results": results,
        "deltas_vs_baseline": deltas,
        "gates": {
            arm: {
                "location_plus_0p5pp_both": all(
                    deltas[readout][arm]["hpa_clustering_ge10"] >= 0.005
                    for readout in READOUTS
                ),
                "retrieval_regression_within_0p1pp": all(
                    deltas[readout][arm]["hpa_retrieval"] >= -0.001
                    for readout in READOUTS
                ),
            }
            for arm in ARMS[1:]
        },
    }


def analyze_cross(root: Path) -> dict:
    results = {
        readout: {
            arm: json.loads(
                (root / arm / readout / "result.json").read_text(encoding="utf-8")
            )["bidirectional_macro"]
            for arm in ARMS
        }
        for readout in READOUTS
    }
    deltas = {
        readout: {
            arm: {
                metric: values[arm][metric] - values["baseline"][metric]
                for metric in CROSS_METRICS
            }
            for arm in ARMS[1:]
        }
        for readout, values in results.items()
    }
    return {
        "mode": "cross",
        "results": results,
        "deltas_vs_baseline": deltas,
        "gates": {
            arm: {
                "map_positive_both_readouts": all(
                    deltas[readout][arm]["map_at_5"] > 0 for readout in READOUTS
                )
            }
            for arm in ARMS[1:]
        },
    }


def main() -> None:
    args = parse_args()
    payload = {
        "generic": analyze_generic,
        "hpa": analyze_hpa,
        "cross": analyze_cross,
    }[args.mode](args.input_root)
    if args.mode == "generic":
        gates = payload["gates"]
        gates["advance_primary"] = all(gates.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
