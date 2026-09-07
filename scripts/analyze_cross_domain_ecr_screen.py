#!/usr/bin/env python3
"""Aggregate retrieval/clustering metrics and apply the causal screen gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ARMS = ("baseline", "single", "consensus", "shuffled")
READOUTS = ("nlb2_avg", "nlb2_cls")
DATASETS = ("lc25000", "nct-crc-he-100", "nct-crc-he-1k", "crc-val-he-7k")
METRICS = ("map_at_5", "recall_at_1", "nmi", "ari")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("dataset") in DATASETS and not row.get("error")]
    if {row["dataset"] for row in rows} != set(DATASETS):
        raise ValueError(f"Incomplete dataset panel in {path}")
    macro = {
        metric: sum(float(row[metric]) for row in rows) / len(rows)
        for metric in METRICS
    }
    return {
        "macro": macro,
        "datasets": {
            row["dataset"]: {metric: float(row[metric]) for metric in METRICS}
            for row in rows
        },
    }


def metric_delta(results: dict, readout: str, arm: str, reference: str, metric: str) -> float:
    return (
        results[readout][arm]["macro"][metric]
        - results[readout][reference]["macro"][metric]
    )


def main() -> None:
    args = parse_args()
    results = {
        readout: {
            arm: load_summary(args.input_root / arm / readout / "summary.csv")
            for arm in ARMS
        }
        for readout in READOUTS
    }
    deltas = {}
    for readout in READOUTS:
        deltas[readout] = {}
        for arm in ("single", "consensus", "shuffled"):
            deltas[readout][arm] = {
                f"{metric}_vs_baseline": metric_delta(
                    results, readout, arm, "baseline", metric
                )
                for metric in METRICS
            }
        deltas[readout]["consensus"]["map_at_5_vs_shuffled"] = metric_delta(
            results, readout, "consensus", "shuffled", "map_at_5"
        )
        deltas[readout]["consensus"]["nmi_vs_shuffled"] = metric_delta(
            results, readout, "consensus", "shuffled", "nmi"
        )

    retrieval_gate = deltas["nlb2_avg"]["consensus"]["map_at_5_vs_baseline"] >= 0.003
    clustering_gate = deltas["nlb2_cls"]["consensus"]["nmi_vs_baseline"] >= 0.01
    causal_gate = (
        deltas["nlb2_avg"]["consensus"]["map_at_5_vs_shuffled"] > 0
        and deltas["nlb2_cls"]["consensus"]["nmi_vs_shuffled"] > 0
    )
    payload = {
        "results": results,
        "deltas": deltas,
        "gates": {
            "retrieval_plus_0p3pp": retrieval_gate,
            "clustering_plus_1pp": clustering_gate,
            "true_beats_shuffled": causal_gate,
            "advance_to_multiseed": retrieval_gate and clustering_gate and causal_gate,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for readout in READOUTS:
        print(readout)
        for arm in ARMS:
            macro = results[readout][arm]["macro"]
            print(
                f"  {arm:9s} mAP@5={macro['map_at_5']:.6f} "
                f"R@1={macro['recall_at_1']:.6f} NMI={macro['nmi']:.6f} "
                f"ARI={macro['ari']:.6f}"
            )
    print(json.dumps(payload["gates"], sort_keys=True))


if __name__ == "__main__":
    main()
