#!/usr/bin/env python3
"""Summarize the matched HPA-Cyclops cross-species retrieval screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ("baseline", "true", "shuffled")
READOUTS = ("nlb2_avg", "nlb2_cls")
METRICS = ("map_at_5", "recall_at_1", "mrr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {
        readout: {
            arm: json.loads(
                (args.input_root / arm / readout / "result.json").read_text(encoding="utf-8")
            )["bidirectional_macro"]
            for arm in ARMS
        }
        for readout in READOUTS
    }
    deltas = {
        readout: {
            metric: {
                "true_vs_baseline": values["true"][metric] - values["baseline"][metric],
                "true_vs_shuffled": values["true"][metric] - values["shuffled"][metric],
            }
            for metric in METRICS
        }
        for readout, values in results.items()
    }
    primary = [deltas[readout]["map_at_5"] for readout in READOUTS]
    payload = {
        "results": results,
        "deltas": deltas,
        "gates": {
            "map_improves_both_readouts": all(
                value["true_vs_baseline"] > 0 for value in primary
            ),
            "true_beats_shuffled_both_readouts": all(
                value["true_vs_shuffled"] > 0 for value in primary
            ),
            "advance": all(
                value["true_vs_baseline"] > 0 and value["true_vs_shuffled"] > 0
                for value in primary
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
