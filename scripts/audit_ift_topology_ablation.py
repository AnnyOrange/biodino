#!/usr/bin/env python3
"""Audit matched sample/patch topology component runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARMS = ("baseline", "sample", "patch", "joint")
FINITE_METRICS = (
    "total_loss",
    "backbone_grad_norm",
    "intervention_factorized_topology_loss",
    "ift_raw_total_loss",
    "ift_sample_topology_loss",
    "ift_sample_topology_abs_error",
    "ift_patch_topology_loss",
    "ift_patch_topology_abs_error",
    "ift_context_head_grad_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--experiment-tag", default="ift_topology_v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "last": values[-1],
    }


def expected_topology_total(row: dict, arm: str) -> float:
    sample = float(row["ift_sample_topology_loss"])
    patch = float(row["ift_patch_topology_loss"])
    return {
        "baseline": 0.0,
        "sample": sample,
        "patch": patch,
        "joint": sample + patch,
    }[arm]


def main() -> None:
    args = parse_args()
    rows_by_arm: dict[str, list[dict]] = {}
    for arm in ARMS:
        run = f"{args.experiment_tag}_{arm}_seed{args.seed}_u{args.updates}"
        rows = load_rows(args.run_root / run / "raw_loss_metrics.jsonl")
        if len(rows) != args.updates:
            raise ValueError(f"{arm} has {len(rows)} updates, expected {args.updates}")
        rows_by_arm[arm] = rows

    reference = [row.get("batch_sample_key_digest") for row in rows_by_arm["baseline"]]
    stream_matches = {
        arm: [row.get("batch_sample_key_digest") for row in rows] == reference
        for arm, rows in rows_by_arm.items()
    }
    nonfinite = {
        arm: [
            {"update": index, "metric": metric, "value": row.get(metric)}
            for index, row in enumerate(rows)
            for metric in FINITE_METRICS
            if metric not in row or not math.isfinite(float(row[metric]))
        ]
        for arm, rows in rows_by_arm.items()
    }
    component_identity_error = {
        arm: max(
            abs(float(row["ift_raw_total_loss"]) - expected_topology_total(row, arm))
            for row in rows
        )
        for arm, rows in rows_by_arm.items()
    }
    baseline_aux_max = max(
        abs(float(row["intervention_factorized_topology_loss"]))
        for row in rows_by_arm["baseline"]
    )
    activity = {
        arm: {
            metric: summary([float(row[metric]) for row in rows])
            for metric in FINITE_METRICS
        }
        for arm, rows in rows_by_arm.items()
    }
    payload = {
        "updates": args.updates,
        "stream_matches_baseline": stream_matches,
        "all_streams_match": all(stream_matches.values()),
        "nonfinite_metrics": nonfinite,
        "all_metrics_finite": not any(nonfinite.values()),
        "baseline_auxiliary_max_abs": baseline_aux_max,
        "baseline_auxiliary_is_zero": baseline_aux_max == 0.0,
        "component_identity_max_abs_error": component_identity_error,
        "component_identities_match": all(
            error < 1.0e-6 for error in component_identity_error.values()
        ),
        "activity": activity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not (
        payload["all_streams_match"]
        and payload["all_metrics_finite"]
        and payload["baseline_auxiliary_is_zero"]
        and payload["component_identities_match"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
