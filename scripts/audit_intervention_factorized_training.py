#!/usr/bin/env python3
"""Audit matched intervention-factorization baseline/true/shuffled runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARMS = ("baseline", "true", "shuffled")
METRICS = (
    "total_loss",
    "backbone_grad_norm",
    "intervention_factorized_topology_loss",
    "ift_raw_total_loss",
    "ift_invariance_loss",
    "ift_positive_cosine",
    "ift_context_loss",
    "ift_context_true_loss",
    "ift_context_shuffled_raw_loss",
    "ift_context_gradient_scale",
    "ift_context_accuracy",
    "ift_context_optimized_target_accuracy",
    "ift_shuffled_target_agreement",
    "ift_decorrelation_loss",
    "ift_sample_topology_loss",
    "ift_sample_topology_abs_error",
    "ift_patch_topology_loss",
    "ift_patch_topology_abs_error",
    "ift_patch_visible_edge_fraction",
    "ift_optimization_active",
    "ift_shuffled_context",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--experiment-tag", default="ift_v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(rows: list[dict], metric: str) -> dict[str, float] | None:
    values = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]
    if not values:
        return None
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "last": values[-1],
    }


def main() -> None:
    args = parse_args()
    rows_by_arm = {}
    for arm in ARMS:
        run_name = f"{args.experiment_tag}_{arm}_seed{args.seed}_u{args.updates}"
        path = args.run_root / run_name / "raw_loss_metrics.jsonl"
        rows = load_rows(path)
        if len(rows) != args.updates:
            raise ValueError(f"{arm} has {len(rows)} updates, expected {args.updates}: {path}")
        rows_by_arm[arm] = rows

    reference_digests = [row.get("batch_sample_key_digest") for row in rows_by_arm["baseline"]]
    stream_matches = {
        arm: [row.get("batch_sample_key_digest") for row in rows] == reference_digests
        for arm, rows in rows_by_arm.items()
    }
    first_mismatch = {
        arm: next(
            (
                index
                for index, (reference, candidate) in enumerate(
                    zip(reference_digests, [row.get("batch_sample_key_digest") for row in rows])
                )
                if reference != candidate
            ),
            None,
        )
        for arm, rows in rows_by_arm.items()
    }
    nonfinite = {
        arm: [
            {"update": index, "metric": metric, "value": row[metric]}
            for index, row in enumerate(rows)
            for metric in METRICS
            if metric in row and not math.isfinite(float(row[metric]))
        ]
        for arm, rows in rows_by_arm.items()
    }
    activity = {
        arm: {
            metric: summary
            for metric in METRICS
            if (summary := summarize(rows, metric)) is not None
        }
        for arm, rows in rows_by_arm.items()
    }
    shuffled_match_error = max(
        abs(float(row["ift_context_loss"]) - float(row["ift_context_true_loss"]))
        for row in rows_by_arm["shuffled"]
    )
    baseline_aux_max = max(
        abs(float(row["intervention_factorized_topology_loss"]))
        for row in rows_by_arm["baseline"]
    )
    payload = {
        "updates": args.updates,
        "stream_matches_baseline": stream_matches,
        "first_stream_mismatch_update": first_mismatch,
        "all_streams_match": all(stream_matches.values()),
        "nonfinite_metrics": nonfinite,
        "all_metrics_finite": not any(nonfinite.values()),
        "shuffled_context_value_match_max_abs_error": shuffled_match_error,
        "shuffled_context_values_match": shuffled_match_error < 1.0e-6,
        "baseline_auxiliary_max_abs": baseline_aux_max,
        "baseline_auxiliary_is_zero": baseline_aux_max == 0.0,
        "activity": activity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not (
        payload["all_streams_match"]
        and payload["all_metrics_finite"]
        and payload["shuffled_context_values_match"]
        and payload["baseline_auxiliary_is_zero"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
