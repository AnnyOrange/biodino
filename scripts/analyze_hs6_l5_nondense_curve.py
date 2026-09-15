#!/usr/bin/env python3
"""Audit non-dense peaks on the HS6-L 5TB 49-checkpoint trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from watch_hs6_l_6m_full_peak import collect_metrics, rank_curve


PREFIXES = ("classification:", "regression:", "retrieval:")


def parse_args() -> argparse.Namespace:
    repo = Path("/mnt/huawei_deepcad/dinov3")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=repo / "outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "outputs/00_reports/hs6_l5_nondense_peak_audit_20260911",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoints = [487 + 488 * index for index in range(49)]
    points: dict[int, dict[str, float]] = {}
    for index, checkpoint in enumerate(checkpoints, start=1):
        point_root = args.eval_root / f"point_{checkpoint}"
        if not point_root.is_dir():
            raise RuntimeError(f"missing checkpoint directory: {point_root}")
        metrics = collect_metrics(point_root, checkpoint)
        points[checkpoint] = {key: value for key, value in metrics.items() if key.startswith(PREFIXES)}
        print(f"[collect] {index:02d}/49 ck{checkpoint}: {len(points[checkpoint])} non-dense metrics", flush=True)

    metric_keys = sorted(set.union(*(set(values) for values in points.values())))
    common = set.intersection(*(set(values) for values in points.values()))
    if len(common) != 33:
        raise RuntimeError(f"expected 33 common non-dense metrics, found {len(common)}")

    common_points = {
        checkpoint: {key: value for key, value in values.items() if key in common}
        for checkpoint, values in points.items()
    }
    curve = rank_curve(common_points)
    aggregate_peak = max(curve, key=lambda row: (float(row["mean_rank_score"]), -int(row["checkpoint"])))

    metric_rows = []
    for key in metric_keys:
        available = [(checkpoint, points[checkpoint][key]) for checkpoint in checkpoints if key in points[checkpoint]]
        peak_value = max(value for _, value in available)
        tied_peaks = [checkpoint for checkpoint, value in available if value == peak_value]
        endpoint_value = points[checkpoints[-1]][key]
        tail = np.asarray([points[checkpoint][key] for checkpoint in checkpoints[-5:]], dtype=np.float64)
        delta = peak_value - endpoint_value
        metric_rows.append(
            {
                "family": key.split(":", 1)[0],
                "metric_key": key,
                "coverage": len(available),
                "peak_checkpoint": tied_peaks[-1],
                "peak_checkpoints": ";".join(str(checkpoint) for checkpoint in tied_peaks),
                "peak_value": peak_value,
                "endpoint_checkpoint": checkpoints[-1],
                "endpoint_value": endpoint_value,
                "absolute_drop": delta,
                "relative_drop_pct": 100.0 * delta / abs(peak_value) if peak_value else None,
                "last5_mean": float(tail.mean()),
                "last5_max": float(tail.max()),
                "last5_below_peak": int(np.sum(tail < peak_value)),
            }
        )
    metric_rows.sort(key=lambda row: (-float(row["absolute_drop"]), row["metric_key"]))

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_metric_peaks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with (args.output / "common33_rank_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)

    endpoint = curve[-1]
    report = {
        "status": "VALID_OBSERVATIONAL_AUDIT",
        "scope": "HS6-L 5TB; classification, regression, and retrieval only",
        "expected_checkpoints": 49,
        "point_directories": len(points),
        "common_metrics": len(common),
        "metrics_with_at_least_48_points": len(metric_rows),
        "metrics_missing_one_checkpoint": [row["metric_key"] for row in metric_rows if row["coverage"] == 48],
        "aggregate_peak": aggregate_peak,
        "aggregate_endpoint": endpoint,
        "aggregate_peak_minus_endpoint": float(aggregate_peak["mean_rank_score"])
        - float(endpoint["mean_rank_score"]),
        "inference_warning": (
            "Per-dataset peaks are retrospective, checkpoint-selection-biased point estimates without "
            "paired predictions or seed replication; they identify targets but do not establish significance."
        ),
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    large = [
        row
        for row in metric_rows
        if float(row["absolute_drop"]) >= 0.02
        and float(row["relative_drop_pct"] or 0.0) >= 2.0
        and int(row["last5_below_peak"]) == 5
    ]
    lines = [
        "# HS6-L 5TB non-dense checkpoint peak audit",
        "",
        "Status: `VALID_OBSERVATIONAL_AUDIT`. This is a target-finding analysis, not a significance claim.",
        "",
        f"All 49 checkpoint directories are present. The aggregate uses the {len(common)} classification/regression/retrieval metrics available at every checkpoint. Two additional metrics have 48/49 coverage because ck3415 is missing them.",
        "",
        f"The common-metric mean-rank peak is ck{aggregate_peak['checkpoint']} ({aggregate_peak['mean_rank_score']:.6f}); ck{endpoint['checkpoint']} is {endpoint['mean_rank_score']:.6f}. The peak-to-endpoint gap is {report['aggregate_peak_minus_endpoint']:+.6f}.",
        "",
        "## Large persistent endpoint gaps",
        "",
        "Post-hoc screen: absolute gap >= 0.02, relative gap >= 2%, and all final five checkpoints below the selected peak.",
        "",
        "| Dataset metric | Coverage | Peak ck | Peak | Endpoint | Gap | Relative gap | Last-5 mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in large:
        lines.append(
            f"| {row['metric_key']} | {row['coverage']}/49 | {row['peak_checkpoint']} | "
            f"{row['peak_value']:.6f} | {row['endpoint_value']:.6f} | {row['absolute_drop']:+.6f} | "
            f"{row['relative_drop_pct']:.2f}% | {row['last5_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Controls against a universal-collapse claim",
            "",
            "HPA-subcellular R@1, CRC-VAL-HE-7K R@1, BBBC005 R2, CHAMMI Allen task 1, and CHAMMI Allen task 2 peak at the endpoint. The degradation is therefore task-selective, not a uniform representation collapse.",
            "",
            "Per-dataset peaks are retrospective and selection-biased. Confirm any claimed mechanism with precommitted checkpoints, paired examples where available, and a matched SSL intervention run.",
        ]
    )
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
