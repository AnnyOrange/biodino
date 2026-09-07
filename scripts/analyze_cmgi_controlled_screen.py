#!/usr/bin/env python3
"""Audit data alignment and auxiliary behavior for a controlled CMGI screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COMMON_METRICS = (
    "dino_local_crops_loss",
    "dino_global_crops_loss",
    "ibot_loss",
    "koleo_loss",
)
CMGI_METRICS = (
    "cmgi_loss",
    "cmgi_graph_loss",
    "cmgi_predictor_loss",
    "cmgi_teacher_innovation",
    "cmgi_edge_gate_fraction",
    "cmgi_teacher_student_edge_error",
    "cmgi_predictor_vs_subset_edge_delta",
    "cmgi_predictor_mode_edge",
)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _values(row: dict, names: tuple[str, ...]) -> dict[str, float | None]:
    return {name: row.get(name) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline_rows = _read_rows(args.root / args.baseline / "raw_loss_metrics.jsonl")
    baseline_digests = [row.get("batch_sample_key_digest") for row in baseline_rows]
    if not baseline_digests or any(digest is None for digest in baseline_digests):
        raise RuntimeError("Raw metrics lack batch_sample_key_digest; rerun with the controlled-stream audit code.")

    report = {
        "root": str(args.root),
        "baseline": args.baseline,
        "baseline_updates": len(baseline_rows),
        "arms": {},
        "all_arms_fully_data_aligned": True,
    }
    for arm in args.arms:
        rows = _read_rows(args.root / arm / "raw_loss_metrics.jsonl")
        digests = [row.get("batch_sample_key_digest") for row in rows]
        matched = sum(left == right for left, right in zip(baseline_digests, digests))
        fully_aligned = len(rows) == len(baseline_rows) and matched == len(baseline_rows)
        report["all_arms_fully_data_aligned"] &= fully_aligned
        report["arms"][arm] = {
            "updates": len(rows),
            "sample_key_matches": matched,
            "fully_data_aligned": fully_aligned,
            "initial_common_metrics": _values(rows[0], COMMON_METRICS),
            "initial_common_minus_baseline": {
                metric: float(rows[0][metric] - baseline_rows[0][metric]) for metric in COMMON_METRICS
            },
            "final_common_metrics": _values(rows[-1], COMMON_METRICS),
            "final_cmgi_metrics": _values(rows[-1], CMGI_METRICS),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_arms_fully_data_aligned"]:
        raise SystemExit("Controlled-screen audit failed: at least one arm saw a different sample sequence.")


if __name__ == "__main__":
    main()
