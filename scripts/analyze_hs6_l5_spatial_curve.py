#!/usr/bin/env python3
"""Validate and summarize the HS6-L label-free spatial-consistency curve."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURVE = ROOT / "outputs/00_reports/hs6_l5_label_free_spatial_curve_20260911"
NON_DENSE = (
    ROOT
    / "outputs/00_reports/hs6_l5_nondense_peak_audit_20260911/common33_rank_curve.csv"
)
RXRX3 = (
    ROOT
    / "outputs/02_eval_runs/rxrx3_core_formal_l5_all49_v3_single3090_20260911/"
    "checkpoint_metrics.csv"
)
LAYERS = (6, 12, 18, 24)
CANDIDATES = (15615, 17079, 20007, 21959, 23911)
STEP_PATTERN = re.compile(r"ck(\d+)\.json$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows_by_step(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {int(row["checkpoint"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate checkpoint rows in {path}")
    return result


def spatial_row(path: Path) -> dict[str, Any]:
    match = STEP_PATTERN.fullmatch(path.name)
    if match is None:
        raise RuntimeError(f"unexpected spatial result name: {path}")
    step = int(match.group(1))
    payload = json.loads(path.read_text())
    required = bool(
        payload.get("diagnostic") == "frozen_nested_local_global_spatial_signal_v1"
        and payload.get("seed") == 20260911
        and payload.get("n_images") == 128
        and payload.get("n_unique_keys") == 128
        and payload["crop_spec"]["photometric_policy"] == "bio_safe"
        and payload.get("global_grid") == 16
        and payload.get("local_grid") == 7
    )
    if not required:
        raise RuntimeError(f"protocol mismatch in {path}")
    checkpoint_step = int(Path(payload["checkpoint"]).parent.name.removeprefix("training_"))
    if checkpoint_step != step:
        raise RuntimeError(f"checkpoint/path mismatch in {path}")

    row: dict[str, Any] = {
        "checkpoint": step,
        "gate_pass": bool(payload["decision_gate"]["pass"]),
        "result_sha256": sha256(path),
    }
    for layer in LAYERS:
        values = payload["layers"][f"block_{layer}"]["all"]
        margin = values["true_minus_shifted"]
        enrichment = values["hit1_enrichment"]
        row[f"block{layer}_margin"] = float(margin["mean"])
        row[f"block{layer}_margin_ci_low"] = float(margin["ci95_low"])
        row[f"block{layer}_margin_ci_high"] = float(margin["ci95_high"])
        row[f"block{layer}_hit1_enrichment"] = float(enrichment["ratio_of_means"])
        row[f"block{layer}_hit1_enrichment_ci_low"] = float(enrichment["ci95_low"])
        row[f"block{layer}_hit1_enrichment_ci_high"] = float(enrichment["ci95_high"])
    return row


def correlation(x: list[float], y: list[float]) -> dict[str, float]:
    result = spearmanr(x, y)
    if not math.isfinite(float(result.statistic)) or not math.isfinite(float(result.pvalue)):
        raise RuntimeError("non-finite Spearman result")
    return {"spearman_rho": float(result.statistic), "two_sided_p": float(result.pvalue)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=Path, default=DEFAULT_CURVE)
    args = parser.parse_args()
    args.curve = args.curve.resolve()

    files = sorted(
        (args.curve / "checkpoints").glob("ck*.json"),
        key=lambda path: int(STEP_PATTERN.fullmatch(path.name).group(1)),  # type: ignore[union-attr]
    )
    if len(files) != 49:
        raise RuntimeError(f"expected 49 spatial results, found {len(files)}")
    rows = [spatial_row(path) for path in files]
    steps = [int(row["checkpoint"]) for row in rows]
    if len(set(steps)) != 49:
        raise RuntimeError("spatial results do not contain 49 unique checkpoints")

    non_dense = csv_rows_by_step(NON_DENSE)
    rxrx3 = csv_rows_by_step(RXRX3)
    if set(steps) != set(non_dense) or set(steps) != set(rxrx3):
        raise RuntimeError("checkpoint sets differ across spatial/non-dense/RxRx3 curves")
    for row in rows:
        step = int(row["checkpoint"])
        row["common33_mean_rank"] = float(non_dense[step]["mean_rank_score"])
        row["rxrx3_recall_at_5"] = float(rxrx3[step]["recall_at_5"])

    primary = np.asarray([float(row["block24_margin"]) for row in rows])
    smoothed: list[float | None] = [None]
    smoothed.extend(
        float(np.median(primary[index - 1 : index + 2]))
        for index in range(1, len(primary) - 1)
    )
    smoothed.append(None)
    for row, value in zip(rows, smoothed):
        row["block24_margin_centered_median3"] = value

    raw_peak_index = int(primary.argmax())
    interior_indices = list(range(1, len(rows) - 1))
    smooth_peak_index = max(
        interior_indices,
        key=lambda index: float(smoothed[index]),  # type: ignore[arg-type]
    )
    secondary = [float(row["block24_hit1_enrichment"]) for row in rows]
    secondary_peak_index = int(np.argmax(secondary))
    endpoint_index = len(rows) - 1

    correlations = {
        "checkpoint_step": correlation(primary.tolist(), [float(step) for step in steps]),
        "common33_mean_rank": correlation(
            primary.tolist(), [float(row["common33_mean_rank"]) for row in rows]
        ),
        "rxrx3_recall_at_5": correlation(
            primary.tolist(), [float(row["rxrx3_recall_at_5"]) for row in rows]
        ),
    }
    by_step = {int(row["checkpoint"]): row for row in rows}
    report = {
        "status": "VALID_COMPLETE",
        "admission": "EXPLORATORY_LABEL_FREE_CURVE",
        "checkpoint_count": len(rows),
        "gate_pass_count": sum(bool(row["gate_pass"]) for row in rows),
        "primary_score": "block24 mean true-minus-shifted cosine",
        "secondary_score": "block24 top1 spatial-hit enrichment over chance",
        "raw_primary_peak": {
            "checkpoint": steps[raw_peak_index],
            "value": float(primary[raw_peak_index]),
        },
        "centered_median3_primary_peak": {
            "checkpoint": steps[smooth_peak_index],
            "value": float(smoothed[smooth_peak_index]),
        },
        "secondary_peak": {
            "checkpoint": steps[secondary_peak_index],
            "value": secondary[secondary_peak_index],
        },
        "endpoint": {
            "checkpoint": steps[endpoint_index],
            "primary": float(primary[endpoint_index]),
            "raw_peak_minus_endpoint": float(primary[raw_peak_index] - primary[endpoint_index]),
        },
        "candidate_rows": {str(step): by_step[step] for step in CANDIDATES},
        "exploratory_correlations": correlations,
        "inputs": {
            "common33_curve": {"path": str(NON_DENSE), "sha256": sha256(NON_DENSE)},
            "rxrx3_curve": {"path": str(RXRX3), "sha256": sha256(RXRX3)},
        },
        "warning": (
            "The ck17079 gate was observed before this full-curve extension. Correlations with "
            "downstream curves are hypothesis-generating and are confounded by training time."
        ),
    }

    csv_path = args.curve / "spatial_curve.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.curve / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# HS6-L 5TB label-free spatial-consistency curve",
        "",
        "Status: `VALID_COMPLETE`; exploratory 49-checkpoint frozen-feature diagnostic.",
        "No labels or trained probes enter the spatial score.",
        "",
        (
            f"Primary raw peak: ck{steps[raw_peak_index]} "
            f"({primary[raw_peak_index]:.6f}); centered-median peak: "
            f"ck{steps[smooth_peak_index]} ({float(smoothed[smooth_peak_index]):.6f}); "
            f"endpoint ck{steps[endpoint_index]}: {primary[endpoint_index]:.6f}."
        ),
        "",
        "| Checkpoint | Block24 margin | Hit@1 enrichment | Common33 mean-rank | RxRx3 R@5 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for step in CANDIDATES:
        row = by_step[step]
        lines.append(
            f"| {step} | {row['block24_margin']:.6f} | "
            f"{row['block24_hit1_enrichment']:.3f}x | "
            f"{row['common33_mean_rank']:.6f} | {row['rxrx3_recall_at_5']:.6f} |"
        )
    lines.extend(["", "| Comparison | Spearman rho | Two-sided p |", "|---|---:|---:|"])
    for name, values in correlations.items():
        lines.append(
            f"| primary vs {name} | {values['spearman_rho']:+.4f} | "
            f"{values['two_sided_p']:.6g} |"
        )
    lines.extend(
        [
            "",
            (
                "The ck17079 gate was observed before this extension. These correlations are "
                "descriptive and training-time-confounded, not confirmatory evidence."
            ),
        ]
    )
    (args.curve / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
