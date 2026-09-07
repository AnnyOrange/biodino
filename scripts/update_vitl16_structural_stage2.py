#!/usr/bin/env python3
"""Append and summarize the MoNuSeg ViT-L/16 Stage 2 decoder screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "instance_seg_tuning"
ADAPTATION = OUT / "adaptation_results.csv"
STRUCTURAL = OUT / "structural_optimization_results.csv"
LEDGER = OUT / "method_gain_ledger.csv"
SUMMARY = OUT / "structural_optimization_summary.md"

SAME_SEED_BASELINE = 0.5272155732919562
THREE_SEED_MEAN = 0.524406
METRIC = "AJI"
METHODS = {
    "current": "Stage2 current decoder",
    "fpn": "Stage2 FPN decoder",
    "unet": "Stage2 U-Net decoder",
    "multi_layer_fpn": "Stage2 multi-layer projection + FPN",
}
STRUCTURAL_FIELDS = [
    "timestamp", "stage", "run_prefix", "dataset", "method", "seed", "metric_name",
    "protocol", "baseline", "optimized_result", "independent_gain",
    "current_best_before_stage", "marginal_gain_over_current_best", "trainable_parameters",
    "peak_memory_gib", "training_time_seconds", "inference_time_seconds", "adopted",
    "statistical_conclusion", "telemetry_note", "results_json", "log_path",
]


def read_csv(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_rows(path: Path, fields, rows) -> int:
    rows = list(rows)
    if not rows:
        return 0
    existing = read_csv(path)
    old_fields = list(existing[0].keys()) if existing else []
    merged_fields = old_fields + [field for field in fields if field not in old_fields]
    if existing and merged_fields != old_fields:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=merged_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fields or list(fields), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def num(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value, digits=6):
    x = num(value)
    return "" if not math.isfinite(x) else f"{x:.{digits}f}"


def signed(value, digits=6):
    x = num(value)
    return "" if not math.isfinite(x) else f"{x:+.{digits}f}"


def meta(row):
    path = Path(row.get("results_json", ""))
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("_meta", {})
    except (OSError, json.JSONDecodeError):
        return {}


def selected_rows(run_prefix: str):
    selected = {}
    for row in read_csv(ADAPTATION):
        path = row.get("results_json", "")
        if run_prefix not in path or row.get("dataset") != "monuseg" or row.get("mode") != "frozen":
            continue
        if row.get("exit_code") != "0" or row.get("decoder_variant") not in METHODS:
            continue
        selected[row["decoder_variant"]] = row
    return selected


def structural_rows(run_prefix: str, selected):
    rows = []
    screen_complete = len(selected) == len(METHODS)
    for variant, row in selected.items():
        value = num(row.get("primary_value"))
        m = meta(row)
        if screen_complete and variant == "current":
            conclusion = "control; within-screen best, but below same-seed and 3-seed Frozen references; no new decoder adopted"
        elif screen_complete:
            conclusion = "rejected: no gain versus same-seed or 3-seed Frozen references; no new decoder adopted"
        else:
            conclusion = "single-seed screening; adoption pending four-way comparison"
        rows.append({
            "timestamp": row.get("timestamp") or datetime.now().astimezone().isoformat(),
            "stage": "stage2_multiscale_decoder", "run_prefix": run_prefix,
            "dataset": "monuseg", "method": METHODS[variant], "seed": 0,
            "metric_name": METRIC,
            "protocol": f"ViT-L/16 strategy screening; Frozen; layers 4,11,17,23; CE+Dice; 30ep; seed0; decoder_variant={variant}",
            "baseline": fmt(SAME_SEED_BASELINE), "optimized_result": fmt(value),
            "independent_gain": signed(value - SAME_SEED_BASELINE),
            "current_best_before_stage": fmt(THREE_SEED_MEAN),
            "marginal_gain_over_current_best": signed(value - THREE_SEED_MEAN),
            "trainable_parameters": row.get("trainable_params", m.get("trainable_params", "")),
            "peak_memory_gib": fmt(row.get("peak_cuda_gib") or m.get("peak_cuda_memory_gib", "")),
            "training_time_seconds": fmt(row.get("training_seconds") or m.get("training_seconds", ""), 3),
            "inference_time_seconds": fmt(row.get("inference_seconds") or m.get("inference_seconds", ""), 3),
            "adopted": 0,
            "statistical_conclusion": conclusion,
            "telemetry_note": f"AJI={fmt(value)}; Dice={fmt(row.get('Dice'))}; bPQ={fmt(row.get('bPQ'))}; same-seed Frozen baseline={SAME_SEED_BASELINE:.6f}; Frozen 3-seed mean={THREE_SEED_MEAN:.6f}",
            "results_json": row.get("results_json", ""), "log_path": row.get("log_path", ""),
        })
    return rows


def append_ledger(run_prefix: str, selected):
    existing = read_csv(LEDGER)
    if not existing:
        return 0
    fields = list(existing[0].keys())
    seen = {(r.get("dataset"), r.get("optimization_method")) for r in existing}
    pending = []
    for variant, row in selected.items():
        method = f"structural_stage2_monuseg_{variant}_seed0"
        if ("monuseg", method) in seen:
            continue
        value = num(row.get("primary_value")); m = meta(row)
        config = {
            "stage": "stage2_multiscale_decoder", "run_prefix": run_prefix,
            "decoder_variant": variant, "seed": 0, "backbone": "Frozen ViT-L/16",
            "layers": [4, 11, 17, 23], "loss": "CE+Dice", "epochs": 30,
            "same_seed_frozen_baseline": SAME_SEED_BASELINE,
            "frozen_3seed_mean": THREE_SEED_MEAN,
            "reference_difference": value - THREE_SEED_MEAN,
        }
        pending.append({
            "timestamp": row.get("timestamp", ""), "dataset": "monuseg", "optimization_method": method,
            "metric": METRIC, "baseline_metric": SAME_SEED_BASELINE, "optimized_metric": value,
            "absolute_gain": value - SAME_SEED_BASELINE, "config": json.dumps(config, sort_keys=True),
            "train_infer_cost": f"Stage2 30ep; train_seconds={m.get('training_seconds','')}; inference_seconds={m.get('inference_seconds','')}; peak_cuda_gib={m.get('peak_cuda_memory_gib','')}",
            "adopted": 0, "gain_type": "single_seed_decoder_screening",
            "metric_definition": "Aggregated Jaccard Index (AJI); ViT-L/16 strategy screening; not DINOv3-7B",
        })
        seen.add(("monuseg", method))
    return append_rows(LEDGER, fields, pending)


def update_summary(run_prefix: str, selected, queue_log: str):
    lines = [
        "# ViT-L/16 Structural Optimization Summary", "",
        "All rows below are ViT-L/16 strategy screening results; none are DINOv3-7B results.", "",
        "## Stage 1 status", "",
        "- MoNuSeg Frozen/Adapter seeds 0,1,2 completed; Adapter was inconclusive and is not adopted.",
        "- LIVECell Frozen seed 1 was interrupted after a `/tmp/pymp-*` multiprocessing deadlock at epoch 27; no result was recorded.",
        "- Remaining LIVECell/Cellpose Stage 1 tasks are preserved but paused.", "",
        "## Stage 2 - MoNuSeg multi-scale decoder screen", "",
        f"- Run prefix: `{run_prefix}`", f"- Queue log: `{queue_log}`",
        "- Frozen ViT-L/16; layers 4, 11, 17, 23; CE+Dice; 30 epochs; seed 0; fixed split/augmentation/crop/stride/postprocess/evaluator.",
        f"- Same-screen Frozen seed-0 baseline: AJI {SAME_SEED_BASELINE:.6f}; Frozen 3-seed reference: AJI {THREE_SEED_MEAN:.6f} +/- 0.004685.", "",
        "| decoder | AJI | Dice | bPQ | gain vs same seed-0 | difference vs Frozen 3-seed mean | params | peak GiB | train s | infer s | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant in METHODS:
        row = selected.get(variant)
        if not row:
            lines.append(f"| {METHODS[variant]} | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | queued |")
            continue
        m = meta(row); value = num(row.get("primary_value"))
        if len(selected) == len(METHODS):
            status = "control; below references" if variant == "current" else "rejected; no gain"
        else:
            status = "single-seed screen"
        lines.append(
            f"| {METHODS[variant]} | {fmt(value)} | {fmt(row.get('Dice'))} | {fmt(row.get('bPQ'))} | {signed(value-SAME_SEED_BASELINE)} | {signed(value-THREE_SEED_MEAN)} | {m.get('trainable_params','')} | {fmt(m.get('peak_cuda_memory_gib'))} | {fmt(m.get('training_seconds'),3)} | {fmt(m.get('inference_seconds'),3)} | {status} |"
        )
    lines += ["", f"Completed decoder variants: {len(selected)}/4."]
    if len(selected) == len(METHODS):
        lines += [
            "The current decoder is the within-screen control winner, but its 30-epoch result remains below both Frozen references.",
            "All three new decoder variants are rejected for this screen; none is adopted or advanced to 50 epochs, and no automatic search expansion is performed.",
        ]
    else:
        lines += ["No decoder is adopted until the four-way comparison is complete; no automatic expansion is performed."]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-prefix", required=True); parser.add_argument("--queue-log", required=True)
    args = parser.parse_args(); selected = selected_rows(args.run_prefix)
    existing = read_csv(STRUCTURAL)
    seen = {(r.get("stage"), r.get("dataset"), r.get("method"), str(r.get("seed"))) for r in existing}
    pending = [r for r in structural_rows(args.run_prefix, selected) if (r["stage"], r["dataset"], r["method"], str(r["seed"])) not in seen]
    structural_added = append_rows(STRUCTURAL, STRUCTURAL_FIELDS, pending)
    ledger_added = append_ledger(args.run_prefix, selected)
    update_summary(args.run_prefix, selected, args.queue_log)
    print(f"stage2_sources={len(selected)}/4 structural_added={structural_added} ledger_added={ledger_added}")


if __name__ == "__main__":
    main()
