#!/usr/bin/env python3
"""Append and summarize the isolated ViT-L/16 CNN spatial adapter A/B."""

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

CURRENT_CONTROL = 0.49660917416606104
METHOD = "Current decoder + CNN spatial adapter (additive)"
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
    merged = old_fields + [f for f in fields if f not in old_fields]
    if existing and merged != old_fields:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=merged, extrasaction="ignore")
            writer.writeheader(); writer.writerows(existing)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged or list(fields), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def as_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value, digits=6):
    x = as_float(value)
    return "" if not math.isfinite(x) else f"{x:.{digits}f}"


def signed(value, digits=6):
    x = as_float(value)
    return "" if not math.isfinite(x) else f"{x:+.{digits}f}"


def find_row(run_prefix: str):
    candidates = [
        row for row in read_csv(ADAPTATION)
        if row.get("dataset") == "monuseg"
        and row.get("mode") == "frozen"
        and run_prefix in row.get("results_json", "")
        and row.get("decoder_variant") == "current"
        and row.get("spatial_adapter") in {"1", "True", "true"}
        and row.get("exit_code") == "0"
    ]
    return candidates[-1] if candidates else None


def meta(row):
    data = json.loads(Path(row["results_json"]).read_text(encoding="utf-8"))
    return data.get("val", {}), data.get("_meta", {})


def conclusion(gain: float) -> str:
    if gain < -0.01:
        return "stop: AJI degradation exceeds 0.01; do not expand CNN spatial adapter"
    if gain < 0.005:
        return "inconclusive: gain is below 0.005 AJI; do not expand or combine"
    return "promising single-seed gain >= 0.005 AJI; seed 1/2 required before adoption"


def update_summary(run_prefix: str, row, val, m, gain: float, queue_log: str):
    existing = SUMMARY.read_text(encoding="utf-8") if SUMMARY.exists() else "# ViT-L/16 Structural Optimization Summary\n"
    marker = "## Stage 3 - CNN spatial adapter A/B"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    text = existing.rstrip() + "\n\n" + "\n".join([
        marker, "",
        "- Isolated A/B on MoNuSeg; Frozen ViT-L/16; original current decoder; seed 0; 30 epochs; CE+Dice; all Stage 2 current-control settings held fixed.",
        f"- Run prefix: `{run_prefix}`; queue log: `{queue_log}`",
        "- CNN adapter: lightweight image pyramid at 1/4, 1/8 and 1/16; additive fusion only; no cross-attention and no decoder/loss/post-processing changes.", "",
        "| method | AJI | Dice | bPQ | gain vs same-screen current | params | peak GiB | train s | infer s | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| Current decoder control | {CURRENT_CONTROL:.6f} | 0.778735 | 0.405569 | +0.000000 | 10239812 | 2.700213 | 588.273 | 39.678 | reused Stage 2 result |",
        f"| {METHOD} | {fmt(val.get('AJI'))} | {fmt(val.get('Dice'))} | {fmt(val.get('bPQ'))} | {signed(gain)} | {m.get('trainable_params','')} | {fmt(m.get('peak_cuda_memory_gib'))} | {fmt(m.get('training_seconds'),3)} | {fmt(m.get('inference_seconds'),3)} | {conclusion(gain)} |",
        "",
        f"The current-control comparison is AJI {CURRENT_CONTROL:.6f}; historical 50-epoch Frozen values are not used as the adapter comparator.",
        f"Statistical conclusion: {conclusion(gain)}.",
    ]) + "\n"
    SUMMARY.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--queue-log", required=True)
    args = parser.parse_args()
    row = find_row(args.run_prefix)
    if row is None:
        raise SystemExit(f"completed CNN spatial adapter row not found for {args.run_prefix}")
    val, m = meta(row)
    value = as_float(val.get("AJI"))
    gain = value - CURRENT_CONTROL
    existing = read_csv(STRUCTURAL)
    key = ("stage3_cnn_spatial_adapter", "monuseg", METHOD, "0")
    seen = {(r.get("stage"), r.get("dataset"), r.get("method"), str(r.get("seed"))) for r in existing}
    structural_added = 0
    if key not in seen:
        structural_added = append_rows(STRUCTURAL, STRUCTURAL_FIELDS, [{
            "timestamp": row.get("timestamp") or datetime.now().astimezone().isoformat(),
            "stage": key[0], "run_prefix": args.run_prefix, "dataset": "monuseg", "method": METHOD, "seed": 0,
            "metric_name": "AJI",
            "protocol": "ViT-L/16 strategy screening; Frozen; current decoder; CNN spatial adapter additive; layers 4,11,17,23; CE+Dice; 30ep; seed0; fixed Stage 2 current-control protocol",
            "baseline": fmt(CURRENT_CONTROL), "optimized_result": fmt(value),
            "independent_gain": signed(gain), "current_best_before_stage": fmt(CURRENT_CONTROL),
            "marginal_gain_over_current_best": signed(gain), "trainable_parameters": m.get("trainable_params", ""),
            "peak_memory_gib": fmt(m.get("peak_cuda_memory_gib")), "training_time_seconds": fmt(m.get("training_seconds"), 3),
            "inference_time_seconds": fmt(m.get("inference_seconds"), 3), "adopted": 0,
            "statistical_conclusion": conclusion(gain),
            "telemetry_note": f"AJI={fmt(value)}; Dice={fmt(val.get('Dice'))}; bPQ={fmt(val.get('bPQ'))}; additive 1/4,1/8,1/16 CNN pyramid; current control={CURRENT_CONTROL:.6f}",
            "results_json": row.get("results_json", ""), "log_path": row.get("log_path", ""),
        }])
    ledger = read_csv(LEDGER)
    ledger_key = ("monuseg", "cnn_spatial_adapter_monuseg_seed0")
    ledger_added = 0
    if ledger and not any((r.get("dataset"), r.get("optimization_method")) == ledger_key for r in ledger):
        ledger_added = append_rows(LEDGER, list(ledger[0].keys()), [{
            "timestamp": row.get("timestamp", ""), "dataset": "monuseg", "optimization_method": ledger_key[1],
            "metric": "AJI", "baseline_metric": CURRENT_CONTROL, "optimized_metric": value,
            "absolute_gain": gain,
            "config": json.dumps({"stage": "stage3_cnn_spatial_adapter", "run_prefix": args.run_prefix, "backbone": "Frozen ViT-L/16", "decoder_variant": "current", "fusion": "additive", "scales": ["1/4", "1/8", "1/16"], "seed": 0, "epochs": 30, "current_control": CURRENT_CONTROL}, sort_keys=True),
            "train_infer_cost": f"30ep; train_seconds={m.get('training_seconds','')}; inference_seconds={m.get('inference_seconds','')}; peak_cuda_gib={m.get('peak_cuda_memory_gib','')}",
            "adopted": 0, "gain_type": "isolated_spatial_adapter_ab", "metric_definition": "Aggregated Jaccard Index (AJI); ViT-L/16 strategy screening; not DINOv3-7B",
        }])
    update_summary(args.run_prefix, row, val, m, gain, args.queue_log)
    print(f"spatial_adapter_structural_added={structural_added} ledger_added={ledger_added} gain={gain:+.6f} conclusion={conclusion(gain)}")


if __name__ == "__main__":
    main()
