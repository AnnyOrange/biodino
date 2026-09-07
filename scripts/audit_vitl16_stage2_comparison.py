#!/usr/bin/env python3
"""Rebase the completed Stage 2 decoder rows on the same-screen current control."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "instance_seg_tuning"
STRUCTURAL = OUT / "structural_optimization_results.csv"
LEDGER = OUT / "method_gain_ledger.csv"
SUMMARY = OUT / "structural_optimization_summary.md"
RUN_PREFIX = "vitl16_structural_stage2_monuseg_20260818_140200"
CURRENT = 0.49660917416606104
HISTORICAL_SEED0 = 0.5272155732919562
HISTORICAL_MEAN = 0.524406


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def signed(x):
    return f"{float(x):+.6f}"


def main():
    rows, fields = read(STRUCTURAL)
    changed = 0
    values = {}
    for row in rows:
        if row.get("stage") != "stage2_multiscale_decoder" or row.get("run_prefix") != RUN_PREFIX:
            continue
        result = json.loads(Path(row["results_json"]).read_text(encoding="utf-8"))
        value = float(result["val"]["AJI"])
        gain = value - CURRENT
        values[row["method"]] = value
        row["baseline"] = f"{CURRENT:.6f}"
        row["current_best_before_stage"] = f"{CURRENT:.6f}"
        row["independent_gain"] = signed(gain)
        row["marginal_gain_over_current_best"] = signed(gain)
        if row["method"] == "Stage2 current decoder":
            row["statistical_conclusion"] = "same-screen current control; formal decoder gain baseline"
        else:
            row["statistical_conclusion"] = "rejected: no gain versus same-screen current decoder; no 50ep extension or dataset transfer"
        row["telemetry_note"] = (
            f"AJI={value:.6f}; Dice={row['telemetry_note'].split('Dice=',1)[1].split(';',1)[0]}; "
            f"bPQ={row['telemetry_note'].split('bPQ=',1)[1].split(';',1)[0]}; "
            f"same-screen current control={CURRENT:.6f}; historical Frozen seed0 50ep={HISTORICAL_SEED0:.6f}; "
            f"historical Frozen 3-seed mean={HISTORICAL_MEAN:.6f}; historical values are reference only"
        )
        changed += 1
    write(STRUCTURAL, fields, rows)

    ledger_rows, ledger_fields = read(LEDGER)
    ledger_changed = 0
    for row in ledger_rows:
        if row.get("dataset") != "monuseg" or not row.get("optimization_method", "").startswith("structural_stage2_monuseg_"):
            continue
        method = row["optimization_method"]
        variant = method.removeprefix("structural_stage2_monuseg_").removesuffix("_seed0")
        value = values.get({
            "current": "Stage2 current decoder", "fpn": "Stage2 FPN decoder",
            "unet": "Stage2 U-Net decoder", "multi_layer_fpn": "Stage2 multi-layer projection + FPN",
        }.get(variant, ""))
        if value is None:
            continue
        gain = value - CURRENT
        row["baseline_metric"] = f"{CURRENT:.15f}"
        row["absolute_gain"] = f"{gain:.15f}"
        config = json.loads(row["config"])
        config["same_screen_current_control"] = CURRENT
        config["historical_frozen_seed0_50ep_reference"] = HISTORICAL_SEED0
        config["historical_frozen_3seed_mean_reference"] = HISTORICAL_MEAN
        config["formal_gain_reference"] = "same_screen_current_decoder_30ep_seed0"
        config["reference_difference"] = gain
        row["config"] = json.dumps(config, sort_keys=True)
        row["gain_type"] = "same_screen_decoder_comparison"
        ledger_changed += 1

    write(LEDGER, ledger_fields, ledger_rows)

    text = SUMMARY.read_text(encoding="utf-8")
    audit = "## Stage 2 comparison audit\n\n- The four Stage 2 runs are matched on epoch=30, seed=0, Frozen ViT-L/16, layers 4/11/17/23, optimizer/LR, weight decay, warmup, gradient clipping, strong augmentation, mosaic probability, crop/stride, CE+Dice, foreground/energy thresholds, evaluator and validation cadence.\n- Formal decoder comparison baseline is the same-screen 30-epoch current decoder: AJI 0.496609. Historical 50-epoch Frozen seed 0 (0.527216) and Frozen 3-seed mean (0.524406 +/- 0.004685) remain reference values only.\n- Formal AJI deltas: current +0.000000; FPN -0.163465; U-Net -0.056171; multi-layer projection + FPN -0.009585.\n\n"
    if "## Stage 2 comparison audit" in text:
        prefix = text.split("## Stage 2 comparison audit", 1)[0].rstrip()
        remainder = text.split("## Stage 2 comparison audit", 1)[1]
        stage3 = ""
        if "## Stage 3 - CNN spatial adapter A/B" in remainder:
            stage3 = "## Stage 3 - CNN spatial adapter A/B" + remainder.split("## Stage 3 - CNN spatial adapter A/B", 1)[1]
        text = prefix + "\n\n" + audit + stage3
    elif "## Stage 3 - CNN spatial adapter A/B" in text:
        text = text.replace("## Stage 3 - CNN spatial adapter A/B", audit + "## Stage 3 - CNN spatial adapter A/B", 1)
    else:
        text = text.rstrip() + "\n\n" + audit.rstrip() + "\n"
    SUMMARY.write_text(text, encoding="utf-8")
    print(f"structural_rows_updated={changed} ledger_rows_updated={ledger_changed}")


if __name__ == "__main__":
    main()
