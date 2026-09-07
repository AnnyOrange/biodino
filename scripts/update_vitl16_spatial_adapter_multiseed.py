#!/usr/bin/env python3
"""Append paired seed 1/2 rows and summarize the three-seed CNN spatial adapter A/B."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "instance_seg_tuning"
ADAPTATION = OUT / "adaptation_results.csv"
STRUCTURAL = OUT / "structural_optimization_results.csv"
LEDGER = OUT / "method_gain_ledger.csv"
SUMMARY = OUT / "structural_optimization_summary.md"
SEED0_CONTROL = OUT / "adaptation_runs/vitl16_structural_stage2_monuseg_20260818_140200_current/monuseg/frozen/results.json"
SEED0_ADAPTER = OUT / "adaptation_runs/vitl16_structural_stage3_monuseg_spatial_20260818_151949/monuseg/frozen/results.json"
STAGE = "stage3_cnn_spatial_adapter_multiseed"
CONTROL_METHOD = "Stage3 current decoder paired control"
ADAPTER_METHOD = "Current decoder + CNN spatial adapter (additive)"
FIELDS = [
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


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def append_rows(path: Path, rows, fields) -> int:
    rows = list(rows)
    if not rows:
        return 0
    existing = read_csv(path)
    old_fields = list(existing[0].keys()) if existing else []
    merged = old_fields + [field for field in fields if field not in old_fields]
    if existing and merged != old_fields:
        write_csv(path, existing, merged)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged or list(fields), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def load_result(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        val, meta = data["val"], data["_meta"]
        if not all(math.isfinite(float(val[key])) for key in ("AJI", "Dice", "bPQ")):
            return None
        return {"path": str(path), "val": val, "meta": meta}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def fmt(value, digits=6):
    return f"{float(value):.{digits}f}"


def signed(value, digits=6):
    return f"{float(value):+.{digits}f}"


def new_sources(run_prefix):
    sources = {}
    for row in read_csv(ADAPTATION):
        if run_prefix not in row.get("results_json", "") or row.get("dataset") != "monuseg" or row.get("mode") != "frozen":
            continue
        if row.get("exit_code") != "0" or row.get("decoder_variant") != "current":
            continue
        try:
            seed = int(row.get("seed", -1))
        except ValueError:
            continue
        if seed not in {1, 2}:
            continue
        method = "adapter" if row.get("spatial_adapter") in {"1", "True", "true"} else "control"
        result = load_result(row.get("results_json", ""))
        if result is not None:
            result["row"] = row
            sources[(seed, method)] = result
    return sources


def all_sources(run_prefix):
    sources = new_sources(run_prefix)
    seed0_control = load_result(SEED0_CONTROL)
    seed0_adapter = load_result(SEED0_ADAPTER)
    if seed0_control:
        sources[(0, "control")] = seed0_control
    if seed0_adapter:
        sources[(0, "adapter")] = seed0_adapter
    return sources


def classify(gains):
    mean_gain = statistics.mean(gains)
    positive = sum(gain > 0 for gain in gains)
    if mean_gain >= 0.005 and positive == 3:
        return "adopt: paired mean gain >= 0.005 AJI and 3/3 seeds are positive", 1
    if mean_gain >= 0.005 and positive >= 2:
        return "promising: paired mean gain >= 0.005 AJI and at least 2/3 seeds are positive", 0
    if mean_gain < 0 and positive <= 1 and abs(mean_gain) >= 0.005:
        return "reject: paired multi-seed result shows a clear negative effect", 0
    return "inconclusive: paired mean gain < 0.005 AJI or direction is unstable", 0


def append_individual_rows(run_prefix, sources):
    existing = read_csv(STRUCTURAL)
    seen = {(row.get("stage"), row.get("method"), str(row.get("seed"))) for row in existing}
    pending = []
    for seed in (1, 2):
        control = sources.get((seed, "control"))
        if control is None:
            continue
        control_value = float(control["val"]["AJI"])
        for method in ("control", "adapter"):
            source = sources.get((seed, method))
            if source is None:
                continue
            label = CONTROL_METHOD if method == "control" else ADAPTER_METHOD
            key = (STAGE, label, str(seed))
            if key in seen:
                continue
            val, meta, row = source["val"], source["meta"], source["row"]
            value = float(val["AJI"])
            gain = 0.0 if method == "control" else value - control_value
            pending.append({
                "timestamp": row.get("timestamp", ""), "stage": STAGE, "run_prefix": run_prefix,
                "dataset": "monuseg", "method": label, "seed": seed, "metric_name": "AJI",
                "protocol": f"ViT-L/16 paired spatial-adapter validation; Frozen; current decoder; 30ep; seed{seed}; layers 4,11,17,23; CE+Dice; fixed Stage3 protocol; spatial_adapter={method == 'adapter'}",
                "baseline": fmt(control_value), "optimized_result": fmt(value), "independent_gain": signed(gain),
                "current_best_before_stage": fmt(control_value), "marginal_gain_over_current_best": signed(gain),
                "trainable_parameters": meta.get("trainable_params", ""), "peak_memory_gib": fmt(meta.get("peak_cuda_memory_gib")),
                "training_time_seconds": fmt(meta.get("training_seconds"), 3), "inference_time_seconds": fmt(meta.get("inference_seconds"), 3),
                "adopted": 0, "statistical_conclusion": "paired control" if method == "control" else "pending complete three-seed paired aggregation",
                "telemetry_note": f"AJI={fmt(value)}; Dice={fmt(val['Dice'])}; bPQ={fmt(val['bPQ'])}; paired control={fmt(control_value)}",
                "results_json": source["path"], "log_path": row.get("log_path", ""),
            })
            seen.add(key)
    return append_rows(STRUCTURAL, pending, FIELDS)


def append_individual_ledger(run_prefix, sources):
    ledger = read_csv(LEDGER)
    if not ledger:
        return 0
    seen = {(row.get("dataset"), row.get("optimization_method")) for row in ledger}
    pending = []
    for seed in (1, 2):
        control = sources.get((seed, "control"))
        if control is None:
            continue
        control_value = float(control["val"]["AJI"])
        for method in ("control", "adapter"):
            source = sources.get((seed, method))
            if source is None:
                continue
            name = f"cnn_spatial_adapter_{method}_seed{seed}"
            if ("monuseg", name) in seen:
                continue
            val, meta, row = source["val"], source["meta"], source["row"]
            value = float(val["AJI"])
            gain = 0.0 if method == "control" else value - control_value
            pending.append({
                "timestamp": row.get("timestamp", ""), "dataset": "monuseg", "optimization_method": name,
                "metric": "AJI", "baseline_metric": control_value, "optimized_metric": value, "absolute_gain": gain,
                "config": json.dumps({"stage": STAGE, "run_prefix": run_prefix, "method": method, "seed": seed, "backbone": "Frozen ViT-L/16", "decoder": "current", "epochs": 30, "spatial_adapter": method == "adapter"}, sort_keys=True),
                "train_infer_cost": f"30ep; train_seconds={meta.get('training_seconds','')}; inference_seconds={meta.get('inference_seconds','')}; peak_cuda_gib={meta.get('peak_cuda_memory_gib','')}",
                "adopted": 0, "gain_type": "paired_seed_validation", "metric_definition": "Aggregated Jaccard Index (AJI); ViT-L/16 strategy screening; not DINOv3-7B",
            })
            seen.add(("monuseg", name))
    return append_rows(LEDGER, pending, list(ledger[0].keys()))


def aggregate(run_prefix, sources):
    if any((seed, method) not in sources for seed in (0, 1, 2) for method in ("control", "adapter")):
        return 0, 0, None
    metrics = {}
    for method in ("control", "adapter"):
        metrics[method] = {
            metric: [float(sources[(seed, method)]["val"][metric]) for seed in (0, 1, 2)]
            for metric in ("AJI", "Dice", "bPQ")
        }
    gains = [metrics["adapter"]["AJI"][seed] - metrics["control"]["AJI"][seed] for seed in (0, 1, 2)]
    decision, adopted = classify(gains)
    control_mean = statistics.mean(metrics["control"]["AJI"])
    adapter_mean = statistics.mean(metrics["adapter"]["AJI"])
    gain_mean = statistics.mean(gains)

    existing = read_csv(STRUCTURAL)
    seen = {(row.get("stage"), row.get("method"), str(row.get("seed"))) for row in existing}
    aggregate_rows = []
    for method, label in (("control", "Stage3 current decoder control mean seed0-2"), ("adapter", "CNN spatial adapter mean seed0-2")):
        key = (STAGE, label, "mean_seed0_2")
        if key in seen:
            continue
        mean = statistics.mean(metrics[method]["AJI"])
        std = statistics.stdev(metrics[method]["AJI"])
        aggregate_rows.append({
            "timestamp": sources[(2, "adapter")]["row"].get("timestamp", ""), "stage": STAGE, "run_prefix": run_prefix,
            "dataset": "monuseg", "method": label, "seed": "mean_seed0_2", "metric_name": "AJI",
            "protocol": "ViT-L/16 paired mean and sample std across seeds 0,1,2; Frozen; current decoder; 30ep; fixed Stage3 protocol",
            "baseline": fmt(control_mean), "optimized_result": fmt(mean),
            "independent_gain": signed(0.0 if method == "control" else gain_mean),
            "current_best_before_stage": fmt(control_mean), "marginal_gain_over_current_best": signed(0.0 if method == "control" else gain_mean),
            "trainable_parameters": sources[(0, method)]["meta"].get("trainable_params", ""),
            "peak_memory_gib": fmt(statistics.mean(float(sources[(seed, method)]["meta"]["peak_cuda_memory_gib"]) for seed in (0, 1, 2))),
            "training_time_seconds": fmt(statistics.mean(float(sources[(seed, method)]["meta"]["training_seconds"]) for seed in (0, 1, 2)), 3),
            "inference_time_seconds": fmt(statistics.mean(float(sources[(seed, method)]["meta"]["inference_seconds"]) for seed in (0, 1, 2)), 3),
            "adopted": adopted if method == "adapter" else 0,
            "statistical_conclusion": "paired control aggregate" if method == "control" else decision,
            "telemetry_note": f"AJI mean={mean:.6f}; sample std={std:.6f}; values={metrics[method]['AJI']}; paired gains={gains}; paired gain mean={gain_mean:.6f}; paired gain sample std={statistics.stdev(gains):.6f}",
            "results_json": ";".join(sources[(seed, method)]["path"] for seed in (0, 1, 2)),
            "log_path": ";".join(sources[(seed, method)].get("row", {}).get("log_path", "historical seed0") for seed in (0, 1, 2)),
        })
    structural_added = append_rows(STRUCTURAL, aggregate_rows, FIELDS)

    ledger = read_csv(LEDGER)
    ledger_added = 0
    aggregate_name = "cnn_spatial_adapter_paired_mean_seed0_2"
    if ledger and not any(row.get("optimization_method") == aggregate_name for row in ledger):
        ledger_added = append_rows(LEDGER, [{
            "timestamp": sources[(2, "adapter")]["row"].get("timestamp", ""), "dataset": "monuseg",
            "optimization_method": aggregate_name, "metric": "AJI", "baseline_metric": control_mean,
            "optimized_metric": adapter_mean, "absolute_gain": gain_mean,
            "config": json.dumps({"stage": STAGE, "run_prefix": run_prefix, "seeds": [0, 1, 2], "paired_gains": gains, "paired_gain_sample_std": statistics.stdev(gains), "positive_seeds": sum(g > 0 for g in gains), "decision": decision}, sort_keys=True),
            "train_infer_cost": "three-seed paired aggregate; per-seed telemetry in structural_optimization_results.csv",
            "adopted": adopted, "gain_type": "paired_multiseed_mean", "metric_definition": "Aggregated Jaccard Index (AJI); ViT-L/16 strategy screening; not DINOv3-7B",
        }], list(ledger[0].keys()))

    rows = read_csv(STRUCTURAL)
    changed = False
    for row in rows:
        if row.get("stage") == STAGE and row.get("method") == ADAPTER_METHOD and row.get("seed") in {"1", "2"}:
            row["statistical_conclusion"] = decision
            row["adopted"] = str(adopted)
            changed = True
    if changed:
        write_csv(STRUCTURAL, rows, list(rows[0].keys()))
    return structural_added, ledger_added, {"metrics": metrics, "gains": gains, "decision": decision, "adopted": adopted}


def update_summary(run_prefix, queue_log, sources, aggregate_data):
    text = SUMMARY.read_text(encoding="utf-8") if SUMMARY.exists() else "# ViT-L/16 Structural Optimization Summary\n"
    marker = "## Stage 3 - CNN spatial adapter A/B"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    lines = [marker, "", "All rows in this section are ViT-L/16 strategy screening results, not DINOv3-7B results.", "",
             f"- Run prefix for paired seed 1/2 completion: `{run_prefix}`", f"- Queue log: `{queue_log}`",
             "- Frozen ViT-L/16; original current decoder; 30 epochs; CE+Dice; paired by seed; the adapter flag is the only intended within-seed difference.", "",
             "| seed | current AJI | CNN adapter AJI | paired gain | current params | adapter params | current peak GiB | adapter peak GiB | current train/infer s | adapter train/infer s |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in (0, 1, 2):
        control, adapter = sources.get((seed, "control")), sources.get((seed, "adapter"))
        if control is None or adapter is None:
            lines.append(f"| {seed} | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |")
            continue
        cval, aval = float(control["val"]["AJI"]), float(adapter["val"]["AJI"])
        cm, am = control["meta"], adapter["meta"]
        lines.append(f"| {seed} | {cval:.6f} | {aval:.6f} | {aval-cval:+.6f} | {cm.get('trainable_params','')} | {am.get('trainable_params','')} | {float(cm.get('peak_cuda_memory_gib')):.6f} | {float(am.get('peak_cuda_memory_gib')):.6f} | {float(cm.get('training_seconds')):.3f}/{float(cm.get('inference_seconds')):.3f} | {float(am.get('training_seconds')):.3f}/{float(am.get('inference_seconds')):.3f} |")
    if aggregate_data is None:
        lines += ["", f"Completed paired sources: {len(sources)}/6. Aggregate decision is pending complete EXIT 0 results."]
    else:
        metrics, gains = aggregate_data["metrics"], aggregate_data["gains"]
        lines += ["", "| metric | current mean +/- sample std | adapter mean +/- sample std |", "|---|---:|---:|"]
        for metric in ("AJI", "Dice", "bPQ"):
            c, a = metrics["control"][metric], metrics["adapter"][metric]
            lines.append(f"| {metric} | {statistics.mean(c):.6f} +/- {statistics.stdev(c):.6f} | {statistics.mean(a):.6f} +/- {statistics.stdev(a):.6f} |")
        lines += ["", f"Paired gains: {[round(g, 6) for g in gains]}; mean +/- sample std = {statistics.mean(gains):.6f} +/- {statistics.stdev(gains):.6f} AJI.",
                  f"Positive seeds: {sum(g > 0 for g in gains)}/3.", f"Decision: {aggregate_data['decision']}.",
                  "With only three seeds, this is an effect-size and direction-consistency assessment; no statistical-significance claim is made."]
    SUMMARY.write_text(text.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--queue-log", required=True)
    args = parser.parse_args()
    sources = all_sources(args.run_prefix)
    structural_individual = append_individual_rows(args.run_prefix, sources)
    ledger_individual = append_individual_ledger(args.run_prefix, sources)
    structural_aggregate, ledger_aggregate, aggregate_data = aggregate(args.run_prefix, sources)
    update_summary(args.run_prefix, args.queue_log, sources, aggregate_data)
    print(f"paired_sources={len(sources)}/6 structural_individual_added={structural_individual} ledger_individual_added={ledger_individual} structural_aggregate_added={structural_aggregate} ledger_aggregate_added={ledger_aggregate}")


if __name__ == "__main__":
    main()
