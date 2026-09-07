#!/usr/bin/env python3
"""Append and summarize ViT-L/16 structural-optimization stage-1 results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "instance_seg_tuning"
ADAPTATION = OUT / "adaptation_results.csv"
STRUCTURAL = OUT / "structural_optimization_results.csv"
SUMMARY = OUT / "structural_optimization_summary.md"
LEDGER = OUT / "method_gain_ledger.csv"

SEED0_RUN = "vitl16_backbone_screen_lr1e5_decay075_20260813_183025"
SPECS = {
    "monuseg": ("frozen", "adapter"),
    "livecell": ("frozen", "finetune"),
    "cellpose": ("frozen", "finetune"),
}
SEEDS = (0, 1, 2)
METRIC = {"monuseg": "AJI", "livecell": "SEG", "cellpose": "CellposeStyleAP"}
ORIGINAL_BASELINE = {
    "monuseg": 0.4753966422532617,
    "livecell": 0.6092096417172832,
    "cellpose": 0.24295972923269685,
}
PRE_BACKBONE_BEST = {
    "monuseg": 0.5272155732919562,
    "livecell": 0.6189078066904272,
    "cellpose": 0.2769550562692866,
}
DISPLAY = {"monuseg": "MoNuSeg", "livecell": "LIVECell", "cellpose": "Cellpose"}
METHOD_DISPLAY = {"frozen": "Frozen", "adapter": "Adapter", "finetune": "Full FT"}
METRIC_DEFINITION = {
    "AJI": "Aggregated Jaccard Index",
    "SEG": "Cell Tracking Challenge SEG; not official LIVECell COCO mask AP",
    "CellposeStyleAP": "Mean TP/(TP+FP+FN) over IoU 0.50:0.05:0.95; not COCO AP",
}

STRUCTURAL_FIELDS = [
    "timestamp", "stage", "run_prefix", "dataset", "method", "seed", "metric_name",
    "protocol", "baseline", "optimized_result", "independent_gain",
    "current_best_before_stage", "marginal_gain_over_current_best", "trainable_parameters",
    "peak_memory_gib", "training_time_seconds", "inference_time_seconds", "adopted",
    "statistical_conclusion", "telemetry_note", "results_json", "log_path",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_rows(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def finite(value: object) -> bool:
    return math.isfinite(number(value))


def fmt(value: object, digits: int = 6) -> str:
    parsed = number(value)
    return "" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def signed(value: object, digits: int = 6) -> str:
    parsed = number(value)
    return "" if not math.isfinite(parsed) else f"{parsed:+.{digits}f}"


def load_meta(row: Mapping[str, str]) -> Mapping[str, object]:
    path = Path(row.get("results_json", ""))
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("_meta", {})
    except (json.JSONDecodeError, OSError):
        return {}


def source_rows(run_prefix: str) -> Dict[Tuple[str, str, int], Dict[str, str]]:
    rows = read_csv(ADAPTATION)
    selected: Dict[Tuple[str, str, int], Dict[str, str]] = {}
    for row in rows:
        dataset = row.get("dataset", "")
        mode = row.get("mode", "")
        if dataset not in SPECS or mode not in SPECS[dataset] or row.get("exit_code") != "0":
            continue
        try:
            seed = int(row.get("seed", ""))
        except ValueError:
            continue
        path = row.get("results_json", "")
        if seed == 0:
            expected_fragment = f"{SEED0_RUN}_{dataset}_{mode}"
            if expected_fragment not in path:
                continue
        elif seed in (1, 2):
            if run_prefix not in path:
                continue
        else:
            continue
        selected[(dataset, mode, seed)] = row
    return selected


def make_structural_row(
    run_prefix: str,
    source: Mapping[str, str],
    sources: Mapping[Tuple[str, str, int], Mapping[str, str]],
) -> Dict[str, object]:
    dataset = source["dataset"]
    mode = source["mode"]
    seed = int(source["seed"])
    value = number(source["primary_value"])
    if mode == "frozen":
        baseline = ORIGINAL_BASELINE[dataset]
    else:
        frozen = sources.get((dataset, "frozen", seed))
        baseline = number(frozen.get("primary_value")) if frozen else float("nan")
    independent = value - baseline if math.isfinite(baseline) else float("nan")
    marginal = value - PRE_BACKBONE_BEST[dataset]
    meta = load_meta(source)
    peak = source.get("peak_cuda_gib") or meta.get("peak_cuda_memory_gib", "")
    training = source.get("training_seconds") or meta.get("training_seconds", "")
    inference = source.get("inference_seconds") or meta.get("inference_seconds", "")
    telemetry_note = "instrumented process peak allocation and timed final validation inference"
    if seed == 0:
        telemetry_note = "historical seed 0; peak memory and separated train/inference timing were not instrumented"
    return {
        "timestamp": source.get("timestamp") or datetime.now().astimezone().isoformat(),
        "stage": "stage1_multiseed_backbone_validation",
        "run_prefix": run_prefix,
        "dataset": dataset,
        "method": METHOD_DISPLAY[mode],
        "seed": seed,
        "metric_name": METRIC[dataset],
        "protocol": "ViT-L/16 val; fixed decoder/loss/50ep/postprocess; seed is the only intended variable",
        "baseline": fmt(baseline),
        "optimized_result": fmt(value),
        "independent_gain": signed(independent),
        "current_best_before_stage": fmt(PRE_BACKBONE_BEST[dataset]),
        "marginal_gain_over_current_best": signed(marginal),
        "trainable_parameters": source.get("trainable_params", ""),
        "peak_memory_gib": fmt(peak),
        "training_time_seconds": fmt(training, 3),
        "inference_time_seconds": fmt(inference, 3),
        "adopted": 0,
        "statistical_conclusion": "historical seed-0 input to multi-seed estimate" if seed == 0 else "pending seeds 0/1/2 aggregation",
        "telemetry_note": telemetry_note,
        "results_json": source.get("results_json", ""),
        "log_path": source.get("log_path", ""),
    }


def statistics_for(
    sources: Mapping[Tuple[str, str, int], Mapping[str, str]], dataset: str
) -> Optional[Dict[str, object]]:
    frozen_mode, candidate_mode = SPECS[dataset]
    required = [(dataset, mode, seed) for mode in (frozen_mode, candidate_mode) for seed in SEEDS]
    if any(key not in sources for key in required):
        return None
    frozen_values = [number(sources[(dataset, frozen_mode, seed)]["primary_value"]) for seed in SEEDS]
    candidate_values = [number(sources[(dataset, candidate_mode, seed)]["primary_value"]) for seed in SEEDS]
    deltas = [candidate - frozen for candidate, frozen in zip(candidate_values, frozen_values)]
    frozen_mean = statistics.mean(frozen_values)
    candidate_mean = statistics.mean(candidate_values)
    frozen_std = statistics.stdev(frozen_values)
    candidate_std = statistics.stdev(candidate_values)
    delta_mean = statistics.mean(deltas)
    delta_std = statistics.stdev(deltas)
    seed_noise = max(frozen_std, candidate_std, delta_std)
    stable = delta_mean > seed_noise and all(delta > 0 for delta in deltas)
    conclusion = (
        "stable positive point difference under the predefined criterion"
        if stable
        else "inconclusive: paired mean gain does not robustly exceed observed seed variation"
    )
    return {
        "frozen_values": frozen_values,
        "candidate_values": candidate_values,
        "deltas": deltas,
        "frozen_mean": frozen_mean,
        "candidate_mean": candidate_mean,
        "frozen_std": frozen_std,
        "candidate_std": candidate_std,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "seed_noise": seed_noise,
        "stable": stable,
        "conclusion": conclusion,
        "candidate_mode": candidate_mode,
    }


def aggregate_structural_rows(
    run_prefix: str,
    sources: Mapping[Tuple[str, str, int], Mapping[str, str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in SPECS:
        stats = statistics_for(sources, dataset)
        if stats is None:
            continue
        for mode in SPECS[dataset]:
            source_set = [sources[(dataset, mode, seed)] for seed in SEEDS]
            values = [number(row["primary_value"]) for row in source_set]
            mean_value = statistics.mean(values)
            std_value = statistics.stdev(values)
            if mode == "frozen":
                baseline = ORIGINAL_BASELINE[dataset]
                independent = mean_value - baseline
                conclusion = f"reference mean={mean_value:.6f}, std={std_value:.6f}"
                adopted = 0
            else:
                baseline = stats["frozen_mean"]
                independent = stats["delta_mean"]
                conclusion = str(stats["conclusion"])
                adopted = int(bool(stats["stable"]) and mean_value > PRE_BACKBONE_BEST[dataset])
            instrumented = source_set[1:]
            peaks = [number(row.get("peak_cuda_gib")) for row in instrumented if finite(row.get("peak_cuda_gib"))]
            train_times = [number(row.get("training_seconds")) for row in instrumented if finite(row.get("training_seconds"))]
            infer_times = [number(row.get("inference_seconds")) for row in instrumented if finite(row.get("inference_seconds"))]
            rows.append(
                {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "stage": "stage1_multiseed_backbone_validation_aggregate",
                    "run_prefix": run_prefix,
                    "dataset": dataset,
                    "method": METHOD_DISPLAY[mode],
                    "seed": "mean_seed0_2",
                    "metric_name": METRIC[dataset],
                    "protocol": "ViT-L/16 val; mean and sample std across seeds 0,1,2; paired by seed",
                    "baseline": fmt(baseline),
                    "optimized_result": fmt(mean_value),
                    "independent_gain": signed(independent),
                    "current_best_before_stage": fmt(PRE_BACKBONE_BEST[dataset]),
                    "marginal_gain_over_current_best": signed(mean_value - PRE_BACKBONE_BEST[dataset]),
                    "trainable_parameters": source_set[-1].get("trainable_params", ""),
                    "peak_memory_gib": fmt(max(peaks)) if peaks else "",
                    "training_time_seconds": fmt(statistics.mean(train_times), 3) if train_times else "",
                    "inference_time_seconds": fmt(statistics.mean(infer_times), 3) if infer_times else "",
                    "adopted": adopted,
                    "statistical_conclusion": conclusion,
                    "telemetry_note": f"metric sample std={std_value:.6f}; seed 0 telemetry unavailable",
                    "results_json": ";".join(row.get("results_json", "") for row in source_set),
                    "log_path": ";".join(row.get("log_path", "") for row in source_set),
                }
            )
    return rows


def append_structural(run_prefix: str, sources: Mapping[Tuple[str, str, int], Mapping[str, str]]) -> int:
    existing = read_csv(STRUCTURAL)
    seen = {(row.get("stage"), row.get("dataset"), row.get("method"), row.get("seed")) for row in existing}
    pending: List[Dict[str, object]] = []
    for key in sorted(sources):
        dataset, mode, seed = key
        record_key = ("stage1_multiseed_backbone_validation", dataset, METHOD_DISPLAY[mode], str(seed))
        if record_key not in seen:
            pending.append(make_structural_row(run_prefix, sources[key], sources))
            seen.add(record_key)
    for row in aggregate_structural_rows(run_prefix, sources):
        record_key = (row["stage"], row["dataset"], row["method"], str(row["seed"]))
        if record_key not in seen:
            pending.append(row)
            seen.add(record_key)
    return append_rows(STRUCTURAL, STRUCTURAL_FIELDS, pending)


def append_ledger(run_prefix: str, sources: Mapping[Tuple[str, str, int], Mapping[str, str]]) -> int:
    ledger_rows = read_csv(LEDGER)
    if not ledger_rows:
        raise RuntimeError(f"Missing ledger: {LEDGER}")
    fields = list(ledger_rows[0].keys())
    seen = {(row.get("dataset"), row.get("optimization_method")) for row in ledger_rows}
    pending: List[Dict[str, object]] = []

    for (dataset, mode, seed), source in sorted(sources.items()):
        if seed == 0:
            continue
        method = f"structural_stage1_multiseed_{mode}_seed{seed}"
        if (dataset, method) in seen:
            continue
        value = number(source["primary_value"])
        if mode == "frozen":
            paired_baseline = ORIGINAL_BASELINE[dataset]
        else:
            frozen = sources.get((dataset, "frozen", seed))
            if frozen is None:
                continue
            paired_baseline = number(frozen["primary_value"])
        config = {
            "run_prefix": run_prefix,
            "stage": "stage1_multiseed_backbone_validation",
            "mode": mode,
            "seed": seed,
            "original_baseline": ORIGINAL_BASELINE[dataset],
            "paired_seed_frozen_baseline": paired_baseline,
            "independent_gain": value - paired_baseline,
            "pre_backbone_best": PRE_BACKBONE_BEST[dataset],
            "marginal_gain_over_pre_backbone_best": value - PRE_BACKBONE_BEST[dataset],
            "statistical_conclusion": "pending seeds 0/1/2 aggregation",
            "results_json": source.get("results_json", ""),
        }
        pending.append(
            {
                "timestamp": source.get("timestamp", ""),
                "dataset": dataset,
                "optimization_method": method,
                "metric": METRIC[dataset],
                "baseline_metric": ORIGINAL_BASELINE[dataset],
                "optimized_metric": value,
                "absolute_gain": value - ORIGINAL_BASELINE[dataset],
                "config": json.dumps(config, sort_keys=True),
                "train_infer_cost": (
                    f"ViT-L/16 stage1 multi-seed validation; 50ep; seed={seed}; "
                    f"wall_seconds={source.get('wall_seconds', '')}; peak_cuda_gib={source.get('peak_cuda_gib', '')}; "
                    f"training_seconds={source.get('training_seconds', '')}; inference_seconds={source.get('inference_seconds', '')}"
                ),
                "adopted": 0,
                "gain_type": "paired_seed_validation_pending",
                "metric_definition": METRIC_DEFINITION[METRIC[dataset]],
            }
        )
        seen.add((dataset, method))

    for dataset in SPECS:
        stats = statistics_for(sources, dataset)
        if stats is None:
            continue
        for mode in SPECS[dataset]:
            method = f"structural_stage1_multiseed_{mode}_mean_seed0_2"
            if (dataset, method) in seen:
                continue
            values = [number(sources[(dataset, mode, seed)]["primary_value"]) for seed in SEEDS]
            mean_value = statistics.mean(values)
            std_value = statistics.stdev(values)
            if mode == "frozen":
                baseline_value = ORIGINAL_BASELINE[dataset]
                gain_type = "multiseed_reference_mean"
                adopted = 0
                conclusion = f"reference sample std={std_value:.6f}"
            else:
                baseline_value = float(stats["frozen_mean"])
                gain_type = "paired_multiseed_mean"
                adopted = int(bool(stats["stable"]) and mean_value > PRE_BACKBONE_BEST[dataset])
                conclusion = str(stats["conclusion"])
            config = {
                "run_prefix": run_prefix,
                "stage": "stage1_multiseed_backbone_validation_aggregate",
                "mode": mode,
                "seeds": list(SEEDS),
                "values": values,
                "mean": mean_value,
                "sample_std": std_value,
                "paired_frozen_mean": stats["frozen_mean"],
                "paired_delta_mean": stats["delta_mean"] if mode != "frozen" else 0.0,
                "paired_delta_std": stats["delta_std"] if mode != "frozen" else 0.0,
                "seed_noise_threshold": stats["seed_noise"],
                "marginal_gain_over_pre_backbone_best": mean_value - PRE_BACKBONE_BEST[dataset],
                "statistical_conclusion": conclusion,
            }
            pending.append(
                {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "dataset": dataset,
                    "optimization_method": method,
                    "metric": METRIC[dataset],
                    "baseline_metric": baseline_value,
                    "optimized_metric": mean_value,
                    "absolute_gain": mean_value - baseline_value,
                    "config": json.dumps(config, sort_keys=True),
                    "train_infer_cost": "ViT-L/16 stage1 mean across seeds 0,1,2; seed 0 lacks separated telemetry",
                    "adopted": adopted,
                    "gain_type": gain_type,
                    "metric_definition": METRIC_DEFINITION[METRIC[dataset]],
                }
            )
            seen.add((dataset, method))
    return append_rows(LEDGER, fields, pending)


def update_summary(run_prefix: str, sources: Mapping[Tuple[str, str, int], Mapping[str, str]], queue_log: str) -> None:
    expected = sum(len(modes) * len(SEEDS) for modes in SPECS.values())
    complete = len(sources)
    lines = [
        "# ViT-L/16 Structural Optimization Summary",
        "",
        "Historical final summary files are not rewritten by this stage. New results are appended to the structural ledger and method-gain ledger.",
        "",
        "## Stage 1 - Existing Strategy Multi-Seed Validation",
        "",
        f"- Run prefix: `{run_prefix}`",
        f"- Queue log: `{queue_log}`",
        f"- Completed seed/method records: {complete}/{expected} (six historical seed-0 rows are included)",
        "- Protocol: ViT-L/16, fixed split/decoder/CE+Dice/50 epochs/postprocess; seeds 0, 1, 2.",
        "- Stability criterion: all paired deltas must be positive and paired mean delta must exceed the maximum of Frozen std, candidate std, and paired-delta std.",
        "",
        "| dataset | method | seed 0 | seed 1 | seed 2 | mean +/- sample std | status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for dataset, modes in SPECS.items():
        for mode in modes:
            values = []
            for seed in SEEDS:
                row = sources.get((dataset, mode, seed))
                values.append(number(row["primary_value"]) if row else float("nan"))
            if all(math.isfinite(value) for value in values):
                aggregate = f"{statistics.mean(values):.6f} +/- {statistics.stdev(values):.6f}"
                status = "Complete"
            else:
                aggregate = "Pending"
                status = "Running/queued"
            lines.append(
                f"| {DISPLAY[dataset]} | {METHOD_DISPLAY[mode]} | {fmt(values[0]) or 'Pending'} | "
                f"{fmt(values[1]) or 'Pending'} | {fmt(values[2]) or 'Pending'} | {aggregate} | {status} |"
            )

    lines.extend(
        [
            "",
            "### Paired Conclusions",
            "",
            "| dataset | candidate vs Frozen | paired deltas by seed | mean delta +/- std | marginal over pre-backbone best | conclusion |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for dataset in SPECS:
        stats = statistics_for(sources, dataset)
        candidate = METHOD_DISPLAY[SPECS[dataset][1]]
        if stats is None:
            lines.append(f"| {DISPLAY[dataset]} | {candidate} | Pending | Pending | Pending | Pending |")
            continue
        deltas = ", ".join(f"{value:+.6f}" for value in stats["deltas"])
        marginal = float(stats["candidate_mean"]) - PRE_BACKBONE_BEST[dataset]
        lines.append(
            f"| {DISPLAY[dataset]} | {candidate} | {deltas} | "
            f"{float(stats['delta_mean']):+.6f} +/- {float(stats['delta_std']):.6f} | "
            f"{marginal:+.6f} | {stats['conclusion']} |"
        )

    lines.extend(
        [
            "",
            "## Later Stages",
            "",
            "- Stage 2 multi-scale decoder: not started; gated on Stage 1 completion and report.",
            "- Stage 3 CNN spatial adapter: not started; gated on Stage 2 completion and benefit check.",
            "- Stage 4 instance geometry auxiliary head: not started; gated on earlier screening.",
        ]
    )
    tmp = SUMMARY.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(SUMMARY)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--queue-log", required=True)
    args = parser.parse_args()
    sources = source_rows(args.run_prefix)
    structural_added = append_structural(args.run_prefix, sources)
    ledger_added = append_ledger(args.run_prefix, sources)
    update_summary(args.run_prefix, sources, args.queue_log)
    print(
        f"stage1_sources={len(sources)}/18 structural_added={structural_added} "
        f"ledger_added={ledger_added}"
    )


if __name__ == "__main__":
    main()
