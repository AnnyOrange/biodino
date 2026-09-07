#!/usr/bin/env python3
"""Idempotently validate and consolidate the MoNuSeg HV auxiliary A/B run."""
from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

ROOT = Path("/mnt/huawei_deepcad/dinov3")
OUT = ROOT / "outputs" / "instance_seg_tuning"


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def valid_result(task_dir: Path, method: str) -> tuple[bool, dict, str]:
    status_path = task_dir / "status.json"
    result_path = task_dir / "results.json"
    log_path = OUT / "distributed_logs" / RUN_ID / "3090-qi" / f"{task_dir.name}.log"
    try:
        status = json.loads(status_path.read_text())
        result = json.loads(result_path.read_text())
        log = log_path.read_text(errors="ignore")
        if status.get("exit_code") != 0 or not re.search(r"Epoch\s+30(?:\s*/\s*30|:|\b)", log) or "Results saved" not in log:
            return False, result, "status/normal-end validation failed"
        for metric in ("AJI", "Dice", "bPQ"):
            if not math.isfinite(float(result["val"][metric])):
                return False, result, f"non-finite {metric}"
        if method == "hv_auxiliary" and float(result["_meta"].get("hv_aux_gradient_l1", 0.0)) <= 0:
            return False, result, "HV auxiliary gradient missing/nonzero check failed"
        non_benign_markers = ("No space left on device", "CUDA out of memory", "RuntimeError:", "ValueError:", "FileNotFoundError", "Killed")
        if any(marker in log for marker in non_benign_markers):
            return False, result, "unhandled traceback"
        return True, result, "valid"
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return False, {}, str(exc)


def append_unique(path: Path, header: list[str], rows: list[dict], key_fields: tuple[str, ...]) -> int:
    existing = []
    if path.exists():
        with path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
    keys = {tuple(row.get(field, "") for field in key_fields) for row in existing}
    additions = [row for row in rows if tuple(row.get(field, "") for field in key_fields) not in keys]
    if additions:
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
            if path.stat().st_size == 0:
                writer.writeheader()
            writer.writerows(additions)
    return len(additions)


def main() -> int:
    global RUN_ID
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    RUN_ID = args.run_id
    run_dir = OUT / "distributed_runs" / RUN_ID
    tasks = [(f"monuseg_control_seed{s}", "control", s) for s in range(3)] + [(f"monuseg_hv_auxiliary_seed{s}", "hv_auxiliary", s) for s in range(3)]
    records = {}
    for task_id, method, seed in tasks:
        task_dir = run_dir / "3090-qi" / task_id
        ok, result, reason = valid_result(task_dir, method)
        status_path = task_dir / "status.json"
        status = json.loads(status_path.read_text())
        if ok:
            status.update({"state": "completed", "epoch": "30/30", "exit_code": 0, "error": "", "heartbeat": stamp()})
            status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
            records[(method, seed)] = result
        else:
            status.update({"state": "failed", "error": reason, "heartbeat": stamp()})
            status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    if len(records) != 6:
        raise SystemExit(f"only {len(records)}/6 tasks valid; no shared consolidation written")

    central_rows = ["task_id\tmethod\tseed\thost\tgpu_index\tgpu_uuid\tstate\tpid\tepoch\theartbeat\texit_code\toutput"]
    for task_id, method, seed in tasks:
        data = json.loads((run_dir / "3090-qi" / task_id / "status.json").read_text())
        central_rows.append("\t".join(str(data.get(key, "")) for key in ("task_id", "method", "seed", "host", "gpu_index", "gpu_uuid", "state", "pid", "epoch", "heartbeat", "exit_code", "result_json")))
    (run_dir / "central_status.tsv").write_text("\n".join(central_rows) + "\n")

    control = [records[("control", s)] for s in range(3)]
    hv = [records[("hv_auxiliary", s)] for s in range(3)]
    gains = [float(hv[s]["val"]["AJI"]) - float(control[s]["val"]["AJI"]) for s in range(3)]
    mean_gain = sum(gains) / 3
    std_gain = (sum((x - mean_gain) ** 2 for x in gains) / 2) ** 0.5
    mean_control = sum(float(x["val"]["AJI"]) for x in control) / 3
    mean_hv = sum(float(x["val"]["AJI"]) for x in hv) / 3
    std = lambda values: (sum((x - sum(values) / len(values)) ** 2 for x in values) / (len(values) - 1)) ** 0.5
    conclusion = "promising" if mean_gain >= 0.005 and all(x > 0 for x in gains) and all(float(hv[s]["val"][m]) >= float(control[s]["val"][m]) - 0.01 for s in range(3) for m in ("Dice", "bPQ")) else "inconclusive"

    summary = run_dir / "hvaux_summary.md"
    lines = [f"# MoNuSeg HV-distance Auxiliary A/B ({RUN_ID})", "", "ViT-L/16 strategy screening; Frozen backbone; current decoder; not DINOv3-7B.", "", "| Seed | Control AJI | HV auxiliary AJI | Paired gain | Control Dice | HV Dice | Control bPQ | HV bPQ |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in range(3):
        lines.append(f"| {s} | {control[s]['val']['AJI']:.6f} | {hv[s]['val']['AJI']:.6f} | {gains[s]:+.6f} | {control[s]['val']['Dice']:.6f} | {hv[s]['val']['Dice']:.6f} | {control[s]['val']['bPQ']:.6f} | {hv[s]['val']['bPQ']:.6f} |")
    lines += ["", f"AJI control mean +/- sample std: {mean_control:.6f} +/- {std([float(x['val']['AJI']) for x in control]):.6f}", f"AJI HV auxiliary mean +/- sample std: {mean_hv:.6f} +/- {std([float(x['val']['AJI']) for x in hv]):.6f}", f"Paired AJI gain mean +/- sample std: {mean_gain:+.6f} +/- {std_gain:.6f}; positive seeds: {sum(x > 0 for x in gains)}/3", f"Conclusion: {conclusion}; adopted=0", "", "HV loss: existing primary HV MSE + MSGE retained, plus independent auxiliary branch supervised by the same horizontal/vertical center-distance GT maps, auxiliary weight=1.0. Primary HV branch remains the sole postprocessing input.", "", "Errno 16 NFS multiprocessing finalizer warnings were excluded as benign because every task reached Results saved and trainer exit code 0."]
    summary.write_text("\n".join(lines) + "\n")

    structural = OUT / "structural_optimization_results.csv"
    with structural.open(newline="") as handle:
        header = list(csv.DictReader(handle).fieldnames or [])
    rows = []
    for method in ("control", "hv_auxiliary"):
        for s in range(3):
            result = records[(method, s)]
            meta = result["_meta"]
            rows.append({"timestamp": stamp(), "stage": "stage4_monuseg_hv_auxiliary_multiseed", "run_prefix": RUN_ID, "dataset": "monuseg", "method": "Current decoder control" if method == "control" else "Current decoder + HV-distance auxiliary head", "seed": str(s), "metric_name": "AJI", "protocol": "ViT-L/16; Frozen; current decoder; 30ep; CE+Dice; independent HV auxiliary MSE+MSGE" if method != "control" else "ViT-L/16; Frozen; current decoder; 30ep; CE+Dice; paired control", "baseline": f"{float(result['val']['AJI']):.6f}", "optimized_result": f"{float(result['val']['AJI']):.6f}", "independent_gain": "+0.000000" if method == "control" else f"{float(result['val']['AJI']) - float(control[s]['val']['AJI']):+.6f}", "current_best_before_stage": f"{float(control[s]['val']['AJI']):.6f}", "marginal_gain_over_current_best": "+0.000000" if method == "control" else f"{float(result['val']['AJI']) - float(control[s]['val']['AJI']):+.6f}", "trainable_parameters": str(meta.get("trainable_params", "")), "peak_memory_gib": f"{float(meta.get('peak_cuda_memory_gib', 0)):.6f}", "training_time_seconds": f"{float(meta.get('training_seconds', 0)):.3f}", "inference_time_seconds": f"{float(meta.get('inference_seconds', 0)):.3f}", "adopted": "0", "statistical_conclusion": "paired control" if method == "control" else conclusion, "telemetry_note": f"Dice={float(result['val']['Dice']):.6f}; bPQ={float(result['val']['bPQ']):.6f}; hv_aux_gradient_l1={meta.get('hv_aux_gradient_l1', 0)}", "results_json": str(run_dir / "3090-qi" / f"monuseg_{method}_seed{s}" / "results.json"), "log_path": str(OUT / "distributed_logs" / RUN_ID / "3090-qi" / f"monuseg_{method}_seed{s}.log")})
    append_unique(structural, header, rows, ("stage", "run_prefix", "dataset", "method", "seed"))

    ledger = OUT / "method_gain_ledger.csv"
    with ledger.open(newline="") as handle:
        ledger_header = list(csv.DictReader(handle).fieldnames or [])
    ledger_rows = []
    for s in range(3):
        for method in ("control", "hv_auxiliary"):
            r = records[(method, s)]
            ledger_rows.append({"timestamp": stamp(), "dataset": "monuseg", "optimization_method": f"stage4_hv_auxiliary_{method}", "metric": "AJI", "baseline_metric": f"{float(control[s]['val']['AJI']):.9f}", "optimized_metric": f"{float(r['val']['AJI']):.9f}", "absolute_gain": f"{(float(r['val']['AJI']) - float(control[s]['val']['AJI'])):.9f}", "config": json.dumps({"run_id": RUN_ID, "seed": s, "backbone": "Frozen ViT-L/16", "decoder": "current", "hv_auxiliary": method == "hv_auxiliary", "hv_aux_weight": 1.0}, sort_keys=True), "train_infer_cost": f"train={float(r['_meta'].get('training_seconds', 0)):.3f}s infer={float(r['_meta'].get('inference_seconds', 0)):.3f}s peak={float(r['_meta'].get('peak_cuda_memory_gib', 0)):.3f}GiB", "adopted": "0", "gain_type": "paired_hv_auxiliary_ab", "metric_definition": "AJI; ViT-L/16 strategy screening; not DINOv3-7B"})
    append_unique(ledger, ledger_header, ledger_rows, ("dataset", "optimization_method", "metric", "config"))
    print(json.dumps({"run_id": RUN_ID, "paired_gains": gains, "paired_mean": mean_gain, "paired_sample_std": std_gain, "positive_seeds": sum(x > 0 for x in gains), "conclusion": conclusion}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
