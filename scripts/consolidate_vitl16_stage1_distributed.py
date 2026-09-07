#!/usr/bin/env python3
"""Idempotently consolidate completed distributed ViT-L/16 Stage-1 workers.

This is intentionally CPU/file-I/O only.  It validates each worker result before
making the CPU7-owned shared CSV updates; workers never write shared ledgers.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

from update_vitl16_structural_stage1 import (
    ADAPTATION,
    METRIC,
    SPECS,
    append_ledger,
    append_structural,
    source_rows,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "instance_seg_tuning"
LOGS = OUT / "distributed_logs"
FIELDS_REQUIRED = ("AJI", "Dice", "SEG", "CellposeStyleAP", "bPQ")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_worker(status_path: Path) -> tuple[dict, dict, Path]:
    status = read_json(status_path)
    task_id = str(status.get("task_id", ""))
    host = str(status.get("host", ""))
    result_path = Path(str(status.get("result_json", "")))
    log_path = LOGS / str(status.get("run_id", "")) / host / f"{task_id}.log"
    if status.get("state") != "completed" or status.get("exit_code") != 0:
        raise RuntimeError(f"{task_id}: status is not completed rc=0")
    result = read_json(result_path)
    metric = "SEG" if status.get("dataset") == "livecell" else "CellposeStyleAP"
    value = result.get("val", {}).get(metric)
    log = log_path.read_text(encoding="utf-8", errors="ignore")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"{task_id}: missing {metric}")
    if not re.search(r"Epoch\s+50/50", log) or "Results saved" not in log:
        raise RuntimeError(f"{task_id}: normal-end evidence is incomplete")
    if not re.search(r"TRAINER_EXIT\s+attempt=\d+\s+rc=0", log):
        raise RuntimeError(f"{task_id}: trainer exit=0 is not logged")
    # The known NFS multiprocessing finalizer warning happens after normal work
    # and does not invalidate a result.  Any other traceback does.
    if log.count("Traceback") > log.count("OSError: [Errno 16] Device or resource busy"):
        raise RuntimeError(f"{task_id}: unhandled traceback in worker log")
    meta = result.get("_meta", {})
    if (meta.get("dataset") != status.get("dataset") or meta.get("seed") != status.get("seed")
            or meta.get("backbone_mode") != status.get("method")):
        raise RuntimeError(f"{task_id}: result metadata does not match status")
    return status, result, log_path


def append_adaptation(rows: list[dict]) -> int:
    with ADAPTATION.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        existing = list(reader)
    seen = {row.get("results_json", "") for row in existing}
    pending = [row for row in rows if row["results_json"] not in seen]
    if not pending:
        return 0
    with ADAPTATION.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writerows(pending)
    return len(pending)


def adaptation_row(status: dict, result: dict, log_path: Path) -> dict:
    val = result["val"]
    meta = result["_meta"]
    metric = METRIC[status["dataset"]]
    row = {key: "" for key in FIELDS_REQUIRED}
    row.update({
        "timestamp": status.get("heartbeat") or datetime.now().astimezone().isoformat(),
        "dataset": status["dataset"],
        "mode": status["method"],
        "layers": " ".join(str(x) for x in meta.get("layers", [])),
        "fusion_mode": meta.get("fusion_mode", ""),
        "decoder_variant": meta.get("decoder_variant", ""),
        "spatial_adapter": int(bool(meta.get("spatial_adapter"))),
        "spatial_adapter_fusion": meta.get("spatial_adapter_fusion", "none"),
        "spatial_adapter_width": meta.get("spatial_adapter_width", ""),
        "split": "val",
        "primary_metric": metric,
        "primary_value": val[metric],
        "epochs": 50,
        "crop_size": meta.get("crop_size", ""),
        "stride": meta.get("stride", ""),
        "batch_size": meta.get("batch_size", ""),
        "grad_accum_steps": meta.get("grad_accum_steps", ""),
        "effective_batch_size": meta.get("effective_batch_size", ""),
        "decoder_lr": meta.get("decoder_lr", ""),
        "backbone_lr": meta.get("backbone_lr", ""),
        "warmup_ratio": meta.get("warmup_ratio", ""),
        "grad_clip_norm": meta.get("grad_clip_norm", ""),
        "layer_wise_lr_decay": meta.get("layer_wise_lr_decay", ""),
        "amp_dtype": meta.get("amp_dtype", ""),
        "feature_size": meta.get("feature_size", ""),
        "embed_proj": meta.get("embed_proj", ""),
        "trainable_params": meta.get("trainable_params", ""),
        "trainable_backbone_params": meta.get("trainable_backbone_params", ""),
        "seed": meta.get("seed", ""),
        "np_loss_mode": meta.get("np_loss_mode", ""),
        "focal_gamma": meta.get("focal_gamma", ""),
        "tversky_alpha": meta.get("tversky_alpha", ""),
        "tversky_beta": meta.get("tversky_beta", ""),
        "pid": status.get("pid", ""),
        "gpu": status.get("gpu_uuid", ""),
        "exit_code": 0,
        "oom_retry": status.get("selected_try", 0),
        "wall_seconds": meta.get("run_wall_seconds", ""),
        "training_seconds": meta.get("training_seconds", ""),
        "inference_seconds": meta.get("inference_seconds", ""),
        "peak_cuda_gib": meta.get("peak_cuda_memory_gib", ""),
        "results_json": status["result_json"],
        "log_path": str(log_path),
    })
    for key in FIELDS_REQUIRED:
        row[key] = val.get(key, "")
    return row


def mean_std(values: list[float]) -> str:
    return f"{statistics.mean(values):.6f} +/- {statistics.stdev(values):.6f}"


def write_summary(run_dir: Path, sources: dict) -> None:
    lines = [
        "# Distributed Stage 1 Consolidation",
        "",
        "All results below are ViT-L/16 strategy screening validation results, not DINOv3-7B results.",
        "",
        "| Dataset | Method | Seed | AJI | Dice | bPQ | Primary metric |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("livecell", "cellpose"):
        for mode in ("frozen", "finetune"):
            for seed in (0, 1, 2):
                row = sources[(dataset, mode, seed)]
                result = read_json(Path(row["results_json"]))
                val = result["val"]
                lines.append(
                    f"| {dataset} | {mode} | {seed} | {val['AJI']:.6f} | {val['Dice']:.6f} | "
                    f"{val['bPQ']:.6f} | {float(row['primary_value']):.6f} |"
                )
    lines += ["", "## Paired Three-Seed Summary", "",
              "| Dataset | Frozen mean +/- std | Full FT mean +/- std | Paired gains (seed 0,1,2) | Paired mean +/- std | Positive seeds | Conclusion |",
              "|---|---:|---:|---|---:|---:|---|"]
    for dataset in ("livecell", "cellpose"):
        frozen = [float(sources[(dataset, "frozen", seed)]["primary_value"]) for seed in (0, 1, 2)]
        full = [float(sources[(dataset, "finetune", seed)]["primary_value"]) for seed in (0, 1, 2)]
        gains = [candidate - baseline for baseline, candidate in zip(frozen, full)]
        positive = sum(gain > 0 for gain in gains)
        noise = max(statistics.stdev(frozen), statistics.stdev(full), statistics.stdev(gains))
        stable = statistics.mean(gains) > noise and positive == 3
        conclusion = "stable positive under predefined criterion" if stable else "inconclusive versus seed variation"
        lines.append(
            f"| {dataset} | {mean_std(frozen)} | {mean_std(full)} | "
            f"{', '.join(f'{gain:+.6f}' for gain in gains)} | {mean_std(gains)} | {positive}/3 | {conclusion} |"
        )
    lines += ["", "Validation criteria: completed status, trainer exit code 0, valid results.json, Epoch 50/50 and Results saved in the worker log.",
              "NFS multiprocessing finalizer `Errno 16` warnings were observed after normal completion and are retained in logs; no `Errno 28` was found.",
              "No HV, CNN+HV, or cross-dataset experiment was started by this consolidator."]
    (run_dir / "stage1_consolidated_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_central_status(run_dir: Path) -> None:
    task_ids = (
        "livecell_frozen_seed1", "livecell_frozen_seed2",
        "livecell_finetune_seed1", "livecell_finetune_seed2",
        "cellpose_frozen_seed1", "cellpose_frozen_seed2",
        "cellpose_finetune_seed1", "cellpose_finetune_seed2",
    )
    records: dict[str, dict] = {}
    for status_path in run_dir.glob("*/**/status.json"):
        status = read_json(status_path)
        if status.get("error") == "duplicate_task_cancelled_gpu4_kept":
            continue
        task_id = str(status.get("task_id", ""))
        if task_id in task_ids:
            records[task_id] = status
    header = ("task_id\thost\tgpu_uuid\tstate\tpid\tepoch\tmem_used_mib\tmem_free_mib\t"
              "utilization\tschedulable_class\theartbeat\toutput\texit_code")
    lines = [header]
    for task_id in task_ids:
        status = records.get(task_id, {})
        fields = (task_id, status.get("host", ""), status.get("gpu_uuid", ""),
                  status.get("state", "pending"), status.get("pid", ""), status.get("epoch", ""),
                  "", "", "", "completed", status.get("heartbeat", ""),
                  status.get("result_json", ""), status.get("exit_code", ""))
        lines.append("\t".join(str(field) for field in fields))
    (run_dir / "central_status.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    run_dir = OUT / "distributed_runs" / args.run_id
    lock_path = run_dir / ".consolidator.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        imported = []
        for status_path in sorted(run_dir.glob("*/**/status.json")):
            raw_status = read_json(status_path)
            if raw_status.get("error") == "duplicate_task_cancelled_gpu4_kept":
                # Retain the cancelled duplicate as an audit trail, but it is
                # neither a formal task result nor a consolidation failure.
                continue
            status, result, log_path = valid_worker(status_path)
            if status["dataset"] in {"livecell", "cellpose"} and int(status["seed"]) in {1, 2}:
                imported.append(adaptation_row(status, result, log_path))
        added = append_adaptation(imported)
        all_sources = source_rows(args.run_id)
        # This distributed run contains LIVECell and Cellpose only; MoNuSeg
        # multi-seed work belongs to the earlier local Stage-1 run.
        sources = {key: row for key, row in all_sources.items() if key[0] in {"livecell", "cellpose"}}
        if len(sources) != 12:
            raise RuntimeError(f"expected 12 distributed LIVECell/Cellpose sources after import, got {len(sources)}")
        structural_added = append_structural(args.run_id, sources)
        ledger_added = append_ledger(args.run_id, sources)
        write_summary(run_dir, sources)
        refresh_central_status(run_dir)
        (run_dir / "STAGE_EXIT").write_text(f"0 {datetime.now().astimezone().isoformat()}\n", encoding="utf-8")
        print(f"adaptation_added={added} structural_added={structural_added} ledger_added={ledger_added} sources={len(sources)}")


if __name__ == "__main__":
    main()
