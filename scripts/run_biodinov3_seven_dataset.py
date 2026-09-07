#!/usr/bin/env python3
"""Atomic, GPU-gated BioDINO Frozen/no-trick seven-dataset runner."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("monuseg", "bbbc038", "cellpose", "tissuenet", "livecell", "pannuke", "conic")
LAYERS = (7, 15, 23, 31)
METRICS = {"monuseg": "AJI", "bbbc038": "CellposeStyleAP", "cellpose": "CellposeStyleAP", "tissuenet": "CellposeStyleAP", "livecell": "SEG", "pannuke": "bPQ", "conic": "mPQ"}


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def snapshot() -> list[dict[str, str]]:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in out.splitlines():
        values = [x.strip() for x in line.split(",")]
        if len(values) == 4:
            rows.append(dict(zip(("index", "uuid", "free", "util"), values)))
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
    active = {line.split(",")[0].strip() for line in apps.splitlines() if line.strip()}
    for row in rows:
        row["compute_pid"] = "" if row["uuid"] not in active else "active"
    return rows


def idle_gpus(min_free: int = 22000) -> list[dict[str, str]]:
    return [row for row in snapshot() if int(row["free"]) >= min_free and int(row["util"]) == 0 and not row["compute_pid"]]


def data_root(dataset: str) -> str:
    return "/mnt/huawei_deepcad/benchmark/segmentation/LIVECell" if dataset == "livecell" else f"/mnt/huawei_deepcad/benchmark/segmentation/{dataset}/extracted"


def trainer_cmd(args: argparse.Namespace, dataset: str, out: Path) -> list[str]:
    return [sys.executable, "-m", "dinov3.eval.bio_segmentation.instance_seg.train", "--dataset", dataset,
            "--data-root", data_root(dataset), "--checkpoint", str(args.checkpoint), "--train-config", str(args.train_config),
            "--output-dir", str(out), "--layers", *map(str, LAYERS), "--freeze-backbone", "--epochs", "20",
            "--batch-size", "1", "--grad-accum-steps", "8", "--crop-size", "256", "--stride", "192",
            "--lr", "1e-3", "--weight-decay", "1e-4", "--amp-dtype", "bf16", "--feature-size", "32",
            "--embed-proj", "384", "--fusion-mode", "bucket_concat", "--decoder-variant", "current",
            "--num-workers", "4", "--eval-every", "5", "--seed", "0", "--aug", "strong",
            "--mosaic-prob", "0.3", "--fg-thresh", "0.5", "--energy-thresh", "0.4",
            "--np-loss-mode", "ce_dice", "--skip-test-eval"]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["dataset", "metric", "no_trick", "validated_tricks", "improvement", "original_dinov3_result", "public_reference", "status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def run_one(args: argparse.Namespace, dataset: str, gpu: dict[str, str], output: Path) -> dict[str, str]:
    log = args.output_root / "logs" / f"{dataset}_no_trick_seed0.log"; log.parent.mkdir(parents=True, exist_ok=True)
    # multiprocessing/resource_sharer uses a UNIX socket under TMPDIR; keep
    # this path short enough for AF_UNIX limits on the shared worker host.
    short_tmp = Path("/tmp") / f"bio7_{dataset}"
    env = dict(os.environ); env.update({"CUDA_VISIBLE_DEVICES": gpu["index"], "PYTHONUNBUFFERED": "1", "TMPDIR": str(short_tmp), "TMP": str(short_tmp), "TEMP": str(short_tmp)})
    short_tmp.mkdir(parents=True, exist_ok=True)
    cmd = trainer_cmd(args, dataset, output)
    with log.open("a") as handle:
        handle.write(f"[{stamp()}] GPU={gpu['index']} UUID={gpu['uuid']} $ {shlex.join(cmd)}\n"); handle.flush()
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"[{stamp()}] PID={process.pid}\n"); handle.flush(); rc = process.wait()
    result = output / "results.json"
    value = ""
    if rc == 0 and result.is_file():
        try: value = str(json.loads(result.read_text())["val"][METRICS[dataset]])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError): pass
    return {"dataset": dataset, "metric": METRICS[dataset], "no_trick": value, "validated_tricks": "", "improvement": "", "original_dinov3_result": "", "public_reference": "", "status": "completed" if rc == 0 and value else f"failed_rc_{rc}", "gpu": gpu["index"], "uuid": gpu["uuid"], "pid": str(process.pid), "log": str(log)}


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--train-config", type=Path, required=True); p.add_argument("--output-root", type=Path, default=ROOT / "outputs/instance_seg_tuning/biodinov3_seven_dataset_results_run"); p.add_argument("--poll-seconds", type=int, default=20); p.add_argument("--min-free-mib", type=int, default=22000); p.add_argument("--no-wait", action="store_true")
    args = p.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True); (args.output_root / "tmp").mkdir(exist_ok=True)
    lock = args.output_root / ".task.lock"
    try: lock.mkdir()
    except FileExistsError: raise SystemExit(f"active atomic lock: {lock}")
    (lock / "owner.json").write_text(json.dumps({"pid": os.getpid(), "started": stamp(), "checkpoint": str(args.checkpoint)}, indent=2) + "\n")
    rows = [{"dataset": d, "metric": METRICS[d], "no_trick": "", "validated_tricks": "", "improvement": "", "original_dinov3_result": "", "public_reference": "", "status": "pending"} for d in DATASETS]
    status_path = ROOT / "outputs/instance_seg_tuning/biodinov3_seven_dataset_results.csv"; status_path.parent.mkdir(parents=True, exist_ok=True); write_rows(status_path, rows)
    try:
        while len(idle_gpus(args.min_free_mib)) < 1:
            if args.no_wait: raise RuntimeError("no eligible GPU for BioDINO smoke")
            print(f"[{stamp()}] waiting for eligible GPU: {snapshot()}", flush=True); time.sleep(args.poll_seconds)
        smoke_gpu = idle_gpus(args.min_free_mib)[0]
        smoke_out = args.output_root / "monuseg" / "smoke" / "smoke.json"; smoke_log = args.output_root / "logs" / "monuseg_smoke.log"; smoke_log.parent.mkdir(parents=True, exist_ok=True)
        smoke_cmd = [sys.executable, "scripts/smoke_biodinov3_instance_seg.py", "--checkpoint", str(args.checkpoint), "--train-config", str(args.train_config), "--output", str(smoke_out), "--layers", *map(str, LAYERS)]
        env = dict(os.environ); env.update({"CUDA_VISIBLE_DEVICES": smoke_gpu["index"], "PYTHONUNBUFFERED": "1"})
        with smoke_log.open("a") as handle:
            handle.write(f"[{stamp()}] GPU={smoke_gpu['index']} UUID={smoke_gpu['uuid']} $ {shlex.join(smoke_cmd)}\n"); handle.flush(); proc = subprocess.Popen(smoke_cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT); handle.write(f"PID={proc.pid}\n"); handle.flush(); rc = proc.wait()
        if rc != 0: raise RuntimeError(f"BioDINO MoNuSeg smoke failed: {smoke_log}")
        # Re-sample after smoke, then require seven simultaneously eligible cards.
        while len(idle_gpus(args.min_free_mib)) < 7:
            if args.no_wait: raise RuntimeError("smoke passed but fewer than seven eligible GPUs")
            print(f"[{stamp()}] waiting for seven eligible GPUs: {snapshot()}", flush=True); time.sleep(args.poll_seconds)
        assigned = idle_gpus(args.min_free_mib)[:7]
        by_dataset = {d: assigned[i] for i, d in enumerate(DATASETS)}
        with ThreadPoolExecutor(max_workers=7) as pool:
            future_map = {pool.submit(run_one, args, d, by_dataset[d], args.output_root / d / "no_trick_seed0"): d for d in DATASETS}
            for future in as_completed(future_map):
                result = future.result(); row = next(r for r in rows if r["dataset"] == result["dataset"]); row.update({k: result[k] for k in ("no_trick", "status")}); write_rows(status_path, rows)
        return 0
    finally:
        for child in lock.iterdir(): child.unlink()
        lock.rmdir()


if __name__ == "__main__": raise SystemExit(main())
