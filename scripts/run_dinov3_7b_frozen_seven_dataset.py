#!/usr/bin/env python3
"""Gated, independently logged Frozen DINOv3-7B seven-dataset coordinator."""

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
LAYERS = (7, 19, 29, 39)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_snapshot() -> list[dict[str, str]]:
    query = "index,uuid,memory.used,memory.free,utilization.gpu"
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True
    )
    rows = []
    for line in out.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 5:
            rows.append(dict(zip(("index", "uuid", "used", "free", "util"), values)))
    return rows


def idle_gpu_ids(min_free_mib: int = 22000) -> list[str]:
    return [row["index"] for row in gpu_snapshot() if int(row["free"]) >= min_free_mib and int(row["util"]) == 0]


def data_root(dataset: str) -> str:
    if dataset == "livecell":
        return "/mnt/huawei_deepcad/benchmark/segmentation/LIVECell"
    return f"/mnt/huawei_deepcad/benchmark/segmentation/{dataset}/extracted"


def command(args: argparse.Namespace, dataset: str, output: Path) -> list[str]:
    return [
        sys.executable, "-m", "dinov3.eval.bio_segmentation.instance_seg.train",
        "--dataset", dataset, "--data-root", data_root(dataset),
        "--checkpoint", str(args.checkpoint), "--train-config", str(args.train_config),
        "--output-dir", str(output), "--layers", *map(str, LAYERS),
        "--freeze-backbone", "--epochs", "50", "--batch-size", "1",
        "--grad-accum-steps", "8", "--crop-size", "256", "--stride", "192",
        "--lr", "1e-3", "--weight-decay", "1e-4", "--amp-dtype", "bf16",
        "--feature-size", "32", "--embed-proj", "384", "--fusion-mode", "bucket_concat",
        "--decoder-variant", "current", "--num-workers", "4", "--eval-every", "10",
        "--seed", "0", "--aug", "strong", "--mosaic-prob", "0.3",
        "--fg-thresh", "0.5", "--energy-thresh", "0.4", "--np-loss-mode", "ce_dice",
        "--skip-test-eval",
    ]


def run_logged(cmd: list[str], gpu: str, log_path: Path, env_extra: dict[str, str]) -> int:
    env = dict(os.environ)
    env.update(env_extra)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"[{stamp()}] GPU={gpu} $ {shlex.join(cmd)}\n")
        log.flush()
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[{stamp()}] PID={process.pid}\n")
        log.flush()
        return process.wait()


def write_status(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["dataset", "metric", "vitl16_frozen", "dinov3_7b_frozen", "7b_gain", "gpu", "pid", "peak_memory", "time_seconds", "status", "log"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/torch_cache/hub/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth")
    parser.add_argument("--train-config", type=Path, default=ROOT / "dinov3/configs/train/dinov3_vit7b16_pretrain.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/instance_seg_tuning/dinov3_7b_frozen_seven_dataset")
    parser.add_argument("--min-free-mib", type=int, default=22000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = args.output_root / ".task.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise SystemExit(f"Existing atomic task lock: {lock}")
    (lock / "owner.json").write_text(json.dumps({"pid": os.getpid(), "started": stamp()}, indent=2) + "\n")
    try:
        rows = [{"dataset": d, "metric": "", "vitl16_frozen": "", "dinov3_7b_frozen": "", "7b_gain": "", "gpu": "", "pid": "", "peak_memory": "", "time_seconds": "", "status": "pending", "log": ""} for d in DATASETS]
        status_path = args.output_root / "results.csv"
        write_status(status_path, rows)
        while not idle_gpu_ids(args.min_free_mib):
            if args.no_wait:
                raise RuntimeError("No truly idle GPU satisfies the 7B memory gate")
            print(f"[{stamp()}] waiting for a truly idle GPU; snapshot={gpu_snapshot()}", flush=True)
            time.sleep(args.poll_seconds)
        gpu = idle_gpu_ids(args.min_free_mib)[0]
        smoke_out = args.output_root / "monuseg" / "smoke" / "smoke.json"
        smoke_log = args.output_root / "logs" / "monuseg_smoke.log"
        smoke_cmd = [sys.executable, "scripts/smoke_dinov3_7b_instance_seg.py", "--checkpoint", str(args.checkpoint), "--train-config", str(args.train_config), "--output", str(smoke_out), "--layers", *map(str, LAYERS)]
        if run_logged(smoke_cmd, gpu, smoke_log, {"PYTHONUNBUFFERED": "1"}) != 0:
            raise RuntimeError(f"MoNuSeg 7B smoke failed; see {smoke_log}")
        formal_out = args.output_root / "monuseg" / "seed0"
        formal_log = args.output_root / "logs" / "monuseg_seed0.log"
        if run_logged(command(args, "monuseg", formal_out), gpu, formal_log, {"PYTHONUNBUFFERED": "1"}) != 0:
            raise RuntimeError(f"MoNuSeg formal training failed; see {formal_log}")
        rows[0]["status"] = "completed"
        rows[0]["gpu"] = gpu
        rows[0]["log"] = str(formal_log)
        write_status(status_path, rows)

        # Only fan out after MoNuSeg completes, and only onto independently
        # re-sampled cards that still satisfy the full idle gate.
        remaining = list(DATASETS[1:])
        while len(idle_gpu_ids(args.min_free_mib)) < len(remaining):
            if args.no_wait:
                raise RuntimeError("MoNuSeg passed but fewer than six idle GPUs remain")
            print(f"[{stamp()}] waiting for six idle GPUs after MoNuSeg; snapshot={gpu_snapshot()}", flush=True)
            time.sleep(args.poll_seconds)
        fanout_gpus = [candidate for candidate in idle_gpu_ids(args.min_free_mib) if candidate != gpu][:len(remaining)]

        def run_dataset(item: tuple[str, str]) -> tuple[str, int, str]:
            dataset, dataset_gpu = item
            out_dir = args.output_root / dataset / "seed0"
            log_path = args.output_root / "logs" / f"{dataset}_seed0.log"
            rc = run_logged(command(args, dataset, out_dir), dataset_gpu, log_path, {"PYTHONUNBUFFERED": "1"})
            return dataset, rc, str(log_path)

        with ThreadPoolExecutor(max_workers=len(remaining)) as executor:
            futures = [executor.submit(run_dataset, item) for item in zip(remaining, fanout_gpus)]
            for future in as_completed(futures):
                dataset, rc, log_path = future.result()
                row = next(row for row in rows if row["dataset"] == dataset)
                row["status"] = "completed" if rc == 0 else f"failed_rc_{rc}"
                row["log"] = log_path
                write_status(status_path, rows)
        return 0
    finally:
        for child in lock.iterdir():
            child.unlink()
        lock.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
