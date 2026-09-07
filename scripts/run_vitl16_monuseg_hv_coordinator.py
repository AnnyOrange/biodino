#!/usr/bin/env python3
"""CPU7-only coordinator for six independent MoNuSeg HV auxiliary A/B tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/mnt/huawei_deepcad/dinov3")
OUT = ROOT / "outputs" / "instance_seg_tuning"
REMOTE = ["-l", "bbnc", "172.16.1.206"]
WORKER = ROOT / "scripts/run_vitl16_monuseg_hv_worker.py"
PYTHON = "/home/bbnc/anaconda3/envs/dinov3/bin/python"
GPUS = ("1", "2", "3", "7")
TASKS = tuple((f"monuseg_{method}_seed{seed}", method, seed) for seed in range(3) for method in ("control", "hv_auxiliary"))


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ssh(command: str, timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", *REMOTE, command], text=True, capture_output=True, check=False, timeout=timeout)


def read_status(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def gpu_idle(gpu: str) -> tuple[bool, dict]:
    command = (
        f"nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader -i {shlex.quote(gpu)}; "
        f"nvidia-smi --query-compute-apps=pid --format=csv,noheader -i {shlex.quote(gpu)}"
    )
    check = ssh(command)
    lines = [line.strip() for line in check.stdout.splitlines() if line.strip()]
    if check.returncode != 0 or not lines:
        return False, {}
    fields = [item.strip() for item in lines[0].split(",")]
    free = float(fields[2].replace("MiB", "").strip())
    util = float(fields[3].replace("%", "").strip())
    # A single-column query returns bare PID lines; multi-column variants may
    # include commas.  Treat both forms as active compute processes.
    pids = []
    for line in lines[1:]:
        candidate = line.split(",", 1)[0].strip()
        if candidate.isdigit():
            pids.append(candidate)
    return not pids and free >= 18432 and util <= 10, {"index": fields[0], "uuid": fields[1], "free_mib": free, "util": util, "pids": pids}


def write_central(run_dir: Path, gpu_observations: dict) -> None:
    rows = ["task_id\tmethod\tseed\thost\tgpu_index\tgpu_uuid\tstate\tpid\tepoch\theartbeat\texit_code\toutput"]
    for task_id, method, seed in TASKS:
        status_path = run_dir / "3090-qi" / task_id / "status.json"
        data = read_status(status_path)
        rows.append("\t".join(str(value) for value in (
            task_id, method, seed, data.get("host", ""), data.get("gpu_index", ""), data.get("gpu_uuid", ""),
            data.get("state", "pending"), data.get("pid", ""), data.get("epoch", ""), data.get("heartbeat", ""),
            data.get("exit_code", ""), data.get("result_json", ""),
        )))
    (run_dir / "central_status.tsv").write_text("\n".join(rows) + "\n")
    (run_dir / "gpu_observations.json").write_text(json.dumps({"at": stamp(), "gpus": gpu_observations}, indent=2) + "\n")


def launch(run_id: str, task: tuple[str, str, int], gpu: str) -> bool:
    task_id, method, seed = task
    session = f"vitl16_hv_{run_id}_{task_id}"[:180]
    command = " ".join(shlex.quote(part) for part in (PYTHON, str(WORKER), run_id, task_id, method, str(seed), "--gpu", gpu, "--host", "3090-qi"))
    remote = f"cd {shlex.quote(str(ROOT))} && tmux new-session -d -s {shlex.quote(session)} {shlex.quote(command)}"
    return ssh(remote).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    run_dir = OUT / "distributed_runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": args.run_id,
        "created_at": stamp(),
        "screening": "ViT-L/16 strategy screening; not DINOv3-7B",
        "protocol": {"dataset": "monuseg", "backbone": "Frozen ViT-L/16", "epochs": 30, "layers": [4, 11, 17, 23], "decoder": "current", "loss": "CE+Dice", "augmentation": "strong", "mosaic_prob": 0.3, "crop": 256, "stride": 192, "batch": 8, "grad_accum": 1, "primary_metric": "AJI", "auxiliary": "independent HV-distance head, MSE + MSGE, weight=1.0"},
        "tasks": [{"task_id": task_id, "method": method, "seed": seed} for task_id, method, seed in TASKS],
        "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (WORKER, ROOT / "dinov3/eval/bio_segmentation/instance_seg/train.py", ROOT / "dinov3/eval/bio_segmentation/instance_seg/decoder.py", ROOT / "dinov3/eval/bio_segmentation/instance_seg/losses.py", ROOT / "dinov3/eval/bio_segmentation/instance_seg/model.py")},
        "shared_writes_forbidden": ["method_gain_ledger.csv", "structural_optimization_results.csv", "adaptation_results.csv", "structural_optimization_summary.md"],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    coordinator_log = OUT / "distributed_logs" / args.run_id / "coordinator.log"
    coordinator_log.parent.mkdir(parents=True, exist_ok=True)
    pending = list(TASKS)
    assigned: dict[str, tuple[str, str, int]] = {}
    with coordinator_log.open("a", buffering=1) as log:
        log.write(f"START coordinator {args.run_id} at {stamp()}\n")
        while pending or assigned:
            observations = {}
            for gpu in GPUS:
                idle, obs = gpu_idle(gpu)
                observations[gpu] = {**obs, "idle": idle}
                task = assigned.get(gpu)
                if task is not None:
                    data = read_status(run_dir / "3090-qi" / task[0] / "status.json")
                    if data.get("state") in {"completed", "failed"}:
                        log.write(f"TERMINAL {task[0]} state={data.get('state')} at {stamp()}\n")
                        assigned.pop(gpu, None)
                    continue
                if not pending or not idle:
                    continue
                next_task = pending.pop(0)
                if launch(args.run_id, next_task, gpu):
                    assigned[gpu] = next_task
                    log.write(f"LAUNCHED {next_task[0]} gpu={gpu} uuid={obs.get('uuid')} at {stamp()}\n")
                else:
                    pending.insert(0, next_task)
                    log.write(f"LAUNCH_FAILED {next_task[0]} gpu={gpu} at {stamp()}\n")
            write_central(run_dir, observations)
            time.sleep(30)
        write_central(run_dir, observations)
        log.write(f"RUN_EXIT coordinator complete at {stamp()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
