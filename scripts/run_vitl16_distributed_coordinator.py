#!/usr/bin/env python3
"""Coordinate independent ViT-L/16 screening jobs across audited single GPUs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path("/mnt/huawei_deepcad/dinov3")
OUT = ROOT / "outputs" / "instance_seg_tuning"
CHECKPOINT = ROOT / "outputs/01_training_runs/5tb_idweak10_vitl16_robust_b1024_8gpu/ckpt/15374/checkpoint.pth"
CONFIG = ROOT / "dinov3/configs/train/microscopy_continual_vitl16_robust_5tb_idweak10.yaml"
TRAIN = ROOT / "dinov3/eval/bio_segmentation/instance_seg/train.py"
WORKER = ROOT / "scripts/run_vitl16_distributed_worker.py"
REMOTE = ["-l", "bbnc", "172.16.1.206"]
BBNC_PYTHON = "/home/bbnc/anaconda3/envs/dinov3/bin/python"
INSPUR_PYTHON = "/home/inspur/anaconda3/envs/dinov3/bin/python"

TASKS = [
    ("livecell_frozen_seed1", "livecell", "frozen", 1),
    ("livecell_frozen_seed2", "livecell", "frozen", 2),
    ("livecell_finetune_seed1", "livecell", "finetune", 1),
    ("livecell_finetune_seed2", "livecell", "finetune", 2),
    ("cellpose_frozen_seed1", "cellpose", "frozen", 1),
    ("cellpose_frozen_seed2", "cellpose", "frozen", 2),
    ("cellpose_finetune_seed1", "cellpose", "finetune", 1),
    ("cellpose_finetune_seed2", "cellpose", "finetune", 2),
]


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ssh(args: List[str], command: str, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}"] + args + [command],
                          capture_output=True, text=True, timeout=timeout + 5, check=False)


def read_status(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def task_status(state_dir: Path, task_id: str) -> Tuple[Optional[Path], dict]:
    matches = list(state_dir.glob(f"*/{task_id}/status.json"))
    if not matches:
        return None, {}
    if len(matches) == 1:
        return matches[0], read_status(matches[0])
    # A duplicate launch is invalid, but keep the live/valid record visible so
    # the scheduler cannot requeue the same task while the retained run exists.
    records = [(path, read_status(path)) for path in matches]
    active = [item for item in records if item[1].get("state") == "running"]
    completed = [item for item in records if item[1].get("state") == "completed"]
    if active:
        return max(active, key=lambda item: item[1].get("heartbeat", ""))
    if completed:
        return max(completed, key=lambda item: item[1].get("heartbeat", ""))
    return records[0]


def terminal(data: dict) -> bool:
    return data.get("state") in {"completed", "failed"}


def reconcile_normal_end(state_dir: Path, task_id: str) -> None:
    path, data = task_status(state_dir, task_id)
    if path is None or data.get("state") != "failed" or data.get("error") != "normal_end_marker_missing" or data.get("exit_code") != 0:
        return
    result_path = Path(data.get("result_json", ""))
    log_path = OUT / "distributed_logs" / state_dir.name / data.get("host", "") / f"{task_id}.log"
    try:
        result = json.loads(result_path.read_text())
        log_text = log_path.read_text(errors="ignore")
        metric = "SEG" if data.get("dataset") == "livecell" else "CellposeStyleAP"
        valid = isinstance(result.get("val", {}).get(metric), (int, float)) and re.search(r"Epoch\s+50/50", log_text) and "Results saved" in log_text
        valid = bool(valid) and not ("Traceback" in log_text and "OSError: [Errno 16] Device or resource busy" not in log_text)
    except (OSError, ValueError, json.JSONDecodeError):
        valid = False
    if valid:
        data.update({"state": "completed", "epoch": "50/50", "error": "", "heartbeat": stamp()})
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def build_workers() -> Dict[str, dict]:
    workers: Dict[str, dict] = {}
    for host in ("cpu8", "cpu9", "cpu10", "cpu11"):
        workers[host] = {"ssh": [host], "python": INSPUR_PYTHON, "gpu": "0", "kind": "cpu", "modes": {"frozen", "finetune"}}
    for index in range(8):
        workers[f"3090-qi-gpu{index}"] = {
            "ssh": REMOTE, "python": BBNC_PYTHON, "gpu": str(index), "kind": "3090", "modes": {"frozen", "finetune"},
        }
    return workers


WORKERS = build_workers()


def gpu_snapshot(worker_id: str) -> Optional[dict]:
    cfg = WORKERS[worker_id]
    command = (
        f"nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw "
        f"--format=csv,noheader -i {shlex.quote(cfg['gpu'])}; "
        f"nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i {shlex.quote(cfg['gpu'])}"
    )
    result = ssh(cfg["ssh"], command)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines:
        return None
    fields = [x.strip() for x in lines[0].split(",")]
    if len(fields) < 7:
        return None
    def mib(value: str) -> float:
        return float(value.replace("MiB", "").strip())
    def percent(value: str) -> float:
        return float(value.replace("%", "").strip())
    pids = []
    for line in lines[1:]:
        try:
            pids.append(int(line.split(",", 1)[0].strip()))
        except (TypeError, ValueError):
            pass
    return {"index": fields[0], "uuid": fields[1], "used": mib(fields[2]), "free": mib(fields[3]),
            "util": percent(fields[4]), "temperature": fields[5], "power": fields[6], "pids": pids}


def capacity_class(worker_id: str, sample: bool = True) -> Tuple[Optional[str], Optional[dict]]:
    cfg = WORKERS[worker_id]
    if cfg["kind"] != "3090":
        return ("cpu", None) if host_idle(worker_id) else (None, None)
    snapshots = []
    for n in range(3 if sample else 1):
        current = gpu_snapshot(worker_id)
        if current is None:
            return None, None
        snapshots.append(current)
        if n < (2 if sample else 0):
            time.sleep(10)
    current = snapshots[-1]
    min_free = min(item["free"] for item in snapshots)
    max_util = max(item["util"] for item in snapshots)
    memory_growth = max(item["used"] for item in snapshots) - min(item["used"] for item in snapshots)
    current["sample_min_free"] = min_free
    current["sample_max_util"] = max_util
    current["memory_growth"] = memory_growth
    if max_util > 90 or min_free < 6144:
        return None, current
    if min_free < 10240:
        return ("frozen_only_shared" if current["pids"] else "frozen_only"), current
    if min_free < 18432:
        return ("frozen_shared" if current["pids"] else "frozen"), current
    return ("full_ft_shared" if current["pids"] else "full_ft"), current


def host_idle(worker_id: str) -> bool:
    cfg = WORKERS[worker_id]
    py = shlex.quote(cfg["python"])
    gpu = shlex.quote(cfg["gpu"])
    check = (
        f"test -x {py} && test -d {shlex.quote(str(ROOT))} && "
        f"test -e /mnt/huawei_deepcad/benchmark/segmentation/LIVECell && "
        f"test -e /mnt/huawei_deepcad/benchmark/segmentation/cellpose/extracted && "
        f"test -e {shlex.quote(str(CHECKPOINT))} && test -e {shlex.quote(str(CONFIG))} && "
        f"test -z \"$(nvidia-smi -i {gpu} --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')\" && "
        f"CUDA_VISIBLE_DEVICES={gpu} {py} -c 'import torch,dinov3; assert torch.cuda.is_available(); torch.tensor([1.0],device=\"cuda\")'"
    )
    return ssh(cfg["ssh"], check).returncode == 0


def write_manifest(state_dir: Path, run_id: str) -> None:
    manifest = {
        "run_id": run_id, "created_at": stamp(), "screening": "ViT-L/16 strategy screening; not DINOv3-7B",
        "code_commit": "d5021e6eaf701dfefdb697dc224961d55d1b00d5",
        "frozen_hashes": {"dinov3/eval/bio_segmentation/instance_seg/train.py": sha256(TRAIN),
                          "scripts/run_vitl16_distributed_worker.py": sha256(WORKER),
                          "scripts/run_vitl16_distributed_coordinator.py": sha256(Path(__file__)),
                          "dinov3/configs/train/microscopy_continual_vitl16_robust_5tb_idweak10.yaml": sha256(CONFIG),
                          "outputs/01_training_runs/5tb_idweak10_vitl16_robust_b1024_8gpu/ckpt/15374/checkpoint.pth": "df92f121c69ac7afe027c9b50cd7eee9b9c7349a61a9063f5d3092f266fadc41"},
        "protocol": {"layers": [4, 11, 17, 23], "epochs": 50, "loss": "CE+Dice", "decoder": "current",
                     "crop_size": 256, "stride": 192, "batch_size": 8, "grad_accum_steps": 1,
                     "augmentation": "strong", "mosaic_prob": 0.3,
                     "data_roots": {"livecell": "/mnt/huawei_deepcad/benchmark/segmentation/LIVECell",
                                    "cellpose": "/mnt/huawei_deepcad/benchmark/segmentation/cellpose/extracted"}},
        "workers": {worker_id: {"ssh": cfg["ssh"], "python": cfg["python"], "gpu_index": cfg["gpu"], "kind": cfg["kind"]}
                    for worker_id, cfg in WORKERS.items()},
        "tasks": [{"task_id": t[0], "dataset": t[1], "mode": t[2], "seed": t[3]} for t in TASKS],
        "shared_writes_forbidden": ["method_gain_ledger.csv", "structural_optimization_results.csv", "adaptation_results.csv", "structural_optimization_summary.md"],
    }
    (state_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def launch(state_dir: Path, log_dir: Path, worker_id: str, task: tuple, assigned: dict) -> bool:
    cfg = WORKERS[worker_id]
    task_id, dataset, mode, seed = task
    session = f"vitl16_{state_dir.name}_{task_id}_{worker_id}"[:180]
    log_dir.joinpath(worker_id).mkdir(parents=True, exist_ok=True)
    (log_dir / worker_id / f"{task_id}.launcher.log").write_text("")
    bootstrap_log = log_dir / worker_id / f"{task_id}.bootstrap.log"
    worker_cmd = " ".join(shlex.quote(x) for x in [str(cfg["python"]), str(WORKER), state_dir.name, worker_id, task_id, dataset, mode, str(seed), "--gpu", cfg["gpu"], "--python", cfg["python"]])
    shell_cmd = f"cd {shlex.quote(str(ROOT))} && exec {worker_cmd} >> {shlex.quote(str(bootstrap_log))} 2>&1"
    remote = f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(shell_cmd)}"
    result = ssh(cfg["ssh"], remote)
    (log_dir / worker_id / f"{task_id}.launcher.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        return False
    assigned[worker_id] = task_id
    return True


def remote_task_alive(worker_id: str, run_id: str, task_id: str) -> bool:
    cfg = WORKERS[worker_id]
    marker = f"run_vitl16_distributed_worker.py {run_id} {worker_id} {task_id}"
    result = ssh(cfg["ssh"], f"ps -ef | grep -F -- {shlex.quote(marker)} | grep -v grep | grep -v pgrep >/dev/null")
    return result.returncode == 0


def refresh_central(state_dir: Path, observations: dict) -> None:
    rows = ["task_id\thost\tgpu_uuid\tstate\tpid\tepoch\tmem_used_mib\tmem_free_mib\tutilization\tschedulable_class\theartbeat\toutput\texit_code"]
    for task_id, *_ in TASKS:
        path, data = task_status(state_dir, task_id)
        if path is None:
            data = {"task_id": task_id, "host": "", "gpu_uuid": "", "state": "pending", "pid": "", "epoch": "", "heartbeat": "", "result_json": "", "exit_code": ""}
        obs = observations.get(data.get("host", ""), {})
        rows.append("\t".join(str(value) for value in (data.get("task_id", task_id), data.get("host", ""), data.get("gpu_uuid", ""), data.get("state", "pending"), data.get("pid", ""), data.get("epoch", ""), obs.get("used", ""), obs.get("free", ""), obs.get("util", ""), obs.get("class", ""), data.get("heartbeat", ""), data.get("result_json", ""), data.get("exit_code", ""))))
    (state_dir / "central_status.tsv").write_text("\n".join(rows) + "\n")


def task_priority(task: tuple) -> tuple:
    # Long LIVECell fine-tuning jobs first, then other fine-tuning, then Frozen.
    return (0 if task[1] == "livecell" and task[2] == "finetune" else 1 if task[2] == "finetune" else 2 if task[1] == "livecell" else 3, task[0])


def run_queue(state_dir: Path, log_dir: Path) -> bool:
    for task, *_ in TASKS:
        reconcile_normal_end(state_dir, task)
    statuses = {task[0]: task_status(state_dir, task[0])[1] for task in TASKS}
    pending = [task for task in TASKS if not statuses[task[0]]]
    assigned: Dict[str, str] = {}
    assigned_at: Dict[str, float] = {}
    observations: Dict[str, dict] = {}
    # Preserve already-running jobs, including the LIVECell task launched by the previous coordinator.
    uuid_to_worker = {}
    for worker_id in WORKERS:
        if WORKERS[worker_id]["kind"] == "3090":
            snap = gpu_snapshot(worker_id)
            if snap:
                uuid_to_worker[snap["uuid"]] = worker_id
    for task in TASKS:
        data = statuses[task[0]]
        if data.get("state") == "running":
            worker_id = uuid_to_worker.get(data.get("gpu_uuid"), data.get("host", ""))
            if worker_id in WORKERS:
                assigned[worker_id] = task[0]
                assigned_at[worker_id] = time.time()
                pending = [item for item in pending if item[0] != task[0]]
    while pending or assigned:
        for worker_id, task_id in list(assigned.items()):
            path, data = task_status(state_dir, task_id)
            # A detached launcher can fail before status.json is written. Retry
            # only after a grace period and only when the remote wrapper is gone.
            if path is None and time.time() - assigned_at.get(worker_id, time.time()) > 90 and not remote_task_alive(worker_id, state_dir.name, task_id):
                assigned.pop(worker_id, None)
                assigned_at.pop(worker_id, None)
                task = next(item for item in TASKS if item[0] == task_id)
                if task not in pending:
                    pending.append(task)
                continue
            if terminal(data):
                assigned.pop(worker_id, None)
                assigned_at.pop(worker_id, None)
        for worker_id in WORKERS:
            if worker_id in assigned:
                continue
            cls, snap = capacity_class(worker_id)
            if snap:
                snap["class"] = cls or "unavailable"
                observations[worker_id] = snap
            if cls is None:
                continue
            for task in sorted(list(pending), key=task_priority):
                if task[2] == "finetune" and not cls.startswith("full_ft"):
                    continue
                if launch(state_dir, log_dir, worker_id, task, assigned):
                    assigned_at[worker_id] = time.time()
                    pending.remove(task)
                    break
        refresh_central(state_dir, observations)
        if pending or assigned:
            time.sleep(60)
    return not any(read_status(path).get("state") == "failed" for path in state_dir.glob("*/*/status.json"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    state_dir = OUT / "distributed_runs" / args.run_id
    log_dir = OUT / "distributed_logs" / args.run_id
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(state_dir, args.run_id)
    (state_dir / "coordinator_started").write_text(stamp() + "\n")
    ok = run_queue(state_dir, log_dir)
    (state_dir / "STAGE_EXIT").write_text(f"{0 if ok else 1} {stamp()}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
