#!/usr/bin/env python3
"""Run the prescribed BioDINO Cellpose/LIVECell paired seed-1/2 queue."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/mnt/huawei_deepcad/dinov3")
ROOT = REPO / "outputs/instance_seg_tuning/biodinov3_paired_multiseed"
LOGS = ROOT / "logs"
LOCKS = ROOT / "locks"
STATUS = ROOT / "scheduler_status.json"
PYTHON = "/home/inspur/anaconda3/envs/dinov3/bin/python"
CHECKPOINT = "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/ckpt/100/checkpoint.pth"
CONFIG = "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/config.yaml"

WORKERS = {
    "cpu8": [
        ("livecell", "finetune", 1),
        ("livecell", "frozen", 1),
        ("cellpose", "frozen", 1),
    ],
    "cpu10": [
        ("livecell", "finetune", 2),
        ("livecell", "frozen", 2),
        ("cellpose", "frozen", 2),
    ],
    "cpu11": [
        ("cellpose", "finetune", 1),
        ("cellpose", "finetune", 2),
    ],
}

state_lock = threading.Lock()
state: dict[str, object] = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "protocol": {
        "layers": [7, 15, 23, 31],
        "decoder": "current",
        "epochs": 50,
        "batch_size": 1,
        "grad_accum_steps": 8,
        "crop_size": 256,
        "stride": 192,
        "amp_dtype": "bf16",
        "augmentation": "strong",
        "mosaic_prob": 0.3,
        "np_loss_mode": "ce_dice",
        "fg_thresh": 0.5,
        "energy_thresh": 0.4,
        "tta": False,
    },
    "tasks": {},
}


def task_name(dataset: str, mode: str, seed: int) -> str:
    return f"{dataset}_{mode}_seed{seed}"


def write_status() -> None:
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(tmp, STATUS)


def update(name: str, **values: object) -> None:
    with state_lock:
        tasks = state["tasks"]
        assert isinstance(tasks, dict)
        tasks.setdefault(name, {}).update(values)
        write_status()


def ssh(host: str, command: str, *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
        check=False,
        capture_output=capture,
        text=True,
    )


def gpu_audit(host: str) -> tuple[bool, dict[str, object]]:
    command = (
        "free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0); "
        "util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0); "
        "uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0); "
        "pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | paste -sd, -); "
        "printf '%s|%s|%s|%s\\n' \"$free\" \"$util\" \"$uuid\" \"$pids\""
    )
    result = ssh(host, command)
    audit: dict[str, object] = {"host": host, "raw": result.stdout.strip(), "ssh_rc": result.returncode}
    if result.returncode != 0:
        return False, audit
    try:
        free, util, uuid, pids = result.stdout.strip().split("|", 3)
        audit.update(free_mib=int(free), utilization=int(util), uuid=uuid, compute_pids=pids)
        eligible = int(free) >= 22000 and int(util) == 0 and not pids
        return eligible, audit
    except (TypeError, ValueError):
        return False, audit


def command_for(dataset: str, mode: str, seed: int, output: Path, tmpdir: str) -> str:
    data_root = (
        "/mnt/huawei_deepcad/benchmark/segmentation/cellpose/extracted"
        if dataset == "cellpose"
        else "/mnt/huawei_deepcad/benchmark/segmentation/LIVECell"
    )
    args = [
        PYTHON,
        "-u",
        "-m",
        "dinov3.eval.bio_segmentation.instance_seg.train",
        "--dataset",
        dataset,
        "--data-root",
        data_root,
        "--checkpoint",
        CHECKPOINT,
        "--train-config",
        CONFIG,
        "--output-dir",
        str(output),
        "--layers",
        "7",
        "15",
        "23",
        "31",
        "--epochs",
        "50",
        "--batch-size",
        "1",
        "--grad-accum-steps",
        "8",
        "--crop-size",
        "256",
        "--stride",
        "192",
        "--lr",
        "1e-3",
        "--weight-decay",
        "1e-4",
        "--amp-dtype",
        "bf16",
        "--feature-size",
        "32",
        "--embed-proj",
        "384",
        "--fusion-mode",
        "bucket_concat",
        "--decoder-variant",
        "current",
        "--num-workers",
        "0",
        "--eval-every",
        "10",
        "--seed",
        str(seed),
        "--aug",
        "strong",
        "--mosaic-prob",
        "0.3",
        "--np-loss-mode",
        "ce_dice",
        "--fg-thresh",
        "0.5",
        "--energy-thresh",
        "0.4",
        "--skip-test-eval",
    ]
    if mode == "finetune":
        args.extend(["--finetune", "--backbone-lr", "2e-5"])
    exports = f"CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 TMPDIR={shlex.quote(tmpdir)} TMP={shlex.quote(tmpdir)} TEMP={shlex.quote(tmpdir)}"
    return f"cd {shlex.quote(str(REPO))} && mkdir -p {shlex.quote(tmpdir)} && {exports} {shlex.join(args)}"


def remote_task_pid(host: str, output: Path) -> int | None:
    result = ssh(
        host,
        f"ps -eo pid=,args= | grep -F -- {shlex.quote(str(output))} | grep 'instance_seg.train' | grep -v grep | head -1",
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return int(result.stdout.strip().split(maxsplit=1)[0])
    except (ValueError, IndexError):
        return None


def wait_for_remote(host: str, remote_pid: int, name: str, exit_path: Path, result_path: Path) -> int:
    while True:
        probe = ssh(host, f"ps -p {remote_pid} -o pid=,stat=")
        if not probe.stdout.strip():
            break
        update(name, last_alive_at=datetime.now(timezone.utc).isoformat())
        time.sleep(30)
    return int(exit_path.read_text().strip()) if exit_path.exists() else 255


def run_task(host: str, dataset: str, mode: str, seed: int) -> None:
    name = task_name(dataset, mode, seed)
    primary_output = ROOT / dataset / f"{mode}_seed{seed}"
    primary_exit = primary_output / "exit_code.txt"
    primary_result = primary_output / "results.json"
    retry = primary_exit.exists() and primary_exit.read_text().strip() != "0" and not primary_result.exists()
    output = ROOT / dataset / (f"{mode}_seed{seed}_retry1" if retry else f"{mode}_seed{seed}")
    result_path = output / "results.json"
    exit_path = output / "exit_code.txt"
    log_path = LOGS / f"{name}.log"
    lock_path = LOCKS / f"{name}.lock"
    tmpdir = f"/tmp/biodino_paired_{name}"

    if result_path.exists() and exit_path.exists() and exit_path.read_text().strip() == "0":
        update(name, status="completed_existing", host=host, results=str(result_path))
        return
    try:
        lock_path.mkdir()
    except FileExistsError:
        existing_pid = remote_task_pid(host, output)
        if existing_pid is None:
            update(name, status="duplicate_lock_blocked", host=host, lock=str(lock_path))
            return
        update(
            name,
            status="running_existing",
            host=host,
            lock=str(lock_path),
            output=str(output),
            log=str(log_path),
            remote_pid=existing_pid,
        )
        rc = wait_for_remote(host, existing_pid, name, exit_path, result_path)
        finished = datetime.now(timezone.utc).isoformat()
        complete = rc == 0 and result_path.exists()
        update(
            name,
            status="completed" if complete else "failed",
            returncode=rc,
            finished_at=finished,
            results=str(result_path) if result_path.exists() else None,
        )
        try:
            lock_path.rmdir()
        except OSError:
            pass
        return

    try:
        eligible, audit = gpu_audit(host)
        update(name, status="audited", audit=audit, host=host, lock=str(lock_path))
        if not eligible:
            update(name, status="blocked_by_gpu_availability")
            return

        output.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        command = command_for(dataset, mode, seed, output, tmpdir)
        task_body = (
            f"{command}; rc=$?; "
            f"printf '%s\\n' \"$rc\" > {shlex.quote(str(exit_path))}.tmp; "
            f"mv {shlex.quote(str(exit_path))}.tmp {shlex.quote(str(exit_path))}; exit $rc"
        )
        wrapped = (
            f"nohup bash -lc {shlex.quote(task_body)} >> {shlex.quote(str(log_path))} 2>&1 < /dev/null & echo $!"
        )
        started = datetime.now(timezone.utc).isoformat()
        update(
            name,
            status="running",
            started_at=started,
            command=command,
            output=str(output),
            log=str(log_path),
            tmpdir=tmpdir,
        )
        with log_path.open("a") as log:
            log.write(f"[{started}] host={host} command={command}\n")
        launch = ssh(host, wrapped)
        if launch.returncode != 0 or not launch.stdout.strip().isdigit():
            update(name, status="launch_failed", launch_rc=launch.returncode, launch_stderr=launch.stderr)
            return
        remote_pid = int(launch.stdout.strip())
        update(name, remote_pid=remote_pid)
        rc = wait_for_remote(host, remote_pid, name, exit_path, result_path)
        finished = datetime.now(timezone.utc).isoformat()
        complete = rc == 0 and result_path.exists() and exit_path.exists() and exit_path.read_text().strip() == "0"
        update(
            name,
            status="completed" if complete else "failed",
            returncode=rc,
            finished_at=finished,
            results=str(result_path) if result_path.exists() else None,
        )
    finally:
        try:
            lock_path.rmdir()
        except OSError:
            pass


def run_worker(host: str, tasks: list[tuple[str, str, int]]) -> None:
    for dataset, mode, seed in tasks:
        run_task(host, dataset, mode, seed)


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    LOCKS.mkdir(parents=True, exist_ok=True)
    for host, tasks in WORKERS.items():
        for dataset, mode, seed in tasks:
            name = task_name(dataset, mode, seed)
            state["tasks"][name] = {"status": "queued", "host": host}
    write_status()
    threads = [threading.Thread(target=run_worker, args=(host, tasks), name=host) for host, tasks in WORKERS.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with state_lock:
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
