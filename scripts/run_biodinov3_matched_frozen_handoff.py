#!/usr/bin/env python3
"""Launch matched 50-epoch Frozen controls after the active Full-FT queue."""

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
LOCKS = ROOT / "locks"
LOGS = ROOT / "logs"
STATUS = ROOT / "matched_handoff_status.json"
CHECKPOINT = "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/ckpt/100/checkpoint.pth"
CONFIG = "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/config.yaml"

TASKS = {
    "livecell_frozen_seed0": {
        "dataset": "livecell",
        "seed": 0,
        "preferred": "cpu8",
        "prerequisite": ROOT / "livecell/finetune_seed1_retry1/results.json",
    },
    "livecell_frozen_seed1": {
        "dataset": "livecell",
        "seed": 1,
        "preferred": "cpu8",
        "prerequisite": ROOT / "livecell/finetune_seed1_retry1/results.json",
    },
    "livecell_frozen_seed2": {
        "dataset": "livecell",
        "seed": 2,
        "preferred": "cpu10",
        "prerequisite": ROOT / "livecell/finetune_seed2_retry1/results.json",
    },
    "cellpose_frozen_seed0": {
        "dataset": "cellpose",
        "seed": 0,
        "preferred": "cpu11",
        "prerequisite": ROOT / "cellpose/finetune_seed2_retry1/results.json",
    },
    "cellpose_frozen_seed1": {
        "dataset": "cellpose",
        "seed": 1,
        "preferred": "cpu11",
        "prerequisite": ROOT / "cellpose/finetune_seed2_retry1/results.json",
    },
    "cellpose_frozen_seed2": {
        "dataset": "cellpose",
        "seed": 2,
        "preferred": "cpu11",
        "prerequisite": ROOT / "cellpose/finetune_seed2_retry1/results.json",
    },
}

FIXED_QUEUES = {
    "cpu8": ["livecell_frozen_seed0", "livecell_frozen_seed1"],
    "cpu10": ["livecell_frozen_seed2"],
    "cpu11": ["cellpose_frozen_seed0", "cellpose_frozen_seed1", "cellpose_frozen_seed2"],
}

state_lock = threading.Lock()
state = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "purpose": "matched Frozen 50ep controls; historical Frozen seed0 is standardized_frozen_probe_20ep",
    "protocol": {
        "checkpoint": CHECKPOINT,
        "layers": [7, 15, 23, 31],
        "decoder": "current",
        "epochs": 50,
        "batch_size": 1,
        "grad_accum_steps": 8,
        "bf16": True,
        "crop": 256,
        "stride": 192,
        "augmentation": "strong",
        "mosaic": 0.3,
        "loss": "ce_dice",
        "fg_thresh": 0.5,
        "energy_thresh": 0.4,
        "tta": False,
    },
    "tasks": {name: {"status": "reserved_pending", **{k: str(v) for k, v in task.items()}} for name, task in TASKS.items()},
}


def write_status() -> None:
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(tmp, STATUS)


def update(name: str, **values: object) -> None:
    with state_lock:
        state["tasks"][name].update(values)
        write_status()


def refresh_summary() -> None:
    subprocess.run(
        [
            "/home/inspur/anaconda3/envs/dinov3/bin/python",
            str(REPO / "scripts/summarize_biodinov3_paired_multiseed.py"),
        ],
        check=False,
        cwd=REPO,
    )


def ssh(host: str, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
        check=False,
        capture_output=True,
        text=True,
    )


def full_ft_complete(path: Path) -> bool:
    exit_path = path.parent / "exit_code.txt"
    if not path.exists() or not exit_path.exists() or exit_path.read_text().strip() != "0":
        return False
    try:
        data = json.loads(path.read_text())
        meta = data["_meta"]
        return meta["global_steps"] == meta["total_steps"] and meta["seed"] in (1, 2)
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def local_gpu0_audit(host: str) -> tuple[bool, dict[str, object]]:
    command = (
        "free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits); "
        "util=$(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits); "
        "uuid=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader); "
        "pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | paste -sd, -); "
        "printf '%s|%s|%s|%s\\n' \"$free\" \"$util\" \"$uuid\" \"$pids\""
    )
    result = ssh(host, command)
    audit: dict[str, object] = {"host": host, "ssh_rc": result.returncode, "raw": result.stdout.strip()}
    if result.returncode != 0:
        return False, audit
    try:
        free, util, uuid, pids = result.stdout.strip().split("|", 3)
        audit.update(free_mib=int(free), utilization=int(util), uuid=uuid, compute_pids=pids)
        return int(free) >= 22000 and int(util) == 0 and not pids, audit
    except (ValueError, TypeError):
        return False, audit


def audit_3090() -> list[dict[str, object]]:
    host = "bbnc@172.16.1.206"
    result = ssh(
        host,
        "nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits; "
        "echo PIDS; nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits",
    )
    if result.returncode != 0:
        return []
    gpu_text, _, pid_text = result.stdout.partition("PIDS\n")
    active = {line.split(",", 1)[0].strip() for line in pid_text.splitlines() if line.strip()}
    rows = []
    for line in gpu_text.splitlines():
        if not line.strip():
            continue
        index, uuid, free, util = (part.strip() for part in line.split(","))
        rows.append(
            {
                "host": host,
                "gpu": index,
                "uuid": uuid,
                "free_mib": int(free),
                "utilization": int(util),
                "active_compute_pid": uuid in active,
                "eligible": int(free) >= 22000 and int(util) == 0 and uuid not in active,
            }
        )
    return rows


def claim(name: str, worker: str, gpu: str, uuid: str) -> bool:
    claim_dir = LOCKS / f"{name}.lock/claimed"
    try:
        claim_dir.mkdir()
    except FileExistsError:
        return False
    (claim_dir / "owner.json").write_text(
        json.dumps({"worker": worker, "gpu": gpu, "uuid": uuid, "claimed_at": datetime.now(timezone.utc).isoformat()}, indent=2)
        + "\n"
    )
    return True


def command_for(name: str, python: str, gpu: str) -> tuple[str, Path, Path, Path]:
    task = TASKS[name]
    dataset = task["dataset"]
    seed = task["seed"]
    output = ROOT / dataset / f"frozen50ep_seed{seed}"
    result_path = output / "results.json"
    exit_path = output / "exit_code.txt"
    log_path = LOGS / f"{name}_50ep.log"
    tmpdir = f"/tmp/biodino_matched_{name}"
    data_root = (
        "/mnt/huawei_deepcad/benchmark/segmentation/cellpose/extracted"
        if dataset == "cellpose"
        else "/mnt/huawei_deepcad/benchmark/segmentation/LIVECell"
    )
    args = [
        python,
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
    command = (
        f"cd {REPO} && mkdir -p {tmpdir} {shlex.quote(str(output))} && "
        f"CUDA_VISIBLE_DEVICES={gpu} PYTHONUNBUFFERED=1 TMPDIR={tmpdir} TMP={tmpdir} TEMP={tmpdir} {shlex.join(args)}"
    )
    return command, result_path, exit_path, log_path


def run_claimed(name: str, host: str, gpu: str, python: str, audit: dict[str, object]) -> None:
    command, result_path, exit_path, log_path = command_for(name, python, gpu)
    if result_path.exists() and exit_path.exists() and exit_path.read_text().strip() == "0":
        update(name, status="completed_existing", results=str(result_path))
        return
    body = (
        f"{command}; rc=$?; printf '%s\\n' \"$rc\" > {exit_path}.tmp; mv {exit_path}.tmp {exit_path}; exit $rc"
    )
    wrapped = f"nohup bash -lc {shlex.quote(body)} >> {shlex.quote(str(log_path))} 2>&1 < /dev/null & echo $!"
    started = datetime.now(timezone.utc).isoformat()
    launch = ssh(host, wrapped)
    if launch.returncode != 0 or not launch.stdout.strip().isdigit():
        update(name, status="launch_failed", audit=audit, launch_stderr=launch.stderr)
        return
    remote_pid = int(launch.stdout.strip())
    update(
        name,
        status="running",
        host=host,
        gpu=gpu,
        uuid=audit["uuid"],
        audit=audit,
        remote_pid=remote_pid,
        started_at=started,
        output=str(result_path.parent),
        log=str(log_path),
    )
    while ssh(host, f"ps -p {remote_pid} -o pid=").stdout.strip():
        update(name, last_alive_at=datetime.now(timezone.utc).isoformat())
        time.sleep(30)
    rc = int(exit_path.read_text().strip()) if exit_path.exists() else 255
    complete = rc == 0 and result_path.exists()
    update(
        name,
        status="completed" if complete else "failed",
        returncode=rc,
        finished_at=datetime.now(timezone.utc).isoformat(),
        results=str(result_path) if result_path.exists() else None,
    )
    refresh_summary()


def wait_prerequisite(name: str) -> None:
    prerequisite = TASKS[name]["prerequisite"]
    assert isinstance(prerequisite, Path)
    while not full_ft_complete(prerequisite):
        update(name, status="waiting_for_full_ft_prerequisite")
        time.sleep(60)


def fixed_worker(host: str, names: list[str]) -> None:
    for name in names:
        wait_prerequisite(name)
        output = ROOT / TASKS[name]["dataset"] / f"frozen50ep_seed{TASKS[name]['seed']}"
        if (output / "results.json").exists() and (output / "exit_code.txt").exists():
            update(name, status="completed_existing", results=str(output / "results.json"))
            continue
        while True:
            eligible, audit = local_gpu0_audit(host)
            update(name, status="waiting_for_eligible_gpu", last_audit=audit)
            if eligible and claim(name, host, "0", str(audit["uuid"])):
                run_claimed(name, host, "0", "/home/inspur/anaconda3/envs/dinov3/bin/python", audit)
                break
            if (LOCKS / f"{name}.lock/claimed").exists():
                break
            time.sleep(60)


def dynamic_3090_worker() -> None:
    while True:
        if all(state["tasks"][name]["status"] in {"completed", "completed_existing", "running", "failed"} for name in TASKS):
            return
        eligible_gpus = [row for row in audit_3090() if row["eligible"]]
        if not eligible_gpus:
            time.sleep(60)
            continue
        selected = None
        for name, task in TASKS.items():
            prerequisite = task["prerequisite"]
            assert isinstance(prerequisite, Path)
            if full_ft_complete(prerequisite) and not (LOCKS / f"{name}.lock/claimed").exists():
                selected = name
                break
        if selected is None:
            time.sleep(60)
            continue
        audit = eligible_gpus[0]
        gpu = str(audit["gpu"])
        if claim(selected, "3090-qi", gpu, str(audit["uuid"])):
            run_claimed(
                selected,
                "bbnc@172.16.1.206",
                gpu,
                "/home/bbnc/anaconda3/envs/dinov3/bin/python",
                audit,
            )


def main() -> int:
    refresh_summary()
    write_status()
    threads = [threading.Thread(target=fixed_worker, args=(host, names), name=host) for host, names in FIXED_QUEUES.items()]
    threads.append(threading.Thread(target=dynamic_3090_worker, name="3090-qi"))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with state_lock:
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_status()
    refresh_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
