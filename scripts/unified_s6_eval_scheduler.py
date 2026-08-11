#!/usr/bin/env python3
"""Result-aware multi-GPU scheduler for the S6 B/L evaluation sweeps."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from reconcile_complete_bio_eval_checkpoints import (  # noqa: E402
    EXPECTED_RESULTS,
    collect_inventory,
    mark_complete,
)


CHECKPOINTS = (1024, 2049, 3074, 4099, 5124, 6149, 7174, 8199, 9224,
               10249, 11274, 12299, 13324, 14349, 15374)
OBJECTIVES = ("nosigreg", "sigreg005")
PREFILL = REPO_ROOT / "docs/scaling_law/bio_sweet_spot/scripts/48_prefill_s6_core_detection_once.sh"
ALPHA_QUEUE = REPO_ROOT / "docs/scaling_law/bio_sweet_spot/scripts/36_queue_s6_ab_alpha.sh"
EVAL_REL = Path("eval/full_taskwise_online_b8_bf16_auto_tta8_20260804")


MODEL_CONFIG = {
    "B": {
        "suffix": "gb1024_lr1e4_wu2_e15_seed1_4x3090qi_20260804",
        "python": "/home/bbnc/anaconda3/envs/dinov3/bin/python",
        "frozen_batch": "8",
        "core_reserve_gib": 18.0,
        "seg_reserve_gib": 5.0,
    },
    "L": {
        "suffix": "gb1024_lr5e5_wu2_e15_seed0_4x5090_20260804",
        "python": "/home/lxy/miniconda3/envs/dinov3/bin/python",
        "frozen_batch": "4",
        "core_reserve_gib": 24.0,
        "seg_reserve_gib": 6.0,
    },
}


@dataclass(frozen=True)
class JobSpec:
    objective: str
    checkpoint: int
    group: str


@dataclass
class RunningJob:
    spec: JobSpec
    gpu: int
    units: int
    process: subprocess.Popen[str]
    log_handle: TextIO
    started_at: float
    counts_before: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CONFIG, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--training-safe-gpus", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--gpu-capacity", type=int, default=6)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--min-free-gib", type=float, default=100)
    parser.add_argument("--quiesce-legacy", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def process_table() -> list[tuple[int, str, str]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,state=,args="], check=True, capture_output=True, text=True
    )
    rows: list[tuple[int, str, str]] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) == 3:
            rows.append((int(fields[0]), fields[1], fields[2]))
    return rows


def run_roots(model: str) -> dict[str, Path]:
    suffix = MODEL_CONFIG[model]["suffix"]
    root = REPO_ROOT / "outputs/01_training_runs"
    return {
        objective: root / f"{model}_s6recipe_{objective}_{suffix}"
        for objective in OBJECTIVES
    }


def legacy_controller_pids(model: str) -> list[int]:
    roots = run_roots(model)
    pids: list[int] = []
    queue_scripts = (
        "49_queue_s6_eval_saturation.sh",
        "50_queue_s6_segmentation_lane.sh",
        "51_queue_s6_core_lane.sh",
    )
    for pid, _, command in process_table():
        if command.startswith("bash docs/") and any(script in command for script in queue_scripts):
            if f" {model} " in command:
                pids.append(pid)
        elif command.startswith("bash docs/") and "25_watch_b_online_full_eval.sh" in command:
            if any(str(root) in command for root in roots.values()):
                pids.append(pid)
    return sorted(set(pids))


def quiesce_legacy(model: str, status_root: Path) -> list[int]:
    pids = legacy_controller_pids(model)
    stopped: list[int] = []
    commands = {pid: command for pid, _, command in process_table()}
    for pid in pids:
        command = commands.get(pid, "")
        if "49_queue_s6_eval_saturation.sh" not in command and "25_watch_b_online_full_eval.sh" not in command:
            continue
        try:
            os.kill(pid, signal.SIGSTOP)
            stopped.append(pid)
        except ProcessLookupError:
            continue
    atomic_json(
        status_root / "quiesced_legacy.json",
        {"created_at_utc": utc_now(), "model": model, "pids": pids, "stopped_pids": stopped},
    )
    return pids


def fence_legacy_locks(model: str) -> list[TextIO]:
    handles: list[TextIO] = []
    for objective, run_root in run_roots(model).items():
        eval_root = run_root / EVAL_REL
        for checkpoint in CHECKPOINTS:
            for group in ("core", "segmentation"):
                spec = JobSpec(objective, checkpoint, group)
                lock_path = status_prefix(model, spec, eval_root).with_suffix(".lock")
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                handle = lock_path.open("a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                else:
                    handles.append(handle)
    return handles


def active_legacy_work(model: str) -> list[dict[str, object]]:
    roots = run_roots(model)
    root_strings = tuple(str(root) for root in roots.values())
    active: list[dict[str, object]] = []
    for pid, state, command in process_table():
        if state.startswith("T"):
            continue
        is_prefill = "48_prefill_s6_core_detection_once.sh" in command and f" {model} " in command
        is_eval_leaf = any(root in command for root in root_strings) and any(
            token in command
            for token in ("01_eval_splus_sweep.sh", "scripts/run_bio_benchmark_all.sh", "dinov3.eval.bio_")
        )
        if is_prefill or is_eval_leaf:
            active.append({"pid": pid, "state": state, "command": command[:500]})
    return active


def cleanup_quiesced(pids: list[int], model: str) -> None:
    current = {pid: command for pid, _, command in process_table()}
    for pid in pids:
        command = current.get(pid, "")
        is_expected = f" {model} " in command and any(
            token in command
            for token in (
                "49_queue_s6_eval_saturation.sh",
                "50_queue_s6_segmentation_lane.sh",
                "51_queue_s6_core_lane.sh",
            )
        )
        is_expected = is_expected or (
            "25_watch_b_online_full_eval.sh" in command
            and any(str(root) in command for root in run_roots(model).values())
        )
        if not is_expected:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if model == "L":
        for pid, _, command in process_table():
            if command.startswith("bash docs/") and "44_resume_s6_bl_train_then_eval.sh L" in command:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


def training_active(model: str) -> bool:
    roots = run_roots(model)
    for _, _, command in process_table():
        if "dinov3/train/train.py" in command and any(str(root) in command for root in roots.values()):
            return True
    return False


def final_training_complete(model: str) -> bool:
    final_checkpoints = [root / "ckpt/15374/checkpoint.pth" for root in run_roots(model).values()]
    return all(path.is_file() and path.stat().st_size > 0 for path in final_checkpoints)


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def result_counts(inventory: dict[int, dict[str, list[Path]]], checkpoint: int) -> dict[str, int]:
    return {family: len(paths) for family, paths in inventory[checkpoint].items()}


def family_complete(counts: dict[str, int], family: str) -> bool:
    return counts[family] == EXPECTED_RESULTS[family][1]


def checkpoint_complete(counts: dict[str, int]) -> bool:
    return all(family_complete(counts, family) for family in EXPECTED_RESULTS)


def core_complete(counts: dict[str, int]) -> bool:
    return all(family_complete(counts, family) for family in EXPECTED_RESULTS if family != "bio_segmentation")


def status_prefix(model: str, spec: JobSpec, eval_root: Path) -> Path:
    prefix = eval_root / "_prefill_status" / f"{model}_{spec.objective}_ckpt_{spec.checkpoint}"
    if spec.group == "segmentation":
        prefix = prefix.with_name(prefix.name + "_segmentation")
    return prefix


def clear_stale_job_status(model: str, spec: JobSpec, eval_root: Path) -> None:
    prefix = status_prefix(model, spec, eval_root)
    for suffix in (".done", ".failed", ".started"):
        prefix.with_suffix(suffix).unlink(missing_ok=True)


def launch_job(
    model: str,
    spec: JobSpec,
    gpu: int,
    attempt: int,
    counts: dict[str, int],
    log_root: Path,
) -> RunningJob:
    eval_root = run_roots(model)[spec.objective] / EVAL_REL
    clear_stale_job_status(model, spec, eval_root)
    tasks = "segmentation" if spec.group == "segmentation" else "classification regression retrieval detection"
    jobs = "1" if spec.group == "segmentation" else "4"
    units = 1 if spec.group == "segmentation" else 4
    log_path = log_root / f"{spec.objective}_ck{spec.checkpoint}_{spec.group}_gpu{gpu}_try{attempt}.log"
    log_handle = log_path.open("a")
    env = os.environ.copy()
    env.update({"PREFILL_TASKS": tasks, "JOBS_PER_GPU": jobs, "MAX_JOBS": jobs})
    process = subprocess.Popen(
        ["bash", str(PREFILL), model, spec.objective, str(spec.checkpoint), str(gpu)],
        cwd=REPO_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return RunningJob(spec, gpu, units, process, log_handle, time.time(), counts)


def alpha_complete(model: str) -> bool:
    root = REPO_ROOT / "outputs/01_training_runs"
    return all(
        (root / f"{model}_s6recipe_{objective}_alpha_tune_20260804/_status/pipeline.done").is_file()
        for objective in OBJECTIVES
    )


def alpha_active(model: str) -> bool:
    return any(
        "36_queue_s6_ab_alpha.sh" in command and command.rstrip().endswith(f" {model}")
        for _, _, command in process_table()
    )


def main() -> None:
    args = parse_args()
    model = args.model
    config = MODEL_CONFIG[model]
    status_root = REPO_ROOT / f"outputs/00_reports/s6_unified_eval_scheduler_20260809/{model}"
    log_root = REPO_ROOT / f"outputs/auto_eval_logs/unified_s6_{model}_20260809"
    status_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (status_root / "scheduler.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Unified scheduler is already active for {model}")

    (status_root / "takeover.enabled").write_text(utc_now() + "\n")
    fence_handles = fence_legacy_locks(model) if args.quiesce_legacy else []
    quiesced = quiesce_legacy(model, status_root) if args.quiesce_legacy else []
    while True:
        active = active_legacy_work(model)
        atomic_json(
            status_root / "drain_status.json",
            {"updated_at_utc": utc_now(), "active_count": len(active), "active": active},
        )
        if not active:
            break
        print(f"[{utc_now()}] waiting for {len(active)} existing evaluation processes", flush=True)
        time.sleep(args.poll_seconds)
    cleanup_quiesced(quiesced, model)
    for handle in fence_handles:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
    (status_root / "takeover.complete").write_text(utc_now() + "\n")

    running: dict[JobSpec, RunningJob] = {}
    attempts: dict[JobSpec, int] = {}
    retry_after: dict[JobSpec, float] = {}
    alpha_process: subprocess.Popen[str] | None = None
    alpha_log: TextIO | None = None

    while True:
        roots = run_roots(model)
        counts_by_objective: dict[str, dict[int, dict[str, int]]] = {}
        complete_by_objective: dict[str, list[int]] = {}
        for objective, run_root in roots.items():
            eval_root = run_root / EVAL_REL
            inventory = collect_inventory(eval_root, set(CHECKPOINTS))
            counts_by_objective[objective] = {
                checkpoint: result_counts(inventory, checkpoint) for checkpoint in CHECKPOINTS
            }
            complete: list[int] = []
            for checkpoint in CHECKPOINTS:
                counts = counts_by_objective[objective][checkpoint]
                done_marker = eval_root / "_online_status" / f"ckpt_{checkpoint}.done"
                if checkpoint_complete(counts):
                    if mark_complete(eval_root, checkpoint, 10, inventory):
                        complete.append(checkpoint)
                elif done_marker.exists():
                    done_marker.unlink()
            complete_by_objective[objective] = complete

        for spec, job in list(running.items()):
            returncode = job.process.poll()
            if returncode is None:
                continue
            job.log_handle.close()
            del running[spec]
            after = counts_by_objective[spec.objective][spec.checkpoint]
            progressed = sum(after.values()) > sum(job.counts_before.values())
            if returncode != 0 or not progressed:
                delay = min(1800, 120 * attempts.get(spec, 1))
                retry_after[spec] = time.time() + delay
                print(
                    f"[{utc_now()}] job ended rc={returncode} progress={progressed}; "
                    f"retry in {delay}s: {spec}",
                    flush=True,
                )
            else:
                # Defer reconsideration until the next inventory scan sees all files.
                retry_after[spec] = time.time() + args.poll_seconds

        raw_complete = all(len(complete_by_objective[objective]) == len(CHECKPOINTS) for objective in OBJECTIVES)
        if raw_complete and not running:
            if alpha_complete(model):
                (status_root / "scheduler.done").write_text(utc_now() + "\n")
                print(f"[{utc_now()}] raw and alpha evaluation complete for {model}", flush=True)
                return
            if alpha_process is not None and alpha_process.poll() is not None:
                returncode = alpha_process.returncode
                if alpha_log is not None:
                    alpha_log.close()
                alpha_process = None
                alpha_log = None
                if returncode != 0:
                    print(f"[{utc_now()}] alpha queue exited rc={returncode}; will retry", flush=True)
                    time.sleep(60)
            if alpha_process is None and not alpha_active(model):
                alpha_log = (log_root / "alpha_queue.log").open("a")
                env = os.environ.copy()
                env.update(
                    {
                        "PYTHON_BIN": str(config["python"]),
                        "BENCHMARK_ROOT": "/mnt/huawei_deepcad/benchmark",
                        "WEIGHTS_DIR": "/mnt/huawei_deepcad/weights",
                    }
                )
                alpha_process = subprocess.Popen(
                    ["bash", str(ALPHA_QUEUE), model], cwd=REPO_ROOT, env=env,
                    stdout=alpha_log, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                )
                print(f"[{utc_now()}] launched alpha queue pid={alpha_process.pid}", flush=True)

        if model == "B" and (training_active(model) or not final_training_complete(model)):
            allowed_gpus = [gpu for gpu in args.gpus if gpu in args.training_safe_gpus]
        else:
            allowed_gpus = list(args.gpus)

        gpu_units = {gpu: 0 for gpu in allowed_gpus}
        for job in running.values():
            if job.gpu in gpu_units:
                gpu_units[job.gpu] += job.units

        candidates: list[JobSpec] = []
        now = time.time()
        for checkpoint in reversed(CHECKPOINTS):
            for objective in OBJECTIVES:
                checkpoint_file = roots[objective] / f"ckpt/{checkpoint}/checkpoint.pth"
                if not checkpoint_file.is_file() or checkpoint_file.stat().st_size == 0:
                    continue
                counts = counts_by_objective[objective][checkpoint]
                if checkpoint_complete(counts):
                    continue
                for group, complete in (
                    ("core", core_complete(counts)),
                    ("segmentation", family_complete(counts, "bio_segmentation")),
                ):
                    spec = JobSpec(objective, checkpoint, group)
                    if not complete and spec not in running and now >= retry_after.get(spec, 0):
                        candidates.append(spec)

        reserved_gib = 0.0
        for spec in candidates:
            units = 1 if spec.group == "segmentation" else 4
            choices = [
                gpu for gpu, used in gpu_units.items()
                if used + units <= args.gpu_capacity
            ]
            if not choices:
                continue
            reserve = float(config["seg_reserve_gib"] if spec.group == "segmentation" else config["core_reserve_gib"])
            if mem_available_gib() - reserved_gib - reserve < args.min_free_gib:
                break
            gpu = min(choices, key=lambda candidate: (gpu_units[candidate], candidate))
            attempts[spec] = attempts.get(spec, 0) + 1
            running[spec] = launch_job(
                model, spec, gpu, attempts[spec], counts_by_objective[spec.objective][spec.checkpoint], log_root
            )
            gpu_units[gpu] += units
            reserved_gib += reserve
            print(f"[{utc_now()}] launched {spec} gpu={gpu} units={units}", flush=True)

        state = {
            "updated_at_utc": utc_now(),
            "model": model,
            "training_active": training_active(model),
            "final_training_complete": final_training_complete(model),
            "allowed_gpus": allowed_gpus,
            "gpu_units": gpu_units,
            "mem_available_gib": round(mem_available_gib(), 2),
            "complete_checkpoints": complete_by_objective,
            "running": [
                {
                    "objective": job.spec.objective,
                    "checkpoint": job.spec.checkpoint,
                    "group": job.spec.group,
                    "gpu": job.gpu,
                    "pid": job.process.pid,
                    "elapsed_seconds": round(time.time() - job.started_at, 1),
                }
                for job in running.values()
            ],
        }
        atomic_json(status_root / "scheduler_status.json", state)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
