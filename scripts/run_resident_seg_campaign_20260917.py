#!/usr/bin/env python3
"""Git-pinned, resource-limited machine-wide queue for resident dense probes."""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_hs6_checkpoint_queue_20260917 import DATASETS, completed_results, sha256

THREAD_ENV = {key: "1" for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)}
REQUIRED_GATES = (
    "approved_plan", "checkpoint_teacher_and_config", "split_identity_counts_and_leakage",
    "extraction_batch_invariance", "protocol_matrix", "resource_and_storage_budget",
    "legacy_reuse_audit",
)


def verify_launch_gates(config, commit):
    report_path = config.get("preflight_report")
    if not report_path:
        raise RuntimeError("No approved preflight report; synchronize first, do not launch")
    report = json.loads(Path(report_path).read_text())
    if (report.get("status") != "PASS" or report.get("git_commit") != commit
            or not all(report.get("checks", {}).get(key) is True for key in REQUIRED_GATES)):
        raise RuntimeError("Evaluation Rules launch gates have not all passed")
    return report


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def gpu_resources():
    lines = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    return {int(row[0]): dict(zip(("used", "total", "free"), map(int, row[1:])))
            for row in (line.replace(" ", "").split(",") for line in lines)}


def available_ram():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def process_commands():
    commands = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            commands.append(path.read_bytes().replace(b"\0", b" ").decode(errors="replace"))
        except OSError:
            pass
    return commands


def extracting(task, commands):
    # The feature stage owns the expensive encoder; cached probe stages do not.
    return any("bio_segmentation.feature_extractor" in command
               and task["cache_run_name"] in command
               and f"--dataset {task['dataset']} " in command for command in commands)


def admission(config, gpu, active, resources, ram_bytes, disk_bytes, commands):
    card = resources[gpu]
    own = [task for task in active if task["gpu"] == gpu]
    if len(active) >= config["max_host_jobs"] or len(own) >= 3:
        return False
    if ram_bytes < config["minimum_ram_gib"] * 1024**3:
        return False
    if disk_bytes < config["minimum_disk_gib"] * 1024**3:
        return False
    if card["free"] < config["minimum_gpu_free_mib"]:
        return False
    # A busy card gets one additional job, never three speculative encoders.
    if card["used"] / card["total"] >= 0.60 and own:
        return False
    # Probe one encoder at a time per card, especially at MoNuSeg resolution.
    if any(not task.get("probe_seen") or extracting(task, commands) for task in own):
        return False
    if any(task["dataset"] == "monuseg" for task in own):
        # The high-resolution probe peak must be measured without co-residents.
        return False
    return True


def run(config):
    root = Path(config["code_root"])
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    queue = output.parent / "git_queue_20260917"
    queue.mkdir(exist_ok=True)
    lock = (queue / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    commit = git(root, "rev-parse", "HEAD")
    dirty = git(root, "status", "--porcelain", "--untracked-files=no")
    if commit != config["git_commit"] or dirty:
        raise RuntimeError(f"Expected clean commit {config['git_commit']}, got {commit}, dirty={dirty!r}")
    gate_report = verify_launch_gates(config, commit)
    for relative, expected in config["code_sha256"].items():
        if sha256(root / relative) != expected:
            raise RuntimeError(f"Code mismatch: {relative}")
    for relative, expected in config["split_sha256"].items():
        if sha256(Path(config["data_root_base"]) / relative) != expected:
            raise RuntimeError(f"Split mismatch: {relative}")
    os.environ.update(THREAD_ENV)
    import torch
    import sklearn
    tasks = []
    for job in config["jobs"]:
        payload = Path(job["checkpoint_root"]) / str(job["checkpoint_id"]) / "checkpoint.pth"
        before = payload.stat()
        digest = sha256(payload)
        after = payload.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Checkpoint changed: {payload}")
        for dataset in DATASETS:
            task = dict(job, dataset=dataset, status="PENDING", checkpoint_sha256=digest,
                        checkpoint_path=str(payload), checkpoint_size=after.st_size,
                        train_config_sha256=sha256(job["train_config"]),
                        cache_run_name=f"{job['model_name']}_last1_budget20_50_bestval")
            tasks.append(task)
    # Spread encoders over all cards before starting long, large feature banks.
    order = {name: index for index, name in enumerate(DATASETS)}
    tasks.sort(key=lambda task: (order[task["dataset"]], config["jobs"].index(
        next(job for job in config["jobs"] if job["model_name"] == task["model_name"]))))
    manifest_path = output / "campaign_manifest.json"
    manifest = {
        "protocol_id": "seg-probe-budget-fairness-v1", "git_commit": commit,
        "git_status_porcelain": git(root, "status", "--porcelain"),
        "hostname": os.uname().nodename, "gpus": config["gpus"],
        "python": sys.version, "python_executable": sys.executable,
        "torch": torch.__version__, "sklearn": sklearn.__version__,
        "code_sha256": config["code_sha256"], "dataset_root": config["data_root_base"],
        "dataset_split_sha256": config["split_sha256"],
        "command": sys.argv, "environment_overrides": THREAD_ENV,
        "preflight_report": config["preflight_report"], "preflight_checks": gate_report["checks"],
        "probe_batch_size": 32, "num_workers": 2, "budgets": [20, 50], "seeds": [0, 1, 2],
        "legacy_output_root": config["legacy_output_root"],
        "legacy_result_policy": "Preserved, not reused as Git-pinned formal results; audit separately.",
        "reportability": "PENDING_PER_CELL_PROTOCOL_AUDIT",
        "started_at": now(), "status": "RUNNING", "pid": os.getpid(), "tasks": tasks,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["git_commit"] != commit:
            raise RuntimeError("Cannot mix different commits in an output root")
        previous_tasks = {(task["model_name"], task["dataset"]): task for task in previous["tasks"]}
        for task in tasks:
            old = previous_tasks.get((task["model_name"], task["dataset"]), {})
            report = Path(old.get("validation_report", "/nonexistent"))
            if (old.get("status") == "COMPLETE" and report.is_file()
                    and old.get("checkpoint_sha256") == task["checkpoint_sha256"]
                    and old.get("train_config_sha256") == task["train_config_sha256"]):
                results = completed_results(output, task["model_name"], task["checkpoint_id"], (task["dataset"],))
                audit = json.loads(report.read_text())
                if results and audit.get("git_commit") == commit and all(
                        sha256(path) == audit["result_sha256"].get(path) for path in results):
                    task.update(old, reused_from_same_commit=True)
        manifest["resumed_at"] = now()
    active = {}
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save():
        manifest["updated_at"] = now()
        atomic_json(manifest_path, manifest)

    save()
    last_resource_log = 0
    failures = 0
    try:
        while not stopping:
            for key, (process, log, task) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                results = []
                if code == 0:
                    try:
                        results = completed_results(output, task["model_name"], task["checkpoint_id"], (task["dataset"],))
                    except (ValueError, KeyError, TypeError):
                        results = []
                    if not results:
                        code = 1
                task.update(status="COMPLETE" if code == 0 else "INVALID_PROTOCOL", exit_code=code, finished_at=now())
                if results:
                    report = output / "validation_reports" / task["model_name"] / f"{task['dataset']}.json"
                    report.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(report, {"status": "COMPUTE_SCHEMA_VALIDATED", "git_commit": commit,
                        "checkpoint_sha256": task["checkpoint_sha256"],
                        "config_sha256": task["train_config_sha256"],
                        "split_sha256": config["split_sha256"], "at": now(),
                        "reportability": manifest["reportability"],
                        "result_sha256": {path: sha256(path) for path in results}})
                    task.update(validation_report=str(report), result_paths=results)
                del active[key]
                if code:
                    failures += 1
                    manifest["status"] = "PAUSED_ON_ERROR"
                    stopping = True
                save()
            pending = [task for task in tasks if task["status"] == "PENDING"]
            if not pending and not active:
                manifest["status"] = "COMPLETE"
                break
            if stopping:
                break
            resources = gpu_resources()
            commands = process_commands()
            for _, _, task in active.values():
                if any("bio_segmentation.linear_probe" in command and task["cache_run_name"] in command
                       and f"--dataset {task['dataset']} " in command
                       for command in commands):
                    task["probe_seen"] = True
            ram = available_ram()
            disk = shutil.disk_usage(output).free
            if time.monotonic() - last_resource_log >= 1800:
                manifest["resource_snapshot"] = {"at": now(), "gpu_memory_mib": resources,
                    "available_ram_bytes": ram, "free_disk_bytes": disk,
                    "own_worker_pids": [entry[0].pid for entry in active.values()]}
                logging.info("Resources: %s", manifest["resource_snapshot"])
                last_resource_log = time.monotonic()
                save()
            active_tasks = [entry[2] for entry in active.values()]
            candidates = sorted(config["gpus"], key=lambda gpu: (
                sum(task["gpu"] == gpu for task in active_tasks), resources[gpu]["used"]))
            for gpu in candidates:
                if not pending or not admission(config, gpu, active_tasks, resources, ram, disk, commands):
                    continue
                task = pending[0]
                if task["dataset"] == "monuseg" and any(task["gpu"] == gpu for task in active_tasks):
                    continue
                command = [sys.executable, str(root / "scripts/run_seg_probe_budget_dinov3_model_20260915.py"),
                    "--model-name", task["model_name"], "--checkpoint-root", task["checkpoint_root"],
                    "--checkpoint-id", str(task["checkpoint_id"]), "--train-config", task["train_config"],
                    "--checkpoint-sha256", task["checkpoint_sha256"], "--data-root-base", config["data_root_base"],
                    "--cache-root", config["cache_root"], "--output-root", str(output),
                    "--gpu", str(gpu), "--feature-batch-size", str(task["feature_batch_size"]),
                    "--feature-num-workers", "2", "--probe-num-workers", "2", "--datasets", task["dataset"]]
                log_path = queue / f"{task['model_name']}_{task['dataset']}.log"
                log = log_path.open("a")
                env = dict(os.environ, **THREAD_ENV)
                process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                task.update(status="RUNNING", gpu=gpu, pid=process.pid, command=command,
                            environment_overrides=THREAD_ENV, log_path=str(log_path), started_at=now())
                active[f"{task['model_name']}:{task['dataset']}"] = (process, log, task)
                logging.info("START gpu=%s pid=%s %s %s", gpu, process.pid, task["model_name"], task["dataset"])
                save()
                # Re-sample RAM and GPU memory before another admission.
                break
            time.sleep(config.get("poll_seconds", 5))
    finally:
        if stopping and manifest["status"] == "RUNNING":
            manifest["status"] = "PAUSED"
        for process, log, task in active.values():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process, log, task in active.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            log.close()
            task.update(status="INTERRUPTED", finished_at=now())
        save()
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(json.loads(Path(args.campaign_config).read_text()))


if __name__ == "__main__":
    raise SystemExit(main())
