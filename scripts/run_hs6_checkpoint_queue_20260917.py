#!/usr/bin/env python3
"""Resident-checkpoint queue for the unchanged segmentation probe protocol."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DATASETS = ("monuseg", "conic", "livecell", "multimodal_cellseg", "cellpose", "tissuenet", "pannuke", "bbbc038")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_results(output_root, model, checkpoint_id, datasets=DATASETS):
    results = []
    for dataset in datasets:
        paths = sorted(Path(output_root).glob(
            f"{model}_last1_budget20_50_bestval*/budget*/seed*/{dataset}/{checkpoint_id}/results.json"
        ))
        if len(paths) != (18 if dataset == "pannuke" else 6):
            return []
        signatures = set()
        for path in paths:
            data = json.loads(path.read_text())
            meta = data.get("_meta", {})
            budget = int(path.parts[-5].removeprefix("budget"))
            seed = int(path.parts[-4].removeprefix("seed"))
            history = meta.get("validation_history", [])
            if not (
                budget in (20, 50) and seed in (0, 1, 2)
                and meta.get("probe_epochs") == budget and meta.get("seed") == seed
                and meta.get("probe_batch_size") == 32 and meta.get("probe_eval_every") == 1
                and len(history) == budget and meta.get("test_evaluations") == 1
                and meta.get("optimizer") == "AdamW" and meta.get("scheduler") == "CosineAnnealingLR"
                and meta.get("scheduler_t_max") == budget and meta.get("learning_rate") == 0.001
                and meta.get("weight_decay") == 0.0001 and meta.get("dropout") == 0.1
                and meta.get("selection_metric") == "val_mIoU"
                and meta.get("class_weight_mode") == ("sqrt_inverse" if dataset == "conic" else "none")
                and "test" in data
            ):
                return []
            best = max(history, key=lambda entry: entry["mIoU"])
            if best["epoch"] != meta.get("best_epoch") or abs(best["mIoU"] - meta.get("best_val_miou", -1)) > 1e-12:
                return []
            signatures.add((path.parts[-6], budget, seed))
        if len(signatures) != len(paths):
            return []
        results.extend(map(str, paths))
    return results


def wait_for_resources(config, state, save):
    while True:
        output = subprocess.check_output([
            "nvidia-smi", f"--id={config['gpu']}",
            "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits",
        ], text=True)
        used, free, utilization = [int(value.strip()) for value in output.strip().split(",")]
        disk_free = shutil.disk_usage(config["output_root"]).free
        available = free >= config["minimum_gpu_free_mib"] and disk_free >= 64 * 1024**3
        if config["exclusive_gpu"]:
            available = available and used < 1024 and utilization < 10
        if available:
            return
        state.update(status="WAITING_FOR_RESOURCES", gpu_memory_used_mib=used,
                     gpu_memory_free_mib=free, gpu_utilization=utilization, disk_free_bytes=disk_free)
        save()
        logging.info("Waiting: gpu=%s used=%s free=%s utilization=%s disk_free_GiB=%.1f exclusive=%s",
                     config["gpu"], used, free, utilization, disk_free / 1024**3, config["exclusive_gpu"])
        time.sleep(30)


def run(config):
    code_root = Path(config["code_root"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    queue_dir = output_root.parent / "queue_20260917"
    queue_dir.mkdir(parents=True, exist_ok=True)
    name = config["queue_name"]
    state_path = queue_dir / f"{name}.state.json"
    lock_handle = (queue_dir / f"{name}.lock").open("a")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Queue {name} is already running") from None
    state = {"queue_name": name, "pid": os.getpid(), "host": os.uname().nodename,
             "gpu": config["gpu"], "started_at": now(), "status": "PREFLIGHT", "jobs": []}

    def save():
        state["updated_at"] = now()
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    save()
    try:
        for relative, expected in config["code_sha256"].items():
            if sha256(code_root / relative) != expected:
                raise RuntimeError(f"Frozen code mismatch: {relative}")
        for relative, expected in config["split_sha256"].items():
            if sha256(Path(config["data_root_base"]) / relative) != expected:
                raise RuntimeError(f"Split manifest mismatch: {relative}")
        state["code_sha256"] = config["code_sha256"]
        state["split_sha256"] = config["split_sha256"]
        failures = 0
        for planned in config["jobs"]:
            job = {**planned, "status": "PENDING", "started_at": now()}
            state["jobs"].append(job)
            save()
            try:
                wait_for_resources(config, state, save)
                state["status"] = "RUNNING"
                job["status"] = "HASHING_CHECKPOINT"
                save()
                payload = Path(job["checkpoint_root"]) / str(job["checkpoint_id"]) / "checkpoint.pth"
                before = payload.stat()
                checkpoint_sha = sha256(payload)
                after = payload.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Checkpoint changed during hashing: {payload}")
                job.update(checkpoint_sha256=checkpoint_sha, checkpoint_path=str(payload),
                           checkpoint_size=after.st_size, train_config_sha256=sha256(job["train_config"]))
                results = completed_results(output_root, job["model_name"], job["checkpoint_id"])
                if results:
                    job.update(status="COMPLETE_VALIDATED_REUSED", validated_results=results, finished_at=now())
                    save()
                    failures = 0
                    continue
                command = [
                    sys.executable, str(code_root / "scripts/run_seg_probe_budget_dinov3_model_20260915.py"),
                    "--model-name", job["model_name"], "--checkpoint-root", job["checkpoint_root"],
                    "--checkpoint-id", str(job["checkpoint_id"]), "--train-config", job["train_config"],
                    "--checkpoint-sha256", checkpoint_sha, "--data-root-base", config["data_root_base"],
                    "--cache-root", config["cache_root"], "--output-root", str(output_root),
                    "--gpu", str(config["gpu"]), "--feature-batch-size", str(job["feature_batch_size"]),
                    "--datasets", *DATASETS,
                ]
                log_dir = queue_dir / "checkpoint_logs"
                log_dir.mkdir(exist_ok=True)
                job_log = log_dir / f"{job['model_name']}.log"
                job.update(status="RUNNING", command=command, log_path=str(job_log))
                save()
                logging.info("Starting %s ck=%s gpu=%s log=%s", job["model_name"], job["checkpoint_id"], config["gpu"], job_log)
                with job_log.open("a") as log:
                    subprocess.run(command, cwd=code_root, stdout=log, stderr=subprocess.STDOUT, check=True)
                results = completed_results(output_root, job["model_name"], job["checkpoint_id"])
                if len(results) != 60:
                    raise RuntimeError(f"Expected 60 validated fits: {job['model_name']}")
                job.update(status="COMPLETE_VALIDATED", validated_results=results, finished_at=now())
                failures = 0
            except Exception as error:
                job.update(status="FAILED", error=f"{type(error).__name__}: {error}", finished_at=now())
                failures += 1
                logging.exception("Checkpoint failed: %s", job["model_name"])
                if failures >= 2:
                    state["status"] = "PAUSED_AFTER_TWO_CONSECUTIVE_FAILURES"
                    save()
                    return 1
            save()
        state["status"] = "COMPLETE_WITH_FAILURES" if any(job["status"] == "FAILED" for job in state["jobs"]) else "COMPLETE_VALIDATED"
        state["finished_at"] = now()
        save()
        return int(state["status"] != "COMPLETE_VALIDATED")
    except Exception as error:
        state.update(status="PREFLIGHT_FAILED", error=f"{type(error).__name__}: {error}")
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(json.loads(Path(args.queue_config).read_text()))


if __name__ == "__main__":
    raise SystemExit(main())
