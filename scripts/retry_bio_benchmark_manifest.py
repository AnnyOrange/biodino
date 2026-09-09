#!/usr/bin/env python3
"""Retry only incomplete jobs from a bio_benchmark command manifest.

The original commands, batches, splits, resolutions, and probe settings are
preserved byte-for-byte.  Only GPU assignment and concurrency are changed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import subprocess
import threading
from pathlib import Path


def _valid_json(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return isinstance(value, dict) and not value.get("error")


def _segmentation_results(job: dict, campaign_root: Path) -> list[Path]:
    dataset = job["dataset"]
    checkpoint = job["ckpt_id"]
    return list(campaign_root.glob(f"bio_segmentation/**/{dataset}/{checkpoint}/results.json"))


def successful(job: dict, campaign_root: Path) -> bool:
    output_dir = Path(job["output_dir"])
    task = job["task"]
    if task in {"classification", "regression", "retrieval"}:
        return _valid_json(output_dir / "last_result.json")
    if task == "detection":
        return _valid_json(output_dir / "results_bio_detection.json")
    if task == "segmentation":
        required = 3 if job["dataset"] == "pannuke" else 1
        return sum(_valid_json(path) for path in _segmentation_results(job, campaign_root)) >= required
    raise ValueError(f"Unsupported task in manifest: {task}")


def _bind_gpu(command: list[str], gpu: str) -> list[str]:
    command = list(command)
    if "--gpu" in command:
        index = command.index("--gpu")
        command[index + 1] = gpu
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--report", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    campaign_root = Path(args.campaign_root).resolve()
    jobs = json.loads(Path(args.manifest).read_text())
    pending = [job for job in jobs if not successful(job, campaign_root)]
    slots = [gpu for gpu in args.gpus for _ in range(max(1, args.jobs_per_gpu))]
    condition = threading.Condition()

    def acquire() -> str:
        with condition:
            while not slots:
                condition.wait()
            return slots.pop(0)

    def release(gpu: str) -> None:
        with condition:
            slots.append(gpu)
            condition.notify()

    def run(job: dict) -> dict:
        gpu = acquire()
        try:
            command = _bind_gpu(job["cmd"], gpu)
            output_dir = Path(job["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / f"retry_formal_{job['task']}_{job['dataset']}_{job['ckpt_id']}.log"
            row = {
                "task": job["task"],
                "dataset": job["dataset"],
                "ckpt_id": job["ckpt_id"],
                "gpu": gpu,
                "log": str(log_path),
            }
            if args.dry_run:
                row.update({"status": "DRY_RUN", "command": command})
                return row
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONUNBUFFERED"] = "1"
            environment.setdefault("OMP_NUM_THREADS", "1")
            environment.setdefault("MKL_NUM_THREADS", "1")
            with log_path.open("w") as log:
                log.write("$ " + " ".join(command) + "\n")
                log.flush()
                process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            valid = successful(job, campaign_root)
            row.update({
                "returncode": process.returncode,
                "status": "VALID" if process.returncode == 0 and valid else "FAILED",
                "result_valid": valid,
            })
            return row
        finally:
            release(gpu)

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    initial = {
        "manifest": str(Path(args.manifest).resolve()),
        "campaign_root": str(campaign_root),
        "total_jobs": len(jobs),
        "already_valid": len(jobs) - len(pending),
        "pending": len(pending),
        "gpus": args.gpus,
        "jobs_per_gpu": args.jobs_per_gpu,
        "status": "RUNNING" if not args.dry_run else "DRY_RUN",
        "rows": [],
    }
    report_path.write_text(json.dumps(initial, indent=2) + "\n")

    rows: list[dict] = []
    max_workers = max(1, len(args.gpus) * max(1, args.jobs_per_gpu))
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted = [executor.submit(run, job) for job in pending]
        for future in futures.as_completed(submitted):
            rows.append(future.result())
            initial["rows"] = rows
            report_path.write_text(json.dumps(initial, indent=2) + "\n")

    failures = [row for row in rows if row["status"] == "FAILED"]
    initial.update({"status": "FAILED" if failures else "COMPLETE", "rows": rows})
    report_path.write_text(json.dumps(initial, indent=2) + "\n")
    print(json.dumps({"pending": len(pending), "completed": len(rows), "failed": len(failures)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
