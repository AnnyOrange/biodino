#!/usr/bin/env python3
"""Run the missing Reg4/Det3 external-FM points sequentially on one GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3")
RUN_ROOT = ROOT / "outputs/02_eval_runs/external_fm_expanded_reg4_det3_3090qi_20260826"
DEPS_ROOT = Path("/mnt/huawei_deepcad/benchmark_model/_vendor/external_gapfill_py311")
MODELS = (
    "dinov2", "mae", "siglip2", "bioclip", "cytoself", "jump_cp",
    "cytoimagenet", "pe", "uni", "conch", "phikon2", "virchow2",
    "gigapath", "hoptimus0",
)
REGRESSION_DATASETS = ("conic-cell-count", "livecell-cell-count")
DETECTION_DATASETS = ("bbbc038", "conic")
BATCH_SIZE = {
    "gigapath": 1, "hoptimus0": 1, "virchow2": 2, "phikon2": 4,
    "uni": 4, "siglip2": 4, "conch": 8, "pe": 8, "dinov2": 16,
    "mae": 16, "bioclip": 16, "jump_cp": 16, "cytoself": 16,
    "cytoimagenet": 1,
}


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    result: Path
    kind: str


def jobs(python: str, models: tuple[str, ...]) -> list[Job]:
    pending: list[Job] = []
    for model in models:
        batch = str(BATCH_SIZE[model])
        regression_dir = RUN_ROOT / "regression" / model
        pending.append(Job(
            f"regression2__{model}",
            (
                python, str(ROOT / "scripts/run_external_fm_linear_probe.py"),
                "--model", model, "--datasets", *REGRESSION_DATASETS,
                "--output-dir", str(regression_dir),
                "--feature-root", str(RUN_ROOT / "feature_cache"),
                "--batch-size", batch, "--num-workers", "0",
            ),
            regression_dir / "summary.csv",
            "regression2",
        ))
        for dataset in DETECTION_DATASETS:
            detection_dir = RUN_ROOT / "detection" / model / dataset
            pending.append(Job(
                f"detection__{model}__{dataset}",
                (
                    python, str(ROOT / "scripts/run_external_fm_livecell_detection.py"),
                    "--model", model, "--dataset", dataset,
                    "--output-dir", str(detection_dir),
                    "--batch-size", batch, "--num-workers", "0",
                ),
                detection_dir / "results_bio_detection.json",
                "json",
            ))
    return pending


def successful(job: Job) -> bool:
    if not job.result.exists():
        return False
    try:
        if job.kind == "json":
            payload = json.loads(job.result.read_text())
            return (
                isinstance(payload, dict)
                and payload.get("dataset") in DETECTION_DATASETS
                and "test_patch_f1" in payload
            )
        with job.result.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        done = {
            row.get("dataset") for row in rows
            if row.get("task") == "regression" and not row.get("error")
        }
        return done == set(REGRESSION_DATASETS)
    except Exception:
        return False


def claim(job: Job, worker: str) -> Path | None:
    claim_dir = RUN_ROOT / "state/claims" / job.name
    try:
        claim_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return None
    (claim_dir / "owner.json").write_text(json.dumps({
        "worker": worker,
        "pid": os.getpid(),
        "claimed_at": time.time(),
        "command": list(job.command),
    }, indent=2) + "\n")
    return claim_dir


def write_marker(directory: str, job: Job, payload: dict) -> None:
    path = RUN_ROOT / "state" / directory / f"{job.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="/home/bbnc/venvs/external_fm/bin/python")
    parser.add_argument("--worker", default=f"{socket.gethostname()}-gpu0")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()

    selected_models = tuple(args.models)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    log_root = RUN_ROOT / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "PYTHONUNBUFFERED": "1",
        "KERAS_BACKEND": "torch",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "DENSE_PROBE_METRIC_PYTHON": args.python,
        "EXTERNAL_GAPFILL_DEPS": str(DEPS_ROOT),
        "PYTHONPATH": ":".join((
            "/mnt/huawei_deepcad/benchmark_model/_vendor",
            "/mnt/huawei_deepcad/benchmark_model",
            str(ROOT),
        )),
    })

    completed = failed = skipped = 0
    for job in jobs(args.python, selected_models):
        if successful(job):
            skipped += 1
            continue
        claim_dir = claim(job, args.worker)
        if claim_dir is None:
            skipped += 1
            continue
        started = time.time()
        log_path = log_root / f"{job.name}.log"
        print(f"[claim] {args.worker}: {job.name}", flush=True)
        print(f"[command] {shlex.join(job.command)}", flush=True)
        with log_path.open("a") as log:
            log.write(f"\n$ {shlex.join(job.command)}\n")
            log.flush()
            process = subprocess.run(
                job.command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        valid = successful(job)
        payload = {
            "job": job.name,
            "worker": args.worker,
            "returncode": process.returncode,
            "elapsed_seconds": time.time() - started,
            "result": str(job.result),
            "result_valid": valid,
        }
        if process.returncode == 0 and valid:
            completed += 1
            write_marker("done", job, payload)
            print(f"[done] {job.name}", flush=True)
        else:
            failed += 1
            write_marker("failed", job, payload)
            print(f"[failed] {job.name} returncode={process.returncode}", flush=True)
        (claim_dir / "owner.json").unlink(missing_ok=True)
        claim_dir.rmdir()

    print(
        f"[queue-complete] completed={completed} failed={failed} skipped={skipped}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
