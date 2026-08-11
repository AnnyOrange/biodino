#!/usr/bin/env python3
"""Shared single-GPU queue for the 14-model H+ protocol gap-fill campaign."""

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
EXTERNAL_ROOT = ROOT / "outputs/02_eval_runs/external_fm_fair_protocol_20260721"
RUN_ROOT = ROOT / "outputs/02_eval_runs/external_fm_hplus_protocol_gapfill_20260811"
DEPS_ROOT = Path("/mnt/huawei_deepcad/benchmark_model/_vendor/external_gapfill_py311")
MODELS = (
    "dinov2", "mae", "siglip2", "bioclip", "cytoself", "jump_cp",
    "cytoimagenet", "pe", "uni", "conch", "phikon2", "virchow2",
    "gigapath", "hoptimus0",
)
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


def jobs(python: str) -> list[Job]:
    pending = []
    for model in MODELS:
        batch = str(BATCH_SIZE[model])
        for dataset in ("cellpose", "multimodal_cellseg"):
            result = EXTERNAL_ROOT / "segmentation/linear_probe" / dataset / model / "results.json"
            pending.append(Job(
                f"seg__{model}__{dataset}",
                (
                    python, str(ROOT / "scripts/run_external_fm_segmentation_gapfill.py"),
                    "--model", model, "--dataset", dataset,
                    "--out-root", str(EXTERNAL_ROOT / "segmentation"),
                    "--extract-batch-size", batch, "--num-workers", "0",
                ),
                result,
                "json",
            ))
        detection = RUN_ROOT / "detection" / model / "results_bio_detection.json"
        pending.append(Job(
            f"detection__{model}__livecell",
            (
                python, str(ROOT / "scripts/run_external_fm_livecell_detection.py"),
                "--model", model, "--output-dir", str(detection.parent),
                "--batch-size", batch, "--num-workers", "0",
            ),
            detection,
            "json",
        ))
        retrieval = RUN_ROOT / "retrieval_clustering" / model / "summary.csv"
        pending.append(Job(
            f"retrieval4__{model}",
            (
                python, str(ROOT / "scripts/run_external_fm_hplus_retrieval.py"),
                "--model", model, "--output-dir", str(retrieval.parent),
                "--batch-size", batch, "--num-workers", "0",
            ),
            retrieval,
            "retrieval4",
        ))
    return pending


def successful(job: Job) -> bool:
    if not job.result.exists():
        return False
    try:
        if job.kind == "json":
            payload = json.loads(job.result.read_text())
            return isinstance(payload, dict) and bool(payload)
        with job.result.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = {"lc25000", "nct-crc-he-100", "nct-crc-he-1k", "crc-val-he-7k"}
        done = {row.get("dataset") for row in rows if row.get("dataset") and not row.get("error")}
        return done == expected
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
    parser.add_argument("--python", required=True)
    parser.add_argument("--worker", default=f"{socket.gethostname()}-gpu0")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

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

    completed = failed = 0
    for job in jobs(args.python):
        if successful(job):
            continue
        claim_dir = claim(job, args.worker)
        if claim_dir is None:
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
        payload = {
            "job": job.name,
            "worker": args.worker,
            "returncode": process.returncode,
            "result": str(job.result),
            "result_valid": successful(job),
            "elapsed_seconds": time.time() - started,
            "log": str(log_path),
        }
        if process.returncode == 0 and payload["result_valid"]:
            write_marker("done", job, payload)
            completed += 1
            print(f"[done] {job.name}", flush=True)
        else:
            write_marker("failed", job, payload)
            failed += 1
            print(f"[failed] {job.name} rc={process.returncode}", flush=True)
    print(f"[worker-exit] {args.worker} completed={completed} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
