#!/usr/bin/env python3
"""Run a resumable shard of FM14 detection-proxy seed repeats."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODELS = (
    "dinov2", "mae", "siglip2", "bioclip", "cytoself", "jump_cp",
    "cytoimagenet", "pe", "uni", "conch", "phikon2", "virchow2",
    "gigapath", "hoptimus0",
)
DATASETS = ("bbbc038", "conic", "livecell")
ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python")


def free_memory_mib(gpu_index: int) -> int:
    output = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    if len(output) != 1:
        raise RuntimeError(f"GPU {gpu_index} did not return one memory value")
    return int(output[0].strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/02_eval_runs/fm14_detection_multiseed_20261016")
    parser.add_argument("--min-free-mib", type=int, default=9000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("shard must be between 0 and shards - 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
        parser.error("CUDA_VISIBLE_DEVICES must match --gpu-index")
    jobs = [(model, dataset, seed) for model in MODELS for dataset in DATASETS for seed in (1, 2, 3, 4)]
    failures = []
    for index, (model, dataset, seed) in enumerate(jobs):
        if index % args.shards != args.shard:
            continue
        output = args.output_root / model / dataset / f"seed{seed}"
        if (output / "component_result.json").exists():
            print(f"[skip] {model}/{dataset}/seed{seed}", flush=True)
            continue
        while free_memory_mib(args.gpu_index) < args.min_free_mib:
            print(f"[wait] {model}/{dataset}/seed{seed}: insufficient free GPU memory", flush=True)
            time.sleep(args.poll_seconds)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env["PYTHONPATH"] = ":".join((
            "/mnt/huawei_deepcad/benchmark_model/_vendor",
            "/mnt/huawei_deepcad/benchmark_model",
            str(ROOT),
        ))
        command = [
            str(PYTHON), "-u", str(ROOT / "scripts/run_fm14_detection_multiseed_20261016.py"),
            "--model", model, "--dataset", dataset, "--seed", str(seed),
            "--output-root", str(args.output_root),
        ]
        print(f"[run] {model}/{dataset}/seed{seed}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if completed.returncode:
            failures.append(dict(model=model, dataset=dataset, seed=seed, exit_code=completed.returncode))
            print(f"[failed] {model}/{dataset}/seed{seed}: {completed.returncode}", flush=True)
    status = args.output_root / f"shard{args.shard}_status.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps(dict(shard=args.shard, shards=args.shards, failures=failures), indent=2) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
