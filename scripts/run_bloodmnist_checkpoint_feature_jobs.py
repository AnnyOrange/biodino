#!/usr/bin/env python3
"""Extract BloodMNIST features for checkpoint families on a remote host."""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--arm-prefix", default="")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value]
    jobs = []
    for checkpoint in args.run_root.glob("*/eval/training_*/teacher_checkpoint.pth"):
        arm = checkpoint.parents[2].name
        if args.arm_prefix and not arm.startswith(args.arm_prefix):
            continue
        jobs.append((arm, checkpoint.parent.name.removeprefix("training_"), checkpoint))
    jobs.sort(key=lambda row: (row[0], int(row[1])))
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    def extract(job: tuple[str, str, Path], gpu: int) -> tuple[str, str, str]:
        arm, ckpt_id, checkpoint = job
        run_root = checkpoint.parents[2]
        out_dir = args.output_root / arm / "bio_classification" / "bloodmnist" / ckpt_id
        features = out_dir / "features" / "bloodmnist"
        if list(features.glob("*_train.npz")) and list(features.glob("*_test.npz")) and (out_dir / "summary.csv").is_file():
            return arm, ckpt_id, "cached"
        out_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(args.repo_root),
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        command = [
            str(args.python),
            "-m",
            "dinov3.eval.bio_frozen_eval.run_classification",
            "--checkpoint",
            str(checkpoint),
            "--train-config",
            str(run_root / "config.yaml"),
            "--benchmark-root",
            str(args.benchmark_root),
            "--datasets",
            "bloodmnist",
            "--output-dir",
            str(out_dir),
            "--model-name",
            f"{arm}_ck{ckpt_id}",
            "--device",
            "cuda:0",
            "--batch-size",
            "64",
            "--num-workers",
            "2",
            "--seed",
            "0",
            "--autocast-dtype",
            "bf16",
            "--n-last-blocks",
            "1",
            "--channel-policy",
            "auto",
            "--channel-tta-samples",
            "8",
            "--resolution-protocol",
            "best",
        ]
        with (log_root / f"{arm}_ck{ckpt_id}.log").open("w") as log:
            result = subprocess.run(command, cwd=args.repo_root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        complete = list(features.glob("*_train.npz")) and list(features.glob("*_test.npz")) and (out_dir / "summary.csv").is_file()
        return arm, ckpt_id, "ok" if result.returncode == 0 and complete else f"failed:{result.returncode}"

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(extract, job, gpus[index % len(gpus)]) for index, job in enumerate(jobs)]
        for number, future in enumerate(as_completed(futures), 1):
            arm, ckpt_id, status = future.result()
            counts[status] = counts.get(status, 0) + 1
            if number % 5 == 0 or number == len(futures):
                print(f"[{number}/{len(futures)}] {arm} ck{ckpt_id}: {status}", flush=True)
    print(f"jobs={len(jobs)} counts={counts}", flush=True)
    return int(any(status.startswith("failed:") for status in counts))


if __name__ == "__main__":
    raise SystemExit(main())
