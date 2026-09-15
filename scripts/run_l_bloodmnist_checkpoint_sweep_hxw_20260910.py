#!/usr/bin/env python3
"""Evaluate BloodMNIST full/5-shot/10-shot across the two L e4 runs and L e8."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from run_hs6_kshot_from_cache import find_feature_pair, kshot_indices, load_npz_xy, probe

CODE = Path("/home/xzj/biodino")
PYTHON = "/home/xzj/miniconda3/envs/dinov3/bin/python"
BENCH = "/mnt/data/benchmark"
RUN_ROOT = Path("/mnt/data/biodino_fixed_pass/outputs/01_training_runs")
LOG_ROOT = Path("/mnt/data/biodino_fixed_pass/outputs/auto_eval_logs/l_blood_ckpt_sweep_20260910")
EVAL_TAG = "bloodmnist_ckpt_sweep_20260910"
GPUS = (4, 5, 6, 7)

RUNS = (
    (
        "L-original-e4-lr1e4",
        "HS6_Cscale_L_robust_biosafe256_gb1024_lr1e4_prop15_nosig_e4_random100_seed0_8x5090hxw_20260903",
        ((4, 4099),),
    ),
    (
        "L-retrained-e4-lr5e5",
        "HS6_Cscale_L_robust_biosafe256_gb1024_lr5e5_prop15_nosig_e4_random100_seed0_4x5090hxw_20260909_tunedlr",
        ((1, 1024), (2, 2049), (3, 3074), (4, 4099)),
    ),
    (
        "L-e8-lr1e4",
        "HS6_Cscale_L_robust_biosafe256_gb1024_lr1e4_prop15_nosig_e8_random100_seed0_4x5090hxw_20260908_epckpt",
        ((1, 1024), (2, 2049), (3, 3074), (4, 4099),
         (5, 5124), (6, 6149), (7, 7174)),
    ),
)


def jobs() -> list[dict]:
    out = []
    for label, dirname, checkpoints in RUNS:
        run = RUN_ROOT / dirname
        for epoch, ckpt in checkpoints:
            checkpoint = run / "ckpt" / str(ckpt) / "checkpoint.pth"
            config = run / "config.yaml"
            if not checkpoint.is_file() or checkpoint.stat().st_size < 10_000_000:
                raise FileNotFoundError(f"invalid checkpoint: {checkpoint}")
            out_dir = run / "eval" / EVAL_TAG / "bio_classification" / "bloodmnist" / str(ckpt)
            out.append({
                "label": label, "run": run, "epoch": epoch, "ckpt": ckpt,
                "checkpoint": checkpoint, "config": config, "out_dir": out_dir,
                "eval_dir": run / "eval" / EVAL_TAG,
            })
    return out


def launch(job: dict, gpu: int) -> tuple[subprocess.Popen, object]:
    job["out_dir"].mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f'{job["label"]}_e{job["epoch"]}_{job["ckpt"]}.log'
    handle = log_path.open("w")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTHONPATH": str(CODE), "OMP_NUM_THREADS": "4"})
    cmd = [
        PYTHON, "-m", "dinov3.eval.bio_frozen_eval.run_classification",
        "--checkpoint", str(job["checkpoint"]), "--train-config", str(job["config"]),
        "--benchmark-root", BENCH, "--datasets", "bloodmnist",
        "--output-dir", str(job["out_dir"]), "--model-name", f'dinov3-{job["ckpt"]}',
        "--resolution-protocol", "best", "--image-size", "224", "--batch-size", "64",
        "--num-workers", "1", "--channel-policy", "auto", "--split-protocol", "current",
        "--autocast-dtype", "bf16", "--overwrite-results",
    ]
    print(f'START gpu={gpu} {job["label"]} e{job["epoch"]} ckpt={job["ckpt"]}', flush=True)
    return subprocess.Popen(cmd, cwd=CODE, env=env, stdout=handle, stderr=subprocess.STDOUT), handle


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    all_jobs = jobs()
    running = []
    for index, job in enumerate(all_jobs):
        gpu = GPUS[index % len(GPUS)]
        proc, handle = launch(job, gpu)
        running.append((proc, handle, job, gpu))

    failures = []
    while running:
        for item in list(running):
            proc, handle, job, gpu = item
            rc = proc.poll()
            if rc is None:
                continue
            handle.close()
            running.remove(item)
            print(f'END rc={rc} gpu={gpu} {job["label"]} e{job["epoch"]}', flush=True)
            if rc:
                failures.append((job["label"], job["epoch"], rc))
        if running:
            time.sleep(5)

    fields = ["run", "epoch", "ckpt", "metric", "seed", "value", "source"]
    rows = []
    for job in all_jobs:
        result_path = job["out_dir"] / "last_result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        if result.get("error") or "macro_f1" not in result:
            continue
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            rows.append({"run": job["label"], "epoch": job["epoch"], "ckpt": job["ckpt"],
                         "metric": f"full_{metric}", "seed": 0, "value": result[metric],
                         "source": result_path})
        train_path, test_path, _ = find_feature_pair(job["eval_dir"], "bloodmnist", job["ckpt"])
        if train_path is None or test_path is None:
            continue
        x_train, y_train = load_npz_xy(train_path)
        x_test, y_test = load_npz_xy(test_path)
        for k in (5, 10):
            for seed in (0, 1, 2):
                selected = kshot_indices(np.asarray(y_train).reshape(-1), k, seed)
                metrics = probe(x_train[selected], y_train[selected], x_test, y_test)
                for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                    rows.append({"run": job["label"], "epoch": job["epoch"], "ckpt": job["ckpt"],
                                 "metric": f"k{k}_{metric}", "seed": seed,
                                 "value": metrics[metric], "source": train_path})

    csv_path = LOG_ROOT / "bloodmnist_checkpoint_sweep.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (LOG_ROOT / "status.json").write_text(json.dumps({"jobs": len(all_jobs), "failures": failures,
                                                       "rows": len(rows)}, indent=2))
    print(f'FINISHED jobs={len(all_jobs)} failures={failures} rows={len(rows)} csv={csv_path}', flush=True)


if __name__ == "__main__":
    main()
