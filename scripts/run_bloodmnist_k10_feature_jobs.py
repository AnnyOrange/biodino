#!/usr/bin/env python3
"""Extract BloodMNIST frozen features for the accessible 5TB/20TB L checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3")
FIVE_TB_RUNS = {
    "5tb_noGRAM": ROOT / (
        "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_"
        "e15_5tb_mix1m03_5tv107_8x5090zxr_20260907"
    ),
    "5tb_pa005": ROOT / (
        "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_"
        "e15_5tb_pa005_mix1m030_5tv1065_he005_8x5090zxr_20260911"
    ),
}
TWENTY_TB_RUNS = {
    "noGRAM20tb": ROOT / (
        "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_"
        "e61x4098_20tb_route2_mix009_021_0604_0096_8x5090zxr_20260924"
    ),
    "cls_slow2_20tb": ROOT / "outputs/01_training_runs/hs6_l_20tb_v2_recovery_fork36111_20261004/cls_slow2",
}
OUT_ROOT = ROOT / "outputs/02_eval_runs/bloodmnist_k10_cache_20261012"
PYTHON = Path("/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python")
GPU_IDS = (0, 3, 4, 5, 6, 7)


def jobs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for arm, run_root in FIVE_TB_RUNS.items():
        for checkpoint in sorted((run_root / "eval").glob("training_*/teacher_checkpoint.pth"), key=lambda p: int(p.parent.name.split("_")[-1])):
            rows.append(
                {
                    "arm": arm,
                    "checkpoint_id": checkpoint.parent.name.split("_")[-1],
                    "checkpoint": str(checkpoint),
                    "config": str(run_root / "config.yaml"),
                }
            )
    for arm, run_root in TWENTY_TB_RUNS.items():
        for checkpoint in sorted((run_root / "eval").glob("training_*/teacher_checkpoint.pth"), key=lambda p: int(p.parent.name.split("_")[-1])):
            rows.append(
                {
                    "arm": arm,
                    "checkpoint_id": checkpoint.parent.name.split("_")[-1],
                    "checkpoint": str(checkpoint),
                    "config": str(run_root / "config.yaml"),
                }
            )
    unique = {(r["arm"], r["checkpoint_id"]): r for r in rows}
    return sorted(unique.values(), key=lambda r: (r["arm"], int(r["checkpoint_id"])))


def extract(job: dict[str, str], gpu: int, args: argparse.Namespace) -> tuple[str, int, str]:
    arm, ckpt = job["arm"], job["checkpoint_id"]
    out_dir = args.out_root / arm / "bio_classification" / "bloodmnist" / ckpt
    train = list((out_dir / "features/bloodmnist").glob("*_train.npz"))
    test = list((out_dir / "features/bloodmnist").glob("*_test.npz"))
    if train and test and (out_dir / "summary.csv").is_file():
        return arm, int(ckpt), "cached"
    if not Path(job["checkpoint"]).is_file() or not Path(job["config"]).is_file():
        return arm, int(ckpt), "missing-checkpoint-or-config"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{arm}_ck{ckpt}.log"
    env = os.environ.copy()
    env.update(
        {
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
        job["checkpoint"],
        "--train-config",
        job["config"],
        "--benchmark-root",
        str(args.benchmark_root),
        "--datasets",
        "bloodmnist",
        "--output-dir",
        str(out_dir),
        "--model-name",
        f"{arm}_ck{ckpt}",
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
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    status = "ok" if result.returncode == 0 and list((out_dir / "features/bloodmnist").glob("*_train.npz")) and list((out_dir / "features/bloodmnist").glob("*_test.npz")) else f"failed:{result.returncode}"
    return arm, int(ckpt), status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=Path("/mnt/huawei_deepcad/benchmark"))
    parser.add_argument("--workers", type=int, default=len(GPU_IDS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_jobs = jobs()
    args.out_root.mkdir(parents=True, exist_ok=True)
    inventory = args.out_root / "feature_job_inventory.csv"
    with inventory.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "checkpoint_id", "checkpoint", "config"])
        writer.writeheader()
        writer.writerows(all_jobs)

    counts: dict[str, dict[str, int]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract, job, GPU_IDS[i % len(GPU_IDS)], args) for i, job in enumerate(all_jobs)]
        for number, future in enumerate(as_completed(futures), start=1):
            arm, ckpt, status = future.result()
            counts.setdefault(arm, {}).setdefault(status, 0)
            counts[arm][status] += 1
            if number % 5 == 0 or number == len(futures):
                print(f"[{number}/{len(futures)}] {arm} ck{ckpt}: {status}", flush=True)

    summary = args.out_root / "feature_job_summary.json"
    summary.write_text(json.dumps({"counts": counts, "inventory": str(inventory)}, indent=2) + "\n")
    print(json.dumps(counts, indent=2), flush=True)
    return 1 if any(status.startswith("failed:") for arm in counts.values() for status in arm) else 0


if __name__ == "__main__":
    raise SystemExit(main())
