#!/usr/bin/env python3
"""Launch the 7-model frozen classification/regression retest on all visible GPUs.

This runs both:
  1. in-repo `dinov3.eval.bio_frozen_eval.run_classification`
  2. `/mnt/huawei_deepcad/benchmark_model/run_dinov3_ckpt_benchmark.py`

with the new split protocol and the 2026-06-23 classification resolution table.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


PYTHON_DEFAULT = "/home/deepcad/anaconda3/envs/dinov3/bin/python"
BENCHMARK_ROOT = "/mnt/huawei_deepcad/benchmark"
BENCHMARK_MODEL_SCRIPT = "/mnt/huawei_deepcad/benchmark_model/run_dinov3_ckpt_benchmark.py"

DEFAULT_DATASETS = [
    # Five datasets with the 2026-06-23 dualroute ep15all resolution sweep.
    "bloodmnist",
    "bbbc048-cellcycle",
    "cyclops-protein-loc",
    "midog25-atypical",
    "chestmnist",
    # Group-aware regression/classification splits.
    "bbbc005",
    "bbbc013",
    # Remaining MedMNIST official train/test splits.
    "breastmnist",
    "dermamnist",
    "octmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "pathmnist",
    "pneumoniamnist",
    "retinamnist",
    "tissuemnist",
    # Added native train/test pathology datasets.
    "nct-crc-he",
    "pcam",
]


@dataclass(frozen=True)
class ModelJob:
    label: str
    ckpt_root: str
    ckpt_iter: int
    config: str
    batch_size: int


MODELS = [
    ModelJob(
        "sp",
        "outputs/bio_continue_vits16_ep15_1025/ckpt",
        15374,
        "outputs/bio_continue_vits16_ep15_1025/config.yaml",
        128,
    ),
    ModelJob(
        "b",
        "outputs/bio_continue_1025_a100_grad_acc_2_base/ckpt",
        16399,
        "outputs/bio_continue_1025_a100_grad_acc_2_base/config.yaml",
        96,
    ),
    ModelJob(
        "l",
        "outputs/bio_continue_vitL16_OEP1025_ep15_b1024_1025/ckpt",
        15374,
        "outputs/bio_continue_vitL16_OEP1025_ep15_b1024_1025/config.yaml",
        32,
    ),
    ModelJob(
        "hplus",
        "outputs/_hplus_consol/ckpt",
        14349,
        "outputs/bio_continue_rgb3_vith16plus/config.yaml",
        16,
    ),
    ModelJob(
        "robust4",
        "outputs/bio_continue_vitl16_robust/ckpt",
        15374,
        "outputs/bio_continue_vitl16_robust/config.yaml",
        32,
    ),
    ModelJob(
        "gram5",
        "outputs/bio_continue_vitl16_robust_hires_gram_1024/ckpt",
        3074,
        "outputs/bio_continue_vitl16_robust_hires_gram_1024/config.yaml",
        32,
    ),
    ModelJob(
        "dualroute",
        "outputs/bio_continue_vitl16_dualroute_ep15_all/ckpt",
        15374,
        "outputs/bio_continue_vitl16_dualroute_ep15_all/config.yaml",
        32,
    ),
]


@dataclass
class LaunchJob:
    mode: str
    model: ModelJob
    output_dir: str
    log_path: str
    cmd: list[str]


def checkpoint_path(model: ModelJob) -> str:
    return str(Path(model.ckpt_root) / str(model.ckpt_iter) / "checkpoint.pth")


def build_jobs(args: argparse.Namespace) -> list[LaunchJob]:
    datasets = args.datasets or DEFAULT_DATASETS
    jobs: list[LaunchJob] = []

    inrepo_root = Path(args.inrepo_output_root)
    bm_root = Path(args.benchmark_model_output_root)
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        if args.mode in {"both", "inrepo"}:
            out_dir = inrepo_root / model.label
            cmd = [
                args.python,
                "-m",
                "dinov3.eval.bio_frozen_eval.run_classification",
                "--checkpoint",
                checkpoint_path(model),
                "--train-config",
                model.config,
                "--benchmark-root",
                args.benchmark_root,
                "--datasets",
                *datasets,
                "--output-dir",
                str(out_dir),
                "--model-name",
                f"{model.label}-{model.ckpt_iter}",
                "--batch-size",
                str(model.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--resolution-protocol",
                args.resolution_protocol,
                "--image-size",
                str(args.image_size),
                "--autocast-dtype",
                args.autocast_dtype,
            ]
            if args.overwrite_results:
                cmd.append("--overwrite-results")
            if args.overwrite_features:
                cmd.append("--overwrite-features")
            if args.no_save_features:
                cmd.append("--no-save-features")
            jobs.append(
                LaunchJob(
                    mode="inrepo",
                    model=model,
                    output_dir=str(out_dir),
                    log_path=str(log_root / f"inrepo_{model.label}.log"),
                    cmd=cmd,
                )
            )

        if args.mode in {"both", "benchmark_model"}:
            out_dir = bm_root / model.label
            cmd = [
                args.python,
                args.benchmark_model_script,
                "--ckpt-root",
                model.ckpt_root,
                "--ckpt-iters",
                str(model.ckpt_iter),
                "--train-config",
                model.config,
                "--benchmark-root",
                args.benchmark_root,
                "--datasets",
                *datasets,
                "--output-dir",
                str(out_dir),
                "--model-prefix",
                model.label,
                "--batch-size",
                str(model.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--resolution-protocol",
                args.resolution_protocol,
                "--image-size",
                str(args.image_size),
                "--autocast-dtype",
                args.autocast_dtype,
            ]
            if args.overwrite_results:
                cmd.append("--overwrite-results")
            if args.overwrite_features:
                cmd.append("--overwrite-features")
            if args.no_save_features:
                cmd.append("--no-save-features")
            jobs.append(
                LaunchJob(
                    mode="benchmark_model",
                    model=model,
                    output_dir=str(out_dir),
                    log_path=str(log_root / f"benchmark_model_{model.label}.log"),
                    cmd=cmd,
                )
            )

    return jobs


def worker(gpu: str, jobs_q: queue.Queue[LaunchJob], results: list[dict], dry_run: bool) -> None:
    while True:
        try:
            job = jobs_q.get_nowait()
        except queue.Empty:
            return
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        Path(job.output_dir).mkdir(parents=True, exist_ok=True)
        Path(job.log_path).parent.mkdir(parents=True, exist_ok=True)
        cmd_str = shlex.join(job.cmd)
        print(f"[gpu {gpu}] {job.mode}/{job.model.label}: {cmd_str}", flush=True)
        if dry_run:
            code = 0
            with open(job.log_path, "w") as f:
                f.write("$ " + cmd_str + "\n")
        else:
            with open(job.log_path, "w") as f:
                f.write(f"[gpu {gpu}] $ {cmd_str}\n")
                f.flush()
                proc = subprocess.run(job.cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
                code = int(proc.returncode)
        results.append({**asdict(job), "gpu": gpu, "returncode": code})
        print(f"[gpu {gpu}] done {job.mode}/{job.model.label} returncode={code}", flush=True)
        jobs_q.task_done()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["both", "inrepo", "benchmark_model"], default="both")
    parser.add_argument("--python", default=PYTHON_DEFAULT)
    parser.add_argument("--benchmark-root", default=BENCHMARK_ROOT)
    parser.add_argument("--benchmark-model-script", default=BENCHMARK_MODEL_SCRIPT)
    parser.add_argument("--inrepo-output-root", default="outputs/classification_splitres_7models_20260624/inrepo")
    parser.add_argument(
        "--benchmark-model-output-root",
        default="/mnt/huawei_deepcad/benchmark_model/benchmark_runs/dinov3_splitres_7models_20260624",
    )
    parser.add_argument("--log-root", default="outputs/classification_splitres_7models_20260624/logs")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--resolution-protocol", choices=["manual", "best"], default="best")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--autocast-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--no-save-features", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("--gpus produced no GPU ids")

    manifest_path = Path(args.log_root) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps([asdict(j) for j in jobs], indent=2))
    print(f"[plan] jobs={len(jobs)} gpus={gpus} manifest={manifest_path}", flush=True)

    jobs_q: queue.Queue[LaunchJob] = queue.Queue()
    for job in jobs:
        jobs_q.put(job)

    results: list[dict] = []
    threads = [threading.Thread(target=worker, args=(gpu, jobs_q, results, args.dry_run)) for gpu in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results_path = Path(args.log_root) / "launcher_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    failed = [r for r in results if r["returncode"] != 0]
    print(f"[done] jobs={len(results)} failed={len(failed)} results={results_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
