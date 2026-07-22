#!/usr/bin/env python3
"""Refit S+ segmentation heads with one deterministic probe protocol."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/02_eval_runs/splus_segmentation_probe_audit_20260716"
DATASETS = (
    "bbbc038",
    "cellpose",
    "livecell",
    "tissuenet",
    "monuseg",
    "pannuke",
    "conic",
    "multimodal_cellseg",
)


@dataclass(frozen=True)
class ModelSource:
    label: str
    checkpoint: str
    passes: int
    roots: tuple[Path, ...]


def eval_root(name: str) -> Path:
    return ROOT / "outputs/02_eval_runs" / name


MODELS = OrderedDict(
    [
        (
            "official",
            ModelSource(
                "Official S+",
                "0",
                0,
                (eval_root("splus_random_data_scaling_seg_ood/official_0"),),
            ),
        ),
        (
            "h_s0",
            ModelSource(
                "H-S0 packwds GB1024",
                "15374",
                15,
                (eval_root("bioseg_best_7models_20260622/sp"),),
            ),
        ),
        (
            "r_s0",
            ModelSource(
                "R-S0 packwds GB4096",
                "3899",
                15,
                (eval_root("S0b_packwds_dino256_b4096_lr2e-4_wu2_e15__S0_e15_seg_all_fill3090_singlecard_20260712_1616"),),
            ),
        ),
        (
            "s1",
            ModelSource(
                "S1 robust+DINO",
                "3899",
                15,
                (eval_root("S1b_robust_dino256_b4096_lr2e-4_wu2_e15__S1_e15_seg_all_fill3090_singlecard_20260712_1616"),),
            ),
        ),
        (
            "s2_wu2",
            ModelSource(
                "S2 robust+BioAug wu2",
                "3899",
                15,
                (eval_root("S2b_robust_biosafe256_b4096_lr2e-4_wu2_e15__S2WU2_e15_seg_all_fill3090_singlecard_20260712_1616"),),
            ),
        ),
        (
            "s2_wu5",
            ModelSource(
                "S2 robust+BioAug wu5",
                "3899",
                15,
                (eval_root("S2b_robust_biosafe256_b4096_lr2e-4_wu5_e15__S2WU5_e15_seg_all_fill3090_singlecard_20260712_1616"),),
            ),
        ),
        (
            "s3",
            ModelSource(
                "S3 BioAug crop224",
                "3899",
                15,
                (eval_root("S3b_robust_biosafe224_b4096_lr2e-4_wu2_e15__S3_e15_seg_all_fill3090_singlecard_20260712_1616"),),
            ),
        ),
        (
            "s6_ck3899",
            ModelSource(
                "S6 robust+BioAug ck3899",
                "3899",
                15,
                (
                    eval_root("S6b_rgb_robust_biosafe256_b4096_lr2e-4_wu2_e30__ckpt3899_dense"),
                    eval_root("S6b_rgb_robust_biosafe256_b4096_lr2e-4_wu2_e30__ckpt3899_5199_6499_7799_seg_monuseg_cellpose_retry_cpu12"),
                    eval_root("S6b_rgb_robust_biosafe256_b4096_lr2e-4_wu2_e30__ckpt3899_5199_6499_7799_seg_multimodal_livecell_deepcad7"),
                ),
            ),
        ),
        (
            "s6_ck5199",
            ModelSource(
                "S6 robust+BioAug ck5199",
                "5199",
                20,
                (eval_root("S6b_rgb_robust_biosafe256_b4096_lr2e-4_wu2_e30__S6_ck5199_seg_all_fill3090_singlecard_20260712_1616"),),
            ),
        ),
    ]
)


@dataclass
class Job:
    model_key: str
    model: ModelSource
    dataset: str
    seed: int
    train_cache: Path
    val_cache: Path
    test_cache: Path
    output_dir: Path


def find_cache(model: ModelSource, dataset: str, split: str) -> Path:
    matches = []
    for source_root in model.roots:
        pattern = f"**/{dataset}/{model.checkpoint}/{dataset}_{split}_*.npz"
        matches.extend(source_root.glob(pattern))
    matches = sorted(set(path.resolve() for path in matches))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {split} cache for {model.label}/{dataset}/{model.checkpoint}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def result_is_complete(path: Path, seed: int, batch_size: int, epochs: int) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
        meta = result["_meta"]
        return (
            "test" in result
            and meta.get("probe_rng_seeded") is True
            and int(meta.get("seed")) == seed
            and int(meta.get("probe_batch_size")) == batch_size
            and int(meta.get("probe_epochs")) == epochs
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_jobs(output_root: Path, seeds: list[int]) -> list[Job]:
    jobs = []
    for model_key, model in MODELS.items():
        for seed in seeds:
            for dataset in DATASETS:
                # The historical H-S0 sweep predates Multimodal CellSeg.
                if model_key == "h_s0" and dataset == "multimodal_cellseg":
                    continue
                jobs.append(
                    Job(
                        model_key=model_key,
                        model=model,
                        dataset=dataset,
                        seed=seed,
                        train_cache=find_cache(model, dataset, "train"),
                        val_cache=find_cache(model, dataset, "val"),
                        test_cache=find_cache(model, dataset, "test"),
                        output_dir=output_root / model_key / f"seed{seed}" / dataset,
                    )
                )
    return jobs


def command(job: Job, batch_size: int, epochs: int, num_workers: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "dinov3.eval.bio_segmentation.linear_probe",
        "--dataset",
        job.dataset,
        "--use-cached-features",
        "--train-cache",
        str(job.train_cache),
        "--val-cache",
        str(job.val_cache),
        "--test-cache",
        str(job.test_cache),
        "--output-dir",
        str(job.output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--eval-every",
        str(epochs),
        "--seed",
        str(job.seed),
        "--semantic-only",
    ]
    if job.dataset == "conic":
        cmd.extend(["--class-weight-mode", "sqrt_inverse"])
    return cmd


def write_manifest(path: Path, jobs: list[Job], batch_size: int, epochs: int) -> None:
    rows = []
    for job in jobs:
        rows.append(
            {
                "model_key": job.model_key,
                "model": job.model.label,
                "checkpoint": job.model.checkpoint,
                "passes": job.model.passes,
                "dataset": job.dataset,
                "seed": job.seed,
                "batch_size": batch_size,
                "epochs": epochs,
                "train_cache": job.train_cache,
                "val_cache": job.val_cache,
                "test_cache": job.test_cache,
                "result": job.output_dir / "results.json",
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--gpus", nargs="+", default=["1", "3", "4", "5", "6", "7"])
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.output_root, args.seeds)
    write_manifest(args.output_root / "manifest.csv", jobs, args.batch_size, args.epochs)

    pending = deque(
        job
        for job in jobs
        if not result_is_complete(job.output_dir / "results.json", job.seed, args.batch_size, args.epochs)
    )
    print(f"jobs={len(jobs)} complete={len(jobs) - len(pending)} pending={len(pending)}")
    if args.dry_run:
        return 0

    running: list[tuple[subprocess.Popen, str, Job, object]] = []
    failures = []
    while pending or running:
        for gpu in args.gpus:
            active = sum(active_gpu == gpu for _, active_gpu, _, _ in running)
            while pending and active < args.workers_per_gpu:
                job = pending.popleft()
                job.output_dir.mkdir(parents=True, exist_ok=True)
                log_handle = (job.output_dir / "probe.log").open("w")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                # Multiple probe processes otherwise each fan out across all
                # host cores and become slower through severe CPU contention.
                for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    env[variable] = "2"
                process = subprocess.Popen(
                    command(job, args.batch_size, args.epochs, args.num_workers),
                    cwd=ROOT,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                running.append((process, gpu, job, log_handle))
                active += 1
                print(f"START gpu={gpu} pid={process.pid} {job.model_key}/seed{job.seed}/{job.dataset}")

        time.sleep(0.5)
        still_running = []
        for process, gpu, job, log_handle in running:
            returncode = process.poll()
            if returncode is None:
                still_running.append((process, gpu, job, log_handle))
                continue
            log_handle.close()
            status = "DONE" if returncode == 0 else "FAIL"
            print(f"{status} gpu={gpu} rc={returncode} {job.model_key}/seed{job.seed}/{job.dataset}")
            if returncode != 0:
                failures.append(job)
        running = still_running

    print(f"finished={len(jobs) - len(failures)} failures={len(failures)}")
    if failures:
        for job in failures:
            print(f"FAILED {job.model_key}/seed{job.seed}/{job.dataset}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
