from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

logger = logging.getLogger("dinov3.bio_benchmark")


@dataclass(frozen=True)
class Job:
    task: str
    dataset: str
    ckpt_id: int
    gpu: str
    cmd: List[str]
    output_dir: Path


def _discover_checkpoints(checkpoints_dir: Path) -> Dict[int, Path]:
    found: Dict[int, Path] = {}
    for child in checkpoints_dir.iterdir():
        if child.is_dir() and child.name.isdigit() and (child / "checkpoint.pth").is_file():
            found[int(child.name)] = child / "checkpoint.pth"
    return dict(sorted(found.items()))


def _select_iters(tokens: Sequence[str], discovered: Dict[int, Path]) -> List[int]:
    if not discovered:
        raise ValueError("No checkpoints found; expected <checkpoints-dir>/<iter>/checkpoint.pth")
    expanded = []
    for token in tokens or ["latest"]:
        expanded.extend([x.strip().lower() for x in str(token).split(",") if x.strip()])
    if "all" in expanded:
        return list(discovered)
    latest = max(discovered)
    selected = []
    for token in expanded:
        selected.append(latest if token == "latest" else int(token))
    missing = [x for x in selected if x not in discovered]
    if missing:
        raise ValueError(f"Requested checkpoints not found: {missing}; available={list(discovered)}")
    return sorted(set(selected))


def _gpus_from_arg(gpus: Sequence[str] | None) -> List[str]:
    if gpus:
        out: List[str] = []
        for g in gpus:
            out.extend([x for x in str(g).split(",") if x != ""])
        return out or ["0"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [x for x in visible.split(",") if x != ""]
    try:
        import torch

        n = torch.cuda.device_count()
        return [str(i) for i in range(n)] if n else ["0"]
    except Exception:
        return ["0"]


def _run_job(job: Job, dry_run: bool) -> Tuple[Job, int, str]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = job.gpu
    env["PYTHONUNBUFFERED"] = "1"
    cmd_str = shlex.join(job.cmd)
    log_path = job.output_dir / "run.log"
    logger.info("[gpu %s] %s/%s ckpt=%s $ %s", job.gpu, job.task, job.dataset, job.ckpt_id, cmd_str)
    if dry_run:
        log_path.write_text(cmd_str + "\n")
        return job, 0, "dry-run"
    with open(log_path, "w") as f:
        f.write("$ " + cmd_str + "\n")
        f.flush()
        proc = subprocess.run(job.cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return job, proc.returncode, str(log_path)


def _write_report(output_dir: Path, task: str, rows: List[Tuple[Job, int, str]]) -> None:
    report = output_dir / f"bio_{task}.md"
    lines = [f"# bio_{task}", "", "| dataset | ckpt | gpu | status | output |", "|---|---:|---:|---:|---|"]
    for job, code, msg in rows:
        status = "ok" if code == 0 else f"failed({code})"
        lines.append(f"| {job.dataset} | {job.ckpt_id} | {job.gpu} | {status} | `{job.output_dir}` |")
    report.write_text("\n".join(lines) + "\n")


def _round_robin_gpus(gpus: Sequence[str]):
    i = 0
    while True:
        yield gpus[i % len(gpus)]
        i += 1


def build_jobs(args, discovered: Dict[int, Path], selected_iters: Sequence[int], gpus: Sequence[str]) -> List[Job]:
    gpu_iter = _round_robin_gpus(gpus)
    jobs: List[Job] = []
    py = sys.executable
    out = Path(args.output_dir).resolve()
    common = ["--benchmark-root", str(Path(args.benchmark_root).resolve())]
    smoke_cap = str(args.smoke_max_samples) if args.smoke else str(args.max_samples_per_split)

    for ckpt_id in selected_iters:
        ckpt = discovered[ckpt_id]
        ckpt_args = ["--checkpoint", str(ckpt), "--train-config", str(Path(args.train_config).resolve())]
        if "segmentation" in args.tasks:
            seg_out = out / "bio_segmentation"
            seg_cache = out / "cache" / "bio_segmentation"
            gpu = next(gpu_iter)
            cmd = [
                py, "-m", "dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline",
                "--datasets", *args.segmentation_datasets,
                "--checkpoints-dir", str(Path(args.checkpoints_dir).resolve()),
                "--checkpoint-iters", str(ckpt_id),
                "--train-config", str(Path(args.train_config).resolve()),
                "--data-root-base", str(Path(args.benchmark_root).resolve() / "segmentation"),
                "--output-root", str(seg_out),
                "--cache-root", str(seg_cache),
                "--run-name", args.run_name,
                "--layer-preset", args.layer_preset,
                "--feature-batch-size", str(args.seg_feature_batch_size),
                "--probe-epochs", str(args.seg_probe_epochs if not args.smoke else 1),
                "--probe-eval-every", str(args.seg_probe_epochs if not args.smoke else 1),
                "--gpu", gpu,
            ]
            if args.smoke:
                cmd.extend(["--fast-eval", "--semantic-only", "--skip-test-eval"])
            jobs.append(Job("segmentation", "+".join(args.segmentation_datasets), ckpt_id, gpu, cmd, seg_out / args.run_name))

        if "classification" in args.tasks:
            for ds in args.classification_datasets:
                od = out / "bio_classification" / ds / str(ckpt_id)
                cmd = [
                    py, "-m", "dinov3.eval.bio_classification.linear",
                    *ckpt_args, *common,
                    "--dataset", ds,
                    "--output-dir", str(od),
                    "--epochs", str(args.cls_epochs if not args.smoke else 1),
                    "--batch-size", str(args.cls_batch_size),
                    "--num-workers", str(args.num_workers),
                    "--learning-rates", *args.cls_learning_rates,
                    "--max-samples-per-split", smoke_cap,
                ]
                jobs.append(Job("classification", ds, ckpt_id, next(gpu_iter), cmd, od))

        if "regression" in args.tasks:
            for ds in args.regression_datasets:
                od = out / "bio_regression" / ds / str(ckpt_id)
                cmd = [
                    py, "-m", "dinov3.eval.bio_regression.linear",
                    *ckpt_args, *common,
                    "--dataset", ds,
                    "--output-dir", str(od),
                    "--batch-size", str(args.reg_batch_size),
                    "--num-workers", str(args.num_workers),
                    "--max-samples-per-split", smoke_cap,
                ]
                jobs.append(Job("regression", ds, ckpt_id, next(gpu_iter), cmd, od))

        if "detection" in args.tasks:
            for ds in args.detection_datasets:
                od = out / "bio_detection" / ds / str(ckpt_id)
                cmd = [
                    py, "-m", "dinov3.eval.bio_detection.center_probe",
                    *ckpt_args, *common,
                    "--dataset", ds,
                    "--output-dir", str(od),
                    "--epochs", str(args.det_epochs if not args.smoke else 1),
                    "--batch-size", str(args.det_batch_size),
                    "--num-workers", str(args.num_workers),
                    "--max-samples-per-split", smoke_cap,
                ]
                jobs.append(Job("detection", ds, ckpt_id, next(gpu_iter), cmd, od))
    return jobs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run bio segmentation/classification/regression/detection probes over a checkpoint folder.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--checkpoint-iters", nargs="+", default=["latest"])
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=["segmentation", "classification", "regression", "detection"], choices=["segmentation", "classification", "regression", "detection"])
    parser.add_argument("--gpus", nargs="+", default=None)
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny validation mode; not for final numbers.")
    parser.add_argument("--smoke-max-samples", type=int, default=8)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", default="bio_eval")
    parser.add_argument("--classification-datasets", nargs="+", default=["bloodmnist", "bbbc048", "cyclops", "midog25"])
    parser.add_argument("--regression-datasets", nargs="+", default=["bbbc013"])
    parser.add_argument("--detection-datasets", nargs="+", default=["livecell"])
    parser.add_argument("--segmentation-datasets", nargs="+", default=["bbbc038", "conic", "monuseg", "pannuke", "tissuenet"])
    parser.add_argument("--layer-preset", default="last1")
    parser.add_argument("--seg-feature-batch-size", type=int, default=32)
    parser.add_argument("--seg-probe-epochs", type=int, default=50)
    parser.add_argument("--cls-epochs", type=int, default=10)
    parser.add_argument("--cls-batch-size", type=int, default=256)
    parser.add_argument("--cls-learning-rates", nargs="+", default=["1e-4", "5e-4", "1e-3", "5e-3", "1e-2"])
    parser.add_argument("--reg-batch-size", type=int, default=128)
    parser.add_argument("--det-epochs", type=int, default=5)
    parser.add_argument("--det-batch-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    discovered = _discover_checkpoints(checkpoints_dir)
    selected = _select_iters(args.checkpoint_iters, discovered)
    gpus = _gpus_from_arg(args.gpus)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args, discovered, selected, gpus)
    (out / "command_manifest.json").write_text(json.dumps([{**job.__dict__, "cmd": job.cmd, "output_dir": str(job.output_dir)} for job in jobs], indent=2, default=str))
    logger.info("Selected checkpoints=%s gpus=%s jobs=%d", selected, gpus, len(jobs))
    max_workers = max(1, len(gpus) * max(1, args.jobs_per_gpu))
    rows_by_task: Dict[str, List[Tuple[Job, int, str]]] = {t: [] for t in args.tasks}
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_job, job, args.dry_run) for job in jobs]
        for fut in futures.as_completed(futs):
            job, code, msg = fut.result()
            rows_by_task[job.task].append((job, code, msg))
            if code != 0:
                logger.error("FAILED %s/%s ckpt=%s log=%s", job.task, job.dataset, job.ckpt_id, msg)
    for task, rows in rows_by_task.items():
        _write_report(out, task, sorted(rows, key=lambda x: (x[0].ckpt_id, x[0].dataset)))
    failed = [(job, code, msg) for rows in rows_by_task.values() for job, code, msg in rows if code != 0]
    if failed:
        raise SystemExit(f"{len(failed)} jobs failed. See logs under {out}")
    logger.info("All jobs finished. Reports: %s", out)


if __name__ == "__main__":
    main()
