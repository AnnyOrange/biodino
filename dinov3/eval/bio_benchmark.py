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
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

logger = logging.getLogger("dinov3.bio_benchmark")

CHANNEL_POLICIES = ("auto", "native", "first3", "compact3", "zerofill3", "mean3", "sample3_tta")

DEFAULT_CLASSIFICATION_DATASETS = [
    "bloodmnist",
    "pathmnist",
    "tissuemnist",
    "breastmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "chestmnist",
    "bbbc048-cellcycle",
    "cyclops-protein-loc",
    "midog25-atypical",
    "pcam",
    "nct-crc-he",
    "lc25000",
    "chammi-allen-task1",
    "chammi-allen-task2",
    "chammi-cp-task1",
    "chammi-cp-task2",
    "chammi-cp-task3",
    "chammi-hpa-task1",
    "chammi-hpa-task2",
]
DEFAULT_REGRESSION_DATASETS = ["bbbc013", "bbbc005", "conic-cell-count", "livecell-cell-count"]
DEFAULT_RETRIEVAL_DATASETS = [
    "lc25000",
    "nct-crc-he-100",
    "nct-crc-he-1k",
    "crc-val-he-7k",
    "hpa-subcellular",
    "rxrx1-cross",
]
DEFAULT_DETECTION_DATASETS = ["livecell", "bbbc038", "conic"]
DEFAULT_SEGMENTATION_DATASETS = [
    "bbbc038",
    "conic",
    "monuseg",
    "pannuke",
    "tissuenet",
    "livecell",
    "multimodal_cellseg",
    "cellpose",
]

CORE_TASKS = {"classification", "regression", "retrieval"}
DENSE_TASKS = {"segmentation", "detection"}
DEFAULT_THREAD_LIMITS = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _channel_policy_tag(channel_policy: str, channel_tta_samples: int) -> str:
    if channel_policy == "auto":
        return ""
    if channel_policy == "sample3_tta":
        return f"_cpsample3tta{channel_tta_samples}"
    return f"_cp{channel_policy}"


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
        if not (child.is_dir() and child.name.isdigit()):
            continue
        if (child / "checkpoint.pth").is_file():
            found[int(child.name)] = child / "checkpoint.pth"
        elif (child / ".metadata").is_file():
            # torch.distributed.checkpoint directory (e.g. ViT-7B runs).
            found[int(child.name)] = child
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


def _successful_result_exists(job: Job) -> bool:
    candidates = [job.output_dir / "last_result.json", job.output_dir / "results_bio_detection.json"]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and not data.get("error"):
            return True
    return False


class _GpuSlotPool:
    """Runtime GPU slots. Jobs take any free card instead of a round-robin pin."""

    def __init__(self, gpus: Sequence[str], jobs_per_gpu: int) -> None:
        self._available: List[str] = []
        for gpu in gpus:
            self._available.extend([gpu] * max(1, int(jobs_per_gpu)))
        self._cv = threading.Condition()

    def acquire(self) -> str:
        with self._cv:
            while not self._available:
                self._cv.wait()
            return self._available.pop(0)

    def release(self, gpu: str) -> None:
        with self._cv:
            self._available.append(gpu)
            self._cv.notify()


def _bind_job_to_gpu(job: Job, gpu: str) -> Job:
    cmd = list(job.cmd)
    if "--gpu" in cmd:
        idx = cmd.index("--gpu")
        if idx + 1 < len(cmd):
            cmd[idx + 1] = gpu
    return replace(job, gpu=gpu, cmd=cmd)


def _run_job(
    job: Job,
    dry_run: bool,
    cpu_slots: threading.Semaphore | None = None,
    gpu_pool: _GpuSlotPool | None = None,
) -> Tuple[Job, int, str]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run and job.task != "segmentation" and _successful_result_exists(job):
        return job, 0, "cached-success"
    assigned = job.gpu
    if gpu_pool is not None and not dry_run:
        assigned = gpu_pool.acquire()
        job = _bind_job_to_gpu(job, assigned)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = assigned
    env["PYTHONUNBUFFERED"] = "1"
    for name, value in DEFAULT_THREAD_LIMITS.items():
        env.setdefault(name, value)
    cmd_str = shlex.join(job.cmd)
    dataset_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", job.dataset)
    log_path = job.output_dir / f"run_{job.task}_{dataset_tag}_{job.ckpt_id}.log"
    logger.info("[gpu %s] %s/%s ckpt=%s $ %s", assigned, job.task, job.dataset, job.ckpt_id, cmd_str)
    if dry_run:
        log_path.write_text(cmd_str + "\n")
        return job, 0, "dry-run"
    if cpu_slots is not None:
        cpu_slots.acquire()
    try:
        with open(log_path, "w") as f:
            f.write("$ " + cmd_str + "\n")
            f.flush()
            proc = subprocess.run(job.cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    finally:
        if cpu_slots is not None:
            cpu_slots.release()
        if gpu_pool is not None:
            gpu_pool.release(assigned)
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


def _chunks(items: Sequence[str], size: int) -> Iterable[List[str]]:
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


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
            seg_job_specs = [(args.segmentation_datasets, False)]
            if args.segmentation_multichannel:
                mc_datasets = [d for d in args.segmentation_datasets if d == "tissuenet"]
                rgb_datasets = [d for d in args.segmentation_datasets if d != "tissuenet"]
                seg_job_specs = []
                if rgb_datasets:
                    seg_job_specs.append((rgb_datasets, False))
                if mc_datasets:
                    seg_job_specs.append((mc_datasets, True))
                else:
                    logger.warning("--segmentation-multichannel requested, but no multichannel-capable dataset is selected")

            for seg_datasets, use_multichannel in seg_job_specs:
                seg_policy = args.segmentation_channel_policy
                if not use_multichannel and seg_policy == "native":
                    logger.info("Using segmentation RGB policy=auto for non-multichannel datasets instead of native")
                    seg_policy = "auto"
                seg_run_name = f"{args.run_name}_mc" if use_multichannel else args.run_name
                seg_run_name = f"{seg_run_name}{_channel_policy_tag(seg_policy, args.segmentation_channel_tta_samples)}"
                datasets_per_job = args.segmentation_datasets_per_job or len(seg_datasets)
                for dataset_chunk in _chunks(seg_datasets, datasets_per_job):
                    gpu = next(gpu_iter)
                    cmd = [
                        py, "-m", "dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline",
                        "--datasets", *dataset_chunk,
                        "--checkpoints-dir", str(Path(args.checkpoints_dir).resolve()),
                        "--checkpoint-iters", str(ckpt_id),
                        "--train-config", str(Path(args.train_config).resolve()),
                        "--data-root-base", str(Path(args.benchmark_root).resolve() / "segmentation"),
                        "--output-root", str(seg_out),
                        "--cache-root", str(seg_cache),
                        "--run-name", args.run_name,
                        "--protocol", args.segmentation_protocol,
                        "--layer-preset", args.layer_preset,
                        "--feature-batch-size", str(args.seg_feature_batch_size),
                        "--feature-num-workers", str(args.seg_feature_num_workers),
                        "--probe-epochs", str(args.seg_probe_epochs if not args.smoke else 1),
                        "--probe-batch-size", str(args.seg_probe_batch_size),
                        "--probe-num-workers", str(args.seg_probe_num_workers),
                        "--probe-eval-every", str(args.seg_probe_epochs if not args.smoke else 1),
                        "--probe-seed", str(args.seed),
                        "--channel-policy", seg_policy,
                        "--channel-tta-samples", str(args.segmentation_channel_tta_samples),
                        "--channel-policy-seed", str(args.segmentation_channel_policy_seed),
                        "--gpu", gpu,
                    ]
                    if args.smoke:
                        cmd.extend(["--fast-eval", "--semantic-only", "--skip-test-eval"])
                    if use_multichannel:
                        cmd.append("--multichannel")
                    jobs.append(
                        Job(
                            "segmentation",
                            "+".join(dataset_chunk),
                            ckpt_id,
                            gpu,
                            cmd,
                            seg_out / seg_run_name,
                        )
                    )

        # Classification / regression / multilabel use the sklearn frozen-feature
        # probes (dinov3.eval.bio_frozen_eval). The entry auto-detects the task
        # and applies the dataset's held-out split protocol (official test,
        # committed group split, or the deterministic 80/20 fallback).
        frozen_common = [
            "--checkpoint", str(ckpt),
            "--train-config", str(Path(args.train_config).resolve()),
            "--benchmark-root", str(Path(args.benchmark_root).resolve()),
            "--model-name", f"dinov3-{ckpt_id}",
            "--n-last-blocks", str(args.frozen_n_last_blocks),
            "--autocast-dtype", args.autocast_dtype,
            "--batch-size", str(args.frozen_batch_size),
            "--num-workers", str(args.num_workers),
            "--train-fraction", str(args.train_fraction),
            "--seed", str(args.seed),
            "--split-protocol", args.frozen_split_protocol,
            "--channel-policy", args.frozen_channel_policy,
            "--channel-tta-samples", str(args.frozen_channel_tta_samples),
            "--channel-policy-seed", str(args.frozen_channel_policy_seed),
        ]
        frozen_cap: List[str] = []
        if args.smoke:
            frozen_cap = ["--max-samples", str(args.smoke_max_samples)]
        elif args.max_samples_per_split:
            frozen_cap = ["--max-samples", str(args.max_samples_per_split)]

        if "classification" in args.tasks:
            for datasets in _chunks(args.classification_datasets, args.frozen_datasets_per_job):
                dataset_tag = datasets[0] if len(datasets) == 1 else f"shard_{datasets[0]}_{datasets[-1]}"
                od = out / "bio_classification" / dataset_tag / str(ckpt_id)
                cmd = [
                    py, "-m", "dinov3.eval.bio_frozen_eval.run_classification",
                    "--datasets", *datasets, "--output-dir", str(od), *frozen_common,
                    "--resolution-protocol", args.classification_resolution_protocol,
                    "--image-size", str(args.classification_image_size),
                    *(
                        ["--resize-size", str(args.classification_resize_size)]
                        if args.classification_resize_size > 0
                        else []
                    ),
                    *frozen_cap,
                ]
                jobs.append(Job("classification", "+".join(datasets), ckpt_id, next(gpu_iter), cmd, od))

        if "regression" in args.tasks:
            for datasets in _chunks(args.regression_datasets, args.frozen_datasets_per_job):
                dataset_tag = datasets[0] if len(datasets) == 1 else f"shard_{datasets[0]}_{datasets[-1]}"
                od = out / "bio_regression" / dataset_tag / str(ckpt_id)
                cmd = [
                    py, "-m", "dinov3.eval.bio_frozen_eval.run_classification",
                    "--datasets", *datasets, "--output-dir", str(od), *frozen_common,
                    "--resolution-protocol", args.regression_resolution_protocol,
                    "--image-size", str(args.regression_image_size),
                    *(
                        ["--resize-size", str(args.regression_resize_size)]
                        if args.regression_resize_size > 0
                        else []
                    ),
                    *frozen_cap,
                ]
                jobs.append(Job("regression", "+".join(datasets), ckpt_id, next(gpu_iter), cmd, od))

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
                    "--channel-policy", args.detection_channel_policy,
                ]
                jobs.append(Job("detection", ds, ckpt_id, next(gpu_iter), cmd, od))

        if "retrieval" in args.tasks:
            for datasets in _chunks(args.retrieval_datasets, args.frozen_datasets_per_job):
                dataset_tag = datasets[0] if len(datasets) == 1 else f"shard_{datasets[0]}_{datasets[-1]}"
                uses_rxrx1_full = args.rxrx1_full and "rxrx1-cross" in datasets
                if uses_rxrx1_full:
                    dataset_tag = f"{dataset_tag}_full"
                od = out / "bio_retrieval" / dataset_tag / str(ckpt_id)
                cmd = [
                    py, "-m", "dinov3.eval.bio_frozen_eval.run_retrieval_clustering",
                    "--checkpoint", str(ckpt),
                    "--train-config", str(Path(args.train_config).resolve()),
                    "--benchmark-root", str(Path(args.benchmark_root).resolve()),
                    "--datasets", *datasets, "--output-dir", str(od),
                    "--model-name", f"dinov3-{ckpt_id}",
                    "--n-last-blocks", str(args.frozen_n_last_blocks),
                    "--autocast-dtype", args.autocast_dtype,
                    "--batch-size", str(args.frozen_batch_size),
                    "--num-workers", str(args.num_workers),
                    "--seed", str(args.seed),
                    "--channel-policy", args.frozen_channel_policy,
                    "--channel-tta-samples", str(args.frozen_channel_tta_samples),
                    "--channel-policy-seed", str(args.frozen_channel_policy_seed),
                ]
                if uses_rxrx1_full:
                    cmd.append("--rxrx1-full")
                if args.smoke:
                    # clustering needs more than n_clusters samples; keep a floor.
                    cmd += ["--max-samples", str(max(args.smoke_max_samples, 64))]
                jobs.append(Job("retrieval", "+".join(datasets), ckpt_id, next(gpu_iter), cmd, od))
    return jobs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run bio segmentation/classification/regression/detection probes over a checkpoint folder.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--checkpoint-iters", nargs="+", default=["latest"])
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=["segmentation", "classification", "regression", "detection", "retrieval"], choices=["segmentation", "classification", "regression", "detection", "retrieval"])
    parser.add_argument("--gpus", nargs="+", default=None)
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--max-concurrent-jobs", type=int, default=0, help="Global subprocess cap; 0 uses GPU count x jobs-per-gpu.")
    parser.add_argument("--max-cpu-jobs", type=int, default=0, help="CPU-heavy subprocess cap; 0 matches --max-concurrent-jobs.")
    parser.add_argument("--concurrent-task-groups", action="store_true", help="Mix core and dense jobs. Default runs core first, then dense jobs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny validation mode; not for final numbers.")
    parser.add_argument("--smoke-max-samples", type=int, default=8)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", default="bio_eval")
    # classification (incl. multilabel chestmnist), regression and retrieval use
    # the sklearn frozen-feature probes in dinov3.eval.bio_frozen_eval. Dataset
    # names must match the bio_frozen_eval registry (e.g. cyclops-protein-loc /
    # bbbc048-cellcycle, not cyclops / bbbc048).
    parser.add_argument("--classification-datasets", nargs="+", default=DEFAULT_CLASSIFICATION_DATASETS)
    parser.add_argument("--regression-datasets", nargs="+", default=DEFAULT_REGRESSION_DATASETS)
    parser.add_argument("--retrieval-datasets", nargs="+", default=DEFAULT_RETRIEVAL_DATASETS)
    parser.add_argument(
        "--rxrx1-full",
        action="store_true",
        help="Use all 112,824 RxRx1 treatment views; default uses the balanced 17,728-view core.",
    )
    parser.add_argument("--detection-datasets", nargs="+", default=DEFAULT_DETECTION_DATASETS)
    parser.add_argument("--segmentation-datasets", nargs="+", default=DEFAULT_SEGMENTATION_DATASETS)
    parser.add_argument(
        "--segmentation-datasets-per-job",
        type=int,
        default=0,
        help="Datasets per segmentation subprocess; 0 keeps the full selected set in one pipeline.",
    )
    # frozen-probe (classification / regression / multilabel / retrieval) settings
    parser.add_argument(
        "--probe-backend",
        default="sklearn",
        choices=["sklearn"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--frozen-batch-size", type=int, default=64, help="Feature-extraction batch size for frozen-probe tasks.")
    parser.add_argument("--frozen-datasets-per-job", type=int, default=1, help="Evaluate this many frozen datasets per model load.")
    parser.add_argument("--frozen-n-last-blocks", type=int, default=1)
    parser.add_argument("--autocast-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--frozen-split-protocol",
        default="current",
        choices=["current", "s0-internal"],
        help="Dataset split used by classification and regression frozen probes.",
    )
    parser.add_argument(
        "--frozen-channel-policy",
        default="auto",
        choices=CHANNEL_POLICIES,
        help="Channel handling for frozen classification/regression/retrieval tensor datasets.",
    )
    parser.add_argument(
        "--frozen-channel-tta-samples",
        type=int,
        default=8,
        help="Number of channel draws for frozen --frozen-channel-policy sample3_tta.",
    )
    parser.add_argument(
        "--frozen-channel-policy-seed",
        type=int,
        default=0,
        help="Seed for frozen stochastic channel policies.",
    )
    parser.add_argument("--classification-resolution-protocol", default="best", choices=["manual", "best"])
    parser.add_argument("--classification-image-size", type=int, default=224, help="Manual/fallback final square crop size for classification/multilabel frozen features.")
    parser.add_argument("--classification-resize-size", type=int, default=0, help="Optional pre-crop resize size for classification; 0 keeps the ImageNet eval ratio.")
    parser.add_argument("--regression-resolution-protocol", default="best", choices=["manual", "best"])
    parser.add_argument("--regression-image-size", type=int, default=224)
    parser.add_argument("--regression-resize-size", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    # segmentation / detection dense linear probe (repo code in bio_segmentation / bio_detection)
    parser.add_argument("--segmentation-protocol", default="best", choices=["manual", "best"])
    parser.add_argument("--segmentation-multichannel", action="store_true", help="Pass --multichannel to the segmentation linear-probe pipeline.")
    parser.add_argument(
        "--segmentation-channel-policy",
        default="auto",
        choices=CHANNEL_POLICIES,
        help="Channel handling for segmentation feature extraction. auto preserves the current RGB path, or native with --segmentation-multichannel and a multichannel stem.",
    )
    parser.add_argument(
        "--segmentation-channel-tta-samples",
        type=int,
        default=8,
        help="Number of channel draws for segmentation --segmentation-channel-policy sample3_tta.",
    )
    parser.add_argument(
        "--segmentation-channel-policy-seed",
        type=int,
        default=0,
        help="Seed for segmentation stochastic channel policies.",
    )
    parser.add_argument("--layer-preset", default="last1")
    parser.add_argument("--seg-feature-batch-size", type=int, default=32)
    parser.add_argument("--seg-feature-num-workers", type=int, default=4)
    parser.add_argument("--seg-probe-epochs", type=int, default=50)
    parser.add_argument("--seg-probe-batch-size", type=int, default=32)
    parser.add_argument("--seg-probe-num-workers", type=int, default=4)
    parser.add_argument("--det-epochs", type=int, default=5)
    parser.add_argument("--det-batch-size", type=int, default=8)
    parser.add_argument("--detection-channel-policy", default="auto", choices=CHANNEL_POLICIES)
    return parser.parse_args(argv)


def _run_jobs(args, jobs: Sequence[Job], gpus: Sequence[str]) -> Dict[str, List[Tuple[Job, int, str]]]:
    rows_by_task: Dict[str, List[Tuple[Job, int, str]]] = {t: [] for t in args.tasks}
    default_workers = len(gpus) * max(1, args.jobs_per_gpu)
    max_workers = max(1, args.max_concurrent_jobs or default_workers)
    cpu_limit = max(1, args.max_cpu_jobs or max_workers)
    cpu_slots = threading.Semaphore(cpu_limit)
    gpu_pool = _GpuSlotPool(gpus, args.jobs_per_gpu)
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_job, job, args.dry_run, cpu_slots, gpu_pool) for job in jobs]
        for fut in futures.as_completed(futs):
            job, code, msg = fut.result()
            rows_by_task[job.task].append((job, code, msg))
            if code != 0:
                logger.error("FAILED %s/%s ckpt=%s log=%s", job.task, job.dataset, job.ckpt_id, msg)
    return rows_by_task


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
    rows_by_task: Dict[str, List[Tuple[Job, int, str]]] = {t: [] for t in args.tasks}
    if args.concurrent_task_groups:
        groups = [jobs]
    else:
        groups = [
            [job for job in jobs if job.task in CORE_TASKS],
            [job for job in jobs if job.task in DENSE_TASKS],
        ]
    for group in groups:
        if not group:
            continue
        group_rows = _run_jobs(args, group, gpus)
        for task, rows in group_rows.items():
            rows_by_task[task].extend(rows)
    for task, rows in rows_by_task.items():
        _write_report(out, task, sorted(rows, key=lambda x: (x[0].ckpt_id, x[0].dataset)))
    failed = [(job, code, msg) for rows in rows_by_task.values() for job, code, msg in rows if code != 0]
    if failed:
        raise SystemExit(f"{len(failed)} jobs failed. See logs under {out}")
    logger.info("All jobs finished. Reports: %s", out)


if __name__ == "__main__":
    main()
