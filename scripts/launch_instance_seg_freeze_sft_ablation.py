#!/usr/bin/env python3
"""Launch a controlled frozen/partial/full instance-seg adaptation ablation.

Each mode is pinned to one GPU. With ``--wait-for-idle`` the launcher waits for
several consecutive low-memory/low-utilization samples before starting, so it
can be submitted safely while other workloads still own the requested cards.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATASETS = ("pannuke", "tissuenet", "conic", "bbbc038", "livecell", "monuseg")
DEFAULT_MODES = ("frozen", "last2", "last4", "finetune")


def gpu_sample(gpu: str) -> tuple[int, int]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    memory, utilization = (int(value.strip()) for value in proc.stdout.strip().split(","))
    return memory, utilization


def wait_for_idle_gpu(args, gpu: str, mode: str) -> None:
    if not args.wait_for_idle:
        return
    consecutive = 0
    while consecutive < args.idle_samples:
        try:
            memory, utilization = gpu_sample(gpu)
            idle = memory <= args.max_memory_used_mb and utilization <= args.max_utilization
            consecutive = consecutive + 1 if idle else 0
            print(
                f"[wait:{mode}] GPU{gpu} memory={memory}MiB util={utilization}% "
                f"idle_samples={consecutive}/{args.idle_samples}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            consecutive = 0
            print(f"[wait:{mode}] GPU{gpu} probe failed: {exc}", flush=True)
        if consecutive < args.idle_samples:
            time.sleep(args.poll_seconds)


def pipeline_command(args, mode: str, gpu: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "dinov3.eval.bio_segmentation.instance_seg.scripts.run_cellvit_pipeline",
        "--datasets",
        *args.datasets,
        "--checkpoints-dir",
        str(Path(args.checkpoints_dir).resolve()),
        "--checkpoint-iters",
        args.checkpoint_iters,
        "--train-config",
        str(Path(args.train_config).resolve()),
        "--data-root-base",
        str(Path(args.data_root_base).resolve()),
        "--output-root",
        str(Path(args.output_root).resolve()),
        "--layers",
        *[str(layer) for layer in args.layers],
        "--modes",
        mode,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--crop-size",
        str(args.crop_size),
        "--stride",
        str(args.stride),
        "--lr",
        str(args.lr),
        "--backbone-lr",
        str(args.backbone_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--amp-dtype",
        args.amp_dtype,
        "--feature-size",
        str(args.feature_size),
        "--embed-proj",
        str(args.embed_proj),
        "--num-workers",
        str(args.num_workers),
        "--eval-every",
        str(args.eval_every),
        "--seed",
        str(args.seed),
        "--aug",
        args.aug,
        "--mosaic-prob",
        str(args.mosaic_prob),
        "--gpu",
        gpu,
        "--skip-completed",
        "--continue-on-error",
    ]
    if args.max_eval_images is not None:
        cmd.extend(["--max-eval-images", str(args.max_eval_images)])
    if args.skip_test_eval:
        cmd.append("--skip-test-eval")
    if args.tta:
        cmd.append("--tta")
    return cmd


def run_mode(args, mode: str, gpu: str, command: list[str], launch_delay: int) -> dict[str, object]:
    wait_for_idle_gpu(args, gpu, mode)
    if launch_delay:
        print(f"[wait:{mode}] staggering launch by {launch_delay}s", flush=True)
        time.sleep(launch_delay)
        wait_for_idle_gpu(args, gpu, mode)
    log_path = Path(args.output_root) / f"launcher_{mode}_gpu{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    print(f"[launch:{mode}] GPU{gpu}: {shlex.join(command)}", flush=True)
    with log_path.open("a") as log_file:
        log_file.write(f"\n[{started}] $ {shlex.join(command)}\n")
        log_file.flush()
        proc = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    finished = datetime.now(timezone.utc).isoformat()
    print(f"[done:{mode}] GPU{gpu} exit={proc.returncode}", flush=True)
    return {
        "mode": mode,
        "gpu": gpu,
        "returncode": proc.returncode,
        "started_at": started,
        "finished_at": finished,
        "log": str(log_path.resolve()),
    }


def write_summary(output_root: Path) -> Path:
    rows = []
    for path in sorted(output_root.glob("*/*/*/results.json")):
        data = json.loads(path.read_text())
        split = "test" if "test" in data else "val"
        metrics = data.get(split, {})
        meta = data.get("_meta", {})
        rows.append(
            {
                "dataset": meta.get("dataset", path.parts[-4]),
                "checkpoint": path.parts[-3],
                "mode": meta.get("backbone_mode", path.parts[-2]),
                "split": split,
                "AJI": metrics.get("AJI"),
                "bPQ": metrics.get("bPQ"),
                "mPQ": metrics.get("mPQ"),
                "AP50": metrics.get("AP50"),
                "AP75": metrics.get("AP75"),
                "results_json": str(path.resolve()),
            }
        )
    summary = output_root / "ablation_summary.csv"
    fields = ("dataset", "checkpoint", "mode", "split", "AJI", "bPQ", "mPQ", "AP50", "AP75", "results_json")
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--checkpoint-iters", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--data-root-base", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=list(DEFAULT_MODES))
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[7, 15, 23, 31])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp-dtype", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--feature-size", type=int, default=64)
    parser.add_argument("--embed-proj", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aug", choices=["none", "strong"], default="strong")
    parser.add_argument("--mosaic-prob", type=float, default=0.3)
    parser.add_argument("--max-eval-images", type=int, default=None)
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--wait-for-idle", action="store_true")
    parser.add_argument("--max-memory-used-mb", type=int, default=2000)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--launch-stagger-seconds",
        type=int,
        default=90,
        help="Delay successive mode launches to avoid concurrent large-checkpoint loads.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.modes) != len(args.gpus):
        parser.error("--modes and --gpus must contain the same number of entries")
    return args


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    commands = {
        mode: {"gpu": gpu, "command": pipeline_command(args, mode, gpu)}
        for mode, gpu in zip(args.modes, args.gpus)
    }
    manifest_path = output_root / "launcher_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "cwd": os.getcwd(),
                "commands": commands,
                "wait_for_idle": args.wait_for_idle,
                "idle_thresholds": {
                    "max_memory_used_mb": args.max_memory_used_mb,
                    "max_utilization": args.max_utilization,
                    "idle_samples": args.idle_samples,
                    "poll_seconds": args.poll_seconds,
                },
            },
            indent=2,
        )
        + "\n"
    )
    if args.dry_run:
        for mode, spec in commands.items():
            print(f"[{mode} -> GPU{spec['gpu']}] {shlex.join(spec['command'])}")
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(commands)) as pool:
        futures = {}
        for index, (mode, spec) in enumerate(commands.items()):
            future = pool.submit(
                run_mode,
                args,
                mode,
                spec["gpu"],
                spec["command"],
                index * args.launch_stagger_seconds,
            )
            futures[future] = mode
        statuses = [future.result() for future in concurrent.futures.as_completed(futures)]

    summary = write_summary(output_root)
    status_path = output_root / "launcher_status.json"
    status_path.write_text(json.dumps(statuses, indent=2) + "\n")
    print(f"[summary] {summary}", flush=True)
    return 1 if any(status["returncode"] != 0 for status in statuses) else 0


if __name__ == "__main__":
    raise SystemExit(main())
