"""
Evaluate saved cpsam fine-tuned models on official test splits.

This is a thin launcher around ``finetune_cpsam --eval-only``. It writes results
where ``summarize_final.py`` expects them:

  outputs/instance_seg/test_eval/<dataset>/cpsam_ft/results.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PY = sys.executable
DATA_ROOTS = {
    "pannuke": "pannuke/extracted",
    "tissuenet": "tissuenet/extracted",
    "conic": "conic/extracted",
    "bbbc038": "bbbc038/extracted",
    "livecell": "LIVECell",
    "monuseg": "monuseg/extracted",
}


def _cmd(args, dataset: str, use_gpu: bool):
    data_root = Path(args.seg_root) / DATA_ROOTS[dataset]
    model_path = Path(args.model_root) / dataset / "models" / f"cpsam_ft_{dataset}"
    output_dir = Path(args.output_root) / dataset / "cpsam_ft"
    cmd = [
        PY,
        "-m",
        "dinov3.eval.bio_segmentation.instance_seg.scripts.finetune_cpsam",
        "--dataset",
        dataset,
        "--data-root",
        str(data_root),
        "--eval-only",
        "--model-path",
        str(model_path),
        "--eval-split",
        "test",
        "--output-dir",
        str(output_dir),
    ]
    cmd.append("--gpu" if use_gpu else "--no-gpu")
    if args.max_eval_images is not None:
        cmd += ["--max-eval-images", str(args.max_eval_images)]
    return cmd, output_dir


def main():
    p = argparse.ArgumentParser(description="Run cpsam-ft eval-only over official test splits.")
    p.add_argument("--datasets", nargs="+", default=list(DATA_ROOTS))
    p.add_argument("--seg-root", default="/mnt/huawei_deepcad/benchmark/segmentation")
    p.add_argument("--model-root", default="outputs/instance_seg/cpsam_ft")
    p.add_argument("--output-root", default="outputs/instance_seg/test_eval")
    p.add_argument("--gpus", default="", help="Comma-separated GPU ids. Empty uses current device/CPU.")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--max-eval-images", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    jobs = []
    for dataset in args.datasets:
        if dataset not in DATA_ROOTS:
            raise SystemExit(f"Unknown dataset: {dataset}")
        cmd, output_dir = _cmd(args, dataset, use_gpu=not args.no_gpu)
        result_path = output_dir / "results.json"
        if result_path.exists() and not args.overwrite:
            print(f"[skip] {dataset}: {result_path} exists", flush=True)
            continue
        model_path = Path(args.model_root) / dataset / "models" / f"cpsam_ft_{dataset}"
        if not model_path.exists():
            print(f"[missing] {dataset}: {model_path}", flush=True)
            continue
        jobs.append((dataset, cmd, output_dir))

    gpus = [g for g in args.gpus.split(",") if g]
    if args.dry_run:
        for _dataset, cmd, _output_dir in jobs:
            print(" ".join(cmd))
        return

    if not gpus:
        for dataset, cmd, output_dir in jobs:
            output_dir.mkdir(parents=True, exist_ok=True)
            log = output_dir.with_suffix(".log")
            print(f"[run] {dataset} -> {output_dir}", flush=True)
            with log.open("w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        return

    running = {}
    free = list(gpus)
    while jobs or running:
        while free and jobs:
            gpu = free.pop(0)
            dataset, cmd, output_dir = jobs.pop(0)
            output_dir.mkdir(parents=True, exist_ok=True)
            log = output_dir.with_suffix(".log")
            f = log.open("w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            running[gpu] = (proc, dataset, f)
            print(f"[launch] {dataset} GPU{gpu} pid={proc.pid}", flush=True)
        time.sleep(10)
        for gpu, (proc, dataset, f) in list(running.items()):
            if proc.poll() is None:
                continue
            f.close()
            del running[gpu]
            free.append(gpu)
            status = "ok" if proc.returncode == 0 else f"FAIL {proc.returncode}"
            print(f"[done] {dataset}: {status}", flush=True)


if __name__ == "__main__":
    main()
