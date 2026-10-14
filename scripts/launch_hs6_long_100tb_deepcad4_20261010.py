#!/usr/bin/env python3
"""Run the exact 100TB finite stream on deepcad GPUs 1,5,6,7."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from launch_hs6_long_finite_quality_20261009 import BASE, ROOT, prepare_resume_metrics


DATA = Path("/mnt/huawei_blm/hs6_long_uniform_100tb_vs_20tb_20261009")
INDEX = DATA / "stream_index_exact_4rank"
WORLD_SIZE = 4
MICROBATCH = 16
ACCUM = 16
UPDATES = 16_000
GPU_GROUP = "1,5,6,7"
RUN_NAME = "HS6_L_quality_long_noreplace_100tb_ddp_gb1024_16k_4xa100_deepcad_20261010"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python-bin", default="/home/deepcad/anaconda3/envs/dinov3/bin/python")
    parser.add_argument("--master-port", type=int, default=31860)
    args = parser.parse_args()
    if args.smoke and args.resume:
        raise ValueError("Smoke runs do not resume")
    assignment = json.loads((INDEX / "assignment.json").read_text())
    target_per_rank = UPDATES * MICROBATCH * ACCUM
    if assignment["status"] != "PASS" or assignment["format"] != "100tb_raw_tar_stream_v2" or \
            assignment["world_size"] != WORLD_SIZE or assignment["candidate_count"] != 16_500_000 or \
            assignment["rank_primary_counts"] != [target_per_rank] * WORLD_SIZE or \
            assignment["target_total"] != UPDATES * 1024:
        raise ValueError("Four-rank 100TB index does not match the training budget")
    if not Path(args.python_bin).is_file():
        raise FileNotFoundError(args.python_bin)
    run = ROOT / "outputs/01_training_runs" / (RUN_NAME + ("_smoke" if args.smoke else ""))
    if not args.dry_run:
        if args.resume and not (run / "launch_manifest.json").is_file():
            raise FileNotFoundError(f"No launch manifest for resuming: {run}")
        if not args.resume and run.exists():
            raise FileExistsError(f"Refusing to overwrite an existing run: {run}")
    overrides = {
        "compute_precision.distributed_mode": "ddp",
        "train.batch_size_per_gpu": MICROBATCH,
        "optim.gradient_accumulation_steps": ACCUM,
        "train.OFFICIAL_EPOCH_LENGTH": 4098,
        "train.max_updates": 1 if args.smoke else UPDATES,
        "train.num_workers": 1,
        "train.wds_shuffle_buffer": 64,
        "train.wds_deterministic_resampling": True,
        "train.prefetch_factor": 1,
        "train.pin_memory": False,
        "train.checkpointing": True,
        "train.checkpointing_full": True,
        "train.checkpointing_blocks": 24,
        "evaluation.eval_period_iterations": 1 if args.smoke else 4000,
        "checkpointing.period": 1 if args.smoke else 4000,
        "checkpointing.max_to_keep": 2,
        "train.dataset_path": "raw100tb_once_robust:" + str(INDEX),
    }
    command = [
        args.python_bin, "-m", "torch.distributed.run", "--nproc_per_node=4",
        f"--master_port={args.master_port}", "dinov3/train/train.py",
        "--config-file", str(BASE), "--output-dir", str(run),
        *([] if args.resume else ["--no-resume"]),
        *(f"{key}={json.dumps(value) if not isinstance(value, str) else value}"
          for key, value in overrides.items()),
    ]
    manifest = {
        "arm": "100tb", "host": "deepcad", "gpu_group": GPU_GROUP,
        "world_size": WORLD_SIZE, "microbatch": MICROBATCH,
        "gradient_accumulation_steps": ACCUM,
        "effective_batch": WORLD_SIZE * MICROBATCH * ACCUM,
        "target_updates": 1 if args.smoke else UPDATES,
        "target_unique_visits": 1024 if args.smoke else UPDATES * 1024,
        "source_population": "already deduplicated 100TB exported stored items",
        "sampling": "first 16,384,000 global random priorities; backup only on read failure",
        "stream_index": str(INDEX),
        "candidate_sha256": assignment["candidate_sha256"],
        "schedule_reference": str(BASE), "warmup_updates": 3 * 4098,
        "command": command, "created_unix": time.time(),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if args.resume:
        prepare_resume_metrics(run)
    else:
        run.mkdir(parents=True)
        (run / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=GPU_GROUP,
        DQ_STREAM_FAILURE_LOG_DIR=str(run),
        PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        MALLOC_MMAP_THRESHOLD_="131072", MALLOC_ARENA_MAX="4",
    )
    with (run / "console.log").open("a" if args.resume else "w", buffering=1) as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (run / "exit.json").write_text(json.dumps({"returncode": result.returncode, "time_unix": time.time()}) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
