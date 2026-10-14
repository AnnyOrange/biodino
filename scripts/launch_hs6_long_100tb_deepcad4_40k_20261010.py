#!/usr/bin/env python3
"""Run 40,000 matched optimizer updates on the audited 100TB random subset."""

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
RUN_NAME = "HS6_L_quality_long_100tb_ddp_gb1024_40k_4xa100_deepcad_20261010"
FULL_POOL_INDEX = DATA / "stream_index_full_pool_balanced_4rank"
FULL_POOL_RUN_NAME = "HS6_L_quality_long_100tb_fullpool_ddp_gb1024_mb128_ga2_40k_4xa100_deepcad_20261010"
GPUS = "1,5,6,7"
UPDATES = 40_000
WORLD_SIZE = 4
MICROBATCH = 32
ACCUM = 8
FULL_POOL_MICROBATCH = 128
FULL_POOL_ACCUM = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-tag", default="")
    parser.add_argument("--smoke-updates", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full-pool", action="store_true")
    parser.add_argument("--python-bin", default="/home/deepcad/anaconda3/envs/dinov3/bin/python")
    parser.add_argument("--master-port", type=int, default=31861)
    args = parser.parse_args()
    if args.smoke and args.resume:
        raise ValueError("Smoke runs cannot resume")
    if args.smoke_tag and (not args.smoke or not args.smoke_tag.isalnum()):
        raise ValueError("Smoke tag requires --smoke and must be alphanumeric")
    if args.smoke_updates < 1 or (args.smoke_updates != 1 and not args.smoke):
        raise ValueError("--smoke-updates requires --smoke and a positive count")
    index = FULL_POOL_INDEX if args.full_pool else INDEX
    run_name = FULL_POOL_RUN_NAME if args.full_pool else RUN_NAME
    microbatch = FULL_POOL_MICROBATCH if args.full_pool else MICROBATCH
    accum = FULL_POOL_ACCUM if args.full_pool else ACCUM
    assignment = json.loads((index / "assignment.json").read_text())
    if args.full_pool:
        if assignment["status"] != "PASS" or assignment["world_size"] != WORLD_SIZE or \
                assignment["format"] != "100tb_full_pool_stream_v1" or \
                assignment["total_items"] != 26_614_979:
            raise ValueError("100TB full-pool index differs from the audited population")
    elif assignment["status"] != "PASS" or assignment["world_size"] != WORLD_SIZE or \
            assignment["target_total"] != 16_384_000 or \
            assignment["rank_primary_counts"] != [4_096_000] * WORLD_SIZE:
        raise ValueError("100TB random subset index differs from the audited four-rank plan")
    if not Path(args.python_bin).is_file():
        raise FileNotFoundError(args.python_bin)
    smoke_suffix = "_smoke_ckpt" + ("_" + args.smoke_tag if args.smoke_tag else "") if args.smoke else ""
    run = ROOT / "outputs/01_training_runs" / (run_name + smoke_suffix)
    if not args.dry_run:
        if args.resume and not (run / "launch_manifest.json").is_file():
            raise FileNotFoundError(f"No launch manifest to resume: {run}")
        if not args.resume and run.exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {run}")
    overrides = {
        "compute_precision.distributed_mode": "ddp",
        "train.batch_size_per_gpu": microbatch,
        "optim.gradient_accumulation_steps": accum,
        "train.OFFICIAL_EPOCH_LENGTH": 4098,
        "train.max_updates": args.smoke_updates if args.smoke else UPDATES,
        "train.num_workers": 2 if args.full_pool else 1,
        "train.wds_shuffle_buffer": 128 if args.full_pool else 64,
        "train.wds_deterministic_resampling": True,
        "train.prefetch_factor": 1,
        "train.pin_memory": False,
        "train.checkpointing": True,
        "train.checkpointing_full": True,
        "train.checkpointing_blocks": 24,
        "evaluation.eval_period_iterations": args.smoke_updates if args.smoke else 4000,
        "checkpointing.period": args.smoke_updates if args.smoke else (500 if args.full_pool else 4000),
        "checkpointing.max_to_keep": None,
        "train.dataset_path": ("raw100tb_full_pool_robust:" if args.full_pool else "raw100tb_repeat_robust:") + str(index),
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
        "arm": "100tb", "host": "deepcad", "gpu_group": GPUS,
        "world_size": WORLD_SIZE, "microbatch": microbatch,
        "gradient_accumulation_steps": accum, "effective_batch": WORLD_SIZE * microbatch * accum,
        "target_updates": args.smoke_updates if args.smoke else UPDATES,
        "target_image_visits": args.smoke_updates * 1024 if args.smoke else UPDATES * 1024,
        "source_population": ("all 100TB exported stored items" if args.full_pool else
                              "audited global-random 100TB exported stored-item subset"),
        "sampling": ("rank-disjoint shards, 4 interleaved random-order shards, 128-item buffer; "
                     "reshuffle each pass" if args.full_pool else
                     "16,384,000 unique items per pass; reshuffle shard and 64-item buffer each pass"),
        "repeat_policy": ("continuous full-pool passes; unreadable items skipped" if args.full_pool else
                          "without replacement within each pass, repeat across passes"),
        "source_subset_size": assignment["total_items"] if args.full_pool else assignment["target_total"],
        "source_pool_size": 26_614_979,
        "stream_index": str(index), "candidate_sha256": assignment.get("candidate_sha256"),
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
        CUDA_VISIBLE_DEVICES=GPUS,
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
