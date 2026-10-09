#!/usr/bin/env python3
"""Prepare and launch matched 20TB/100TB finite-stream HS6-L runs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3")
DATA = Path("/mnt/huawei_blm/hs6_long_uniform_100tb_vs_20tb_20261009")
BASE = ROOT / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_15tb07_5tbboundary03_ablation_8x3090qi_20260928/config.yaml"
TWENTY_INVENTORY = Path("/mnt/huawei_blm/deepcad_20tb_quality_1m_20261002/inventory.json")
HUNDRED_STREAM = DATA / "stream_index/assignment.json"
UPDATES = 16_000
WORLD_SIZE = 8
MICROBATCH = 8
ACCUM = 16
TARGET_PER_RANK = UPDATES * MICROBATCH * ACCUM
TARGET_TOTAL = TARGET_PER_RANK * WORLD_SIZE
EXPECTED_100TB = 16_500_000


def source_items(arm: str) -> list[tuple[str, int]]:
    if arm == "20tb":
        inventory = json.loads(TWENTY_INVENTORY.read_text())
        rows = inventory["rows"]
        if inventory["total_samples"] != 16_625_610 or len(rows) != 5526:
            raise ValueError("20TB full-pool inventory changed")
        items = [(row["path"], int(row["samples"])) for row in rows]
        if sum(count for _, count in items) != inventory["total_samples"]:
            raise ValueError("20TB per-shard counts do not sum to inventory")
        for row in rows:
            if Path(row["path"]).stat().st_size != row["bytes"]:
                raise ValueError(f"20TB tar changed: {row['path']}")
        return items

    raise ValueError("100TB uses the raw-tar stream index, not materialized WDS tars")


def assignment(arm: str) -> dict:
    if arm == "100tb":
        index = json.loads(HUNDRED_STREAM.read_text())
        if index["status"] != "PASS" or index["format"] != "100tb_raw_tar_stream_v1" or \
                index["candidate_count"] != EXPECTED_100TB or index["world_size"] != WORLD_SIZE or \
                min(index["rank_sample_counts"]) < TARGET_PER_RANK:
            raise ValueError("100TB streaming index does not cover the requested budget")
        return {
            "status": "PASS", "arm": arm,
            "rank_shards": [[index["shards"][shard_id] for shard_id in group]
                            for group in index["rank_shards"]],
            "rank_sample_counts": index["rank_sample_counts"],
            "target_per_rank": TARGET_PER_RANK, "sample_count": index["candidate_count"],
            "source_shards": len(index["shards"]),
            "inventory_sha256": index["candidate_sha256"], "seed": index["seed"],
            "sampling_unit": "100TB exported stored item",
        }
    items = source_items(arm)
    if len({path for path, _ in items}) != len(items):
        raise ValueError("Repeated source tar path")
    if sum(count for _, count in items) < TARGET_TOTAL:
        raise ValueError("Source pool cannot cover the requested unique image budget")
    shuffled = items.copy()
    random.Random(20261009).shuffle(shuffled)
    shuffled.sort(key=lambda row: row[1], reverse=True)
    groups = [[] for _ in range(WORLD_SIZE)]
    totals = [0] * WORLD_SIZE
    for path, count in shuffled:
        rank = min(range(WORLD_SIZE), key=lambda index: totals[index])
        groups[rank].append(path)
        totals[rank] += count
    if min(totals) < TARGET_PER_RANK:
        raise ValueError(f"Rank assignment cannot cover 16,000 updates: {totals}")
    digest = hashlib.sha256(
        "\n".join(f"{path}:{Path(path).stat().st_size}:{count}" for path, count in sorted(items)).encode()
    ).hexdigest()
    return {
        "status": "PASS", "arm": arm, "rank_shards": groups,
        "rank_sample_counts": totals, "target_per_rank": TARGET_PER_RANK,
        "sample_count": sum(totals), "source_shards": len(items),
        "inventory_sha256": digest, "seed": 20261009,
        "sampling_unit": "20TB packed image occurrence" if arm == "20tb" else "100TB exported stored item",
    }


def prepare_resume_metrics(run: Path) -> None:
    checkpoints = sorted(
        (int(path.name), path / "checkpoint.pth")
        for path in (run / "ckpt").iterdir() if path.is_dir() and path.name.isdigit()
    )
    if not checkpoints:
        raise ValueError(f"No optimizer checkpoint to resume: {run}")
    step, checkpoint = checkpoints[-1]
    if not checkpoint.is_file() or checkpoint.stat().st_size < 3_000_000_000:
        raise ValueError(f"Latest optimizer checkpoint is incomplete: {checkpoint}")
    metrics_path = run / "raw_loss_metrics.jsonl"
    lines = metrics_path.read_text().splitlines(keepends=True)
    if len(lines) < step + 1:
        raise ValueError(f"Metrics end before optimizer checkpoint {step}")
    for index, line in enumerate(lines[:step + 1]):
        if json.loads(line)["optimizer_update"] != index:
            raise ValueError(f"Optimizer metrics diverge at update {index}")
    if len(lines) > step + 1:
        stamp = int(time.time())
        abandoned = run / f"raw_loss_metrics.abandoned_after_{step}_{stamp}.jsonl"
        abandoned.write_text("".join(lines[step + 1:]))
        temporary = run / "raw_loss_metrics.resume_tmp"
        temporary.write_text("".join(lines[:step + 1]))
        os.replace(temporary, metrics_path)
        with (run / "resume_events.jsonl").open("a") as stream:
            stream.write(json.dumps({"checkpoint_step": step, "discarded_updates": len(lines) - step - 1,
                                     "abandoned_metrics": str(abandoned), "time_unix": time.time()}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("20tb", "100tb"), required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu-group", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--master-port", type=int, default=31859)
    parser.add_argument("--python-bin", default="/home/bbnc/anaconda3/envs/dinov3/bin/python")
    args = parser.parse_args()
    allocation = assignment(args.arm)
    allocation_path = DATA / f"{args.arm}_rank_assignment.json"
    if allocation_path.exists():
        prior = json.loads(allocation_path.read_text())
        if prior != allocation:
            raise ValueError(f"Existing rank assignment differs: {allocation_path}")
    else:
        allocation_path.parent.mkdir(parents=True, exist_ok=True)
        allocation_path.write_text(json.dumps(allocation, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps({"assignment": str(allocation_path), "rank_sample_counts": allocation["rank_sample_counts"]}))
        return

    gpus = args.gpu_group.split(",")
    if len(gpus) != WORLD_SIZE or len(set(gpus)) != WORLD_SIZE:
        raise ValueError("This protocol requires eight distinct GPUs")
    if args.smoke and args.resume:
        raise ValueError("Smoke runs do not resume")
    run = ROOT / f"outputs/01_training_runs/HS6_L_quality_long_noreplace_{args.arm}_ddp_gb1024_16k_20261009{'_smoke' if args.smoke else ''}"
    if not args.resume and run.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {run}")
    if args.resume and not (run / "launch_manifest.json").is_file():
        raise FileNotFoundError(f"No launch manifest for resuming: {run}")
    if not Path(args.python_bin).is_file():
        raise FileNotFoundError(args.python_bin)
    overrides = {
        "compute_precision.distributed_mode": "ddp",
        "train.batch_size_per_gpu": MICROBATCH,
        "optim.gradient_accumulation_steps": ACCUM,
        "train.OFFICIAL_EPOCH_LENGTH": 4098,
        "train.max_updates": 1 if args.smoke else UPDATES,
        "train.num_workers": 1,
        "train.wds_shuffle_buffer": 64 if args.arm == "100tb" else 3000,
        "train.wds_deterministic_resampling": True,
        "train.prefetch_factor": 1,
        "train.pin_memory": False,
        "train.checkpointing": True,
        "train.checkpointing_full": True,
        "train.checkpointing_blocks": 24,
        "evaluation.eval_period_iterations": 1 if args.smoke else 4000,
        "checkpointing.period": 1 if args.smoke else 4000,
        "checkpointing.max_to_keep": 2,
    }
    # The source list is read from this glob; rank assignment validates its exact members.
    if args.arm == "20tb":
        source_patterns = [
            "/mnt/deepcad_nfs/deepcad_100t/final-data/20TB_takeover_20260922_2350/route2_strict_pathology_20260923/route2_15tb_no_old5_micro_tars/filtered_projection_20TB_nested-r*.tar",
            "/mnt/deepcad_nfs/deepcad_100t/final-data/20TB_takeover_20260922_2350/route2_strict_pathology_20260923/route2_5tb_boundary_tars/filtered_projection_20TB_nested-r*.tar",
        ]
        if sorted(set(path for pattern in source_patterns for path in glob.glob(pattern))) != sorted(
            path for group in allocation["rank_shards"] for path in group
        ):
            raise ValueError("Source glob and audited rank assignment differ")
        overrides["train.dataset_path"] = "packwds_once_robust:" + ";".join(source_patterns) + "::pct=1,99"
    else:
        overrides["train.dataset_path"] = "raw100tb_once_robust:" + str(HUNDRED_STREAM.parent)
    command = [
        args.python_bin, "-m", "torch.distributed.run", "--nproc_per_node=8",
        f"--master_port={args.master_port}", "dinov3/train/train.py",
        "--config-file", str(BASE), "--output-dir", str(run),
        *(["--no-resume"] if not args.resume else []),
        *(f"{key}={json.dumps(value) if not isinstance(value, str) else value}" for key, value in overrides.items()),
    ]
    manifest = {
        "arm": args.arm, "objective": "matched finite no-replacement quality comparison",
        "source_population": allocation["sampling_unit"],
        "sampling": "global SRS candidate set, rank-disjoint raw-tar streaming" if args.arm == "100tb" else "random finite pass over complete 20TB route-2 pool",
        "source_shards": allocation["source_shards"],
        "source_images": allocation["sample_count"],
        "inventory_sha256": allocation["inventory_sha256"],
        "assignment": str(allocation_path), "target_updates": 1 if args.smoke else UPDATES,
        "effective_batch": WORLD_SIZE * MICROBATCH * ACCUM,
        "target_unique_visits": 1024 if args.smoke else TARGET_TOTAL,
        "schedule_reference": str(BASE),
        "schedule_total_updates": 61 * 4098,
        "warmup_updates": 3 * 4098,
        "natural_20tb_mix_note": "Full 20TB pool is about 23% boundary, not the old resampled 30% boundary recipe",
        "100tb_candidate_failure_fraction": None,
        "command": command, "created_unix": time.time(),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if args.resume:
        prepare_resume_metrics(run)
    if not args.resume:
        run.mkdir(parents=True)
        (run / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(gpus), DQ_FINITE_SHARD_ASSIGNMENT=str(allocation_path),
        DQ_STREAM_FAILURE_LOG_DIR=str(run),
        PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
        PYTORCH_ALLOC_CONF="expandable_segments:True", NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1",
        MALLOC_MMAP_THRESHOLD_="131072", MALLOC_ARENA_MAX="4",
    )
    with (run / "console.log").open("a" if args.resume else "w", buffering=1) as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (run / "exit.json").write_text(json.dumps({"returncode": result.returncode, "time_unix": time.time()}) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
