#!/usr/bin/env python3
"""Audit completed HS6-L finite-stream training and its actual image visits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


ROOT = Path("/mnt/huawei_deepcad/dinov3/outputs/01_training_runs")
STREAM_INDEX = Path("/mnt/huawei_blm/hs6_long_uniform_100tb_vs_20tb_20261009/stream_index_exact")


def audit(arm: str, smoke: bool, run_path: Path | None = None) -> dict:
    suffix = "_smoke" if smoke else ""
    run = run_path or ROOT / f"HS6_L_quality_long_noreplace_{arm}_ddp_gb1024_16k_20261009{suffix}"
    launch = json.loads((run / "launch_manifest.json").read_text())
    completion = json.loads((run / "exit.json").read_text())
    updates = 1 if smoke else 16_000
    world_size = int(launch.get("world_size", 8))
    microbatch = int(launch.get("microbatch", 8))
    accumulation = int(launch.get("gradient_accumulation_steps", 16))
    if world_size * microbatch * accumulation != 1024:
        raise ValueError("Effective batch does not equal 1024")
    if launch["target_updates"] != updates or completion["returncode"] != 0:
        raise ValueError("Launch budget or training exit code is invalid")
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    if cfg["compute_precision"]["distributed_mode"] != "ddp" or cfg["gram"]["use_loss"]:
        raise ValueError("Expected DDP without GRAM")
    if cfg["train"]["batch_size_per_gpu"] != microbatch or \
            cfg["optim"]["gradient_accumulation_steps"] != accumulation:
        raise ValueError("Microbatch or gradient accumulation differs from launch manifest")
    if cfg["train"]["OFFICIAL_EPOCH_LENGTH"] != 4098 or cfg["optim"]["epochs"] != 61:
        raise ValueError("Schedule differs from the 20TB no-GRAM reference")
    if cfg["optim"]["warmup_epochs"] != 3 or cfg["optim"]["lr"] != 0.0001:
        raise ValueError("LR or warmup differs from the 20TB no-GRAM reference")
    expected_prefix = "raw100tb_once_robust:" if arm == "100tb" else "packwds_once_robust:"
    if cfg["train"]["max_updates"] != updates or not cfg["train"]["dataset_path"].startswith(expected_prefix):
        raise ValueError("Finite-stream budget or loader is wrong")
    metrics_path = run / "raw_loss_metrics.jsonl"
    with metrics_path.open() as stream:
        metrics = [json.loads(line) for line in stream if line.strip()]
    if len(metrics) != updates or [row["optimizer_update"] for row in metrics] != list(range(updates)):
        raise ValueError(f"Expected {updates} consecutive optimizer updates, got {len(metrics)}")
    if any(row["effective_global_batch_size"] != 1024 or row["samples_seen"] != (index + 1) * 1024
           for index, row in enumerate(metrics)):
        raise ValueError("Actual batch or image visit count differs from the protocol")
    unique = set()
    counts = []
    for rank in range(world_size):
        count = 0
        with (run / f"consumed_sample_keys_rank{rank:02d}.jsonl").open() as stream:
            for line in stream:
                keys = json.loads(line)
                if len(keys) != microbatch:
                    raise ValueError(f"Rank {rank} has an incomplete microbatch")
                for key in keys:
                    identity = key.rsplit("::", 1)[-1] if arm == "100tb" else key
                    if identity in unique:
                        raise ValueError(f"Image key repeated across the run: {key}")
                    unique.add(identity)
                    count += 1
        if count != updates * accumulation * microbatch:
            raise ValueError(f"Rank {rank} consumed {count} images, expected {updates * accumulation * microbatch}")
        counts.append(count)
    if len(unique) != updates * 1024:
        raise ValueError("Global unique image count differs from the optimizer budget")
    primary_missing = 0
    backup_used = 0
    if arm == "100tb":
        stream_index = Path(launch.get("stream_index", STREAM_INDEX))
        assignment = json.loads((stream_index / "assignment.json").read_text())
        if assignment["world_size"] != world_size:
            raise ValueError("100TB stream index world size differs from training")
        primary = set()
        backup = set()
        for rank in range(world_size):
            with (stream_index / f"rank{rank:02d}.tsv").open(newline="") as source:
                for _shard, item_id, _hint, priority in csv.reader(source, delimiter="\t"):
                    if item_id in primary or item_id in backup:
                        raise ValueError(f"100TB index repeats item {item_id}")
                    destination = primary if int(priority) < assignment["target_total"] else backup
                    destination.add(item_id)
        if len(primary) != 16_384_000 or len(primary) + len(backup) != 16_500_000:
            raise ValueError("100TB indexed candidate counts changed")
        extras = unique - primary
        if smoke:
            if extras:
                raise ValueError("100TB smoke consumed a backup or unknown item")
        else:
            missing = primary - unique
            if extras - backup or len(missing) != len(extras):
                raise ValueError("100TB actual visits differ from primary and backup candidate sets")
            primary_missing = len(missing)
            backup_used = len(extras)
            if primary_missing:
                logged_failures = set()
                for log_path in run.glob("raw100tb_failures_rank*.jsonl"):
                    with log_path.open() as source:
                        logged_failures.update(str(json.loads(line)["item_id"]) for line in source if line.strip())
                if not missing <= logged_failures:
                    raise ValueError("100TB used backups without logging every failed primary item")
    checkpoints = [0] if smoke else [3999, 7999, 11999, 15999]
    for step in checkpoints:
        teacher = run / f"eval/training_{step}/teacher_checkpoint.pth"
        if not teacher.is_file() or teacher.stat().st_size < 600_000_000:
            raise ValueError(f"Missing or short teacher checkpoint at update {step + 1}")
    result = {
        "status": "PASS", "arm": arm, "smoke": smoke,
        "optimizer_updates": updates, "effective_batch": 1024, "world_size": world_size,
        "unique_image_visits": len(unique), "rank_counts": counts,
        "100tb_primary_missing": primary_missing, "100tb_backup_used": backup_used,
        "first_loss": metrics[0]["total_loss"], "last_loss": metrics[-1]["total_loss"],
        "checkpoints": checkpoints,
    }
    (run / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("20tb", "100tb"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.arm, args.smoke, args.run), indent=2))


if __name__ == "__main__":
    main()
