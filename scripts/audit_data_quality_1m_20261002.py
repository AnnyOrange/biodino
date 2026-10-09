#!/usr/bin/env python3
"""Verify a completed matched-budget data-quality training arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path("/mnt/huawei_deepcad/dinov3/plot/fig2/data_quality")
ARMS = ("1tb", "5tb", "20tb", "100tb", "1pb", "20tb_route1", "20tb_route1_ddp", "20tb_route2_ddp")
UPDATES = 976
PER_RANK = 249_856
TOTAL = 999_424


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(arm: str) -> dict:
    run = ROOT / "training" / arm
    launch = json.loads((run / "launch_manifest.json").read_text())
    exit_record = json.loads((run / "exit.json").read_text())
    if launch["arm"] != arm or launch["optimizer_updates"] != UPDATES or launch["target_image_visits"] != TOTAL:
        raise ValueError("Launch manifest budget mismatch")
    if launch["effective_batch"] != 1024 or exit_record.get("returncode") != 0:
        raise ValueError("Training did not finish with effective batch 1024")
    metrics = [json.loads(line) for line in (run / "raw_loss_metrics.jsonl").read_text().splitlines() if line]
    if len(metrics) != UPDATES or [row["optimizer_update"] for row in metrics] != list(range(UPDATES)):
        raise ValueError(f"Expected consecutive updates 0..975, got {len(metrics)} rows")
    if any(row["effective_global_batch_size"] != 1024 or row["samples_seen"] != (i + 1) * 1024
           for i, row in enumerate(metrics)):
        raise ValueError("Metric batch or image-visit count mismatch")
    config = run / "config.yaml"
    cfg = yaml.safe_load(config.read_text())
    if arm in ("20tb_route1_ddp", "20tb_route2_ddp") and (cfg["compute_precision"]["distributed_mode"] != "ddp" or
                                                              launch.get("distributed_mode") != "ddp"):
        raise ValueError("20TB DDP arm did not train with DDP")
    if cfg["train"]["batch_size_per_gpu"] != 8 or cfg["optim"]["gradient_accumulation_steps"] != 32:
        raise ValueError("Configured microbatch or accumulation mismatch")
    if cfg["train"]["OFFICIAL_EPOCH_LENGTH"] != UPDATES or cfg["optim"]["epochs"] != 1:
        raise ValueError("Configured schedule mismatch")
    if cfg["crops"]["rgb_mean"] != [0.5] * 3 or cfg["crops"]["rgb_std"] != [0.35] * 3:
        raise ValueError("Configured image normalization mismatch")
    if not cfg["train"]["dataset_path"].startswith("packwds_once_robust:"):
        raise ValueError("Training did not use finite robust WDS")
    all_keys: set[str] = set()
    rank_counts = {}
    for rank in range(4):
        path = run / f"consumed_sample_keys_rank{rank:02d}.jsonl"
        count = 0
        with path.open() as stream:
            for line in stream:
                keys = json.loads(line)
                if len(keys) != 8:
                    raise ValueError(f"Rank {rank}: invalid microbatch size")
                for key in keys:
                    if key in all_keys:
                        raise ValueError(f"Repeated image across ranks: {key}")
                    all_keys.add(key)
                    count += 1
        if count != PER_RANK:
            raise ValueError(f"Rank {rank}: {count} consumed images, expected {PER_RANK}")
        rank_counts[str(rank)] = count
    if len(all_keys) != TOTAL:
        raise ValueError(f"Unique image count {len(all_keys)}, expected {TOTAL}")
    checkpoint = run / "eval/training_975/teacher_checkpoint.pth"
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000_000:
        raise ValueError("Final teacher checkpoint missing or truncated")
    result = {
        "status": "PASS", "arm": arm, "optimizer_updates": UPDATES,
        "effective_batch": 1024, "consumed_unique_images": len(all_keys),
        "per_rank_consumed_images": rank_counts,
        "first_loss": metrics[0]["total_loss"], "last_loss": metrics[-1]["total_loss"],
        "source_inventory_sha256": launch["source_inventory_sha256"],
        "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(config), "config_sha256": sha256(config),
    }
    (run / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.arm), indent=2))


if __name__ == "__main__":
    main()
