#!/usr/bin/env python3
"""Audit completed HS6-L finite-stream training and its actual image visits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path("/mnt/huawei_deepcad/dinov3/outputs/01_training_runs")


def audit(arm: str, smoke: bool) -> dict:
    suffix = "_smoke" if smoke else ""
    run = ROOT / f"HS6_L_quality_long_noreplace_{arm}_ddp_gb1024_16k_20261009{suffix}"
    launch = json.loads((run / "launch_manifest.json").read_text())
    completion = json.loads((run / "exit.json").read_text())
    updates = 1 if smoke else 16_000
    if launch["target_updates"] != updates or completion["returncode"] != 0:
        raise ValueError("Launch budget or training exit code is invalid")
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    if cfg["compute_precision"]["distributed_mode"] != "ddp" or cfg["gram"]["use_loss"]:
        raise ValueError("Expected DDP without GRAM")
    if cfg["train"]["batch_size_per_gpu"] != 8 or cfg["optim"]["gradient_accumulation_steps"] != 16:
        raise ValueError("Effective batch is not eight ranks x eight images x sixteen microsteps")
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
    for rank in range(8):
        count = 0
        with (run / f"consumed_sample_keys_rank{rank:02d}.jsonl").open() as stream:
            for line in stream:
                keys = json.loads(line)
                if len(keys) != 8:
                    raise ValueError(f"Rank {rank} has an incomplete microbatch")
                for key in keys:
                    identity = key.rsplit("::", 1)[-1] if arm == "100tb" else key
                    if identity in unique:
                        raise ValueError(f"Image key repeated across the run: {key}")
                    unique.add(identity)
                    count += 1
        if count != updates * 16 * 8:
            raise ValueError(f"Rank {rank} consumed {count} images, expected {updates * 16 * 8}")
        counts.append(count)
    if len(unique) != updates * 1024:
        raise ValueError("Global unique image count differs from the optimizer budget")
    checkpoints = [0] if smoke else [3999, 7999, 11999, 15999]
    for step in checkpoints:
        teacher = run / f"eval/training_{step}/teacher_checkpoint.pth"
        if not teacher.is_file() or teacher.stat().st_size < 600_000_000:
            raise ValueError(f"Missing or short teacher checkpoint at update {step + 1}")
    result = {
        "status": "PASS", "arm": arm, "smoke": smoke,
        "optimizer_updates": updates, "effective_batch": 1024,
        "unique_image_visits": len(unique), "rank_counts": counts,
        "first_loss": metrics[0]["total_loss"], "last_loss": metrics[-1]["total_loss"],
        "checkpoints": checkpoints,
    }
    (run / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("20tb", "100tb"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.arm, args.smoke), indent=2))


if __name__ == "__main__":
    main()
