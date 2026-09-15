#!/usr/bin/env python3
"""Run a deterministic shard of the HS6-L label-free spatial curve."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_RUN = ROOT / (
    "outputs/01_training_runs/"
    "HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_"
    "5tv107_8x5090zxr_20260907"
)
DEFAULT_OUTPUT = ROOT / "outputs/00_reports/hs6_l5_label_free_spatial_curve_20260911"
STEP_PATTERN = re.compile(r"training_(\d+)$")


def checkpoint_rows() -> list[tuple[int, Path]]:
    rows = []
    for directory in (TRAIN_RUN / "eval").glob("training_*"):
        match = STEP_PATTERN.fullmatch(directory.name)
        checkpoint = directory / "teacher_checkpoint.pth"
        if match and checkpoint.is_file():
            rows.append((int(match.group(1)), checkpoint))
    rows.sort()
    if len(rows) != 49:
        raise RuntimeError(f"expected 49 checkpoints, found {len(rows)}")
    return rows


def valid_result(path: Path, checkpoint: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
        final = payload["layers"]["block_24"]["all"]
        return bool(
            payload.get("diagnostic") == "frozen_nested_local_global_spatial_signal_v1"
            and Path(payload["checkpoint"]).resolve() == checkpoint.resolve()
            and payload.get("seed") == 20260911
            and payload.get("n_images") == 128
            and payload.get("n_unique_keys") == 128
            and payload["crop_spec"]["photometric_policy"] == "bio_safe"
            and "true_minus_shifted" in final
            and "hit1_enrichment" in final
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")

    args.output.mkdir(parents=True, exist_ok=True)
    selected = checkpoint_rows()[args.shard_index :: args.num_shards]
    for index, (step, checkpoint) in enumerate(selected, start=1):
        destination = args.output / "checkpoints" / f"ck{step}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if valid_result(destination, checkpoint):
            print(f"[{index}/{len(selected)}] ck{step}: reuse", flush=True)
            continue
        command = [
            sys.executable,
            str(ROOT / "scripts/diagnose_local_global_spatial_signal.py"),
            "--checkpoint",
            str(checkpoint),
            "--train-config",
            str(TRAIN_RUN / "config.yaml"),
            "--output",
            str(destination),
            "--layers",
            "5,11,17,23",
            "--batch-size",
            "8",
            "--batches",
            "16",
            "--workers",
            "0",
            "--global-size",
            "256",
            "--local-size",
            "112",
            "--photometric-policy",
            "bio_safe",
            "--bootstrap-reps",
            "2000",
            "--seed",
            "20260911",
            "--device",
            args.device,
        ]
        print(f"[{index}/{len(selected)}] ck{step}: run", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        if not valid_result(destination, checkpoint):
            raise RuntimeError(f"invalid result after ck{step}: {destination}")
    print(f"shard {args.shard_index}/{args.num_shards}: COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
