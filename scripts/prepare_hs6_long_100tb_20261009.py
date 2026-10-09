#!/usr/bin/env python3
"""Continue the 100TB global-SRS finite training data preparation."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3")
DATA = Path("/mnt/huawei_blm/hs6_long_uniform_100tb_vs_20tb_20261009")
MANIFEST = DATA / "manifests/100tb_candidates_16500000.csv"
OLD_PREFIX = Path("/mnt/huawei_blm/random_1pb_100tb_global_v3/manifests/100tb_candidates_2m.csv")
STAGE = DATA / "100tb_stage_16500000"
REPAIR = DATA / "100tb_repair_16500000"
FINAL = DATA / "100tb_final_16400000"
STATUS = DATA / "PREPARATION_STATUS.json"
PYTHON = "/home/inspur/anaconda3/envs/siglip2_env/bin/python"


def record(state: str, **detail) -> None:
    STATUS.write_text(json.dumps({"state": state, "time_unix": time.time(), **detail}, indent=2) + "\n")


def run(stage: str, args: list[str], result: Path) -> None:
    if result.is_file():
        print(f"{stage}: complete, using {result}", flush=True)
        return
    record(stage, command=args)
    log_path = DATA / "logs" / f"{stage}.log"
    with log_path.open("a", buffering=1) as log:
        code = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    if code or not result.is_file():
        raise RuntimeError(f"{stage} failed ({code}); inspect {log_path}")


def verify_historical_prefix() -> None:
    with MANIFEST.open("rb") as current, OLD_PREFIX.open("rb") as previous:
        for line_index in range(2_000_001):
            if current.readline() != previous.readline():
                raise ValueError(f"100TB candidate permutation changed at CSV line {line_index + 1}")
        if previous.readline():
            raise ValueError("Historical 2M reference has unexpected extra rows")


def main() -> None:
    try:
        while not MANIFEST.is_file() or not MANIFEST.with_suffix(".json").is_file():
            record("WAITING_FOR_CANDIDATES", manifest=str(MANIFEST))
            if subprocess.run(["tmux", "has-session", "-t", "hs6_100tb_srs16m"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
                raise RuntimeError("100TB candidate sampler stopped without a complete manifest")
            time.sleep(60)
        sample = json.loads(MANIFEST.with_suffix(".json").read_text())
        if sample["candidate_count"] != 16_500_000 or sample["universe"]["storage_count"] != 26_614_979:
            raise ValueError("100TB sampled population or candidate count changed")
        record("VERIFYING_HISTORICAL_PREFIX")
        verify_historical_prefix()
        run("MATERIALIZING", [
            PYTHON, "-u", "scripts/materialize_global_100tb_wds.py",
            "--candidates", str(MANIFEST), "--max-candidates", "16500000",
            "--out-root", str(STAGE), "--workers", "8",
            "--samples-per-shard", "2000", "--seed", "20260924",
        ], STAGE / "materialization.json")
        run("REPAIRING", [
            PYTHON, "-u", "scripts/repair_global_100tb_failed_frames.py",
            "--candidates", str(MANIFEST), "--stage-root", str(STAGE),
            "--out-root", str(REPAIR), "--max-candidates", "16500000",
            "--seed", "20260924",
        ], REPAIR / "repair.json")
        run("FINALIZING", [
            PYTHON, "-u", "scripts/finalize_global_priority_wds.py",
            "--stage-root", str(STAGE), "--repair-root", str(REPAIR),
            "--out-root", str(FINAL), "--expected-candidates", "16500000",
            "--samples", "16400000", "--samples-per-shard", "2000",
            "--max-failure-fraction", "0.01",
            "--sampling-population", "readable exported stored items in the full 100TB index pool",
        ], FINAL / "finalization.json")
        final = json.loads((FINAL / "finalization.json").read_text())
        if final["samples"] != 16_400_000 or final["wds_shards"] != 8200:
            raise ValueError("100TB final sample count or shard count mismatch")
        run("ASSIGNING", [
            "/home/inspur/anaconda3/envs/dinov3/bin/python",
            "scripts/launch_hs6_long_finite_quality_20261009.py",
            "--arm", "100tb", "--prepare-only",
        ], DATA / "100tb_rank_assignment.json")
        record("READY", finalization=str(FINAL / "finalization.json"),
               assignment=str(DATA / "100tb_rank_assignment.json"),
               samples=final["samples"], candidate_failure_fraction=final["candidate_failure_fraction"])
    except Exception as exc:
        record("ERROR", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
