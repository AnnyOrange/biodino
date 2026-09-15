#!/usr/bin/env python3
"""Lock the batch-64 OOD correction for the legacy Gram campaign."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/02_eval_runs/hs6_l5_gram_u488_legacy_dataset_test_20260912"
PARENT = CAMPAIGN / "campaign_manifest.json"
AMENDMENT = CAMPAIGN / "campaign_manifest.ood_batch64_amendment.json"
PLAN = ROOT / "Evaluation Rules/plans/hs6_l5_gram_legacy_ood_batch64_amendment_20260912.md"
WORKER = ROOT / "scripts/run_hs6_l_6m_full_eval_fleet_worker.py"
LAUNCHER = ROOT / "scripts/launch_hs6_l5_gram_legacy_ood_batch64_worker_20260912.sh"
RUNNER = ROOT / "outputs/02_eval_runtime/dinov3_a029eef0_full_registry.GOjODo/scripts/run_bio_benchmark_all.sh"
ARMS = ("control", "anchor7807", "anchor17079")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    if AMENDMENT.exists():
        payload = json.loads(AMENDMENT.read_text())
        if payload.get("status") != "LOCKED_BEFORE_VALID_OOD_RUN":
            raise RuntimeError(f"incompatible amendment: {AMENDMENT}")
        print(f"[amendment] already locked: {AMENDMENT}")
        return 0
    for path in (PARENT, PLAN, WORKER, LAUNCHER, RUNNER):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    parent = json.loads(PARENT.read_text())
    if parent["evaluation"]["frozen_batch_size"] != 64:
        raise RuntimeError("parent campaign did not lock frozen batch 64")
    invalid = {}
    for arm in ARMS:
        archive = CAMPAIGN / arm / "_invalid_protocol/ood_batch32_20260912T041800Z"
        if not archive.is_dir():
            raise FileNotFoundError(archive)
        state = CAMPAIGN / arm / "_state"
        if (state / "done/ckpt_20495__ood.json").exists():
            raise RuntimeError(f"refusing amendment after a valid OOD marker: {arm}")
        claim = state / "claims/ckpt_20495__ood.lock"
        if claim.exists() and not (claim / "PROTOCOL_HOLD_BATCH64").is_file():
            raise RuntimeError(f"OOD is still claimed by a worker: {arm}")
        invalid[arm] = str(archive.resolve())
    payload = {
        "status": "LOCKED_BEFORE_VALID_OOD_RUN",
        "created_at_unix": time.time(),
        "created_host": platform.node(),
        "reason": "fleet worker overrode locked OOD batch 64 with invalid batch 32",
        "parent_manifest": file_record(PARENT),
        "invalid_batch32_archives": invalid,
        "corrected_protocol": {
            "lane": "ood",
            "batch_size": 64,
            "frozen_batch_size": 64,
            "seed": 0,
            "channel_policy": "auto",
            "autocast_dtype": "bf16",
            "tasks": ["xray", "cryo"],
            "reservation": "PROTOCOL_HOLD_BATCH64; corrected workers replace it atomically",
        },
        "code": {
            "plan": file_record(PLAN),
            "worker": file_record(WORKER),
            "launcher": file_record(LAUNCHER),
            "runner": file_record(RUNNER),
        },
    }
    atomic_json(AMENDMENT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
