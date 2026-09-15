#!/usr/bin/env python3
"""Lock explicit-lane OOD dispatch after the pre-result scheduling correction."""

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
V1 = CAMPAIGN / "campaign_manifest.ood_batch64_amendment.json"
OUTPUT = CAMPAIGN / "campaign_manifest.ood_batch64_dispatch_v2.json"
INCIDENT = CAMPAIGN / "_runtime_incidents/ood_batch64_dispatch_race_20260912T042100Z"
PLAN = ROOT / "Evaluation Rules/plans/hs6_l5_gram_legacy_ood_batch64_dispatch_v2_20260912.md"
WORKER = ROOT / "scripts/run_hs6_l_6m_full_eval_fleet_worker.py"
LAUNCHER = ROOT / "scripts/launch_hs6_l5_gram_legacy_ood_batch64_worker_20260912.sh"
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


def main() -> int:
    if OUTPUT.exists():
        payload = json.loads(OUTPUT.read_text())
        if payload.get("status") != "LOCKED_BEFORE_VALID_OOD_RUN":
            raise RuntimeError(f"incompatible dispatch manifest: {OUTPUT}")
        print(f"[dispatch-v2] already locked: {OUTPUT}")
        return 0
    for path in (V1, PLAN, WORKER, LAUNCHER):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if not INCIDENT.is_dir():
        raise FileNotFoundError(INCIDENT)
    reservations = {}
    for arm in ARMS:
        state = CAMPAIGN / arm / "_state"
        hold = state / "claims/ckpt_20495__ood.lock/PROTOCOL_HOLD_BATCH64"
        if not hold.is_file():
            raise RuntimeError(f"missing batch-64 OOD reservation: {arm}")
        if (state / "done/ckpt_20495__ood.json").exists():
            raise RuntimeError(f"valid OOD result already exists: {arm}")
        reservations[arm] = str(hold.resolve())
    payload = {
        "status": "LOCKED_BEFORE_VALID_OOD_RUN",
        "created_at_unix": time.time(),
        "created_host": platform.node(),
        "supersedes_dispatch_only": file_record(V1),
        "scientific_protocol_unchanged": True,
        "batch_size": 64,
        "included_lanes": ["ood"],
        "reservation_marker": "PROTOCOL_HOLD_BATCH64",
        "reservations": reservations,
        "invalid_misdispatch_archive": str(INCIDENT.resolve()),
        "affected_output_files_created_or_modified_after_dispatch": 0,
        "code": {
            "plan": file_record(PLAN),
            "worker": file_record(WORKER),
            "launcher": file_record(LAUNCHER),
        },
    }
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
