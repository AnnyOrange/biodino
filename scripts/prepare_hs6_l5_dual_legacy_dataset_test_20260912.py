#!/usr/bin/env python3
"""Lock the dual-anchor endpoint before its observational frozen evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = (
    ROOT / "outputs/02_eval_runs/hs6_l5_dualgram_u488_legacy_dataset_test_20260912"
)
PARENT_CAMPAIGN = (
    ROOT / "outputs/02_eval_runs/hs6_l5_gram_u488_legacy_dataset_test_20260912"
)
RUN_NAME = "HS6_L5_ck20007_dualgram_pa7807_ga20007_u488_gb64_4x3090qi_dual_v3_20260912"
TRAIN_RUN = ROOT / "outputs/01_training_runs" / RUN_NAME
CONTROL_RUN = (
    ROOT
    / "outputs/01_training_runs"
    / "HS6_L5_ck20007_control_gram_a7807_u488_gb64_4x3090qi_screen_20260911"
)
PLAN = (
    ROOT
    / "Evaluation Rules/plans/hs6_l5_dual_anchor_legacy_dataset_test_addendum_20260912.md"
)
LAUNCHER = ROOT / "scripts/launch_hs6_l5_dual_legacy_dataset_test_worker_20260912.sh"
WORKER = ROOT / "scripts/run_hs6_l_6m_full_eval_fleet_worker.py"


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


def read_metrics(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def training_integrity() -> dict[str, Any]:
    dual_path = TRAIN_RUN / "raw_loss_metrics.jsonl"
    control_path = CONTROL_RUN / "raw_loss_metrics.jsonl"
    dual = read_metrics(dual_path)
    control_by_update = {int(row["optimizer_update"]): row for row in read_metrics(control_path)}
    expected = list(range(20008, 20496))
    updates = [int(row["optimizer_update"]) for row in dual]
    if updates != expected:
        raise RuntimeError(f"dual update sequence is incomplete or unordered: {len(updates)} rows")
    mismatched = [
        update
        for update, row in zip(updates, dual, strict=True)
        if control_by_update.get(update, {}).get("batch_sample_key_digest")
        != row.get("batch_sample_key_digest")
    ]
    relation = [float(row["gram_global_relation_loss"]) for row in dual]
    batches = {int(row["gram_global_relation_batch"]) for row in dual}
    if mismatched:
        raise RuntimeError(f"sample-digest mismatch at updates: {mismatched[:8]}")
    if batches != {64} or not all(math.isfinite(value) and value > 0 for value in relation):
        raise RuntimeError("global relation loss failed the finite/nontrivial batch-64 check")
    return {
        "updates": len(updates),
        "first_update": updates[0],
        "last_update": updates[-1],
        "sample_digest_matches_control": len(updates),
        "global_relation_batch": 64,
        "global_relation_loss_min": min(relation),
        "global_relation_loss_max": max(relation),
        "raw_metrics": file_record(dual_path),
        "control_raw_metrics": file_record(control_path),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    manifest_path = CAMPAIGN_ROOT / "campaign_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("status") != "LOCKED_AFTER_ENDPOINT_BEFORE_PROBES":
            raise RuntimeError(f"incompatible existing manifest: {manifest_path}")
        print(f"[manifest] already locked: {manifest_path}")
        return 0

    parent_manifest = PARENT_CAMPAIGN / "campaign_manifest.json"
    checkpoint = TRAIN_RUN / "eval/training_20495/teacher_checkpoint.pth"
    config = TRAIN_RUN / "config.yaml"
    for path in (parent_manifest, checkpoint, config, PLAN, LAUNCHER, WORKER):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    parent = json.loads(parent_manifest.read_text())
    integrity = training_integrity()
    payload = {
        "status": "LOCKED_AFTER_ENDPOINT_BEFORE_PROBES",
        "admission": "OBSERVATIONAL_LEGACY_PROTOCOL_REPRODUCTION",
        "created_at_unix": time.time(),
        "created_host": platform.node(),
        "parent_campaign_manifest": file_record(parent_manifest),
        "evaluation": parent["evaluation"],
        "arm": {
            "name": "dual_anchor_ck20495",
            "run": RUN_NAME,
            "student_start_checkpoint": 20007,
            "endpoint_checkpoint": 20495,
            "patch_gram_anchor_checkpoint": 7807,
            "global_relation_anchor_checkpoint": 20007,
            "patch_gram_weight": 2.0,
            "global_relation_weight": 0.5,
            "checkpoint": file_record(checkpoint),
            "config": file_record(config),
            "training_integrity": integrity,
        },
        "code": {
            "plan": file_record(PLAN),
            "launcher": file_record(LAUNCHER),
            "worker": file_record(WORKER),
        },
        "label_policy": (
            "No label entered SSL training or selected anchors/weights; frozen probes are "
            "evaluation-only and this addendum remains observational."
        ),
    }
    atomic_json(manifest_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
