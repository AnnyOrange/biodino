#!/usr/bin/env python3
"""Fail-closed validation for the ck29279 official-Gram trajectory arms."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/huawei_deepcad/dinov3")
COMMON_NAME = "ck29279_u2440_gb1024_realgb256_4xdeepcad_curve_20260915"
RUNS = {
    "official": ROOT / "outputs/01_training_runs" / f"HS6_L5_{COMMON_NAME}_official_gram_a7807",
    "control": ROOT / "outputs/01_training_runs" / f"HS6_L5_{COMMON_NAME}_matched_control",
}
BASE = 29279
ENDPOINT = 31719
UPDATES = 2440
SNAPSHOTS = (29767, 30255, 30743, 31231, 31719)
REPORT = ROOT / "outputs/03_comparisons/hs6_l5_ck29279_official_gram_curve_20260915/training_validation.json"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object row at {path}:{line_number}")
            rows.append(row)
    return rows


def validate_arm(name: str, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_rows(root / "raw_loss_metrics.jsonl")
    expected = list(range(BASE + 1, ENDPOINT + 1))
    observed = [int(row["optimizer_update"]) for row in rows]
    if observed != expected or len(rows) != UPDATES:
        raise RuntimeError(f"{name} update coverage mismatch: {len(rows)} rows")

    for row in rows:
        if row.get("controlled_data_stream") is not True:
            raise RuntimeError(f"{name} has an uncontrolled data-stream row")
        if row.get("batch_sample_key_digest") in (None, ""):
            raise RuntimeError(f"{name} has a missing sample digest")
        if int(row["real_global_batch_size"]) != 256:
            raise RuntimeError(f"{name} real global batch mismatch")
        if int(row["effective_global_batch_size"]) != 1024:
            raise RuntimeError(f"{name} effective global batch mismatch")
        for metric in ("total_loss", "dino_local_crops_loss", "ibot_loss"):
            if not math.isfinite(float(row[metric])):
                raise RuntimeError(f"{name} has non-finite {metric}")

    missing = []
    for checkpoint in SNAPSHOTS:
        teacher = root / f"eval/training_{checkpoint}/teacher_checkpoint.pth"
        full = root / f"ckpt/{checkpoint}/checkpoint.pth"
        if not teacher.is_file() or teacher.stat().st_size < 1_000_000_000:
            missing.append(str(teacher))
        if not full.is_file() or full.stat().st_size < 6_000_000_000:
            missing.append(str(full))
    if missing:
        raise RuntimeError(f"{name} missing complete checkpoints: {missing}")

    return rows, {
        "updates": len(rows),
        "first_update": observed[0],
        "last_update": observed[-1],
        "snapshots": list(SNAPSHOTS),
        "gram_enabled": bool(rows[-1].get("gram_loss") is not None),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    arm_rows = {}
    arm_reports = {}
    for name, root in RUNS.items():
        arm_rows[name], arm_reports[name] = validate_arm(name, root)

    digest_mismatches = []
    for official, control in zip(arm_rows["official"], arm_rows["control"], strict=True):
        if official["batch_sample_key_digest"] != control["batch_sample_key_digest"]:
            digest_mismatches.append(int(official["optimizer_update"]))
    if digest_mismatches:
        raise RuntimeError(f"sample-digest mismatches: {digest_mismatches[:20]}")

    report = {
        "status": "VALID_MATCHED_TRAINING_COMPLETE",
        "base_checkpoint": BASE,
        "endpoint_checkpoint": ENDPOINT,
        "sample_digest_matches": UPDATES,
        "sample_digest_total": UPDATES,
        "arms": arm_reports,
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

