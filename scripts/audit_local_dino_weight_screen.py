#!/usr/bin/env python3
"""Audit causal comparability and health of a local-DINO-weight screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import yaml


ARMS = {"w1": 1.0, "w025": 0.25, "w0": 0.0}
FINITE_METRICS = (
    "total_loss",
    "dino_local_crops_loss",
    "dino_global_crops_loss",
    "ibot_loss",
    "koleo_loss",
    "backbone_grad_norm",
    "dino_head_grad_norm",
    "ibot_head_grad_norm",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON at {path}:{line_number}: {error}") from error
    return rows


def _normalized_config(path: Path) -> tuple[dict[str, Any], float]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    weight = float(config["dino"]["local_loss_weight_schedule"]["start"])
    normalized = copy.deepcopy(config)
    normalized["train"].pop("output_dir", None)
    schedule = normalized["dino"]["local_loss_weight_schedule"]
    for field in ("start", "peak", "end"):
        schedule[field] = "<ARM_WEIGHT>"
    return normalized, weight


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_inventory(run_dir: Path) -> list[dict[str, Any]]:
    checkpoints = []
    for path in sorted((run_dir / "ckpt").glob("*/checkpoint.pth"), key=lambda x: int(x.parent.name)):
        checkpoints.append(
            {
                "iteration": int(path.parent.name),
                "bytes": path.stat().st_size,
                "is_readable_zip": zipfile.is_zipfile(path),
            }
        )
    return checkpoints


def _metric_range(rows: list[dict[str, Any]], name: str) -> dict[str, float]:
    values = [float(row[name]) for row in rows]
    return {"min": min(values), "max": max(values), "last": values[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1024, help="Expected completed optimizer updates")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs: dict[str, dict[str, Any]] = {}
    normalized_digests: dict[str, str] = {}
    failures: list[str] = []
    for arm, expected_weight in ARMS.items():
        run_dir = args.root / f"{arm}_u{args.updates}_gb1024_seed{args.seed}"
        metrics_path = run_dir / "raw_loss_metrics.jsonl"
        config_path = run_dir / "config.yaml"
        if not metrics_path.is_file() or not config_path.is_file():
            failures.append(f"{arm}: missing config or raw metrics")
            continue
        rows = _load_jsonl(metrics_path)
        config, configured_weight = _normalized_config(config_path)
        normalized_digests[arm] = _canonical_digest(config)
        updates = [int(row["optimizer_update"]) for row in rows]
        contiguous = updates == list(range(len(rows)))
        finite = all(
            metric in row and math.isfinite(float(row[metric]))
            for row in rows
            for metric in FINITE_METRICS
        )
        logged_weights = sorted({float(row["dino_local_loss_weight"]) for row in rows})
        checkpoints = _checkpoint_inventory(run_dir)
        runs[arm] = {
            "run_dir": str(run_dir.resolve()),
            "completed_updates": len(rows),
            "first_update": updates[0] if updates else None,
            "last_update": updates[-1] if updates else None,
            "updates_contiguous": contiguous,
            "configured_weight": configured_weight,
            "logged_weights": logged_weights,
            "all_health_metrics_finite": finite,
            "batch_digest_first": rows[0].get("batch_sample_key_digest") if rows else None,
            "batch_digest_last": rows[-1].get("batch_sample_key_digest") if rows else None,
            "metric_ranges": {
                metric: _metric_range(rows, metric) for metric in FINITE_METRICS if rows
            },
            "checkpoints": checkpoints,
        }
        if len(rows) != args.updates:
            failures.append(f"{arm}: completed {len(rows)}/{args.updates} updates")
        if not contiguous:
            failures.append(f"{arm}: optimizer updates are not contiguous from zero")
        if configured_weight != expected_weight or logged_weights != [expected_weight]:
            failures.append(
                f"{arm}: expected weight {expected_weight}, config={configured_weight}, logs={logged_weights}"
            )
        if not finite:
            failures.append(f"{arm}: non-finite or missing health metric")
        if not checkpoints or not all(item["is_readable_zip"] for item in checkpoints):
            failures.append(f"{arm}: no complete readable checkpoint")

    config_match = len(set(normalized_digests.values())) == 1 and len(normalized_digests) == len(ARMS)
    if not config_match:
        failures.append("normalized configs differ outside output_dir and local weight")

    common_updates = min((run["completed_updates"] for run in runs.values()), default=0)
    digest_mismatches: list[dict[str, Any]] = []
    raw_rows = {
        arm: _load_jsonl(Path(run["run_dir"]) / "raw_loss_metrics.jsonl")
        for arm, run in runs.items()
    }
    if len(raw_rows) == len(ARMS):
        for index in range(common_updates):
            digests = {arm: rows[index].get("batch_sample_key_digest") for arm, rows in raw_rows.items()}
            if len(set(digests.values())) != 1:
                digest_mismatches.append({"optimizer_update": index, "digests": digests})
    if digest_mismatches:
        failures.append(f"batch-key digest differs at {len(digest_mismatches)} common updates")

    update_zero_identity: dict[str, Any] = {"checked": False}
    if len(raw_rows) == len(ARMS) and all(raw_rows.values()):
        first = {arm: rows[0] for arm, rows in raw_rows.items()}
        invariant_metrics = ("dino_local_crops_loss", "dino_global_crops_loss", "ibot_loss", "koleo_loss")
        max_invariant_delta = max(
            max(float(first[arm][metric]) for arm in ARMS)
            - min(float(first[arm][metric]) for arm in ARMS)
            for metric in invariant_metrics
        )
        local_scale = 16.0 / 18.0  # 8 local x 2 global terms versus 2 global off-diagonal terms.
        expected_w1_minus_w0 = local_scale * float(first["w1"]["dino_local_crops_loss"])
        observed_w1_minus_w0 = float(first["w1"]["total_loss"]) - float(first["w0"]["total_loss"])
        residual = observed_w1_minus_w0 - expected_w1_minus_w0
        update_zero_identity = {
            "checked": True,
            "max_invariant_loss_delta": max_invariant_delta,
            "expected_total_w1_minus_w0": expected_w1_minus_w0,
            "observed_total_w1_minus_w0": observed_w1_minus_w0,
            "residual": residual,
            "pass": max_invariant_delta <= 1e-6 and abs(residual) <= 1e-4,
        }
        if not update_zero_identity["pass"]:
            failures.append("update-zero loss decomposition is inconsistent with the configured intervention")

    report = {
        "audit": "local_dino_weight_causal_screen_v1",
        "root": str(args.root.resolve()),
        "expected_updates": args.updates,
        "expected_seed": args.seed,
        "normalized_config_match": config_match,
        "normalized_config_digests": normalized_digests,
        "common_updates_checked": common_updates,
        "batch_digest_mismatch_count": len(digest_mismatches),
        "batch_digest_mismatches": digest_mismatches[:20],
        "update_zero_loss_identity": update_zero_identity,
        "runs": runs,
        "failures": failures,
        "pass": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pass": report["pass"], "failures": failures}, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
