#!/usr/bin/env python3
"""Audit the matched dual-anchor Gram screen and its label-free gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs/01_training_runs"
CONTROL_RUN = RUN_ROOT / "HS6_L5_ck20007_control_gram_a7807_u488_gb64_4x3090qi_screen_20260911"
OFFICIAL_RUN = RUN_ROOT / "HS6_L5_ck20007_official_gram_a7807_u488_gb64_4x3090qi_screen_20260911"
DUAL_RUN = RUN_ROOT / "HS6_L5_ck20007_dualgram_pa7807_ga20007_u488_gb64_4x3090qi_dual_v3_20260912"
SPATIAL_ROOT = ROOT / "outputs/00_reports/hs6_l5_official_gram_spatial_u488_20260911"
RELATION = ROOT / "outputs/00_reports/hs6_l5_dual_anchor_gram_screen_20260912/relation_drift.json"
OUTPUT = ROOT / "outputs/00_reports/hs6_l5_dual_anchor_gram_screen_20260912"
EXPECTED_UPDATES = 488
FIRST_UPDATE = 20008
LAST_UPDATE = 20495
BOOTSTRAP_REPS = 50_000
SEED = 20260912


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(child) for child in value)
    return True


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "first_100_mean": float(values[:100].mean()),
        "last_100_mean": float(values[-100:].mean()),
        "last": float(values[-1]),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def paired_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPS, 1000):
        stop = min(start + 1000, BOOTSTRAP_REPS)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def load_training_rows(run: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (run / "raw_loss_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != EXPECTED_UPDATES:
        raise RuntimeError(f"{run.name} has {len(rows)} rows, expected {EXPECTED_UPDATES}")
    updates = [int(row["optimizer_update"]) for row in rows]
    if updates != list(range(FIRST_UPDATE, LAST_UPDATE + 1)):
        raise RuntimeError(f"{run.name} has a non-contiguous update sequence")
    if not finite_tree(rows):
        raise RuntimeError(f"{run.name} contains a non-finite metric")
    return rows


def audit_training(control_run: Path, official_run: Path, dual_run: Path) -> dict[str, Any]:
    runs = {"control": control_run, "official_a7807": official_run, "dual": dual_run}
    rows = {name: load_training_rows(run) for name, run in runs.items()}
    control_digests = [row["batch_sample_key_digest"] for row in rows["control"]]
    digest_matches = {
        name: sum(
            row["batch_sample_key_digest"] == expected
            for row, expected in zip(arm_rows, control_digests, strict=True)
        )
        for name, arm_rows in rows.items()
    }
    if any(count != EXPECTED_UPDATES for count in digest_matches.values()):
        raise RuntimeError(f"sample stream mismatch: {digest_matches}")

    critical_config = {
        "dataset_path": "train.dataset_path",
        "batch_size_per_gpu": "train.batch_size_per_gpu",
        "num_workers": "train.num_workers",
        "seed": "train.seed",
        "global_crop": "crops.global_crops_size",
        "local_crop": "crops.local_crops_size",
        "augmentation": "crops.augmentation_policy",
        "effective_accumulation": "optim.gradient_accumulation_steps",
    }
    configs = {
        name: yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        for name, run in runs.items()
    }

    def get_path(payload: dict[str, Any], dotted: str) -> Any:
        value: Any = payload
        for part in dotted.split("."):
            value = value[part]
        return value

    config_values = {
        label: {name: get_path(config, path) for name, config in configs.items()}
        for label, path in critical_config.items()
    }
    mismatched = {
        label: values for label, values in config_values.items() if len(set(values.values())) != 1
    }
    if mismatched:
        raise RuntimeError(f"critical matched config mismatch: {mismatched}")

    dual = rows["dual"]
    relation_loss = np.asarray(
        [float(row["gram_global_relation_loss"]) for row in dual], dtype=np.float64
    )
    relation_batch = {int(row["gram_global_relation_batch"]) for row in dual}
    if relation_batch != {64} or not np.all(relation_loss > 0):
        raise RuntimeError(
            f"invalid global relation branch: batch={relation_batch}, min={relation_loss.min()}"
        )
    patch_loss = np.asarray([float(row["gram_loss"]) for row in dual], dtype=np.float64)
    train_metrics = {}
    for name, arm_rows in rows.items():
        keys = [
            "total_loss",
            "dino_global_crops_loss",
            "dino_local_crops_loss",
            "ibot_loss",
            "koleo_loss",
            "backbone_grad_norm",
        ]
        train_metrics[name] = {
            key: summary(np.asarray([float(row[key]) for row in arm_rows])) for key in keys
        }
    train_metrics["dual"]["gram_loss"] = summary(patch_loss)
    train_metrics["dual"]["gram_global_relation_loss"] = summary(relation_loss)
    return {
        "status": "VALID_MATCHED_DUAL_ANCHOR_SCREEN",
        "updates": EXPECTED_UPDATES,
        "optimizer_update_range": [FIRST_UPDATE, LAST_UPDATE],
        "sample_digest_matches": digest_matches,
        "critical_matched_config": config_values,
        "global_relation_batch": 64,
        "global_relation_loss_nontrivial": True,
        "metrics": train_metrics,
        "inputs": {
            name: {
                "run": str(run),
                "metrics_sha256": sha256(run / "raw_loss_metrics.jsonl"),
                "config_sha256": sha256(run / "config.yaml"),
            }
            for name, run in runs.items()
        },
    }


def audit_spatial(spatial_root: Path) -> dict[str, Any]:
    records = {
        name: read_json(spatial_root / filename)
        for name, filename in {
            "control": "control_records.json",
            "official_a7807": "official_records.json",
            "dual": "dual_records.json",
        }.items()
    }
    keys = records["control"]["keys"]
    areas = records["control"]["local_area"]
    for name, record in records.items():
        if record["keys"] != keys or record["local_area"] != areas:
            raise RuntimeError(f"spatial sample/crop mismatch for {name}")
    values = {
        name: np.asarray(
            record["layers"]["block_24"]["true_minus_shifted"], dtype=np.float64
        )
        for name, record in records.items()
    }
    if any(value.shape != (128,) for value in values.values()):
        raise RuntimeError("spatial gate requires exactly 128 paired images")
    control, official, dual = (
        values["control"],
        values["official_a7807"],
        values["dual"],
    )
    official_gain = float((official - control).mean())
    dual_gain = float((dual - control).mean())
    gate = {
        "locked_before_dual_endpoint": True,
        "criteria": {
            "dual_minus_control_at_least": 0.0,
            "retained_official_gain_at_least": 0.5,
        },
        "observed": {
            "control_mean": float(control.mean()),
            "official_a7807_mean": float(official.mean()),
            "dual_mean": float(dual.mean()),
            "official_a7807_minus_control": official_gain,
            "dual_minus_control": dual_gain,
            "retained_official_gain": dual_gain / official_gain,
            "dual_minus_control_ci95": paired_ci(dual - control, SEED),
            "dual_minus_official_a7807_ci95": paired_ci(dual - official, SEED + 1),
        },
    }
    gate["pass"] = dual_gain >= 0 and dual_gain >= 0.5 * official_gain
    return {
        "status": "VALID_PAIRED_LABEL_FREE_SPATIAL_GATE",
        "n_images": 128,
        "ordered_keys_exact_match": True,
        "local_areas_exact_match": True,
        "decision_gate": gate,
        "input_sha256": {
            name: sha256(
                spatial_root
                / {
                    "control": "control_records.json",
                    "official_a7807": "official_records.json",
                    "dual": "dual_records.json",
                }[name]
            )
            for name in records
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", type=Path, default=CONTROL_RUN)
    parser.add_argument("--official-run", type=Path, default=OFFICIAL_RUN)
    parser.add_argument("--dual-run", type=Path, default=DUAL_RUN)
    parser.add_argument("--spatial-root", type=Path, default=SPATIAL_ROOT)
    parser.add_argument("--relation", type=Path, default=RELATION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    training = audit_training(
        args.control_run.resolve(), args.official_run.resolve(), args.dual_run.resolve()
    )
    spatial = audit_spatial(args.spatial_root.resolve())
    relation = read_json(args.relation.resolve())
    if relation.get("status") != "VALID_COMPLETE_LABEL_FREE_RELATION_DIAGNOSTIC":
        raise RuntimeError("relation diagnostic is not valid and complete")
    relation_gate = relation.get("decision_gate")
    if not isinstance(relation_gate, dict):
        raise RuntimeError("relation diagnostic does not contain the locked decision gate")
    label_free_pass = bool(spatial["decision_gate"]["pass"] and relation_gate["pass"])
    payload = {
        "status": "VALID_COMPLETE_DUAL_ANCHOR_SCREEN",
        "admission": "LABEL_FREE_METHOD_GATE_WITH_FROZEN_EVAL_SEPARATE",
        "training": training,
        "spatial": spatial,
        "relation": relation,
        "label_free_gate_pass": label_free_pass,
        "gb1024_method_gate": (
            "PASS_PENDING_SEPARATE_LEGACY_EVAL_AUDIT"
            if label_free_pass
            else "FAIL_DO_NOT_SCALE"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_json = args.output / "analysis.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    relation_observed = relation_gate["observed"]
    spatial_observed = spatial["decision_gate"]["observed"]
    markdown = [
        "# HS6-L5 dual-anchor Gram screen",
        "",
        f"Status: `{payload['status']}`.",
        f"Label-free gate: `{'PASS' if label_free_pass else 'FAIL'}`.",
        "",
        "## Integrity",
        "",
        f"- Updates: {EXPECTED_UPDATES}/{EXPECTED_UPDATES}, ck{FIRST_UPDATE}--ck{LAST_UPDATE}.",
        f"- Sample digests: {training['sample_digest_matches']['dual']}/{EXPECTED_UPDATES} matched.",
        "- Global relation graph: 64 samples per view; every recorded loss is finite and nonzero.",
        "",
        "## Label-free gates",
        "",
        f"- Relation drift dual/official: {relation_observed['dual_over_official_a7807']:.4f} "
        "(required <= 0.8).",
        f"- Relation drift dual/control: {relation_observed['dual_over_control']:.4f} "
        "(required <= 1.1).",
        f"- Spatial dual-control: {spatial_observed['dual_minus_control']:+.6f}; retained official "
        f"gain: {spatial_observed['retained_official_gain']:.4f} (required >= 0.5).",
        "",
        "No labels or trained probes enter these gates. The legacy frozen evaluation is audited "
        "separately and cannot tune the anchors, weights, or thresholds.",
    ]
    (args.output / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_json), "label_free_gate_pass": label_free_pass}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
