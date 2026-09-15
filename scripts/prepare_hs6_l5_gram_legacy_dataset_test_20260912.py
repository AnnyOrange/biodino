#!/usr/bin/env python3
"""Lock inputs for the observational legacy dataset-test Gram comparison."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = (
    ROOT / "outputs/02_eval_runs/hs6_l5_gram_u488_legacy_dataset_test_20260912"
)
REFERENCE_ROOT = (
    ROOT
    / "outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908"
)
EVAL_REPO = ROOT / "outputs/02_eval_runtime/dinov3_a029eef0_full_registry.GOjODo"
PLAN = ROOT / "Evaluation Rules/plans/hs6_l5_gram_legacy_dataset_test_20260912.md"
WORKER = ROOT / "scripts/run_hs6_l_6m_full_eval_fleet_worker.py"
LAUNCHER = ROOT / "scripts/launch_hs6_l5_gram_legacy_dataset_test_worker_20260912.sh"
RUNNER = EVAL_REPO / "scripts/run_bio_benchmark_all.sh"
ARMS = {
    "control": (
        "HS6_L5_ck20007_control_gram_a7807_u488_gb64_4x3090qi_screen_20260911"
    ),
    "anchor7807": (
        "HS6_L5_ck20007_official_gram_a7807_u488_gb64_4x3090qi_screen_20260911"
    ),
    "anchor17079": (
        "HS6_L5_ck20007_official_gram_a17079_u488_gb64_4x3090qi_anchor_ablation_20260911"
    ),
}
LANES = {
    "classification_a": "bloodmnist pathmnist tissuemnist breastmnist organamnist organcmnist organsmnist",
    "classification_b": "dermamnist octmnist pneumoniamnist retinamnist chestmnist bbbc048-cellcycle",
    "classification_c": "cyclops-protein-loc midog25-atypical pcam nct-crc-he lc25000 chammi-allen-task1",
    "classification_d": "chammi-allen-task2 chammi-cp-task1 chammi-cp-task2 chammi-cp-task3 chammi-hpa-task1 chammi-hpa-task2",
    "regression": "bbbc013 bbbc005 conic-cell-count livecell-cell-count",
    "retrieval": "lc25000 nct-crc-he-100 nct-crc-he-1k crc-val-he-7k hpa-subcellular rxrx1-cross",
    "detection": "livecell bbbc038 conic",
    "segmentation_a": "bbbc038 conic",
    "segmentation_b": "pannuke tissuenet",
    "segmentation_c": "livecell multimodal_cellseg",
    "segmentation_d": "monuseg cellpose",
    "ood": "xray cryo",
}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_manifest() -> dict[str, Any]:
    arms = {}
    for arm, run_name in ARMS.items():
        run = ROOT / "outputs/01_training_runs" / run_name
        checkpoint = run / "eval/training_20495/teacher_checkpoint.pth"
        config = run / "config.yaml"
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError(f"missing {arm} input: {checkpoint} or {config}")
        print(f"[manifest] hashing {arm}", flush=True)
        arms[arm] = {
            "run": run_name,
            "student_start_checkpoint": 20007,
            "endpoint_checkpoint": 20495,
            "gram_anchor_checkpoint": {
                "control": None,
                "anchor7807": 7807,
                "anchor17079": 17079,
            }[arm],
            "checkpoint": file_record(checkpoint),
            "config": file_record(config),
        }

    reference_status = REFERENCE_ROOT / "_online_status/ckpt_20007.status.json"
    status = json.loads(reference_status.read_text())
    if len(status.get("done_lanes", [])) != len(LANES) or status.get("terminal_lanes"):
        raise RuntimeError("historical ck20007 reference is not complete")

    return {
        "status": "LOCKED_BEFORE_RUN",
        "admission": "OBSERVATIONAL_LEGACY_PROTOCOL_REPRODUCTION",
        "created_at_unix": time.time(),
        "created_host": platform.node(),
        "question": (
            "Does corrected official-structure Gram preserve the historical "
            "5TB retrieval/clustering gains under the exact old dataset test?"
        ),
        "training": {
            "matched_updates": 488,
            "effective_global_batch": 64,
            "identical_sample_digests": 488,
            "student_crop": 256,
            "gram_teacher_crop": 512,
            "gram_img_level": True,
            "gram_normalized": True,
            "gram_weight": 2.0,
            "teacher_refresh_completed_updates": [20200, 20400],
        },
        "evaluation": {
            "runtime_registry": file_record(
                EVAL_REPO / "dinov3/eval/bio_frozen_eval/registry.py"
            ),
            "frozen_batch_size": 64,
            "autocast_dtype": "bf16",
            "seed": 0,
            "channel_policy": "auto",
            "split_protocol": "current",
            "resolution_protocol": "best",
            "lanes": LANES,
        },
        "historical_reference": {
            "checkpoint": 20007,
            "root": str(REFERENCE_ROOT.resolve()),
            "status": file_record(reference_status),
        },
        "arms": arms,
        "code": {
            "plan": file_record(PLAN),
            "worker": file_record(WORKER),
            "launcher": file_record(LAUNCHER),
            "runner": file_record(RUNNER),
        },
    }


def main() -> int:
    manifest_path = CAMPAIGN_ROOT / "campaign_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("status") != "LOCKED_BEFORE_RUN":
            raise RuntimeError(f"incompatible existing manifest: {manifest_path}")
        print(f"[manifest] already locked: {manifest_path}")
        return 0
    manifest = build_manifest()
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
