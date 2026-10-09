#!/usr/bin/env python3
"""Complete the admitted Union-v4 gaps for four HS6-L 1M checkpoints.

The queue is intentionally limited to cells that are missing from the already
validated v3/shared campaigns.  CTC and OOD remain explicit admission blockers.
Each single-GPU worker claims one task at a time through an atomic directory on
the shared filesystem.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/mnt/huawei_deepcad/dinov3")
ROOT = REPO / "outputs/02_eval_runs/hs6_l_1tb_5tb_1pb_100tb_v4_completion_20260924"
SOURCE = Path("/mnt/huawei_deepcad/dinov3_selective_retention_snapshot_20260923")
PYTHON = Path("/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python")
BENCHMARK = Path("/mnt/huawei_deepcad/benchmark")
RXRX3_CACHE = REPO / "outputs/02_eval_inputs/formal_v3/rxrx3-core"
PROTOCOL = REPO / "Evaluation Rules/protocol_v4.json"
PROTOCOL_ID = "bio-eval-union-v4"
RXRX3_PROTOCOL = "crispr-query-guide-plate-disjoint-all-eligible-genes-v1"

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

ASSETS = (
    {
        "arm": "hs6_l_1tb_1m_e1_ck1024",
        "checkpoint_id": 1024,
        "checkpoint": REPO / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_seed0_20260818/ckpt/1024/checkpoint.pth",
        "config": REPO / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_seed0_20260818/config.yaml",
        "checkpoint_sha256": "0c873f8d388ec48bf572d88d4a149505004de2a80f2cc29b2046487ffba74d6f",
        "shared_evidence": REPO / "outputs/02_eval_runs/old_v3_protocol_union",
    },
    {
        "arm": "hs6_l_5tb_1m_ck975",
        "checkpoint_id": 975,
        "checkpoint": REPO / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907/eval/training_975/teacher_checkpoint.pth",
        "config": REPO / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907/config.yaml",
        "checkpoint_sha256": "554ee7b1f750072f51408786a78dd78b2d8d1479946b67cd9bfda43c6cfa44bc",
        "shared_evidence": REPO / "outputs/02_eval_runs/old_v3_protocol_union",
    },
    {
        "arm": "hs6_l_1pb_1m_e1_ck976",
        "checkpoint_id": 976,
        "checkpoint": REPO / "outputs/01_training_runs/HS6_L_random1pb1m_clean_robust_biosafe256_gb1024_lr1e4_e1_seed0_4x3090qi_20260916/eval/training_976/teacher_checkpoint.pth",
        "config": REPO / "outputs/01_training_runs/HS6_L_random1pb1m_clean_robust_biosafe256_gb1024_lr1e4_e1_seed0_4x3090qi_20260916/config.yaml",
        "checkpoint_sha256": "c6027758cf52748748af1b0e3454be2671d1b2eefde99094f94179d819cc1955",
        "shared_evidence": REPO / "outputs/02_eval_runs/random_100tb_vs_1pb_1m_v4_20260922",
    },
    {
        "arm": "hs6_l_100tb_1m_e1_ck976",
        "checkpoint_id": 976,
        "checkpoint": REPO / "outputs/01_training_runs/HS6_L_random100tb1m_robust_biosafe256_gb1024_lr1e4_e1_seed0_ddp_b128acc4_2xA100deepcad_20260922/eval/training_976/teacher_checkpoint.pth",
        "config": REPO / "outputs/01_training_runs/HS6_L_random100tb1m_robust_biosafe256_gb1024_lr1e4_e1_seed0_ddp_b128acc4_2xA100deepcad_20260922/config.yaml",
        "checkpoint_sha256": "f536d786f74c45bc9579352b08713d337bb4fb2b3f06d2797eca633b65f9f93b",
        "shared_evidence": REPO / "outputs/02_eval_runs/random_100tb_vs_1pb_1m_v4_20260922",
    },
)

LOCKED_INPUTS = {
    "/mnt/huawei_deepcad/benchmark_model/benchmark_runs/hs6_5tb_protocol_union_nonseg_20260921/locked_inputs/lc25000_classification_split.json": "84edcdf2d30bb2b989d7d4dcc42dad1e058c59f49e167693626bf72bdcc85fd8",
    "/mnt/huawei_deepcad/benchmark/Regression/CoNIC_Cell_Count/conic_cell_count.csv": "8640e1fd6365a5bd51269bb107725d86538a3cbfce59a79a9df3292a0c2ac0d1",
    "/mnt/huawei_deepcad/benchmark/Regression/LIVECell_Cell_Count/livecell_cell_count.csv": "ea9073c3edd5efc575c15732a2fb4c0b772f462b4c9cc56bbd52f43a536717d0",
    "/mnt/huawei_deepcad/benchmark/Retrieval_Clustering/NCT-CRC-HE/owkin_hf_parquet/data/nct_crc_he_100-00000-of-00001-25a54abad9e9e379.parquet": "f145a1e5c6aaa23bbab1743903b12cc784275550d5d51a98e32fdc4dce66af77",
    "/mnt/huawei_deepcad/dinov3/outputs/02_eval_inputs/formal_v3/rxrx3-core/split_manifest.jsonl": "94d570cb66d71de20e9ded203d8727623fe85a807d8b5b9cdbe0de3fce1318f5",
    "/mnt/huawei_deepcad/benchmark/segmentation/bbbc038/extracted/bbbc038_splits.npz": "4eb72dc1e58453126893261fedff661b7eb58dc9863ae04d991e7a5282f57ac4",
    "/mnt/huawei_deepcad/benchmark/segmentation/LIVECell/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json": "6788902479bc0ffd2838fb7160a778078f464ea6034ed76bb5a0e40263cb9e78",
    "/mnt/huawei_deepcad/benchmark/segmentation/LIVECell/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json": "eb72a68f5065d9ade1805b496eecdcd14ca275e5f1bc6aae7e07279bddadfdc7",
    "/mnt/huawei_deepcad/benchmark/segmentation/LIVECell/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json": "9eb5805ca031cebcd70bab4a4aa0c89d98646c206d333803a8d5b10ea7fe9960",
}

SOURCE_ENTRIES = (
    "dinov3/eval/bio_frozen_eval/run_classification.py",
    "dinov3/eval/bio_frozen_eval/run_retrieval_clustering.py",
    "dinov3/eval/bio_detection/center_probe.py",
    "dinov3/eval/bio_segmentation/scripts/run_linear_probe_pipeline.py",
    "scripts/run_external4_fixedbudget_model.py",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def frozen_task(asset: dict, dataset: str, order: int) -> dict:
    arm = asset["arm"]
    out = ROOT / "extension" / dataset / arm
    command = [
        str(PYTHON), "-B", "-m", "dinov3.eval.bio_frozen_eval.run_classification",
        "--checkpoint", str(asset["checkpoint"]), "--train-config", str(asset["config"]),
        "--benchmark-root", str(BENCHMARK), "--datasets", dataset,
        "--output-dir", str(out), "--model-name", arm,
        "--batch-size", "64", "--num-workers", "2", "--n-last-blocks", "1",
        "--autocast-dtype", "bf16", "--channel-policy", "auto",
        "--channel-tta-samples", "8", "--resolution-protocol", "best",
        "--split-protocol", "current", "--save-paths",
    ]
    family = "classification" if dataset == "lc25000" else "regression"
    return {
        "id": f"{arm}__{family}__{dataset}", "arm": arm, "family": family,
        "dataset": dataset, "order": order, "command": command,
        "cwd": str(SOURCE), "pythonpath": str(SOURCE),
        "output": str(out), "done_kind": "frozen_csv",
    }


def retrieval_task(asset: dict, dataset: str, order: int) -> dict:
    arm = asset["arm"]
    out = ROOT / "retrieval" / dataset / arm
    command = [
        str(PYTHON), "-B", "-m", "dinov3.eval.bio_frozen_eval.run_retrieval_clustering",
        "--checkpoint", str(asset["checkpoint"]), "--train-config", str(asset["config"]),
        "--benchmark-root", str(BENCHMARK), "--datasets", dataset,
        "--output-dir", str(out), "--model-name", arm,
        "--batch-size", "64", "--num-workers", "2", "--autocast-dtype", "bf16",
        "--n-last-blocks", "1", "--channel-policy", "auto",
        "--channel-tta-samples", "8", "--metric-device", "cpu", "--seed", "0",
    ]
    return {
        "id": f"{arm}__retrieval_clustering__{dataset}", "arm": arm,
        "family": "retrieval_clustering", "dataset": dataset, "order": order,
        "command": command, "cwd": str(SOURCE), "pythonpath": str(SOURCE),
        "output": str(out), "done_kind": "retrieval_csv",
    }


def rxrx3_task(asset: dict, order: int) -> dict:
    arm = asset["arm"]
    out = ROOT / "retrieval" / "rxrx3-core" / arm
    command = [
        str(PYTHON), str(SOURCE / "scripts/run_external4_fixedbudget_model.py"),
        "--model", arm, "--campaign", str(out),
        "--cache-root", str(RXRX3_CACHE.parent), "--datasets", "rxrx3",
        "--checkpoint", str(asset["checkpoint"]), "--train-config", str(asset["config"]),
        "--device", "cuda:0", "--batch-size", "64",
    ]
    return {
        "id": f"{arm}__retrieval_clustering__rxrx3-core", "arm": arm,
        "family": "retrieval_clustering", "dataset": "rxrx3-core", "order": order,
        "command": command, "cwd": str(SOURCE), "pythonpath": str(SOURCE),
        "output": str(out), "done_kind": "rxrx3_json",
    }


def detection_task(asset: dict, dataset: str, order: int) -> dict:
    arm = asset["arm"]
    out = ROOT / "detection_proxy" / dataset / arm
    command = [
        str(PYTHON), "-B", "-m", "dinov3.eval.bio_detection.center_probe",
        "--checkpoint", str(asset["checkpoint"]), "--train-config", str(asset["config"]),
        "--benchmark-root", str(BENCHMARK), "--dataset", dataset,
        "--output-dir", str(out), "--batch-size", "8", "--num-workers", "2",
        "--image-size", "224", "--epochs", "5", "--lr", "0.001",
        "--autocast-dtype", "bf16", "--channel-policy", "auto", "--seed", "0",
        "--max-samples-per-split", "0",
        "--conic-split-protocol", "official-baseline-fold0-nested-v1",
    ]
    return {
        "id": f"{arm}__detection_proxy__{dataset}", "arm": arm,
        "family": "detection_proxy", "dataset": dataset, "order": order,
        "command": command, "cwd": str(SOURCE), "pythonpath": str(SOURCE),
        "output": str(out), "done_kind": "detection_json",
    }


def monuseg_task(asset: dict, order: int) -> dict:
    arm = asset["arm"]
    out = ROOT / "segmentation" / "monuseg" / arm
    command = [
        str(PYTHON), "-B", "-m", "dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline",
        "--datasets", "monuseg", "--checkpoint-file", str(asset["checkpoint"]),
        "--checkpoint-id", str(asset["checkpoint_id"]), "--train-config", str(asset["config"]),
        "--protocol", "manual", "--dataset-split-protocol", "formal-v1",
        "--feature-img-size", "768", "--resize-mode", "pad", "--layer-preset", "last1",
        "--feature-batch-size", "32", "--feature-num-workers", "2",
        "--autocast-dtype", "bf16", "--channel-policy", "auto", "--channel-tta-samples", "8",
        "--probe-epoch-grid", "20", "50", "--probe-seeds", "0", "1", "2",
        "--probe-batch-size", "32", "--probe-lr", "0.001", "--probe-weight-decay", "0.0001",
        "--probe-eval-every", "1", "--probe-num-workers", "0", "--gpu", "0",
        "--probe-class-weight-mode", "none", "--chunked-cache", "--no-compress-cache",
        "--cache-root", str(out / "cache"), "--output-root", str(out / "results"),
        "--run-name", f"{arm}_v4_monuseg",
    ]
    return {
        "id": f"{arm}__segmentation__monuseg", "arm": arm,
        "family": "segmentation", "dataset": "monuseg", "order": order,
        "command": command, "cwd": str(SOURCE), "pythonpath": str(SOURCE),
        "output": str(out), "done_kind": "monuseg_results",
    }


def make_tasks() -> list[dict]:
    tasks: list[dict] = []
    order = 0

    def add(task: dict) -> None:
        nonlocal order
        order += 1
        task["order"] = order
        tasks.append(task)

    # Interleave checkpoints within each family.  This exposes failures early
    # and prevents a slow family from monopolising the tail of the fleet.
    for asset in ASSETS:
        add(retrieval_task(asset, "nct-crc-he-100", order))
    for asset in ASSETS:
        add(frozen_task(asset, "conic-cell-count", order))
    for asset in ASSETS:
        add(monuseg_task(asset, order))
    for asset in ASSETS:
        add(rxrx3_task(asset, order))
    for asset in ASSETS:
        add(detection_task(asset, "bbbc038", order))
    for asset in ASSETS:
        add(frozen_task(asset, "lc25000", order))
    for asset in ASSETS:
        add(retrieval_task(asset, "lc25000", order))
    for asset in ASSETS:
        add(frozen_task(asset, "livecell-cell-count", order))
    for asset in ASSETS:
        add(detection_task(asset, "conic", order))
    for asset in ASSETS:
        add(detection_task(asset, "livecell", order))
    if len(tasks) != 40 or len({task["id"] for task in tasks}) != 40:
        raise RuntimeError("Expected exactly 40 unique admitted completion tasks")
    asset_by_arm = {asset["arm"]: asset for asset in ASSETS}
    for task in tasks:
        asset = asset_by_arm[task["arm"]]
        task.update(
            protocol_id=PROTOCOL_ID,
            checkpoint=str(asset["checkpoint"]),
            checkpoint_sha256=asset["checkpoint_sha256"],
            config=str(asset["config"]),
            config_sha256=sha256(asset["config"]),
            source_entry_sha256=source_digest_for(task),
        )
    return tasks


def source_digest_for(task: dict) -> str:
    if task["done_kind"] == "frozen_csv":
        rel = "dinov3/eval/bio_frozen_eval/run_classification.py"
    elif task["done_kind"] == "retrieval_csv":
        rel = "dinov3/eval/bio_frozen_eval/run_retrieval_clustering.py"
    elif task["done_kind"] == "detection_json":
        rel = "dinov3/eval/bio_detection/center_probe.py"
    elif task["done_kind"] == "monuseg_results":
        rel = "dinov3/eval/bio_segmentation/scripts/run_linear_probe_pipeline.py"
    else:
        rel = "scripts/run_external4_fixedbudget_model.py"
    return sha256(SOURCE / rel)


def validate_frozen(task: dict) -> tuple[bool, str]:
    path = Path(task["output"]) / "summary.csv"
    if not path.is_file():
        return False, "missing summary.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    valid = [row for row in rows if row.get("dataset") == task["dataset"] and not row.get("error")]
    if len(valid) != 1:
        return False, f"valid rows={len(valid)} expected=1"
    row = valid[0]
    if row.get("batch_size") != "64" or row.get("seed") != "0":
        return False, "frozen batch/seed mismatch"
    expected = {
        "lc25000": ("classification", "20000", "5000"),
        "conic-cell-count": ("regression", "3940", "517"),
        "livecell-cell-count": ("regression", "3253", "1564"),
    }[task["dataset"]]
    if (row.get("task"), row.get("n_train"), row.get("n_test")) != expected:
        return False, "task/sample-count mismatch"
    metric = "balanced_accuracy" if task["dataset"] == "lc25000" else "r2"
    try:
        if not math.isfinite(float(row[metric])):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return False, f"nonfinite {metric}"
    if task["dataset"] == "lc25000" and row.get("split") != "internal-80-20":
        return False, "LC25000 locked split mismatch"
    if task["dataset"] == "conic-cell-count" and row.get("fold_protocol") != "source-image-grouped-CoNIC-10fold-v1":
        return False, "CoNIC fold protocol mismatch"
    return True, "VALID_COMPLETE"


def validate_retrieval(task: dict) -> tuple[bool, str]:
    path = Path(task["output"]) / "summary.csv"
    if not path.is_file():
        return False, "missing summary.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row.get("dataset") == task["dataset"] and not row.get("error")]
    if len(rows) != 1:
        return False, f"valid rows={len(rows)} expected=1"
    row = rows[0]
    expected = {"lc25000": ("25000", "5"), "nct-crc-he-100": ("99", "9")}[task["dataset"]]
    if row.get("task") != "retrieval_clustering" or (row.get("n_samples"), row.get("n_classes")) != expected:
        return False, "retrieval/clustering identity mismatch"
    if row.get("protocol") != "within-set-leave-one-out":
        return False, "self-match-excluding protocol mismatch"
    for metric in ("recall_at_1", "nmi", "ari", "cluster_accuracy"):
        try:
            if not math.isfinite(float(row[metric])):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            return False, f"nonfinite {metric}"
    return True, "VALID_COMPLETE retrieval+clustering"


def validate_detection(task: dict) -> tuple[bool, str]:
    path = Path(task["output"]) / "results_bio_detection.json"
    if not path.is_file():
        return False, "missing detection JSON"
    try:
        data = json.loads(path.read_text())
        valid = (
            data.get("dataset") == task["dataset"]
            and data.get("batch_size") == 8
            and data.get("epochs") == 5
            and data.get("image_size") == 224
            and data.get("seed") == 0
            and math.isfinite(float(data["test_patch_f1"]))
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        valid = False
    return (True, "VALID_COMPLETE matched-B8") if valid else (False, "invalid detection JSON")


def validate_monuseg(task: dict) -> tuple[bool, str]:
    paths = list((Path(task["output"]) / "results").rglob("results.json"))
    signatures = set()
    valid_paths = []
    for path in paths:
        if "monuseg" not in path.parts:
            continue
        try:
            data = json.loads(path.read_text())
            meta = data["_meta"]
            signature = (int(meta["probe_epochs"]), int(meta["seed"]))
            if (
                meta.get("full_train_samples") == 24
                and meta.get("used_train_samples") == 24
                and meta.get("probe_batch_size") == 32
                and meta.get("probe_eval_every") == 1
                and meta.get("test_evaluations") == 1
                and math.isfinite(float(data["test"]["mDice"]))
            ):
                signatures.add(signature)
                valid_paths.append(path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    expected = {(epoch, seed) for epoch in (20, 50) for seed in (0, 1, 2)}
    if signatures != expected or len(valid_paths) != 6:
        return False, f"valid MoNuSeg fits={len(valid_paths)}/6 signatures={sorted(signatures)}"
    return True, "VALID_COMPLETE official30/test14, six probe fits"


def validate_rxrx3(task: dict) -> tuple[bool, str]:
    path = Path(task["output"]) / "models" / task["arm"] / "results.json"
    if not path.is_file():
        return False, "missing RxRx3 results JSON"
    try:
        data = json.loads(path.read_text())
        result = data["tests"]["rxrx3"]
        valid = (
            data.get("status") == "VALID_COMPLETE"
            and data.get("batch_size") == 64
            and data.get("teacher_branch") == "teacher"
            and result.get("status") == "FORMAL"
            and result.get("proxy") is False
            and result.get("protocol_id") == RXRX3_PROTOCOL
            and result.get("n_query") == 734
            and result.get("n_gallery") == 734
            and math.isfinite(float(result["recall_at_1"]))
            and math.isfinite(float(result["nmi"]))
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        valid = False
    return (True, "VALID_COMPLETE full 734-gene retrieval+clustering") if valid else (False, "invalid RxRx3 result")


def validate_task(task: dict) -> tuple[bool, str]:
    return {
        "frozen_csv": validate_frozen,
        "retrieval_csv": validate_retrieval,
        "detection_json": validate_detection,
        "monuseg_results": validate_monuseg,
        "rxrx3_json": validate_rxrx3,
    }[task["done_kind"]](task)


def prepare() -> None:
    if (ROOT / "campaign_manifest.json").exists():
        raise RuntimeError(f"Campaign already exists: {ROOT}")
    if not SOURCE.is_dir() or not PYTHON.is_file():
        raise RuntimeError("Pinned source snapshot or Python environment is missing")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Protocol identity mismatch")
    if protocol["admission_gates"].get("segmentation/monuseg", "").split(":", 1)[0] != "READY_IDENTITY_LOCKED_20260923":
        raise RuntimeError("MoNuSeg is not admitted by the current v4 protocol")
    for path, expected in LOCKED_INPUTS.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Locked input mismatch: {path}: {actual} != {expected}")
    for asset in ASSETS:
        if not asset["checkpoint"].is_file() or not asset["config"].is_file():
            raise FileNotFoundError(f"Missing checkpoint/config for {asset['arm']}")
        if not asset["shared_evidence"].is_dir():
            raise FileNotFoundError(f"Missing shared evidence for {asset['arm']}")
    tasks = make_tasks()
    for task in tasks:
        atomic_json(ROOT / "tasks" / f"{task['id']}.json", task)
        if task["done_kind"] == "rxrx3_json":
            out = Path(task["output"])
            atomic_json(out / "campaign_manifest.json", {
                "protocol_id": PROTOCOL_ID,
                "protocol_sha256": sha256(PROTOCOL),
                "fixed_split_protocol_id": RXRX3_PROTOCOL,
                "fixed_split_sha256": LOCKED_INPUTS[str(RXRX3_CACHE / "split_manifest.jsonl")],
                "checkpoint": task["checkpoint"],
                "checkpoint_sha256": task["checkpoint_sha256"],
                "config": task["config"],
                "config_sha256": task["config_sha256"],
                "teacher_branch": True,
                "batch_size": 64,
                "query": 734,
                "gallery": 734,
                "created_utc": now(),
            })
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256(PROTOCOL),
        "campaign": ROOT.name,
        "created_utc": now(),
        "purpose": "Complete admitted Union-v4 gaps for HS6-L 1TB/5TB/1PB/100TB 1M checkpoints",
        "source_snapshot": str(SOURCE),
        "source_entries_sha256": {rel: sha256(SOURCE / rel) for rel in SOURCE_ENTRIES},
        "interpreter": str(PYTHON),
        "assets": [
            {**{key: (str(value) if isinstance(value, Path) else value) for key, value in asset.items()},
             "config_sha256": sha256(asset["config"])}
            for asset in ASSETS
        ],
        "locked_inputs_sha256": LOCKED_INPUTS,
        "expected_inventory_per_model": protocol["expected_unique_dataset_counts"],
        "new_execution_tasks_per_model": 10,
        "new_execution_tasks_total": 40,
        "reused_shared_cells": {
            "classification": 24, "regression": 2, "retrieval": 4,
            "clustering": "same four validated feature identities/readouts",
            "segmentation": "six datasets, with PanNuke three rotations",
        },
        "explicit_blockers": {
            "cell_tracking/ctc": protocol["admission_gates"]["cell_tracking/ctc"],
            "ood/xray": "BLOCKED_NOT_TESTED: common OOD selection/config freeze absent",
            "ood/cryo": "BLOCKED_NOT_TESTED: common OOD selection/config freeze absent",
        },
        "lc25000_classification_status": "PROVISIONAL_LEGACY_ONLY",
        "nct_crc_he_100_status": "VALIDATABLE_LOW_N_99",
        "aggregate_policy": "Do not claim an all-56 aggregate while CTC/OOD are blocked; report admitted coverage and blockers explicitly.",
        "task_ids": [task["id"] for task in tasks],
    }
    atomic_json(ROOT / "campaign_manifest.json", manifest)
    summary()
    print(f"PREPARED {len(tasks)} tasks at {ROOT}", flush=True)


def gpu_free_mib() -> int:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"
    ], text=True)
    return int(output.strip().splitlines()[0])


def worker(label: str, min_free_mib: int) -> None:
    supervisor = ROOT / "supervisors" / f"{platform.node()}_gpu0.lock"
    supervisor.parent.mkdir(parents=True, exist_ok=True)
    lock = supervisor.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = [json.loads(path.read_text()) for path in sorted((ROOT / "tasks").glob("*.json"))]
    tasks.sort(key=lambda task: (task["order"], task["id"]))
    while True:
        selected = None
        for task in tasks:
            claim = ROOT / "claims" / task["id"]
            if claim.exists():
                continue
            try:
                ok, _ = validate_task(task)
            except Exception:
                ok = False
            if ok:
                claim.mkdir(parents=True, exist_ok=True)
                atomic_json(claim / "status.json", {
                    "state": "DONE", "validation": "adopted pre-existing valid output",
                    "host": platform.node(), "label": label, "end_utc": now(),
                })
                continue
            try:
                claim.mkdir(parents=True)
            except FileExistsError:
                continue
            selected = (task, claim)
            break
        if selected is None:
            summary()
            print(f"{now()} NO_UNCLAIMED_TASKS {label}", flush=True)
            return
        task, claim = selected
        while gpu_free_mib() < min_free_mib:
            print(f"{now()} GPU_ADMISSION_WAIT {label} free_mib={gpu_free_mib()}", flush=True)
            time.sleep(30)
        output = Path(task["output"])
        output.mkdir(parents=True, exist_ok=True)
        invocation = {
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": sha256(PROTOCOL),
            "task": task,
            "host": platform.node(),
            "worker_label": label,
            "gpu": 0,
            "start_utc": now(),
        }
        atomic_json(output / "v4_invocation.json", invocation)
        log_path = ROOT / "logs" / f"{task['id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        status = {
            "state": "RUNNING", "host": platform.node(), "label": label,
            "gpu": 0, "pid": os.getpid(), "start_utc": now(), "log": str(log_path),
        }
        atomic_json(claim / "status.json", status)
        env = os.environ.copy()
        env.update(THREAD_ENV)
        env.update(CUDA_VISIBLE_DEVICES="0", PYTHONPATH=task["pythonpath"], PYTHONUNBUFFERED="1")
        print(f"{now()} START {label} {task['id']}", flush=True)
        with log_path.open("a") as log:
            completed = subprocess.run(task["command"], cwd=task["cwd"], env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
        try:
            ok, reason = validate_task(task)
        except Exception as error:
            ok, reason = False, f"validator exception: {type(error).__name__}: {error}"
        status.update(
            state="DONE" if completed.returncode == 0 and ok else "FAILED",
            returncode=completed.returncode, validation=reason, end_utc=now(),
        )
        atomic_json(claim / "status.json", status)
        if status["state"] == "DONE":
            atomic_json(output / "validation_report.json", {
                "status": "VALID_COMPLETE", "protocol_id": PROTOCOL_ID,
                "task_id": task["id"], "validation": reason,
                "checkpoint_sha256": task["checkpoint_sha256"],
                "config_sha256": task["config_sha256"],
                "source_entry_sha256": task["source_entry_sha256"],
                "validated_utc": now(),
            })
        print(f"{now()} {status['state']} {label} {task['id']} rc={completed.returncode} {reason}", flush=True)
        summary()


def requeue_failed() -> None:
    archive = ROOT / "attempts"
    archive.mkdir(parents=True, exist_ok=True)
    moved = 0
    for status_path in (ROOT / "claims").glob("*/status.json"):
        status = json.loads(status_path.read_text())
        if status.get("state") != "FAILED":
            continue
        claim = status_path.parent
        task_path = ROOT / "tasks" / f"{claim.name}.json"
        task = json.loads(task_path.read_text())
        if task.get("done_kind") == "monuseg_results":
            command = task["command"]
            command[command.index("--probe-num-workers") + 1] = "0"
            task["runtime_retry_amendment"] = (
                "Probe DataLoader num_workers 2->0 after torch_shm_manager/NFS socket failure; "
                "cached features and all scientific parameters are unchanged"
            )
            atomic_json(task_path, task)
        destination = archive / f"{claim.name}__{int(time.time())}"
        claim.rename(destination)
        moved += 1
    summary()
    print(f"REQUEUED_FAILED {moved}", flush=True)


def summary() -> None:
    task_paths = list((ROOT / "tasks").glob("*.json"))
    states = Counter()
    by_arm: dict[str, Counter] = {asset["arm"]: Counter() for asset in ASSETS}
    failures = []
    for task_path in task_paths:
        task = json.loads(task_path.read_text())
        status_path = ROOT / "claims" / task["id"] / "status.json"
        if not status_path.exists():
            state = "QUEUED"
        else:
            status = json.loads(status_path.read_text())
            state = status.get("state", "UNKNOWN")
            if state == "FAILED":
                failures.append({"task": task["id"], **status})
        states[state] += 1
        by_arm[task["arm"]][state] += 1
    value = {
        "time_utc": now(), "task_count": len(task_paths), "counts": dict(states),
        "by_arm": {arm: dict(counts) for arm, counts in by_arm.items()},
        "failures": failures,
        "blocked_not_tested_per_model": ["cell_tracking/ctc", "ood/xray", "ood/cryo"],
    }
    atomic_json(ROOT / "STATUS.json", value)
    print(json.dumps(value, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "summary", "requeue-failed"))
    parser.add_argument("--label", default=platform.node())
    parser.add_argument("--min-free-mib", type=int, default=14000)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "worker":
        worker(args.label, args.min_free_mib)
    elif args.mode == "requeue-failed":
        requeue_failed()
    else:
        summary()


if __name__ == "__main__":
    main()
