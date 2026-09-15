#!/usr/bin/env python3
"""Run and validate the formal RxRx3-core protocol on all 49 HS6-L 5TB checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "outputs/02_eval_runs/rxrx3_core_formal_l5_all49_v3_single3090_20260911"
TRAIN_RUN = ROOT / (
    "outputs/01_training_runs/"
    "HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907"
)
CACHE_ROOT = ROOT / "outputs/02_eval_inputs/formal_v3"
DATASET_ROOT = CACHE_ROOT / "rxrx3-core"
WORKER = ROOT / "scripts/run_external4_fixedbudget_model.py"
PROTOCOL_ID = "crispr-query-guide-plate-disjoint-all-eligible-genes-v1"
CHECKPOINTS = tuple(range(487, 23912, 488))


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def checkpoint_path(step: int) -> Path:
    return TRAIN_RUN / "eval" / f"training_{step}" / "teacher_checkpoint.pth"


def file_record(path: Path, *, include_hash: bool = True) -> dict:
    resolved = path.resolve()
    stat = resolved.stat()
    record = {
        "path": str(resolved),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        record["sha256"] = sha256(resolved)
    return record


def git_head() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return proc.stdout.strip() or None


def prepare_manifest(campaign: Path) -> Path:
    manifest_path = campaign / "campaign_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        if payload.get("status") != "LOCKED_BEFORE_RUN":
            raise RuntimeError(f"refusing unlocked manifest: {manifest_path}")
        return manifest_path

    config = TRAIN_RUN / "config.yaml"
    required = [config, WORKER, DATASET_ROOT / "images.npy", DATASET_ROOT / "labels.npy",
                DATASET_ROOT / "metadata.json", DATASET_ROOT / "split_manifest.jsonl"]
    required.extend(checkpoint_path(step) for step in CHECKPOINTS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing formal inputs:\n" + "\n".join(missing))

    metadata = json.loads((DATASET_ROOT / "metadata.json").read_text())
    if metadata.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(f"wrong RxRx3 protocol: {metadata.get('protocol_id')}")
    rows = metadata.get("rows", [])
    gallery = sum(row.get("split") == "gallery" for row in rows)
    query = sum(row.get("split") == "query" for row in rows)
    if (gallery, query) != (734, 734):
        raise RuntimeError(f"expected 734/734 RxRx3 rows, found {gallery}/{query}")

    models = []
    for index, step in enumerate(CHECKPOINTS, start=1):
        path = checkpoint_path(step)
        print(f"[manifest] hashing checkpoint {index}/{len(CHECKPOINTS)}: {step}", flush=True)
        models.append({
            "model": f"hs6_l_5tb_ck{step}",
            "checkpoint_step": step,
            "checkpoint": file_record(path),
        })
    payload = {
        "campaign": campaign.name,
        "status": "LOCKED_BEFORE_RUN",
        "created_unix": time.time(),
        "created_host": platform.node(),
        "git_head": git_head(),
        "protocol_id": PROTOCOL_ID,
        "teacher_branch": "teacher",
        "dataset": "rxrx3-core",
        "dataset_protocol": {
            "input_mapping": metadata.get("input_mapping"),
            "manifest_sha256": sha256(DATASET_ROOT / "split_manifest.jsonl"),
            "metadata_sha256": sha256(DATASET_ROOT / "metadata.json"),
            "images_sha256": sha256(DATASET_ROOT / "images.npy"),
            "labels_sha256": sha256(DATASET_ROOT / "labels.npy"),
            "n_gallery": gallery,
            "n_query": query,
        },
        "evaluation": {
            "resolution": 224,
            "resize_size": 256,
            "layers": "final_cls_plus_final_patch_mean_l2",
            "logical_batch_size": 64,
            "inference_microbatch": 64,
            "seed": 0,
            "clustering": "MiniBatchKMeans(n_init=5,max_iter=200,seed=0)",
        },
        "config": file_record(config),
        "worker": file_record(WORKER),
        "models": models,
    }
    atomic_json(manifest_path, payload)
    print(f"[manifest] locked {manifest_path} sha256={sha256(manifest_path)}", flush=True)
    return manifest_path


def valid_result(path: Path, manifest_sha: str, model: dict) -> bool:
    try:
        result = json.loads(path.read_text())
        test = result["tests"]["rxrx3"]
        metrics = [test[key] for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "nmi")]
        return (
            result.get("status") == "VALID_COMPLETE"
            and result.get("model") == model["model"]
            and Path(result.get("checkpoint", "")).resolve() == Path(model["checkpoint"]["path"])
            and result.get("campaign_manifest_sha256_at_start") == manifest_sha
            and result.get("batch_size") == 64
            and result.get("teacher_branch") == "teacher"
            and test.get("status") == "FORMAL"
            and test.get("proxy") is False
            and test.get("protocol_id") == PROTOCOL_ID
            and test.get("n_gallery") == 734
            and test.get("n_query") == 734
            and all(math.isfinite(float(value)) for value in metrics)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_models(args: argparse.Namespace, manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256(manifest_path)
    selected = manifest["models"][args.lane_index::args.num_lanes]
    log_root = args.campaign / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for index, model in enumerate(selected, start=1):
        model_dir = args.campaign / "models" / model["model"]
        result_path = model_dir / "results.json"
        if valid_result(result_path, manifest_sha, model):
            print(f"[skip-valid] {model['model']}", flush=True)
            continue
        command = [
            args.python_bin, "-u", str(WORKER),
            "--model", model["model"],
            "--campaign", str(args.campaign),
            "--cache-root", str(CACHE_ROOT),
            "--datasets", "rxrx3",
            "--checkpoint", model["checkpoint"]["path"],
            "--train-config", manifest["config"]["path"],
            "--device", args.device,
            "--batch-size", "64",
        ]
        print(f"[run {index}/{len(selected)}] {model['model']}", flush=True)
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(ROOT) if not existing_pythonpath else f"{ROOT}{os.pathsep}{existing_pythonpath}"
        )
        with (log_root / f"{model['model']}.log").open("a") as log:
            log.write("COMMAND " + " ".join(command) + "\n")
            log.flush()
            proc = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode or not valid_result(result_path, manifest_sha, model):
            failures += 1
            print(f"[failed] {model['model']} rc={proc.returncode}", flush=True)
    return failures


def validate(campaign: Path) -> dict:
    manifest_path = campaign / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256(manifest_path)
    rows = []
    errors = []
    for model in manifest["models"]:
        path = campaign / "models" / model["model"] / "results.json"
        if not valid_result(path, manifest_sha, model):
            errors.append(f"invalid or missing result: {model['model']}")
            continue
        result = json.loads(path.read_text())
        test = result["tests"]["rxrx3"]
        rows.append({"checkpoint": model["checkpoint_step"], **{
            key: test[key] for key in
            ("recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "mrr", "map", "nmi")
        }})
    if rows:
        with (campaign / "checkpoint_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row["checkpoint"]))
    report = {
        "status": "VALID_COMPLETE" if not errors and len(rows) == len(CHECKPOINTS) else "INVALID_INCOMPLETE",
        "protocol_id": PROTOCOL_ID,
        "campaign_manifest_sha256": manifest_sha,
        "expected_checkpoints": len(CHECKPOINTS),
        "valid_checkpoints": len(rows),
        "errors": errors,
    }
    atomic_json(campaign / "validation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lane-index", type=int, default=0)
    parser.add_argument("--num-lanes", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.num_lanes <= 0 or not 0 <= args.lane_index < args.num_lanes:
        parser.error("require 0 <= lane-index < num-lanes")
    manifest_path = prepare_manifest(args.campaign)
    if args.prepare_only:
        return
    if args.validate_only:
        raise SystemExit(0 if validate(args.campaign)["status"] == "VALID_COMPLETE" else 1)
    failures = run_models(args, manifest_path)
    report = validate(args.campaign)
    raise SystemExit(0 if failures == 0 and report["status"] == "VALID_COMPLETE" else 1)


if __name__ == "__main__":
    main()
