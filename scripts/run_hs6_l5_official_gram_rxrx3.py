#!/usr/bin/env python3
"""Run the matched HS6-L5 official-Gram screen on formal RxRx3-core."""

from __future__ import annotations

import argparse
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
RUN_ROOT = ROOT / "outputs/01_training_runs"
CACHE_ROOT = ROOT / "outputs/02_eval_inputs/formal_v3"
WORKER = ROOT / "scripts/run_external4_fixedbudget_model.py"
PROTOCOL_ID = "crispr-query-guide-plate-disjoint-all-eligible-genes-v1"
FINAL_STEP = 20495
EXPERIMENTS = {
    "control-vs-anchor7807": {
        "campaign": ROOT
        / "outputs/02_eval_runs/hs6_l5_official_gram_u488_rxrx3_formal_20260911",
        "comparison": "matched control versus official Gram",
        "baseline_arm": "control",
        "candidate_arm": "official",
        "runs": {
            "control": "HS6_L5_ck20007_control_gram_a7807_u488_gb64_4x3090qi_screen_20260911",
            "official": "HS6_L5_ck20007_official_gram_a7807_u488_gb64_4x3090qi_screen_20260911",
        },
        "gram_anchor_checkpoint": 7807,
    },
    "anchor7807-vs-anchor17079": {
        "campaign": ROOT
        / "outputs/02_eval_runs/hs6_l5_official_gram_anchor7807_vs_17079_u488_rxrx3_formal_20260911",
        "comparison": "matched official Gram anchor ck7807 versus ck17079",
        "baseline_arm": "anchor7807",
        "candidate_arm": "anchor17079",
        "runs": {
            "anchor7807": "HS6_L5_ck20007_official_gram_a7807_u488_gb64_4x3090qi_screen_20260911",
            "anchor17079": "HS6_L5_ck20007_official_gram_a17079_u488_gb64_4x3090qi_anchor_ablation_20260911",
        },
        "gram_anchor_checkpoint_by_arm": {"anchor7807": 7807, "anchor17079": 17079},
    },
    "anchor17079-vs-bilevel17079": {
        "campaign": ROOT
        / "outputs/02_eval_runs/hs6_l5_gram_anchor17079_vs_bilevel_u488_rxrx3_formal_20260911",
        "comparison": "matched official Gram anchor ck17079 versus bi-level relation Gram",
        "baseline_arm": "anchor17079",
        "candidate_arm": "bilevel17079",
        "runs": {
            "anchor17079": "HS6_L5_ck20007_official_gram_a17079_u488_gb64_4x3090qi_anchor_ablation_20260911",
            "bilevel17079": "HS6_L5_ck20007_bilevel_gram_a17079_u488_gb64_4x3090qi_bilevel_20260911",
        },
        "gram_anchor_checkpoint_by_arm": {"anchor17079": 17079, "bilevel17079": 17079},
        "inter_image_loss_weight_by_arm": {"anchor17079": 0.0, "bilevel17079": 0.25},
    },
    "control-vs-dual": {
        "campaign": ROOT
        / "outputs/02_eval_runs/hs6_l5_dual_anchor_u488_rxrx3_formal_20260912",
        "comparison": "matched control versus decoupled dual-anchor Gram",
        "baseline_arm": "control",
        "candidate_arm": "dual",
        "runs": {
            "control": "HS6_L5_ck20007_control_gram_a7807_u488_gb64_4x3090qi_screen_20260911",
            "dual": "HS6_L5_ck20007_dualgram_pa7807_ga20007_u488_gb64_4x3090qi_dual_v3_20260912",
        },
        "gram_anchor_checkpoint_by_arm": {"control": None, "dual": 7807},
        "training_metadata": {
            "global_relation_anchor_checkpoint_by_arm": {"control": None, "dual": 20007},
            "global_relation_loss_weight_by_arm": {"control": 0.0, "dual": 0.5},
            "label_free_ssl_training": True,
        },
        "plan": ROOT
        / "Evaluation Rules/plans/hs6_l5_gram_new_v3_datasets_addendum_20260912.md",
    },
}
METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "map", "nmi")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_record(path: Path) -> dict:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(resolved),
    }


def inputs_for(run_name: str) -> tuple[Path, Path]:
    run = RUN_ROOT / run_name
    return run / "eval" / f"training_{FINAL_STEP}" / "teacher_checkpoint.pth", run / "config.yaml"


def prepare_manifest(campaign: Path, experiment: str) -> Path:
    specification = EXPERIMENTS[experiment]
    manifest_path = campaign / "campaign_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "LOCKED_BEFORE_RUN":
            raise RuntimeError(f"refusing incompatible manifest: {manifest_path}")
        if manifest.get("comparison") != specification["comparison"]:
            raise RuntimeError(f"manifest comparison does not match {experiment}: {manifest_path}")
        return manifest_path

    models = []
    for arm, run_name in specification["runs"].items():
        checkpoint, config = inputs_for(run_name)
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError(f"missing completed {arm} inputs: {checkpoint} or {config}")
        print(f"[manifest] hashing {arm} checkpoint", flush=True)
        models.append(
            {
                "arm": arm,
                "model": f"hs6_l5_{arm}_gram_u488",
                "run": run_name,
                "checkpoint": file_record(checkpoint),
                "config": file_record(config),
            }
        )

    dataset = CACHE_ROOT / "rxrx3-core"
    metadata = json.loads((dataset / "metadata.json").read_text())
    rows = metadata["rows"]
    n_gallery = sum(row["split"] == "gallery" for row in rows)
    n_query = sum(row["split"] == "query" for row in rows)
    if metadata.get("protocol_id") != PROTOCOL_ID or (n_gallery, n_query) != (734, 734):
        raise RuntimeError("formal RxRx3-core cache does not match the locked protocol")

    manifest = {
        "campaign": campaign.name,
        "status": "LOCKED_BEFORE_RUN",
        "created_unix": time.time(),
        "created_host": platform.node(),
        "protocol_id": PROTOCOL_ID,
        "experiment": experiment,
        "comparison": specification["comparison"],
        "baseline_arm": specification["baseline_arm"],
        "candidate_arm": specification["candidate_arm"],
        "training": {
            "branch_checkpoint": 20007,
            "continuation_updates": 488,
            "effective_global_batch": 64,
            "student_crop": 256,
            "gram_teacher_crop": 512,
            "gram_img_level": True,
            "gram_weight": 2.0,
            "teacher_refresh_completed_updates": [20200, 20400],
        },
        "evaluation": {
            "teacher_branch": True,
            "batch_size": 64,
            "resolution": 224,
            "readout": "final_cls_plus_final_patch_mean_l2",
            "seed": 0,
        },
        "dataset": {
            "protocol_id": PROTOCOL_ID,
            "n_gallery": n_gallery,
            "n_query": n_query,
            "images": file_record(dataset / "images.npy"),
            "labels": file_record(dataset / "labels.npy"),
            "metadata": file_record(dataset / "metadata.json"),
            "split_manifest": file_record(dataset / "split_manifest.jsonl"),
        },
        "worker": file_record(WORKER),
        "models": models,
    }
    if "gram_anchor_checkpoint" in specification:
        manifest["training"]["gram_anchor_checkpoint"] = specification["gram_anchor_checkpoint"]
    else:
        manifest["training"]["gram_anchor_checkpoint_by_arm"] = specification[
            "gram_anchor_checkpoint_by_arm"
        ]
    if "inter_image_loss_weight_by_arm" in specification:
        manifest["training"]["inter_image_loss_weight_by_arm"] = specification[
            "inter_image_loss_weight_by_arm"
        ]
    manifest["training"].update(specification.get("training_metadata", {}))
    if "plan" in specification:
        manifest["plan"] = file_record(specification["plan"])
    atomic_json(manifest_path, manifest)
    return manifest_path


def valid_result(campaign: Path, manifest_sha: str, model: dict) -> bool:
    try:
        result = json.loads((campaign / "models" / model["model"] / "results.json").read_text())
        test = result["tests"]["rxrx3"]
        return bool(
            result.get("status") == "VALID_COMPLETE"
            and result.get("model") == model["model"]
            and Path(result["checkpoint"]).resolve() == Path(model["checkpoint"]["path"])
            and result.get("campaign_manifest_sha256_at_start") == manifest_sha
            and result.get("teacher_branch") == "teacher"
            and result.get("batch_size") == 64
            and test.get("status") == "FORMAL"
            and test.get("proxy") is False
            and test.get("protocol_id") == PROTOCOL_ID
            and test.get("n_gallery") == 734
            and test.get("n_query") == 734
            and all(math.isfinite(float(test[key])) for key in METRICS)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_lane(args: argparse.Namespace, manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256(manifest_path)
    selected = manifest["models"][args.lane_index :: args.num_lanes]
    logs = args.campaign / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    failures = 0
    for model in selected:
        if valid_result(args.campaign, manifest_sha, model):
            print(f"[skip-valid] {model['model']}", flush=True)
            continue
        command = [
            args.python_bin,
            "-u",
            str(WORKER),
            "--model",
            model["model"],
            "--campaign",
            str(args.campaign),
            "--cache-root",
            str(CACHE_ROOT),
            "--datasets",
            "rxrx3",
            "--checkpoint",
            model["checkpoint"]["path"],
            "--train-config",
            model["config"]["path"],
            "--device",
            args.device,
            "--batch-size",
            "64",
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT)
        with (logs / f"{model['model']}.log").open("a") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode or not valid_result(args.campaign, manifest_sha, model):
            failures += 1
            print(f"[failed] {model['model']} rc={process.returncode}", flush=True)
    return failures


def validate(campaign: Path) -> dict:
    manifest_path = campaign / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256(manifest_path)
    results = {}
    errors = []
    for model in manifest["models"]:
        if not valid_result(campaign, manifest_sha, model):
            errors.append(f"invalid result: {model['model']}")
            continue
        payload = json.loads((campaign / "models" / model["model"] / "results.json").read_text())
        results[model["arm"]] = payload["tests"]["rxrx3"]
    baseline_arm = manifest.get("baseline_arm", "control")
    candidate_arm = manifest.get("candidate_arm", "official")
    deltas = {}
    if set(results) == {baseline_arm, candidate_arm}:
        deltas = {
            key: results[candidate_arm][key] - results[baseline_arm][key] for key in METRICS
        }
    report = {
        "status": "VALID_COMPLETE" if not errors and len(results) == 2 else "INVALID_INCOMPLETE",
        "protocol_id": PROTOCOL_ID,
        "comparison": manifest["comparison"],
        "baseline_arm": baseline_arm,
        "candidate_arm": candidate_arm,
        "manifest_sha256": manifest_sha,
        "results": results,
        "candidate_minus_baseline": deltas,
        "errors": errors,
    }
    if (baseline_arm, candidate_arm) == ("control", "official"):
        report["official_minus_control"] = deltas
    atomic_json(campaign / "comparison.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default="control-vs-anchor7807",
    )
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lane-index", type=int, default=0)
    parser.add_argument("--num-lanes", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.campaign is None:
        args.campaign = EXPERIMENTS[args.experiment]["campaign"]
    args.campaign = args.campaign.resolve()
    if args.num_lanes < 1 or not 0 <= args.lane_index < args.num_lanes:
        parser.error("require 0 <= lane-index < num-lanes")
    manifest_path = prepare_manifest(args.campaign, args.experiment)
    if args.prepare_only:
        return 0
    failures = 0 if args.validate_only else run_lane(args, manifest_path)
    if args.num_lanes > 1 and not args.validate_only:
        return 0 if failures == 0 else 1
    report = validate(args.campaign)
    return 0 if failures == 0 and report["status"] == "VALID_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
