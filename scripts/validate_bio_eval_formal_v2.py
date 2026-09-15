#!/usr/bin/env python3
"""Hard preflight for the approved biological evaluation protocol v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from validate_bio_eval_formal_v1 import (
    _sha256,
    _validate_conic,
    _validate_manifest,
    _validate_pannuke,
    _validate_registry,
)


PROTOCOL_PATH = REPO / "Evaluation Rules" / "protocol_v2.json"


def _task_dataset_pairs(section: dict) -> set[str]:
    pairs: set[str] = set()
    for task, value in section.items():
        if isinstance(value, list):
            pairs.update(f"{task}/{item}" for item in value)
    return pairs


def _validate_dataset_policy(protocol: dict) -> dict:
    formal = _task_dataset_pairs(protocol["tier_a"]) | _task_dataset_pairs(
        protocol["tier_b"]
    )
    observational = _task_dataset_pairs(protocol["observational"])
    excluded = _task_dataset_pairs(protocol["excluded_from_formal"]["task_datasets"])
    forbidden_datasets = set(protocol["forbidden_formal_datasets"])
    forbidden_pairs = set(protocol["forbidden_formal_task_datasets"])

    if formal & observational:
        raise AssertionError(f"formal/observational overlap: {sorted(formal & observational)}")
    if formal & excluded:
        raise AssertionError(f"formal/excluded overlap: {sorted(formal & excluded)}")
    formal_on_forbidden_dataset = {
        pair for pair in formal if pair.split("/", 1)[1] in forbidden_datasets
    }
    if formal_on_forbidden_dataset:
        raise AssertionError(
            f"formal task uses forbidden dataset: {sorted(formal_on_forbidden_dataset)}"
        )
    observational_on_forbidden_datasets = {
        pair
        for pair in observational
        if pair.split("/", 1)[1] in forbidden_datasets
    }
    if observational != observational_on_forbidden_datasets:
        raise AssertionError("observational datasets must be forbidden in formal manifests")
    if not excluded <= forbidden_pairs:
        raise AssertionError("excluded task/dataset pairs must be forbidden in formal manifests")
    if forbidden_pairs != excluded:
        raise AssertionError(
            "forbidden task/dataset pairs must exactly match excluded_from_formal: "
            f"extra={sorted(forbidden_pairs - excluded)} missing={sorted(excluded - forbidden_pairs)}"
        )

    expected_tier_a_segmentation = {
        "cellpose",
        "conic",
        "livecell",
        "monuseg",
        "pannuke",
        "tissuenet",
    }
    actual_tier_a_segmentation = set(protocol["tier_a"]["segmentation"])
    if actual_tier_a_segmentation != expected_tier_a_segmentation:
        raise AssertionError(
            "Tier A segmentation matrix mismatch: "
            f"missing={sorted(expected_tier_a_segmentation - actual_tier_a_segmentation)} "
            f"extra={sorted(actual_tier_a_segmentation - expected_tier_a_segmentation)}"
        )

    approved_retrieval = {"hpa-subcellular", "rxrx1-cross"}
    actual_retrieval = set(protocol["tier_b"]["retrieval"])
    if not approved_retrieval <= actual_retrieval:
        raise AssertionError(
            f"missing approved Tier B retrieval datasets: "
            f"{sorted(approved_retrieval - actual_retrieval)}"
        )
    if protocol["tier_a"].get("detection") or protocol["tier_b"].get("detection"):
        raise AssertionError("formal v2 detection matrix must be empty")
    if protocol["observational"].get("aggregate") is not False:
        raise AssertionError("observational datasets must be excluded from aggregates")
    if protocol["observational"].get("main_ranking") is not False:
        raise AssertionError("observational datasets must be excluded from main ranking")

    return {
        "formal_task_datasets": sorted(formal),
        "observational_task_datasets": sorted(observational),
        "excluded_task_datasets": sorted(excluded),
    }


def _validate_manifest_v2(path: Path, protocol: dict) -> dict:
    rows = json.loads(path.read_text())
    forbidden_pairs = set(protocol["forbidden_formal_task_datasets"])
    for row in rows:
        task = str(row["task"])
        datasets = set(str(row.get("dataset", "")).split("+"))
        bad = sorted(
            f"{task}/{dataset}"
            for dataset in datasets
            if f"{task}/{dataset}" in forbidden_pairs
        )
        if bad:
            raise AssertionError(f"forbidden formal task/dataset pair(s) {bad}")
    return _validate_manifest(path, protocol)


def _validate_livecell(protocol: dict, benchmark_root: Path) -> dict:
    split_policy = protocol["dense_splits"]["livecell"]
    if protocol["segmentation"]["livecell"]["split"] != split_policy["protocol"]:
        raise AssertionError("LIVECell segmentation split does not match dense_splits policy")
    annotations = (
        benchmark_root
        / "segmentation"
        / "LIVECell"
        / "LIVECell_dataset_2021"
        / "annotations"
        / "LIVECell"
    )
    actual_hashes = {
        split: _sha256(annotations / f"livecell_coco_{split}.json")
        for split in ("train", "val", "test")
    }
    if actual_hashes != split_policy["annotation_sha256"]:
        raise AssertionError(
            f"LIVECell official COCO annotation hash mismatch: {actual_hashes}"
        )
    return {
        "protocol": split_policy["protocol"],
        "image_records": split_policy["image_records"],
        "unique_image_files": split_policy["unique_image_files"],
        "train_val_shared_filenames": split_policy["train_val_shared_filenames"],
        "annotation_sha256": actual_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--command-manifest", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL_PATH.read_text())
    report = {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "dataset_policy": _validate_dataset_policy(protocol),
        "registry": _validate_registry(protocol),
        "conic": _validate_conic(protocol, Path(args.benchmark_root)),
        "livecell": _validate_livecell(protocol, Path(args.benchmark_root)),
        "pannuke": _validate_pannuke(protocol, Path(args.benchmark_root)),
    }
    if args.command_manifest:
        report["commands"] = _validate_manifest_v2(Path(args.command_manifest), protocol)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"PRECHECK FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise
