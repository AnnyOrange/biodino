#!/usr/bin/env python3
"""Audit native CTC input coverage and FM18 result completeness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "ctc-labelled-training-sequence-domain-heldout-v1"
DATA = ROOT / "outputs/02_eval_inputs/formal_v3/ctc_native_full/data_manifest.json"
SPLIT = ROOT / "outputs/02_eval_inputs/formal_v3/ctc/split_manifest.jsonl"
OUT = ROOT / "outputs/00_reports/ctc_fm18_readiness_20261021.json"
FM14 = (
    "dinov2", "mae", "siglip2", "pe", "bioclip", "cytoself", "jump_cp",
    "cytoimagenet", "uni", "conch", "phikon2", "virchow2", "gigapath", "hoptimus0",
)
NEW4 = ("distillpath", "morphem", "genbio_pathfm", "openpath")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = json.loads(DATA.read_text())
    errors = []
    if data.get("status") != "READY" or data.get("protocol_id") != PROTOCOL_ID:
        errors.append("cache status or protocol mismatch")
    if data.get("source_split_manifest_sha256") != sha256(SPLIT):
        errors.append("locked split manifest hash mismatch")
    domains = set(data["domains"])
    if len(domains) != 20:
        errors.append("20-domain inventory mismatch")
    held_out = []
    for fold in data["folds"]:
        train = set(fold["train_domains"])
        test = set(fold["test_domains"])
        if len(train) != 16 or len(test) != 4 or train & test or train | test != domains:
            errors.append(f"invalid fold {fold['fold']}")
        held_out.extend(test)
    if len(data["folds"]) != 5 or sorted(held_out) != sorted(domains):
        errors.append("five-fold coverage mismatch")

    missing_files = []
    wrong_sizes = []
    for record in data["files"]:
        path = Path(record["path"])
        if not path.is_file():
            missing_files.append(str(path))
        elif path.stat().st_size != record["bytes"]:
            wrong_sizes.append(str(path))
    if missing_files or wrong_sizes:
        errors.append("missing or size-mismatched extracted files")

    # A count proxy, segmentation mask, or partial fold is never a native result.
    result_roots = ROOT / "outputs/02_eval_runs"
    models = []
    for name in (*FM14, *NEW4):
        matches = []
        for path in result_roots.glob(f"ctc_native_full_*/models/{name}/results.json"):
            try:
                result = json.loads(path.read_text())
                domains = result["domain_rows"]
                valid = (
                    result.get("status") == "VALID_COMPLETE"
                    and result.get("protocol_id") == PROTOCOL_ID
                    and len(result.get("folds", [])) == 5
                    and len(domains) == 20
                    and {row["domain"] for row in domains} == set(data["domains"])
                    and all(int(row["ctc_metrics"]["Valid"]) == 1 for row in domains)
                )
                if valid:
                    matches.append(str(path))
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                pass
        models.append({"model": name, "native_complete": bool(matches), "results": matches})

    report = {
        "status": "INPUTS_READY" if not errors else "INPUTS_INVALID",
        "protocol_id": PROTOCOL_ID,
        "data_manifest": str(DATA),
        "data_manifest_sha256": sha256(DATA),
        "split_manifest": str(SPLIT),
        "split_manifest_sha256": sha256(SPLIT),
        "domain_count": len(data["domains"]),
        "fold_count": len(data["folds"]),
        "extracted_file_count": len(data["files"]),
        "missing_files": missing_files,
        "wrong_sizes": wrong_sizes,
        "errors": errors,
        "models": models,
        "native_complete_models": sum(row["native_complete"] for row in models),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "domain_count", "fold_count", "extracted_file_count",
        "native_complete_models", "errors",
    )}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
