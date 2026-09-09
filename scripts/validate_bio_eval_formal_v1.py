#!/usr/bin/env python3
"""Hard preflight for the approved biological evaluation protocol v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO / "Evaluation Rules" / "protocol_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_arg(cmd: list[str], name: str, expected: str) -> None:
    if name not in cmd:
        raise AssertionError(f"missing {name}: {' '.join(cmd)}")
    pos = cmd.index(name)
    actual = cmd[pos + 1] if pos + 1 < len(cmd) else None
    if actual != expected:
        raise AssertionError(f"{name}={actual!r}, expected {expected!r}: {' '.join(cmd)}")


def _validate_manifest(path: Path, protocol: dict) -> dict:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise AssertionError(f"empty command manifest: {path}")
    forbidden = set(protocol["forbidden_formal_datasets"])
    counts: dict[str, int] = {}
    datasets_by_task: dict[str, set[str]] = {}
    for row in rows:
        task = str(row["task"])
        cmd = [str(x) for x in row["cmd"]]
        counts[task] = counts.get(task, 0) + 1
        if "--smoke" in cmd or "--fast-eval" in cmd:
            raise AssertionError(f"formal manifest contains smoke/fast eval: {' '.join(cmd)}")
        dataset_tokens = set(str(row.get("dataset", "")).split("+"))
        datasets_by_task.setdefault(task, set()).update(dataset_tokens)
        bad = sorted(forbidden & dataset_tokens)
        if bad:
            raise AssertionError(f"forbidden formal dataset(s) {bad}: {' '.join(cmd)}")
        if task in {"classification", "regression", "retrieval"}:
            _require_arg(cmd, "--batch-size", str(protocol["batch_sizes"]["frozen"]))
        elif task == "segmentation":
            _require_arg(cmd, "--feature-batch-size", str(protocol["batch_sizes"]["segmentation_feature"]))
            _require_arg(cmd, "--probe-batch-size", str(protocol["batch_sizes"]["segmentation_probe"]))
            _require_arg(cmd, "--probe-epochs", str(protocol["probe"]["segmentation_epochs"]))
            _require_arg(cmd, "--probe-eval-every", str(protocol["probe"]["segmentation_eval_every"]))
            _require_arg(cmd, "--dataset-split-protocol", "formal-v1")
        elif task == "detection":
            _require_arg(cmd, "--batch-size", str(protocol["batch_sizes"]["detection"]))
            _require_arg(cmd, "--epochs", str(protocol["probe"]["detection_epochs"]))
            _require_arg(cmd, "--conic-split-protocol", protocol["dense_splits"]["conic"])
    for task in ("classification", "regression", "retrieval", "segmentation", "detection"):
        expected = set(protocol["tier_a"].get(task, [])) | set(protocol["tier_b"].get(task, []))
        actual = datasets_by_task.get(task, set())
        if actual != expected:
            raise AssertionError(
                f"{task} dataset matrix mismatch: missing={sorted(expected-actual)} "
                f"extra={sorted(actual-expected)}"
            )
    return {"jobs": len(rows), "jobs_by_task": counts, "manifest": str(path)}


def _validate_conic(protocol: dict, benchmark_root: Path) -> dict:
    from dinov3.eval.bio_segmentation.datasets.conic import get_conic_paths

    root = benchmark_root / "segmentation" / "conic" / "extracted"
    patch_info = next(root.rglob("patch_info.csv"))
    with patch_info.open(newline="") as handle:
        reader = csv.DictReader(handle)
        column = "patch_info" if "patch_info" in (reader.fieldnames or []) else (reader.fieldnames or [""])[0]
        sources = [str(row[column]).split("-")[0] for row in reader]
    split_indices = {}
    split_sources = {}
    split_protocol = protocol["dense_splits"]["conic"]
    for split in ("train", "val", "test"):
        _, _, indices = get_conic_paths(str(root), split=split, split_protocol=split_protocol)
        split_indices[split] = set(indices)
        split_sources[split] = {sources[i] for i in indices}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_indices[left] & split_indices[right]:
            raise AssertionError(f"CoNIC index overlap: {left}/{right}")
        if split_sources[left] & split_sources[right]:
            raise AssertionError(f"CoNIC source overlap: {left}/{right}")
    sizes = {key: len(value) for key, value in split_indices.items()}
    if sizes != {"train": 3469, "val": 494, "test": 1018}:
        raise AssertionError(f"unexpected CoNIC formal split sizes: {sizes}")
    return {
        "protocol": split_protocol,
        "sizes": sizes,
        "source_sizes": {key: len(value) for key, value in split_sources.items()},
        "patch_info_sha256": _sha256(patch_info),
    }


def _validate_pannuke(protocol: dict, benchmark_root: Path) -> dict:
    from dinov3.eval.bio_segmentation.feature_extractor import _build_dataset

    root = benchmark_root / "segmentation" / "pannuke" / "extracted"
    expected = {
        "pannuke-fold1-train-fold2-val-fold3-test": [2656, 2523, 2722],
        "pannuke-fold2-train-fold1-val-fold3-test": [2523, 2656, 2722],
        "pannuke-fold3-train-fold2-val-fold1-test": [2722, 2523, 2656],
    }
    actual = {}
    for split_protocol in protocol["dense_splits"]["pannuke"]:
        actual[split_protocol] = [
            len(
                _build_dataset(
                    "pannuke",
                    str(root),
                    split,
                    256,
                    dataset_split_protocol=split_protocol,
                )
            )
            for split in ("train", "val", "test")
        ]
    if actual != expected:
        raise AssertionError(f"unexpected PanNuke fold sizes: {actual}")
    return {"protocols": actual}


def _validate_registry(protocol: dict) -> dict:
    from dinov3.eval.bio_frozen_eval.registry import ALL_DATASETS

    expected = set(protocol["tier_a"]["classification"] + protocol["tier_a"]["regression"])
    expected.update(protocol["tier_b"]["classification"] + protocol["tier_b"]["regression"])
    missing = sorted(expected - set(ALL_DATASETS))
    if missing:
        raise AssertionError(f"frozen registry missing formal datasets: {missing}")
    return {"registered": len(ALL_DATASETS), "missing": []}


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
        "registry": _validate_registry(protocol),
        "conic": _validate_conic(protocol, Path(args.benchmark_root)),
        "pannuke": _validate_pannuke(protocol, Path(args.benchmark_root)),
    }
    if args.command_manifest:
        report["commands"] = _validate_manifest(Path(args.command_manifest), protocol)
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
