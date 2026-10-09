#!/usr/bin/env python3
"""Collect only validated, checkpoint-matched v4 ID scores for the 1M arms."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from audit_data_quality_1m_20261002 import ARMS, ROOT


EXPECTED = {"classification": 25, "regression": 4, "retrieval": 7,
            "clustering": 7, "segmentation": 7, "detection_proxy": 3}
FAMILIES = tuple(EXPECTED)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def one_csv(path: Path, task: str | None = None) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if task == "retrieval":
        rows = [row for row in rows if row.get("task") in ("retrieval", "retrieval_clustering")
                and row.get("aggregation") not in ("HEPG2", "HUVEC", "RPE", "U2OS", "macro-cell-type")]
    if task == "clustering":
        rows = [row for row in rows if row.get("task") in ("clustering", "retrieval_clustering")
                and "ge10" not in row.get("protocol", "")]
    if len(rows) != 1:
        raise ValueError(f"Expected one selected row in {path}, got {len(rows)}")
    return rows[0]


def scalar(row: dict, family: str, path: Path) -> float:
    key = {"classification": "macro_f1", "regression": "spearman",
           "retrieval": "map_at_5", "clustering": "nmi",
           "detection_proxy": "test_patch_f1"}[family]
    value = float(row[key])
    if family == "detection_proxy":
        value /= 100.0
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {family} score in {path}")
    return value


def retrieval_result(result: dict, family: str) -> dict:
    if "rows" not in result:
        return result
    rows = result["rows"]
    if family == "retrieval":
        selected = [row for row in rows if row.get("task") == "retrieval"
                    and row.get("aggregation") == "global"]
    else:
        selected = [row for row in rows if row.get("task") == "clustering"
                    and "ge10" not in row.get("protocol", "")]
    if len(selected) != 1:
        raise ValueError(f"Ambiguous {family} rows: {[(r.get('task'), r.get('aggregation')) for r in rows]}")
    return selected[0]


def dense_score(directory: Path, checkpoint: str, checkpoint_sha: str,
                expected_train: int | None = None) -> tuple[float, str]:
    report = read_json(directory / "validation_report.json")
    if report["status"] != "VALID_COMPLETE":
        raise ValueError(f"Unvalidated dense result: {directory}")
    invocation = read_json(directory / "invocation_manifest.json")
    if invocation["checkpoint"]["path"] != checkpoint or invocation["checkpoint"]["sha256"] != checkpoint_sha:
        raise ValueError(f"Dense result used another checkpoint: {directory}")
    paths = sorted(path for path in (directory / "results").rglob("results.json")
                   if "budget50" in path.parts)
    if len(paths) != 3:
        raise ValueError(f"Expected three budget50 seeds in {directory}, got {len(paths)}")
    scores = []
    for path in paths:
        result = read_json(path)
        if expected_train is not None and (result["_meta"]["full_train_samples"] != expected_train or
                                           result["_meta"]["used_train_samples"] != expected_train):
            raise ValueError(f"Wrong MoNuSeg training split count: {path}")
        score = float(result["test"]["mDice"])
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite mDice: {path}")
        scores.append(score)
    return statistics.mean(scores), ";".join(map(str, paths))


def collect_arm(arm: str) -> list[dict]:
    root = ROOT / "eval" / arm
    prepared = read_json(root / "prepared.json")
    audit = read_json(ROOT / "training" / arm / "audit.json")
    checkpoint = audit["checkpoint"]
    digest = audit["checkpoint_sha256"]
    if prepared["checkpoint_sha256"] != digest or prepared["monuseg_counts"] != {"train": 30, "val": 7, "test": 14}:
        raise ValueError(f"Arm {arm} provenance or MoNuSeg split mismatch")
    rows = []

    def add(family: str, dataset: str, split: str, score: float, source: str) -> None:
        rows.append(dict(arm=arm, family=family, dataset=dataset, split=split,
                         score=score, checkpoint=checkpoint,
                         checkpoint_sha256=digest, source=source))

    shared = read_json(root / "shared/campaign_manifest.json")
    if len(shared["tasks"]) != 38 or shared["checkpoint_teacher_sha256"] != {f"dq_{arm}": digest}:
        raise ValueError(f"Arm {arm}: shared task inventory or checkpoint mismatch")
    for task in shared["tasks"]:
        spec = task["dataset"]
        directory = root / "shared/cells" / task["key"]
        report = read_json(directory / "validation_report.json")
        if report["status"] != "VALID_COMPLETE":
            raise ValueError(f"Unvalidated shared task: {task['key']}")
        invocation = read_json(directory / "invocation_manifest.json")
        if invocation["checkpoint"]["path"] != checkpoint or invocation["checkpoint"]["sha256"] != digest:
            raise ValueError(f"Shared task used another checkpoint: {task['key']}")
        family, dataset = spec["task"], spec["dataset"]
        split = spec.get("split") or ""
        if family == "segmentation":
            score, source = dense_score(directory, checkpoint, digest)
            add(family, dataset, split, score, source)
            continue
        path = directory / "component_result.json"
        result = read_json(path)
        result_rows = result.get("rows", [result])
        provenance_checkpoint = result.get("_component_provenance", {}).get("checkpoint", {})
        if not result_rows or any(row.get("checkpoint") != checkpoint for row in result_rows) or \
                provenance_checkpoint.get("path") != checkpoint or provenance_checkpoint.get("sha256") != digest:
            raise ValueError(f"Wrong checkpoint in {path}")
        if family == "retrieval":
            for target in ("retrieval", "clustering"):
                chosen = retrieval_result(result, target)
                add(target, dataset, split, scalar(chosen, target, path), str(path))
        else:
            add(family, dataset, split, scalar(result, family, path), str(path))

    extension = read_json(root / "extension/campaign_manifest.json")
    if len(extension["task_ids"]) != 9 or extension["checkpoint_sha256"] != digest:
        raise ValueError(f"Arm {arm}: extension inventory mismatch")
    for task_path in sorted((root / "extension/tasks").glob("*.json")):
        task = read_json(task_path)
        if task["id"] not in extension["task_ids"] or task["dataset"] == "monuseg":
            raise ValueError(f"Unexpected extension task: {task_path}")
        directory = Path(task["output"])
        report = read_json(directory / "validation_report.json")
        if report["status"] != "VALID_COMPLETE" or report["checkpoint_sha256"] != digest:
            raise ValueError(f"Unvalidated extension task: {task['id']}")
        dataset = task["dataset"]
        kind = task["done_kind"]
        if kind == "frozen_csv":
            path = directory / "summary.csv"
            row = one_csv(path)
            if row["checkpoint"] != checkpoint:
                raise ValueError(f"Wrong checkpoint in {path}")
            family = "classification" if dataset == "lc25000" else "regression"
            add(family, dataset, row.get("split", ""), scalar(row, family, path), str(path))
        elif kind == "retrieval_csv":
            path = directory / "summary.csv"
            for family in ("retrieval", "clustering"):
                row = one_csv(path, family)
                if row["checkpoint"] != checkpoint:
                    raise ValueError(f"Wrong checkpoint in {path}")
                add(family, dataset, row.get("protocol", ""), scalar(row, family, path), str(path))
        elif kind == "rxrx3_json":
            paths = list((directory / "models").rglob("results.json"))
            if len(paths) != 1:
                raise ValueError(f"Expected one RxRx3 result under {directory}")
            result = read_json(paths[0])["tests"]["rxrx3"]
            add("retrieval", dataset, "plate-disjoint", float(result["map"]), str(paths[0]))
            add("clustering", dataset, "plate-disjoint", float(result["nmi"]), str(paths[0]))
        elif kind == "detection_json":
            path = directory / "results-detection.csv"
            row = one_csv(path)
            add("detection_proxy", dataset, row.get("conic_split_protocol", ""),
                scalar(row, "detection_proxy", path), str(path))
        else:
            raise ValueError(f"Unknown v4 task type: {kind}")

    monuseg = read_json(root / "monuseg/campaign_manifest.json")
    counts = {"train": 30, "val": 7, "test": 14}
    split_id = "monuseg2018-train30-extra7val-test14-v1"
    if len(monuseg["tasks"]) != 1 or monuseg["datasets"][0]["counts"] != counts or \
            monuseg["datasets"][0]["split_protocol_id"] != split_id:
        raise ValueError(f"Arm {arm}: MoNuSeg is not the official 30/7/14 split")
    directory = root / "monuseg/cells" / monuseg["tasks"][0]["key"]
    report = read_json(directory / "validation_report.json")
    invocation = read_json(directory / "invocation_manifest.json")
    if report.get("expected_counts") != counts or invocation["dataset"].get("counts") != counts or \
            invocation["dataset"].get("split_protocol_id") != split_id:
        raise ValueError(f"Arm {arm}: MoNuSeg result used another split")
    score, source = dense_score(directory, checkpoint, digest, expected_train=30)
    add("segmentation", "monuseg", "train30-val7-test14", score, source)

    pannuke = [row for row in rows if row["family"] == "segmentation" and row["dataset"] == "pannuke"]
    if len(pannuke) != 3:
        raise ValueError(f"Arm {arm}: expected three PanNuke folds")
    rows = [row for row in rows if row not in pannuke]
    add("segmentation", "pannuke", "three-fold-mean", statistics.mean(row["score"] for row in pannuke),
        ";".join(row["source"] for row in pannuke))
    for family, expected in EXPECTED.items():
        actual = sum(row["family"] == family for row in rows)
        if actual != expected:
            raise ValueError(f"Arm {arm}: incomplete {family}: {actual}/{expected}")
    if any(not math.isfinite(row["score"]) for row in rows):
        raise ValueError(f"Arm {arm}: nonfinite score")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS[:5]))
    parser.add_argument("--stem", default="v4_id")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.stem):
        raise ValueError("Output stem must contain only lowercase letters, digits, and underscores")
    if args.stem == "v4_id_two20tb" and tuple(args.arms) != (
        "1tb", "5tb", "20tb_route2_ddp", "20tb_route1_ddp", "100tb", "1pb"
    ):
        raise ValueError("The two-20TB figure requires both independently audited DDP arms")
    rows = [row for arm in args.arms for row in collect_arm(arm)]
    score_path = ROOT / f"{args.stem}_cell_scores.csv"
    with score_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    by_family = defaultdict(list)
    for row in rows:
        by_family[(row["arm"], row["family"])].append(row["score"])
    family_rows = []
    for arm in args.arms:
        means = []
        for family in FAMILIES:
            scores = by_family[(arm, family)]
            mean = statistics.mean(scores)
            means.append(mean)
            family_rows.append(dict(arm=arm, family=family, n_datasets=len(scores), mean_score=mean))
        family_rows.append(dict(arm=arm, family="ID family mean", n_datasets=sum(EXPECTED.values()),
                                mean_score=statistics.mean(means)))
    family_path = ROOT / f"{args.stem}_family_scores.csv"
    with family_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=family_rows[0].keys())
        writer.writeheader()
        writer.writerows(family_rows)
    (ROOT / f"{args.stem}_summary.json").write_text(json.dumps(dict(
        protocol="bio-eval-union-v4 ID only, MoNuSeg train30/val7/test14",
        arms=args.arms, cells_per_arm=sum(EXPECTED.values()),
        source_score_csv=str(score_path), source_family_csv=str(family_path),
        family_rows=family_rows), indent=2) + "\n")
    print(f"Collected {len(rows)} validated ID scores from {len(args.arms)} arms")


if __name__ == "__main__":
    main()
