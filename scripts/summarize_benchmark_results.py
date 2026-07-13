#!/usr/bin/env python3
"""Build benchmark summary tables and best-checkpoint indices from outputs/02_eval_runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TASK_DIRS = {
    "bio_classification": "classification",
    "bio_regression": "regression",
    "bio_retrieval": "retrieval",
    "bio_detection": "detection",
    "bio_segmentation": "segmentation",
    "bio_segmentation_best": "segmentation",
}

RESULT_FILES = {"last_result.json", "results.json", "results_bio_detection.json"}
UNSUPPORTED_RESULT_DATASETS = {"chammi-cp-task4", "chammi-hpa-task3"}
PRUNE_DIRS = {
    ".git",
    ".heavy_slots",
    "__pycache__",
    "cache",
    "ckpt",
    "features",
    "logs",
    "master_logs",
    "nan_logs",
    "remote_logs",
    "remote_scripts",
}

SUMMARY_COLUMNS = [
    "model_key",
    "train_run",
    "eval_run",
    "subrun",
    "task",
    "task_dir",
    "dataset",
    "protocol",
    "ckpt",
    "primary_metric",
    "metric_value",
    "higher_is_better",
    "split",
    "accuracy",
    "balanced_accuracy",
    "label_accuracy",
    "macro_f1",
    "micro_f1",
    "macro_auc",
    "micro_auc",
    "r2",
    "spearman",
    "mae",
    "mrr",
    "recall_at_1",
    "recall_at_5",
    "nmi",
    "cluster_accuracy",
    "test_patch_f1",
    "test_patch_precision",
    "test_patch_recall",
    "seg_mDice",
    "seg_mIoU",
    "seg_AJI",
    "seg_bPQ",
    "n_train",
    "n_test",
    "checkpoint_path",
    "train_config",
    "result_path",
]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def fmt(value: Any) -> Any:
    if is_number(value):
        return f"{float(value):.8g}"
    return value if value is not None else ""


def discover_result_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        for name in filenames:
            if name in RESULT_FILES:
                yield Path(dirpath) / name


def find_task(parts: Tuple[str, ...]) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    for i, part in enumerate(parts):
        if part in TASK_DIRS:
            return i, part, TASK_DIRS[part]
    return None, None, None


def parse_result_path(path: Path, eval_root: Path) -> Optional[Dict[str, str]]:
    try:
        rel = path.relative_to(eval_root)
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    task_idx, task_dir, task = find_task(parts)
    if task_idx is None or task_dir is None or task is None:
        return None

    eval_run = parts[0]
    subrun = "/".join(parts[1:task_idx])
    after = parts[task_idx + 1 : -1]
    dataset = protocol = ckpt = ""

    if task == "segmentation":
        if len(after) >= 3:
            protocol, dataset, ckpt = after[0], after[1], after[2]
        elif len(after) >= 2:
            dataset, ckpt = after[0], after[1]
    else:
        if len(after) >= 2:
            dataset, ckpt = after[0], after[1]
        elif len(after) == 1:
            dataset = after[0]

    return {
        "eval_run": eval_run,
        "subrun": subrun,
        "task_dir": task_dir,
        "task": task,
        "dataset": dataset,
        "protocol": protocol,
        "ckpt": ckpt,
    }


def basename_before_marker(path_value: str, marker: str) -> str:
    if not path_value:
        return ""
    parts = path_value.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part == marker and i > 0:
            return parts[i - 1]
    return ""


def resolve_relocated_output_path(path_value: str) -> str:
    """Map pre-cleanup outputs/<run>/... paths to outputs/01_training_runs/<run>/... when possible."""
    if not path_value:
        return ""
    path = Path(path_value)
    if path.exists():
        return str(path)

    parts = path_value.replace("\\", "/").split("/")
    if "outputs" not in parts:
        return path_value
    out_idx = parts.index("outputs")
    if out_idx + 1 >= len(parts):
        return path_value
    if parts[out_idx + 1] in {"00_reports", "01_training_runs", "02_eval_runs", "03_comparisons", "04_recovery_reruns", "05_debug_smoke", "06_data_prep_transfer", "07_scripts", "08_logs", "09_archives", "99_cache_misc"}:
        return path_value

    prefix = parts[: out_idx + 1]
    suffix = parts[out_idx + 1 :]
    candidates = [
        Path("/".join(prefix + ["01_training_runs"] + suffix)),
        Path("outputs/01_training_runs").joinpath(*suffix),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return path_value


def ckpt_from_checkpoint(path_value: str) -> str:
    if not path_value:
        return ""
    if str(path_value).isdigit():
        return str(path_value)
    parts = str(path_value).replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part == "ckpt" and i + 1 < len(parts):
            return parts[i + 1]
    for part in reversed(parts):
        if part.isdigit():
            return part
    return ""


def normalize_subrun(subrun: str, ckpt: str) -> str:
    """Drop path wrappers that only repeat the checkpoint id."""
    if not subrun or not ckpt:
        return subrun
    checkpoint_aliases = {ckpt, f"ckpt_{ckpt}", f"training_{ckpt}"}
    parts = [part for part in subrun.split("/") if part]
    if parts and all(part in checkpoint_aliases for part in parts):
        return ""
    return subrun


def train_run_from_data(data: Dict[str, Any]) -> str:
    checkpoint = str(data.get("checkpoint", "") or "")
    train_config = str(data.get("train_config", "") or "")
    train_run = basename_before_marker(checkpoint, "ckpt")
    if train_run:
        return train_run
    if train_config:
        p = Path(train_config)
        if p.name in {"config.yaml", "config.yml"}:
            return p.parent.name
    return ""


def pick_primary(task: str, dataset: str, data: Dict[str, Any], flat: Dict[str, Any]) -> Tuple[str, Optional[float], str]:
    if task == "classification":
        result_task = str(data.get("task", ""))
        for key in ("macro_auc", "macro_auroc", "auroc"):
            if (result_task == "multilabel_classification" or dataset == "chestmnist") and is_number(flat.get(key)):
                return key, float(flat[key]), ""
        for key in ("balanced_accuracy", "accuracy", "macro_f1"):
            if is_number(flat.get(key)):
                return key, float(flat[key]), ""
    elif task == "regression":
        for key in ("r2", "spearman"):
            if is_number(flat.get(key)):
                return key, float(flat[key]), ""
    elif task == "retrieval":
        for key in ("recall_at_1", "mrr", "nmi"):
            if is_number(flat.get(key)):
                return key, float(flat[key]), ""
    elif task == "detection":
        for key in ("test_patch_f1", "val_patch_f1"):
            if is_number(flat.get(key)):
                return key, float(flat[key]), ""
    elif task == "segmentation":
        for split in ("test", "val"):
            section = data.get(split)
            if isinstance(section, dict):
                for key in ("mDice", "mIoU", "bPQ", "AJI"):
                    if is_number(section.get(key)):
                        return f"{split}.{key}", float(section[key]), split
    return "", None, ""


def flatten_metrics(data: Dict[str, Any], task: str, split: str) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if task == "segmentation":
        section = data.get(split) if split else data.get("test") or data.get("val") or {}
        if isinstance(section, dict):
            flat["seg_mDice"] = section.get("mDice")
            flat["seg_mIoU"] = section.get("mIoU")
            flat["seg_AJI"] = section.get("AJI")
            flat["seg_bPQ"] = section.get("bPQ")
        return flat

    for key in SUMMARY_COLUMNS:
        if key in data and is_number(data[key]):
            flat[key] = data[key]
    # Common aliases from older scripts.
    if "macro_auroc" in data and "macro_auc" not in flat:
        flat["macro_auc"] = data["macro_auroc"]
    return flat


def row_from_result(path: Path, eval_root: Path) -> Optional[Dict[str, Any]]:
    parsed = parse_result_path(path, eval_root)
    if parsed is None:
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("error"):
        return None

    task = parsed["task"]
    dataset = str(data.get("dataset") or parsed["dataset"])
    # Older sweeps may contain open-set CHAMMI rows. They are not valid
    # closed-set supervised-probe results, so keep them out of aggregates.
    if task == "classification" and dataset in UNSUPPORTED_RESULT_DATASETS:
        return None
    checkpoint_path = resolve_relocated_output_path(str(data.get("checkpoint", "") or ""))
    train_config = resolve_relocated_output_path(str(data.get("train_config", "") or ""))
    train_run = train_run_from_data(data)
    ckpt = ckpt_from_checkpoint(checkpoint_path) or parsed["ckpt"]
    subrun = normalize_subrun(parsed["subrun"], ckpt)
    model_key = train_run or subrun or parsed["eval_run"]
    if subrun:
        model_key = f"{model_key}/{subrun}" if train_run else model_key

    provisional_flat = flatten_metrics(data, task, "test")
    primary_metric, metric_value, split = pick_primary(task, dataset, data, {**data, **provisional_flat})
    flat = flatten_metrics(data, task, split)

    row: Dict[str, Any] = {col: "" for col in SUMMARY_COLUMNS}
    row.update(parsed)
    row["subrun"] = subrun
    row.update(
        {
            "model_key": model_key,
            "train_run": train_run,
            "dataset": dataset,
            "ckpt": ckpt,
            "primary_metric": primary_metric,
            "metric_value": metric_value if metric_value is not None else "",
            "higher_is_better": "1" if metric_value is not None else "",
            "split": split,
            "checkpoint_path": checkpoint_path,
            "train_config": train_config,
            "result_path": str(path),
        }
    )
    for key, value in flat.items():
        if key in row:
            row[key] = value
    for key in ("n_train", "n_test"):
        if key in data:
            row[key] = data[key]
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k, "")) for k in fieldnames})


def metric_float(row: Dict[str, Any]) -> Optional[float]:
    value = row.get("metric_value")
    if is_number(value):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def build_best_by_dataset(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = metric_float(row)
        if value is None:
            continue
        key = (
            str(row["model_key"]),
            str(row["eval_run"]),
            str(row["subrun"]),
            str(row["task"]),
            str(row["dataset"]),
            str(row["protocol"]),
        )
        groups[key].append(row)

    best_rows: List[Dict[str, Any]] = []
    for key, members in groups.items():
        best = max(members, key=lambda r: metric_float(r) if metric_float(r) is not None else float("-inf"))
        out = dict(best)
        out["num_candidates"] = len({str(r["ckpt"]) for r in members})
        best_rows.append(out)
    return sorted(best_rows, key=lambda r: (r["model_key"], r["eval_run"], r["subrun"], r["task"], r["dataset"], r["protocol"]))


def build_checkpoint_index(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Rank checkpoints within each eval sweep and dataset, then average ranks
    # across datasets. This avoids averaging incomparable metrics directly.
    dataset_groups: Dict[Tuple[str, str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if metric_float(row) is None or not row.get("ckpt"):
            continue
        key = (
            str(row["model_key"]),
            str(row["eval_run"]),
            str(row["subrun"]),
            str(row["task"]),
            str(row["dataset"]),
            str(row["protocol"]),
        )
        dataset_groups[key].append(row)

    aggregate: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for key, members in dataset_groups.items():
        unique = {}
        for row in members:
            ckpt = str(row["ckpt"])
            value = metric_float(row)
            if value is None:
                continue
            if ckpt not in unique or value > unique[ckpt][0]:
                unique[ckpt] = (value, row)
        ordered = sorted(unique.values(), key=lambda item: item[0], reverse=True)
        n = len(ordered)
        if n == 0:
            continue
        for rank, (value, row) in enumerate(ordered, start=1):
            score = 1.0 if n == 1 else 1.0 - ((rank - 1) / (n - 1))
            agg_key = (str(row["model_key"]), str(row["eval_run"]), str(row["subrun"]), str(row["ckpt"]))
            agg = aggregate.setdefault(
                agg_key,
                {
                    "model_key": row["model_key"],
                    "train_run": row["train_run"],
                    "eval_run": row["eval_run"],
                    "subrun": row["subrun"],
                    "ckpt": row["ckpt"],
                    "rank_score_sum": 0.0,
                    "metrics_covered": 0,
                    "best_on_metrics": 0,
                    "tasks": set(),
                    "datasets": set(),
                    "checkpoint_path": row["checkpoint_path"],
                },
            )
            agg["rank_score_sum"] += score
            agg["metrics_covered"] += 1
            agg["best_on_metrics"] += 1 if rank == 1 else 0
            agg["tasks"].add(str(row["task"]))
            agg["datasets"].add(str(row["dataset"]))
            if not agg.get("checkpoint_path") and row.get("checkpoint_path"):
                agg["checkpoint_path"] = row["checkpoint_path"]

    rows_out: List[Dict[str, Any]] = []
    for agg in aggregate.values():
        metrics_covered = int(agg["metrics_covered"])
        rows_out.append(
            {
                "model_key": agg["model_key"],
                "train_run": agg["train_run"],
                "eval_run": agg["eval_run"],
                "subrun": agg["subrun"],
                "ckpt": agg["ckpt"],
                "mean_rank_score": agg["rank_score_sum"] / metrics_covered if metrics_covered else 0.0,
                "metrics_covered": metrics_covered,
                "tasks_covered": len(agg["tasks"]),
                "datasets_covered": len(agg["datasets"]),
                "best_on_metrics": agg["best_on_metrics"],
                "tasks": " ".join(sorted(agg["tasks"])),
                "datasets": " ".join(sorted(agg["datasets"])),
                "checkpoint_path": agg["checkpoint_path"],
                "recommended": "",
            }
        )

    by_sweep: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows_out:
        by_sweep[(str(row["model_key"]), str(row["eval_run"]), str(row["subrun"]))].append(row)
    for members in by_sweep.values():
        best = max(members, key=lambda r: (float(r["mean_rank_score"]), int(r["metrics_covered"]), int(r["best_on_metrics"])))
        best["recommended"] = "1"

    return sorted(rows_out, key=lambda r: (r["model_key"], r["eval_run"], r["subrun"], -float(r["mean_rank_score"]), str(r["ckpt"])))


def write_markdown_index(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recommended = [r for r in rows if str(r.get("recommended")) == "1"]
    lines = [
        "# Best checkpoint index",
        "",
        "Recommended checkpoints are selected by mean rank across available task/dataset primary metrics within each eval sweep.",
        "",
        "| model_key | eval_run | subrun | recommended_ckpt | mean_rank_score | metrics | tasks | datasets |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in recommended:
        lines.append(
            "| {model_key} | {eval_run} | {subrun} | {ckpt} | {score:.4f} | {metrics} | {tasks} | {datasets} |".format(
                model_key=row["model_key"],
                eval_run=row["eval_run"],
                subrun=row["subrun"],
                ckpt=row["ckpt"],
                score=float(row["mean_rank_score"]),
                metrics=row["metrics_covered"],
                tasks=row["tasks_covered"],
                datasets=row["datasets_covered"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize BioDINO benchmark outputs and build best-checkpoint indices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-root", default="outputs/02_eval_runs")
    parser.add_argument("--reports-dir", default="outputs/00_reports")
    parser.add_argument("--prefix", default="benchmark")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_root = Path(args.eval_root)
    reports_dir = Path(args.reports_dir)
    rows: List[Dict[str, Any]] = []
    for path in discover_result_files(eval_root):
        row = row_from_result(path, eval_root)
        if row is not None and row.get("metric_value") != "":
            rows.append(row)

    rows = sorted(rows, key=lambda r: (r["model_key"], r["eval_run"], r["subrun"], r["task"], r["dataset"], str(r["ckpt"]), r["protocol"]))
    summary_path = reports_dir / f"{args.prefix}_results_summary.csv"
    write_csv(summary_path, rows, SUMMARY_COLUMNS)

    best_dataset = build_best_by_dataset(rows)
    best_dataset_path = reports_dir / "best_checkpoint_by_dataset.csv"
    write_csv(best_dataset_path, best_dataset, SUMMARY_COLUMNS + ["num_candidates"])

    index_rows = build_checkpoint_index(rows)
    index_columns = [
        "recommended",
        "model_key",
        "train_run",
        "eval_run",
        "subrun",
        "ckpt",
        "mean_rank_score",
        "metrics_covered",
        "tasks_covered",
        "datasets_covered",
        "best_on_metrics",
        "tasks",
        "datasets",
        "checkpoint_path",
    ]
    index_path = reports_dir / "best_checkpoint_index.csv"
    write_csv(index_path, index_rows, index_columns)
    md_path = reports_dir / "best_checkpoint_index.md"
    write_markdown_index(md_path, index_rows)

    print(f"results: {len(rows)} rows -> {summary_path}")
    print(f"best by dataset: {len(best_dataset)} rows -> {best_dataset_path}")
    print(f"checkpoint index: {len(index_rows)} rows -> {index_path}")
    print(f"markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
