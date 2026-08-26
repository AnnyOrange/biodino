#!/usr/bin/env python3
"""Select an HS6-L teacher after sustained decline on the complete benchmark."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import time
from collections import defaultdict
from pathlib import Path


OOD_METRICS = (
    "xray_dose_r2",
    "xray_resolution_r2",
    "xray_sample_balanced_accuracy",
    "xray_variant_balanced_accuracy",
    "xray_pair_recall_at_1",
    "xray_ood_auroc",
    "cryo_class_balanced_accuracy",
    "cryo_project_balanced_accuracy",
    "cryo_quality_auroc",
    "cryo_quality_score_r2",
    "cryo_retrieval_recall_at_1",
    "cryo_cluster_nmi",
    "cryo_ood_auroc",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_symlink(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def number(payload: dict, keys: tuple[str, ...]) -> tuple[str, float] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return key, float(value)
    return None


def add_standard_result(metrics: dict[str, float], result: dict) -> None:
    if result.get("error"):
        return
    dataset = str(result.get("dataset", ""))
    rows = result.get("rows")
    if isinstance(rows, list):
        candidates = [
            row for row in rows
            if isinstance(row, dict)
            and row.get("task") == "retrieval"
            and row.get("aggregation") == "global"
        ]
        if candidates:
            picked = number(candidates[0], ("recall_at_1", "mrr", "map_at_10"))
            if picked:
                metric, value = picked
                metrics[f"retrieval:{dataset}:{metric}"] = value
        return

    task = str(result.get("task", ""))
    if task in {"classification", "multilabel_classification"}:
        keys = ("macro_auc", "macro_auroc", "auroc") if task == "multilabel_classification" else (
            "balanced_accuracy",
            "accuracy",
            "macro_f1",
        )
        family = "classification"
    elif task == "regression":
        keys = ("r2", "spearman")
        family = "regression"
    elif task in {"retrieval", "retrieval_clustering"}:
        keys = ("recall_at_1", "mrr", "nmi")
        family = "retrieval"
    else:
        return
    picked = number(result, keys)
    if picked:
        metric, value = picked
        metrics[f"{family}:{dataset}:{metric}"] = value


def collect_metrics(point_root: Path, checkpoint: int) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for path in point_root.glob("**/last_result.json"):
        result = load_json(path)
        ood_values = {key: result.get(key) for key in OOD_METRICS if isinstance(result.get(key), (int, float))}
        if ood_values:
            metrics.update({f"ood:{key}": float(value) for key, value in ood_values.items()})
        else:
            add_standard_result(metrics, result)

    for path in point_root.glob("**/results_bio_detection.json"):
        result = load_json(path)
        picked = number(result, ("test_patch_f1", "val_patch_f1"))
        if picked and not result.get("error"):
            metric, value = picked
            metrics[f"detection:{result.get('dataset', path.parent.parent.name)}:{metric}"] = value

    for path in point_root.glob(f"**/{checkpoint}/results.json"):
        result = load_json(path)
        test = result.get("test")
        if not isinstance(test, dict):
            continue
        picked = number(test, ("mDice", "mIoU", "bPQ", "AJI"))
        if picked:
            metric, value = picked
            dataset = path.parent.parent.name
            metrics[f"segmentation:{dataset}:{metric}"] = value
    return metrics


def average_ranks(values: list[tuple[int, float]]) -> dict[int, float]:
    ordered = sorted(values, key=lambda item: item[1], reverse=True)
    ranks: dict[int, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = ((index + 1) + end) / 2.0
        for checkpoint, _ in ordered[index:end]:
            ranks[checkpoint] = rank
        index = end
    return ranks


def rank_curve(points: dict[int, dict[str, float]]) -> list[dict[str, object]]:
    if not points:
        return []
    common = set.intersection(*(set(metrics) for metrics in points.values()))
    rank_sum = defaultdict(float)
    wins = defaultdict(int)
    for metric in sorted(common):
        ranks = average_ranks([(checkpoint, values[metric]) for checkpoint, values in points.items()])
        count = len(ranks)
        for checkpoint, rank in ranks.items():
            score = 1.0 if count == 1 else 1.0 - ((rank - 1.0) / (count - 1.0))
            rank_sum[checkpoint] += score
            wins[checkpoint] += int(rank == 1.0)
    return [
        {
            "checkpoint": checkpoint,
            "image_visits": (checkpoint + 1) * 1024,
            "mean_rank_score": rank_sum[checkpoint] / len(common),
            "metrics_covered": len(common),
            "wins": wins[checkpoint],
        }
        for checkpoint in sorted(points)
    ]


def write_curve_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("checkpoint", "image_visits", "mean_rank_score", "metrics_covered", "wins"),
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def decline_evidence(
    points: dict[int, dict[str, float]], peak: int, later: list[int]
) -> list[dict[str, object]]:
    common = set(points[peak])
    for checkpoint in later:
        common &= set(points[checkpoint])
    evidence: list[dict[str, object]] = []
    for checkpoint in later:
        wins = ties = losses = 0
        for metric in common:
            value = points[checkpoint][metric]
            peak_value = points[peak][metric]
            wins += int(value > peak_value)
            ties += int(value == peak_value)
            losses += int(value < peak_value)
        evidence.append(
            {
                "checkpoint": checkpoint,
                "loss_fraction_vs_peak": losses / len(common),
                "losses_vs_peak": losses,
                "metrics_compared": len(common),
                "ties_vs_peak": ties,
                "wins_vs_peak": wins,
            }
        )
    return evidence


def parse_args() -> argparse.Namespace:
    repo = Path("/mnt/huawei_deepcad/dinov3")
    run_name = "HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_6m_mix1m03_10tv107_8x5090zxr_20260826"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=repo / "outputs/01_training_runs" / run_name)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=repo / "outputs/02_eval_runs/hs6_l_6m_full_1m_3090fleet_20260826",
    )
    parser.add_argument("--expected-points", type=int, default=90)
    parser.add_argument("--min-metrics", type=int, default=46)
    parser.add_argument("--decline-window", type=int, default=3)
    parser.add_argument("--loss-fraction-required", type=float, default=0.60)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def update(args: argparse.Namespace) -> int:
    state_root = args.eval_root / "_full_peak_monitor"
    completed = sorted(
        int(path.name.removeprefix("ckpt_").removesuffix(".done"))
        for path in (args.eval_root / "_online_status").glob("ckpt_*.done")
    )
    points: dict[int, dict[str, float]] = {}
    incomplete: dict[int, int] = {}
    for checkpoint in completed:
        metrics = collect_metrics(args.eval_root / f"point_{checkpoint}", checkpoint)
        if len(metrics) >= args.min_metrics:
            points[checkpoint] = metrics
        else:
            incomplete[checkpoint] = len(metrics)

    curve = rank_curve(points)
    if curve:
        write_curve_csv(state_root / "curve.csv", curve)
        atomic_json(state_root / "curve.json", curve)
        best = max(curve, key=lambda row: (float(row["mean_rank_score"]), -int(row["checkpoint"])))
        peak = int(best["checkpoint"])
        teacher = args.train_root / f"eval/training_{peak}/teacher_checkpoint.pth"
        provisional = {
            **best,
            "complete_full_suite_points": len(points),
            "source": str(teacher),
            "updated_at_utc": utc_now(),
        }
        atomic_json(state_root / "provisional_peak.json", provisional)
        atomic_symlink(state_root / "provisional_peak_teacher_checkpoint.pth", teacher)

        later = [checkpoint for checkpoint in sorted(points) if checkpoint > peak]
        window = later[-args.decline_window :]
        if len(window) == args.decline_window:
            evidence = decline_evidence(points, peak, window)
            sustained = all(
                float(row["loss_fraction_vs_peak"]) >= args.loss_fraction_required
                for row in evidence
            )
            if sustained:
                decision = {
                    **provisional,
                    "criterion": (
                        f"last {args.decline_window} later complete-suite checkpoints each lose on at least "
                        f"{args.loss_fraction_required:.0%} of common metrics versus peak"
                    ),
                    "detected_at_utc": utc_now(),
                    "evidence": evidence,
                    "handoff": "Use selected_peak_teacher_checkpoint.pth as the teacher for Gram training.",
                }
                atomic_json(state_root / "decline_detected.json", decision)
                atomic_json(state_root / "gram_teacher_handoff.json", decision)
                atomic_symlink(state_root / "selected_peak_teacher_checkpoint.pth", teacher)

    atomic_json(
        state_root / "status.json",
        {
            "complete_checkpoint_markers": len(completed),
            "expected_points": args.expected_points,
            "incomplete_metric_counts": incomplete,
            "ranked_full_suite_points": len(points),
            "updated_at_utc": utc_now(),
        },
    )
    return len(completed)


def main() -> int:
    args = parse_args()
    args.train_root = args.train_root.resolve()
    args.eval_root = args.eval_root.resolve()
    while True:
        completed = update(args)
        if args.once or completed >= args.expected_points:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
