#!/usr/bin/env python3
"""Plot task-family checkpoint peaks for the complete HS6-L 5TB trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


TASKS = (
    "classification",
    "regression",
    "retrieval",
    "clustering",
    "segmentation",
    "detection",
)
TASK_LABELS = {
    "classification": "Classification",
    "regression": "Regression",
    "retrieval": "Retrieval",
    "clustering": "Clustering",
    "segmentation": "Segmentation",
    "detection": "Detection",
}
TASK_COLORS = {
    "classification": "#2563A6",
    "regression": "#9A4EAE",
    "retrieval": "#16807A",
    "clustering": "#D17A22",
    "segmentation": "#3A8F5C",
    "detection": "#C64B4B",
}
TASK_RAW_LABELS = {
    "classification": "Macro BA / macro-AUC",
    "regression": "Macro R2",
    "retrieval": "Macro R@1",
    "clustering": "Macro NMI",
    "segmentation": "Macro mDice",
    "detection": "Macro patch-F1",
}
EXPECTED_MIN_METRICS = {
    "classification": 20,
    "regression": 2,
    "retrieval": 4,
    "clustering": 4,
    "segmentation": 5,
    "detection": 1,
}


def parse_args() -> argparse.Namespace:
    repo = Path("/mnt/huawei_deepcad/dinov3")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=repo
        / "outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908",
    )
    parser.add_argument(
        "--complete-curve",
        type=Path,
        default=repo
        / "outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908"
        / "_full_peak_audit_20260911/curve.csv",
        help="Audited complete-suite curve whose checkpoint set defines the common support.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "outputs/00_reports/hs6_l5_5tb_task_peak_curves_20260912",
    )
    parser.add_argument("--anchor-checkpoints", type=int, nargs="*", default=(7807, 17079))
    parser.add_argument("--endpoint", type=int, default=23911)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def finite_number(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def add_metric(
    metrics: dict[str, dict[str, float]],
    family: str,
    key: str,
    value: float | None,
) -> None:
    if value is None:
        return
    if key in metrics[family]:
        raise ValueError(f"Duplicate metric {family}:{key}")
    metrics[family][key] = value


def collect_checkpoint(eval_root: Path, checkpoint: int) -> dict[str, dict[str, float]]:
    point = eval_root / f"point_{checkpoint}"
    if not point.is_dir():
        raise FileNotFoundError(point)
    metrics: dict[str, dict[str, float]] = defaultdict(dict)

    for lane in sorted(point.glob("classification_*")):
        for path in lane.glob("**/last_result.json"):
            result = read_json(path)
            if result.get("error"):
                continue
            dataset = str(result.get("dataset", path.parent.parent.name))
            task = str(result.get("task", ""))
            if task == "multilabel_classification":
                value = next(
                    (
                        number
                        for key in ("macro_auc", "macro_auroc", "auroc")
                        if (number := finite_number(result, key)) is not None
                    ),
                    None,
                )
                metric = "macro_auc"
            else:
                value = next(
                    (
                        number
                        for key in ("balanced_accuracy", "accuracy", "macro_f1")
                        if (number := finite_number(result, key)) is not None
                    ),
                    None,
                )
                metric = "balanced_accuracy"
            add_metric(metrics, "classification", f"{dataset}:{metric}", value)

    for path in point.glob("regression/**/last_result.json"):
        result = read_json(path)
        if result.get("error"):
            continue
        dataset = str(result.get("dataset", path.parent.parent.name))
        add_metric(metrics, "regression", f"{dataset}:r2", finite_number(result, "r2"))

    for path in point.glob("retrieval/**/last_result.json"):
        result = read_json(path)
        if result.get("error"):
            continue
        dataset = str(result.get("dataset", path.parent.parent.name))
        add_metric(
            metrics,
            "retrieval",
            f"{dataset}:recall_at_1",
            finite_number(result, "recall_at_1"),
        )
        add_metric(metrics, "clustering", f"{dataset}:nmi", finite_number(result, "nmi"))

    for lane in sorted(point.glob("segmentation_*")):
        for path in lane.glob(f"**/{checkpoint}/results.json"):
            result = read_json(path)
            test = result.get("test")
            if not isinstance(test, dict):
                continue
            dataset = path.parent.parent.name
            add_metric(metrics, "segmentation", f"{dataset}:mDice", finite_number(test, "mDice"))

    for path in point.glob("detection/**/results_bio_detection.json"):
        result = read_json(path)
        if result.get("error"):
            continue
        dataset = str(result.get("dataset", path.parent.parent.name))
        value = finite_number(result, "test_patch_f1")
        if value is None:
            value = finite_number(result, "val_patch_f1")
        if value is not None and value > 1.0:
            value /= 100.0
        add_metric(metrics, "detection", f"{dataset}:patch_f1", value)

    return {family: dict(values) for family, values in metrics.items()}


def load_checkpoint_support(path: Path) -> list[int]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    checkpoints = [int(row["checkpoint"]) for row in rows]
    if checkpoints != sorted(set(checkpoints)):
        raise ValueError("Complete-suite checkpoint support must be unique and sorted")
    if len(checkpoints) < 20:
        raise ValueError(f"Too few complete checkpoints for a trajectory: {len(checkpoints)}")
    return checkpoints


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


def task_curves(
    checkpoints: list[int],
    collected: dict[int, dict[str, dict[str, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    peaks: list[dict[str, Any]] = []
    for family in TASKS:
        common = set.intersection(*(set(collected[checkpoint].get(family, {})) for checkpoint in checkpoints))
        if len(common) < EXPECTED_MIN_METRICS[family]:
            raise ValueError(
                f"{family} has only {len(common)} common metrics; expected at least "
                f"{EXPECTED_MIN_METRICS[family]}"
            )
        rank_sum = defaultdict(float)
        wins = defaultdict(int)
        for metric in sorted(common):
            ranks = average_ranks(
                [(checkpoint, collected[checkpoint][family][metric]) for checkpoint in checkpoints]
            )
            denominator = len(checkpoints) - 1
            for checkpoint, rank in ranks.items():
                rank_sum[checkpoint] += 1.0 - ((rank - 1.0) / denominator)
                wins[checkpoint] += int(rank == 1.0)
        family_rows = []
        for checkpoint in checkpoints:
            raw_values = [collected[checkpoint][family][metric] for metric in sorted(common)]
            row = {
                "task_family": family,
                "checkpoint": checkpoint,
                "image_visits": (checkpoint + 1) * 1024,
                "mean_percentile_rank": rank_sum[checkpoint] / len(common),
                "raw_macro_mean": sum(raw_values) / len(raw_values),
                "metrics_covered": len(common),
                "metric_wins": wins[checkpoint],
            }
            rows.append(row)
            family_rows.append(row)
        peak = max(
            family_rows,
            key=lambda row: (
                float(row["mean_percentile_rank"]),
                float(row["raw_macro_mean"]),
                -int(row["checkpoint"]),
            ),
        )
        raw_peak = max(
            family_rows,
            key=lambda row: (float(row["raw_macro_mean"]), -int(row["checkpoint"])),
        )
        endpoint = family_rows[-1]
        peaks.append(
            {
                **peak,
                "raw_peak_checkpoint": raw_peak["checkpoint"],
                "raw_peak_macro_mean": raw_peak["raw_macro_mean"],
                "endpoint_raw_macro_mean": endpoint["raw_macro_mean"],
                "raw_peak_minus_endpoint": float(raw_peak["raw_macro_mean"])
                - float(endpoint["raw_macro_mean"]),
                "endpoint_checkpoint": endpoint["checkpoint"],
                "endpoint_mean_percentile_rank": endpoint["mean_percentile_rank"],
                "peak_minus_endpoint_rank": float(peak["mean_percentile_rank"])
                - float(endpoint["mean_percentile_rank"]),
            }
        )
    return rows, peaks


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_axis(endpoint: int) -> tuple[int, tuple[int, ...], tuple[str, ...]]:
    tick_step = 5000
    axis_max = max(tick_step, ((endpoint + tick_step - 1) // tick_step) * tick_step)
    ticks = tuple(range(0, axis_max + 1, tick_step))
    labels = tuple("0" if tick == 0 else f"{tick // 1000}k" for tick in ticks)
    return axis_max, ticks, labels


def plot(
    rows: list[dict[str, Any]],
    peaks: list[dict[str, Any]],
    anchors: tuple[int, ...],
    endpoint: int,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#27313A",
            "text.color": "#20272D",
            "xtick.color": "#44515C",
            "ytick.color": "#44515C",
        }
    )
    peak_by_task = {row["task_family"]: row for row in peaks}
    checkpoint_count = len({int(row["checkpoint"]) for row in rows})
    axis_max, ticks, tick_labels = checkpoint_axis(endpoint)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharex=True, sharey=True)
    anchor_colors = ("#3977C5", "#D9892B")
    for axis, family in zip(axes.flat, TASKS, strict=True):
        family_rows = [row for row in rows if row["task_family"] == family]
        x = [int(row["checkpoint"]) for row in family_rows]
        y = [float(row["mean_percentile_rank"]) for row in family_rows]
        color = TASK_COLORS[family]
        axis.plot(x, y, color=color, linewidth=1.8, alpha=0.92, zorder=2)
        axis.scatter(x, y, s=15, color=color, edgecolor="white", linewidth=0.35, zorder=3)
        for anchor, anchor_color in zip(anchors, anchor_colors, strict=False):
            axis.axvline(anchor, color=anchor_color, linestyle=(0, (4, 3)), linewidth=1.25, alpha=0.8)
        axis.axvline(endpoint, color="#7A858E", linestyle=":", linewidth=1.1, alpha=0.8)
        peak = peak_by_task[family]
        peak_x = int(peak["checkpoint"])
        peak_y = float(peak["mean_percentile_rank"])
        axis.scatter(
            [peak_x],
            [peak_y],
            marker="*",
            s=155,
            color="#E33D3D",
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )
        horizontal = 16 if peak_x < 19000 else -72
        axis.annotate(
            f"peak ck{peak_x}",
            (peak_x, peak_y),
            xytext=(horizontal, -17),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="#9D2727",
            arrowprops={"arrowstyle": "-", "color": "#C85757", "lw": 0.7},
        )
        axis.set_title(
            f"{TASK_LABELS[family]}  |  {int(peak['metrics_covered'])} dataset-metrics",
            fontsize=11.5,
        )
        axis.set_ylim(-0.03, 1.04)
        axis.set_xlim(0, axis_max)
        axis.set_xticks(ticks)
        axis.set_xticklabels(tick_labels)
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.8, alpha=0.75)
        axis.grid(axis="x", color="#EDF0F2", linewidth=0.55, alpha=0.7)

    axes[0, 0].set_ylabel("Within-task mean percentile rank")
    axes[1, 0].set_ylabel("Within-task mean percentile rank")
    for axis in axes[1]:
        axis.set_xlabel("5TB SSL checkpoint (optimizer updates)")

    fig.suptitle(
        "HS6-L 5TB: downstream task families peak at different checkpoints",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.942,
        f"{checkpoint_count} checkpoints with a complete common suite; each dataset-metric is percentile-ranked over the same trajectory",
        ha="center",
        fontsize=10.5,
        color="#4B5862",
    )
    handles = [
        plt.Line2D([0], [0], color=anchor_colors[index], linestyle=(0, (4, 3)), lw=1.5, label=f"official-style anchor ck{anchor}")
        for index, anchor in enumerate(anchors[:2])
    ]
    handles.extend(
        [
            plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#E33D3D", markeredgecolor="white", markersize=12, label="task-family peak"),
            plt.Line2D([0], [0], color="#7A858E", linestyle=":", lw=1.3, label=f"trajectory endpoint ck{endpoint}"),
        ]
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.text(
        0.5,
        0.052,
        "Descriptive downstream trajectory, not a fitted scaling law. Frozen probes are evaluation only and never update SSL.",
        ha="center",
        fontsize=9,
        color="#68747D",
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.12, top=0.90, hspace=0.28, wspace=0.17)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output / f"hs6_l5_5tb_task_family_peak_curves.{suffix}",
            dpi=240 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(fig)


def plot_raw(
    rows: list[dict[str, Any]],
    peaks: list[dict[str, Any]],
    anchors: tuple[int, ...],
    endpoint: int,
    output: Path,
) -> None:
    peak_by_task = {row["task_family"]: row for row in peaks}
    checkpoint_count = len({int(row["checkpoint"]) for row in rows})
    axis_max, ticks, tick_labels = checkpoint_axis(endpoint)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharex=True)
    anchor_colors = ("#3977C5", "#D9892B")
    for axis, family in zip(axes.flat, TASKS, strict=True):
        family_rows = [row for row in rows if row["task_family"] == family]
        x = [int(row["checkpoint"]) for row in family_rows]
        y = [float(row["raw_macro_mean"]) for row in family_rows]
        color = TASK_COLORS[family]
        axis.plot(x, y, color=color, linewidth=1.8, alpha=0.92, zorder=2)
        axis.scatter(x, y, s=15, color=color, edgecolor="white", linewidth=0.35, zorder=3)
        for anchor, anchor_color in zip(anchors, anchor_colors, strict=False):
            axis.axvline(anchor, color=anchor_color, linestyle=(0, (4, 3)), linewidth=1.25, alpha=0.8)
        axis.axvline(endpoint, color="#7A858E", linestyle=":", linewidth=1.1, alpha=0.8)
        peak = peak_by_task[family]
        peak_x = int(peak["raw_peak_checkpoint"])
        peak_y = float(peak["raw_peak_macro_mean"])
        axis.scatter(
            [peak_x],
            [peak_y],
            marker="*",
            s=155,
            color="#E33D3D",
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )
        horizontal = 16 if peak_x < 19000 else -72
        axis.annotate(
            f"raw peak ck{peak_x}",
            (peak_x, peak_y),
            xytext=(horizontal, -17),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="#9D2727",
            arrowprops={"arrowstyle": "-", "color": "#C85757", "lw": 0.7},
        )
        span = max(y) - min(y)
        padding = max(span * 0.16, max(abs(value) for value in y) * 0.002, 1e-4)
        axis.set_ylim(min(y) - padding, max(y) + padding)
        axis.set_xlim(0, axis_max)
        axis.set_xticks(ticks)
        axis.set_xticklabels(tick_labels)
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.8, alpha=0.75)
        axis.grid(axis="x", color="#EDF0F2", linewidth=0.55, alpha=0.7)
        axis.set_title(
            f"{TASK_LABELS[family]}  |  {int(peak['metrics_covered'])} dataset-metrics",
            fontsize=11.5,
        )
        axis.set_ylabel(TASK_RAW_LABELS[family])
    for axis in axes[1]:
        axis.set_xlabel("5TB SSL checkpoint (optimizer updates)")

    fig.suptitle(
        "HS6-L 5TB: raw task-macro performance peaks are checkpoint-specific",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.942,
        f"Same {checkpoint_count} complete checkpoints; equal mean over each task family's frozen-evaluation datasets",
        ha="center",
        fontsize=10.5,
        color="#4B5862",
    )
    handles = [
        plt.Line2D([0], [0], color=anchor_colors[index], linestyle=(0, (4, 3)), lw=1.5, label=f"official-style anchor ck{anchor}")
        for index, anchor in enumerate(anchors[:2])
    ]
    handles.extend(
        [
            plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#E33D3D", markeredgecolor="white", markersize=12, label="raw task-family peak"),
            plt.Line2D([0], [0], color="#7A858E", linestyle=":", lw=1.3, label=f"trajectory endpoint ck{endpoint}"),
        ]
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.text(
        0.5,
        0.052,
        "Raw metrics are shown on separate axes and must not be averaged across task families. Frozen probes never update SSL.",
        ha="center",
        fontsize=9,
        color="#68747D",
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.90, hspace=0.28, wspace=0.22)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output / f"hs6_l5_5tb_task_family_raw_macro_curves.{suffix}",
            dpi=240 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(fig)


def render_readme(
    checkpoints: list[int],
    peaks: list[dict[str, Any]],
    eval_root: Path,
    complete_curve: Path,
) -> str:
    lines = [
        "# HS6-L 5TB task-family checkpoint peaks",
        "",
        "This is a descriptive downstream trajectory over one fixed 5TB SSL run. It is not a fitted scaling law and does not use labels to train or select an SSL intervention.",
        "",
        f"- Common complete checkpoints: {len(checkpoints)} (`ck{checkpoints[0]}` through `ck{checkpoints[-1]}`).",
        f"- Evaluation root: `{eval_root}`.",
        f"- Audited support: `{complete_curve}`.",
        "- Aggregation: rank every dataset-metric over the same checkpoints, map best to 1 and worst to 0, then take an equal mean within each task family.",
        "",
        "| Task family | Metrics | Raw peak ck | Rank peak ck | Raw peak - endpoint | Peak rank - endpoint |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in peaks:
        lines.append(
            f"| {TASK_LABELS[str(row['task_family'])]} | {int(row['metrics_covered'])} | "
            f"{int(row['raw_peak_checkpoint'])} | {int(row['checkpoint'])} | "
            f"{float(row['raw_peak_minus_endpoint']):+.6f} | "
            f"{float(row['peak_minus_endpoint_rank']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "The dispersion of task-family peaks is evidence for the motivation that one fixed checkpoint anchor cannot represent every downstream optimum. It is not, by itself, a causal proof that the official Gram loss is incomplete; that requires matched Gram interventions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.eval_root = args.eval_root.resolve()
    args.complete_curve = args.complete_curve.resolve()
    args.output = args.output.resolve()
    checkpoints = load_checkpoint_support(args.complete_curve)
    if args.endpoint not in checkpoints:
        raise ValueError(f"Endpoint ck{args.endpoint} is absent from complete support")
    missing_anchors = sorted(set(args.anchor_checkpoints) - set(checkpoints))
    if missing_anchors:
        raise ValueError(f"Anchor checkpoints absent from complete support: {missing_anchors}")
    collected = {
        checkpoint: collect_checkpoint(args.eval_root, checkpoint) for checkpoint in checkpoints
    }
    rows, peaks = task_curves(checkpoints, collected)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "task_family_curve.csv", rows)
    write_csv(args.output / "task_family_peaks.csv", peaks)
    manifest = {
        "status": "VALID_COMPLETE_DESCRIPTIVE_TASK_FAMILY_TRAJECTORY",
        "eval_root": str(args.eval_root),
        "complete_curve": str(args.complete_curve),
        "complete_curve_sha256": sha256(args.complete_curve),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "anchor_checkpoints": list(args.anchor_checkpoints),
        "endpoint": args.endpoint,
        "aggregation": "per-metric percentile rank over common checkpoints, equal mean within task family",
        "label_policy": "Frozen downstream evaluation only; no downstream label updates SSL.",
    }
    atomic_write(args.output / "input_manifest.json", json.dumps(manifest, indent=2) + "\n")
    atomic_write(
        args.output / "README.md",
        render_readme(checkpoints, peaks, args.eval_root, args.complete_curve),
    )
    plot(rows, peaks, tuple(args.anchor_checkpoints), args.endpoint, args.output)
    plot_raw(rows, peaks, tuple(args.anchor_checkpoints), args.endpoint, args.output)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(args.output),
                "peaks": {
                    row["task_family"]: int(row["checkpoint"]) for row in peaks
                },
                "raw_peaks": {
                    row["task_family"]: int(row["raw_peak_checkpoint"])
                    for row in peaks
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
