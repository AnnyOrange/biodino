#!/usr/bin/env python3
"""Plot strict 15-epoch compute-scaling curves for S+, B, and L."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "outputs/02_eval_runs/compute_scaling_strict_nosigreg_raw_20260813"
OUT = ROOT / "outputs/00_reports/compute_scaling_strict_sbl_20260814"

MODELS = {
    "S+": {"directory": "Splus", "color": "#168C84", "marker": "o"},
    "B": {"directory": "B", "color": "#D95D39", "marker": "s"},
    "L": {"directory": "L", "color": "#263D52", "marker": "^"},
}
CHECKPOINTS = {
    epoch: 1024 + (epoch - 1) * 1025
    for epoch in range(1, 16)
}
CLASSIFICATION = (
    "bbbc048-cellcycle",
    "bloodmnist",
    "breastmnist",
    "chestmnist",
    "cyclops-protein-loc",
    "dermamnist",
    "midog25-atypical",
    "octmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "pathmnist",
    "pneumoniamnist",
    "retinamnist",
    "tissuemnist",
)
RETRIEVAL = ("lc25000", "nct-crc-he-1k", "crc-val-he-7k")
SEGMENTATION = (
    "bbbc038",
    "cellpose",
    "conic",
    "livecell",
    "monuseg",
    "pannuke",
    "tissuenet",
)
METRICS = (
    ("classification_c15_primary", "C15 primary", "BA; ChestMNIST AUROC"),
    ("regression_bbbc005_r2", "BBBC005", r"$R^2$"),
    ("retrieval_ret3_r1", "Ret3 retrieval", "Recall@1"),
    ("clustering_ret3_nmi", "Ret3 clustering", "NMI"),
    ("segmentation_seg7_mdice", "Seg7", "test mDice"),
    ("detection_livecell_f1", "LiveCell detection", "test patch F1"),
)
PAPER = "#F4F0E6"
PANEL = "#FFFDF8"
INK = "#263238"
GRID = "#D9D4C9"
MUTED = "#77736C"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text())
    if not isinstance(result, dict) or result.get("error"):
        raise ValueError(f"Invalid result: {path}")
    return result


def unique(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {pattern} under {root}, got {matches}")
    return matches[0]


def scalar_result(root: Path, family: str, dataset: str, checkpoint: int) -> dict:
    candidates = list(root.glob(f"{family}/{dataset}/{checkpoint}/last_result.json"))
    candidates += list(root.glob(f"{family}/**/{checkpoint}/{dataset}/last_result.json"))
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one {family}/{dataset} result for ck{checkpoint} under {root}, "
            f"got {candidates}"
        )
    path = candidates[0]
    result = load_json(path)
    if result.get("dataset") != dataset:
        raise RuntimeError(f"Dataset mismatch in {path}")
    if result.get("channel_policy") != "auto":
        raise RuntimeError(f"Non-auto channel policy in {path}")
    return result


def audited_scalar_result(
    root: Path, family: str, dataset: str, checkpoint: int
) -> tuple[dict | None, str | None]:
    try:
        return scalar_result(root, family, dataset, checkpoint), None
    except (FileNotFoundError, json.JSONDecodeError, KeyError, RuntimeError, ValueError) as error:
        return None, f"{family}/{dataset}: {error}"


def audited_result(root: Path, pattern: str) -> tuple[dict | None, str | None]:
    try:
        path = unique(root, pattern)
        return load_json(path), None
    except (FileNotFoundError, json.JSONDecodeError, KeyError, RuntimeError, ValueError) as error:
        return None, f"{pattern}: {error}"


def summarize_epoch(model: str, epoch: int) -> dict:
    spec = MODELS[model]
    checkpoint = CHECKPOINTS[epoch]
    root = EVAL_ROOT / spec["directory"] / f"epoch_{epoch:02d}"
    scalar_root = root / "scalar"
    segmentation_root = root / "segmentation"
    scalar_complete = (scalar_root / "._strict_complete").is_file()
    segmentation_complete = (segmentation_root / "._strict_complete").is_file()
    row = {
        "model": model,
        "epoch": epoch,
        "checkpoint": checkpoint,
        "image_visits": epoch * 1025 * 1024,
        "scalar_complete": int(scalar_complete),
        "segmentation_complete": int(segmentation_complete),
        "classification_coverage": 0,
        "regression_coverage": 0,
        "retrieval_coverage": 0,
        "segmentation_coverage": 0,
        "detection_coverage": 0,
        "audit_issues": "",
        **{key: None for key, _, _ in METRICS},
        "overall5_family_mean": None,
    }
    issues = []

    classification_values = []
    for dataset in CLASSIFICATION:
        result, issue = audited_scalar_result(
            scalar_root, "bio_classification", dataset, checkpoint
        )
        if result is None:
            issues.append(issue)
        else:
            metric = "macro_auc" if dataset == "chestmnist" else "balanced_accuracy"
            try:
                classification_values.append(float(result[metric]))
            except (KeyError, TypeError, ValueError) as error:
                issues.append(f"bio_classification/{dataset}/{metric}: {error}")
    row["classification_coverage"] = len(classification_values)
    if len(classification_values) == len(CLASSIFICATION):
        row["classification_c15_primary"] = statistics.fmean(classification_values)

    regression, issue = audited_scalar_result(
        scalar_root, "bio_regression", "bbbc005", checkpoint
    )
    if regression is None:
        issues.append(issue)
    else:
        try:
            row["regression_bbbc005_r2"] = float(regression["r2"])
            row["regression_coverage"] = 1
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"bio_regression/bbbc005/r2: {error}")

    retrieval_values = []
    clustering_values = []
    for dataset in RETRIEVAL:
        result, issue = audited_scalar_result(
            scalar_root, "bio_retrieval", dataset, checkpoint
        )
        if result is None:
            issues.append(issue)
        else:
            try:
                retrieval_values.append(float(result["recall_at_1"]))
                clustering_values.append(float(result["nmi"]))
            except (KeyError, TypeError, ValueError) as error:
                issues.append(f"bio_retrieval/{dataset}: {error}")
    row["retrieval_coverage"] = min(len(retrieval_values), len(clustering_values))
    if row["retrieval_coverage"] == len(RETRIEVAL):
        row["retrieval_ret3_r1"] = statistics.fmean(retrieval_values)
        row["clustering_ret3_nmi"] = statistics.fmean(clustering_values)

    detection, issue = audited_result(
        scalar_root,
        f"bio_detection/livecell/{checkpoint}/results_bio_detection.json",
    )
    if detection is None:
        issues.append(issue)
    else:
        try:
            detection_f1 = float(detection["test_patch_f1"])
            if detection_f1 > 1:
                detection_f1 /= 100.0
            row["detection_livecell_f1"] = detection_f1
            row["detection_coverage"] = 1
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"bio_detection/livecell/test_patch_f1: {error}")

    segmentation_values = []
    for dataset in SEGMENTATION:
        result, issue = audited_result(
            segmentation_root,
            f"bio_segmentation/**/{dataset}/{checkpoint}/results.json",
        )
        if result is None:
            issues.append(issue)
        else:
            try:
                meta = result.get("_meta", {})
                if meta.get("probe_batch_size") != 32 or meta.get("probe_epochs") != 50:
                    raise ValueError("expected probe_batch_size=32 and probe_epochs=50")
                segmentation_values.append(float(result["test"]["mDice"]))
            except (KeyError, TypeError, ValueError) as error:
                issues.append(f"bio_segmentation/{dataset}: {error}")
    row["segmentation_coverage"] = len(segmentation_values)
    if len(segmentation_values) == len(SEGMENTATION):
        row["segmentation_seg7_mdice"] = statistics.fmean(segmentation_values)

    overall_inputs = (
        row["classification_c15_primary"],
        row["regression_bbbc005_r2"],
        None
        if row["retrieval_ret3_r1"] is None
        else statistics.fmean(
            (row["retrieval_ret3_r1"], row["clustering_ret3_nmi"])
        ),
        row["segmentation_seg7_mdice"],
        row["detection_livecell_f1"],
    )
    if all(value is not None for value in overall_inputs):
        row["overall5_family_mean"] = statistics.fmean(overall_inputs)
    row["audit_issues"] = " | ".join(issue for issue in issues if issue)
    return row


def finite_points(rows: list[dict], model: str, key: str) -> tuple[list[int], list[float]]:
    selected = [row for row in rows if row["model"] == model and row[key] is not None]
    return [int(row["epoch"]) for row in selected], [float(row[key]) for row in selected]


def full_series(rows: list[dict], model: str, key: str) -> tuple[list[int], list[float]]:
    selected = sorted(
        (row for row in rows if row["model"] == model), key=lambda row: row["epoch"]
    )
    return (
        [int(row["epoch"]) for row in selected],
        [math.nan if row[key] is None else float(row[key]) for row in selected],
    )


def style_axis(axis: plt.Axes, values: list[float]) -> None:
    axis.set_facecolor(PANEL)
    axis.grid(axis="y", color=GRID, linewidth=0.85)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9D988E")
    axis.tick_params(colors=INK, labelsize=8.6, length=0)
    axis.set_xlim(0.55, 15.45)
    axis.set_xticks(range(1, 16))
    axis.set_xlabel("training epoch", fontsize=9.6, color=INK)
    low, high = min(values), max(values)
    span = max(high - low, 0.015)
    axis.set_ylim(max(0.0, low - 0.16 * span), min(1.0, high + 0.22 * span))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))


def plot_task_curves(rows: list[dict], coverage_note: str) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.4, 8.3), facecolor=PAPER)
    all_handles = []
    for panel_index, (axis, (key, title, ylabel)) in enumerate(zip(axes.flat, METRICS)):
        values = []
        for model, spec in MODELS.items():
            x, y = full_series(rows, model, key)
            finite_x, finite_y = finite_points(rows, model, key)
            values.extend(finite_y)
            (line,) = axis.plot(
                x,
                y,
                color=spec["color"],
                marker=spec["marker"],
                markersize=5.8,
                markeredgecolor=PANEL,
                markeredgewidth=0.9,
                linewidth=2.0,
                label=model,
                zorder=3,
            )
            if panel_index == 0:
                all_handles.append(line)
            if finite_y:
                best_index = max(range(len(finite_y)), key=finite_y.__getitem__)
                axis.scatter(
                    [finite_x[best_index]],
                    [finite_y[best_index]],
                    s=92,
                    facecolor="none",
                    edgecolor=spec["color"],
                    linewidth=1.5,
                    zorder=4,
                )
        style_axis(axis, values)
        axis.set_title(title, loc="left", fontsize=12.0, fontweight="bold", color=INK)
        axis.set_ylabel(ylabel, fontsize=9.7, color=INK)

    figure.legend(
        all_handles,
        MODELS.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
        fontsize=10.8,
    )
    figure.suptitle(
        "Strict compute scaling — S+ / B / L",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    subtitle = "no-SIGReg · raw EMA · fixed ~1.05M-image pool · all epoch endpoints"
    if coverage_note:
        subtitle += " · gaps mark incomplete coverage"
    figure.text(0.055, 0.947, subtitle, ha="left", fontsize=10.1, color=MUTED)
    if coverage_note:
        figure.text(0.985, 0.012, coverage_note, ha="right", fontsize=8.1, color=MUTED)
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.07, top=0.86, wspace=0.25, hspace=0.34)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(OUT / f"strict_compute_scaling_tasks.{suffix}", dpi=230, facecolor=PAPER)
    plt.close(figure)


def plot_overall(rows: list[dict], coverage_note: str) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 6.2), facecolor=PAPER)
    values = []
    for model, spec in MODELS.items():
        x, y = full_series(rows, model, "overall5_family_mean")
        finite_x, finite_y = finite_points(rows, model, "overall5_family_mean")
        values.extend(finite_y)
        axis.plot(
            x,
            y,
            color=spec["color"],
            marker=spec["marker"],
            markersize=7.2,
            markeredgecolor=PANEL,
            markeredgewidth=1.0,
            linewidth=2.35,
            label=model,
            zorder=3,
        )
        if finite_y:
            best_index = max(range(len(finite_y)), key=finite_y.__getitem__)
            axis.scatter(
                [finite_x[best_index]],
                [finite_y[best_index]],
                s=130,
                facecolor="none",
                edgecolor=spec["color"],
                linewidth=1.8,
                zorder=4,
            )
    style_axis(axis, values)
    axis.set_ylabel("five-family balanced mean", fontsize=11, color=INK)
    axis.legend(loc="lower right", frameon=False, fontsize=10.5)
    figure.text(
        0.11,
        0.955,
        "Overall downstream trajectory",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.11,
        0.91,
        "mean(C15, BBBC005, mean(Ret3 R@1, NMI), Seg7, LiveCell F1)",
        ha="left",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    if coverage_note:
        figure.text(
            0.97,
            0.025,
            coverage_note,
            ha="right",
            va="bottom",
            color=MUTED,
            fontsize=8.4,
        )
    figure.subplots_adjust(left=0.11, right=0.97, bottom=0.15, top=0.82)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(OUT / f"strict_compute_scaling_overall.{suffix}", dpi=230, facecolor=PAPER)
    plt.close(figure)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def best_rows(rows: list[dict]) -> list[dict]:
    result = []
    for key, title, _ in (*METRICS, ("overall5_family_mean", "Overall-5", "")):
        for model in MODELS:
            x, y = finite_points(rows, model, key)
            if not y:
                continue
            best_index = max(range(len(y)), key=y.__getitem__)
            result.append(
                {
                    "metric": title,
                    "model": model,
                    "best_epoch": x[best_index],
                    "best_value": y[best_index],
                    "last_available_epoch": x[-1],
                    "last_available_value": y[-1],
                }
            )
    return result


def write_readme(rows: list[dict], best: list[dict], coverage_gaps: list[str]) -> None:
    lines = [
        "# Strict compute scaling: S+ / B / L",
        "",
        "All points use the same no-SIGReg/raw-EMA training protocol and fixed ~1.05M-image pool.",
        "The x-axis is the direct epoch index from the same 15-epoch trajectory.",
        "",
        "The overall curve is the equal mean of five task families: C15 classification,",
        "BBBC005 regression, Ret3 retrieval/clustering (internally averaged), Seg7 segmentation,",
        "and LiveCell detection.",
        "",
    ]
    if coverage_gaps:
        lines.extend(
            [
                "Curves are accepted by actual result-file coverage, not by launcher completion markers.",
                "Incomplete task-family points are shown as gaps:",
                "",
            ]
        )
        lines.extend(f"- {gap}" for gap in coverage_gaps)
        lines.append("")
    lines.extend(
        [
            "| metric | model | best epoch | best value | last epoch | last value |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best:
        lines.append(
            f"| {row['metric']} | {row['model']} | {row['best_epoch']} | "
            f"{row['best_value']:.6f} | {row['last_available_epoch']} | "
            f"{row['last_available_value']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `strict_compute_scaling_tasks.{png,pdf,svg}`",
            "- `strict_compute_scaling_overall.{png,pdf,svg}`",
            "- `task_family_curves.csv`",
            "- `best_epochs.csv`",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [summarize_epoch(model, epoch) for model in MODELS for epoch in range(1, 16)]
    coverage_gaps = []
    expected = {
        "classification_coverage": len(CLASSIFICATION),
        "regression_coverage": 1,
        "retrieval_coverage": len(RETRIEVAL),
        "segmentation_coverage": len(SEGMENTATION),
        "detection_coverage": 1,
    }
    labels = {
        "classification_coverage": "C15",
        "regression_coverage": "BBBC005",
        "retrieval_coverage": "Ret3",
        "segmentation_coverage": "Seg7",
        "detection_coverage": "LiveCell-det",
    }
    for row in rows:
        for key, total in expected.items():
            if row[key] != total:
                coverage_gaps.append(
                    f"{row['model']} e{row['epoch']} {labels[key]} "
                    f"{row[key]}/{total}"
                )
    coverage_note = "; ".join(coverage_gaps)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    plot_task_curves(rows, coverage_note)
    plot_overall(rows, coverage_note)
    best = best_rows(rows)
    write_csv(OUT / "task_family_curves.csv", rows)
    write_csv(OUT / "best_epochs.csv", best)
    write_readme(rows, best, coverage_gaps)
    print(f"Wrote {len(rows)} epoch rows to {OUT}")
    if coverage_gaps:
        print("Coverage gaps:")
        for gap in coverage_gaps:
            print(f"  - {gap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
