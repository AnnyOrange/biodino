#!/usr/bin/env python3
"""Plot model-size scaling on the S+ scaling-task suite in Image-2 style."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(".")
OUT = ROOT / "outputs/00_reports/20260708_taskwise_fm_figures_vertical_white"
COMMON = ROOT / "outputs/03_comparisons/7b_vs_hplus_l_b_splus_compare_20260616/task_summary_common_best.csv"
PER_BEST = ROOT / "outputs/03_comparisons/2026-7-1-test-overall/per_dataset_best.csv"
ALL_SUMMARY = Path("/mnt/huawei_deepcad/benchmark_model/benchmark_runs/_summary_20260527/all_summary_rows.csv")
SEVEN_B_RETRIEVAL = ROOT / "outputs/02_eval_runs/dino_eval_20999/raw_csv/retr_20999_summary.csv"

OUTPUT_NAME = "scaling_size_s_b_l_h_7b_splus_tasks_image2_style"
ID5_OUTPUT_NAME = "scaling_size_s_b_l_h_7b_id5_overall_image2_style"
MODELS = ("S+", "B", "L", "H+", "7B")
PARAM_M = {"S+": 22, "B": 86, "L": 300, "H+": 840, "7B": 7000}
X = np.array([math.log10(PARAM_M[model]) for model in MODELS])
BUBBLE = np.array([90, 150, 230, 330, 470])

MODEL_KEYS = {
    "S+": "bio_continue_vits16_ep15_1025",
    "B": "bio_continue_1025_a100_grad_acc_2_base",
    "L": "bio_continue_vitL16_OEP1025_ep15_b1024_1025",
    "H+": "bio_continue_rgb3_vith16plus",
}
MODEL_PREFIXES = {
    "S+": "dinov3-splus-",
    "B": "dinov3-b-",
    "L": "dinov3-l-",
    "H+": "dinov3-hplus-",
}

BLUE = "#4B87B9"
GRID = "#BEBEBE"
TEXT = "#111111"
MISSING = "#A6A6A6"


@dataclass(frozen=True)
class Panel:
    task: str
    metric: str
    title: str
    caption: str
    values: dict[str, float | None]
    sources: dict[str, str]
    ylim: tuple[float, float]
    decimals: int = 4


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def classification_panel() -> Panel:
    rows = read_csv(COMMON)
    values = {
        row["model"]: float(row["mean_best"])
        for row in rows
        if row["task"] == "classification_balanced_accuracy"
    }
    return Panel(
        task="Classification",
        metric="Balanced accuracy",
        title="Classification",
        caption="(a) Classification (common-4)",
        values={model: values.get(model) for model in MODELS},
        sources={model: str(COMMON) for model in MODELS},
        ylim=(0.690, 0.730),
    )


def regression_panel() -> Panel:
    rows = read_csv(PER_BEST)
    values: dict[str, float | None] = {}
    sources: dict[str, str] = {}
    for model in MODELS:
        key = MODEL_KEYS.get(model)
        matches = [
            row
            for row in rows
            if key is not None
            and row["model"] == key
            and row["metric_key"] == "regression_r2"
            and row["dataset"] == "bbbc005"
        ]
        values[model] = float(matches[0]["value"]) if matches else None
        sources[model] = matches[0]["source"] if matches else "not evaluated under the common protocol"
    return Panel(
        task="Regression",
        metric="R2",
        title="Regression",
        caption="(b) Regression (BBBC005)",
        values=values,
        sources=sources,
        ylim=(0.920, 0.975),
    )


def segmentation_panel() -> Panel:
    rows = read_csv(COMMON)
    values = {
        row["model"]: float(row["mean_best"])
        for row in rows
        if row["task"] == "segmentation_mDice"
    }
    return Panel(
        task="Segmentation",
        metric="mDice",
        title="Segmentation",
        caption="(e) Segmentation (common-5)",
        values={model: values.get(model) for model in MODELS},
        sources={model: str(COMMON) for model in MODELS},
        ylim=(0.660, 0.725),
    )


def overall_panel(task_panels: list[Panel]) -> Panel:
    values: dict[str, float | None] = {}
    sources: dict[str, str] = {}
    for model in MODELS:
        task_values = [panel.values[model] for panel in task_panels]
        values[model] = (
            float(np.mean(task_values))
            if all(value is not None for value in task_values)
            else None
        )
        sources[model] = "equal-weight raw mean of the five task panels"
    return Panel(
        task="ID-5 Overall",
        metric="Raw mean",
        title="Overall",
        caption="(f) ID-5 Overall",
        values=values,
        sources=sources,
        ylim=(0.825, 0.858),
    )


def lc25000_panels() -> tuple[Panel, Panel]:
    rows = read_csv(ALL_SUMMARY)
    values: dict[str, dict[str, float | None]] = {"map_at_5": {}, "nmi": {}}
    sources: dict[str, dict[str, str]] = {"map_at_5": {}, "nmi": {}}
    for metric in values:
        for model, prefix in MODEL_PREFIXES.items():
            candidates = [
                row
                for row in rows
                if row["dataset"] == "lc25000"
                and row["task"] == "retrieval_clustering"
                and row["model"].startswith(prefix)
                and row.get(metric)
            ]
            if candidates:
                best = max(candidates, key=lambda row: float(row[metric]))
                values[metric][model] = float(best[metric])
                sources[metric][model] = best["source_summary"]
            else:
                values[metric][model] = None
                sources[metric][model] = "not evaluated"

    seven_b = next(row for row in read_csv(SEVEN_B_RETRIEVAL) if row["dataset"] == "lc25000")
    for metric in values:
        values[metric]["7B"] = float(seven_b[metric])
        sources[metric]["7B"] = str(SEVEN_B_RETRIEVAL)

    retrieval = Panel(
        task="Retrieval",
        metric="mAP@5",
        title="Retrieval",
        caption="(c) Retrieval (LC25000)",
        values=values["map_at_5"],
        sources=sources["map_at_5"],
        ylim=(0.99930, 1.00004),
        decimals=5,
    )
    clustering = Panel(
        task="Clustering",
        metric="NMI",
        title="Clustering",
        caption="(d) Clustering (LC25000)",
        values=values["nmi"],
        sources=sources["nmi"],
        ylim=(0.825, 0.890),
    )
    return retrieval, clustering


def add_axis_arrows(axis: plt.Axes) -> None:
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.annotate(
        "",
        xy=(xmax, ymin),
        xytext=(xmin, ymin),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8, "shrinkA": 0, "shrinkB": 0},
        annotation_clip=False,
    )
    axis.annotate(
        "",
        xy=(xmin, ymax),
        xytext=(xmin, ymin),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8, "shrinkA": 0, "shrinkB": 0},
        annotation_clip=False,
    )


def plot_panel(axis: plt.Axes, panel: Panel) -> None:
    y = np.array([panel.values[model] if panel.values[model] is not None else np.nan for model in MODELS])
    valid = np.isfinite(y)
    axis.plot(X[valid], y[valid], linestyle=(0, (5, 5)), color="black", linewidth=0.8, zorder=1)
    axis.scatter(
        X[valid],
        y[valid],
        s=BUBBLE[valid],
        color=BLUE,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.96,
        zorder=2,
    )

    span = panel.ylim[1] - panel.ylim[0]
    for index, (model, xx, yy) in enumerate(zip(MODELS, X, y)):
        if np.isfinite(yy):
            axis.text(xx, yy, model, ha="center", va="center", fontsize=8.2, color="white", fontweight="bold", zorder=3)
            value_offset = 0.040 * span if index % 2 == 0 else 0.050 * span
            axis.text(xx, yy + value_offset, f"{yy:.{panel.decimals}f}", ha="center", va="bottom", fontsize=7.4)
        else:
            axis.text(xx, panel.ylim[0] + 0.18 * span, f"{model}\nn/a", ha="center", va="center", fontsize=8, color=MISSING)

    axis.set_xlim(X[0] - 0.18, X[-1] + 0.22)
    axis.set_ylim(*panel.ylim)
    axis.set_ylabel(r"$R^2$" if panel.metric == "R2" else panel.metric)
    axis.set_xlabel("log-FLOPs proxy")
    axis.set_xticks(X, [""] * len(MODELS))
    axis.grid(True, color=GRID, linewidth=0.75, alpha=0.75)
    axis.set_title(panel.title, fontsize=11, pad=6, fontweight="normal")
    axis.text(0.5, -0.27, panel.caption, transform=axis.transAxes, ha="center", va="top", fontsize=11.5)
    add_axis_arrows(axis)


def write_values(panels: list[Panel], output_name: str) -> None:
    rows = []
    for panel in panels:
        for model in MODELS:
            rows.append(
                {
                    "task": panel.task,
                    "metric": panel.metric,
                    "model_size": model,
                    "parameters_millions": PARAM_M[model],
                    "log_params_proxy": math.log10(PARAM_M[model]),
                    "score": "" if panel.values[model] is None else panel.values[model],
                    "source": panel.sources[model],
                }
            )
    with (OUT / f"{output_name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    retrieval, clustering = lc25000_panels()
    panels = [classification_panel(), regression_panel(), retrieval, clustering]

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Serif",
            "axes.labelsize": 11.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 4, figsize=(16.2, 4.2))
    for axis, panel in zip(axes, panels):
        plot_panel(axis, panel)
    figure.subplots_adjust(left=0.055, right=0.995, top=0.92, bottom=0.28, wspace=0.42)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(OUT / f"{OUTPUT_NAME}.{suffix}", dpi=260 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)
    write_values(panels, OUTPUT_NAME)

    segmentation = segmentation_panel()
    task_panels = [panels[0], panels[1], panels[2], panels[3], segmentation]
    id5_panels = [*task_panels, overall_panel(task_panels)]
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 7.7))
    for axis, panel in zip(axes.flat, id5_panels):
        plot_panel(axis, panel)
    figure.text(
        0.5,
        0.012,
        "ID-5 Overall is the equal-weight raw mean of the five panels; 7B is N/A because BBBC005 R2 is unavailable.",
        ha="center",
        fontsize=8.7,
        color="#666666",
    )
    figure.subplots_adjust(left=0.065, right=0.99, top=0.96, bottom=0.16, wspace=0.42, hspace=0.62)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(OUT / f"{ID5_OUTPUT_NAME}.{suffix}", dpi=260 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)
    write_values(id5_panels, ID5_OUTPUT_NAME)
    print(f"Wrote {OUT / OUTPUT_NAME}.png/.svg/.pdf/.csv")
    print(f"Wrote {OUT / ID5_OUTPUT_NAME}.png/.svg/.pdf/.csv")


if __name__ == "__main__":
    main()
