#!/usr/bin/env python3
"""Render Fig. 3 panels A-E as separate white-background publication figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec

from make_fig3_complete_from_outputs import (
    DEFAULT_OUT_DIR,
    load_json,
    plot_panel_b,
    plot_panel_d,
    plot_panel_e,
    scatter_umap,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT = "#1d2b30"
GRID = "#e6e8e8"
OURS = "#ef3b33"
OFFICIAL = "#345c8c"
EXTERNAL_COLORS = [
    "#78c6b8",
    "#ab91c5",
    "#82a6ce",
    "#7fbd96",
    "#f0b186",
    "#78becd",
    "#d7c76d",
    "#c796b6",
    "#bd9c8a",
    "#8fa7a2",
    "#d2bf73",
    "#c2a2d2",
    "#a0b5ca",
    "#b4b4ad",
]
TASKWISE_SOURCE_DIR = (
    REPO_ROOT / "outputs/00_reports/20260708_taskwise_fm_figures_vertical_white"
)
TASKWISE_COLORS = {
    "BioDINOv3": OURS,
    "Virchow2": "#78c6b8",
    "GigaPath": "#ab91c5",
    "BioCLIP": "#82a6ce",
    "SigLIP2": "#7fbd96",
    "DINOv2": "#78becd",
    "MAE": "#f0b186",
    "CytoSelf": "#bd9c8a",
    "CytoImageNet": "#8fa7a2",
    "UNI": "#d7c76d",
    "JUMP-CP": "#c796b6",
    "CONCH": "#f3ca63",
    "PE": "#b99a89",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#26363d",
            "axes.labelcolor": TEXT,
            "xtick.color": "#36464c",
            "ytick.color": "#36464c",
            "text.color": TEXT,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = 300
        fig.savefig(out_dir / f"{stem}.{extension}", **kwargs)
    plt.close(fig)


def load_inputs(out_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    metrics_path = out_dir / "metrics.json"
    pair_path = out_dir / "panel_b_pair_scores.csv"
    alignment_path = out_dir / "panel_b_model_alignment.csv"
    if not metrics_path.is_file() or not pair_path.is_file():
        raise FileNotFoundError("Panel A/B cached inputs are missing under " + str(out_dir))
    metrics = load_json(metrics_path)
    pair_scores = pd.read_csv(pair_path)
    alignment = pd.read_csv(alignment_path) if alignment_path.is_file() else None
    return metrics, pair_scores, alignment, None


def make_panel_a(out_dir: Path) -> dict[str, Any]:
    umap_path = out_dir / "panel_a_crossmodality_umap.csv"
    if not umap_path.is_file():
        raise FileNotFoundError(umap_path)
    umap = pd.read_csv(umap_path)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=300)
    fig.subplots_adjust(left=0.07, right=0.89, top=0.82, bottom=0.14, wspace=0.52)
    scatter_umap(axes[0], umap, "modality", "Colored by imaging modality", max_labels=4)
    scatter_umap(axes[1], umap, "biology_id", "Colored by biology identity", max_labels=8)
    fig.suptitle(
        "A  Biological embedding landscape",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.025,
        f"H+ Biodino embeddings; cross-modality-balanced held-out pool (n={len(umap)}).",
        fontsize=8.5,
        color="#4d5b60",
    )
    save_figure(fig, out_dir, "fig3_panel_a_embedding_landscape_white")
    return {"input": str(umap_path), "n_samples": int(len(umap))}


def make_panel_b(out_dir: Path) -> dict[str, Any]:
    metrics, pair_scores, alignment, _ = load_inputs(out_dir)
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=300)
    fig.subplots_adjust(left=0.25, right=0.96, top=0.86, bottom=0.20)
    plot_panel_b(ax, pair_scores, metrics, model_alignment=alignment)
    ax.set_facecolor("white")
    fig.text(
        0.25,
        0.045,
        "All methods score the same deterministic held-out positive and negative pair mask.",
        fontsize=8.5,
        color="#4d5b60",
    )
    save_figure(fig, out_dir, "fig3_panel_b_crossmodal_correspondence_white")
    return {
        "pair_scores": str(out_dir / "panel_b_pair_scores.csv"),
        "model_alignment": str(out_dir / "panel_b_model_alignment.csv"),
        "n_pairs_per_model": int(metrics["panel_b"]["n_pairs"]),
    }


def model_colors(matrix: pd.DataFrame) -> dict[str, str]:
    colors: dict[str, str] = {}
    external_index = 0
    for _, row in matrix.iterrows():
        group = str(row["group"])
        model_key = str(row["model_key"])
        if group == "ours":
            colors[model_key] = OURS
        elif group == "official":
            colors[model_key] = OFFICIAL
        else:
            colors[model_key] = EXTERNAL_COLORS[external_index % len(EXTERNAL_COLORS)]
            external_index += 1
    return colors


def task_axis_limits(values: np.ndarray) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    spread = max(high - low, 0.02)
    lower = max(0.0, low - 0.18 * spread)
    # Some task metrics (for example MAE) are not bounded by one.
    upper = high + 0.12 * spread
    return lower, upper


def plot_taskwise_bars(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    colors: dict[str, str],
    score_column: str,
    title: str,
    ylabel: str,
    n_datasets: int,
) -> None:
    data = matrix.sort_values([score_column, "model"], ascending=[False, True]).reset_index(drop=True)
    values = data[score_column].to_numpy(dtype=float)
    bars = ax.bar(
        np.arange(len(data)),
        values,
        color=[colors[key] for key in data["model_key"]],
        edgecolor="#6d777a",
        linewidth=0.7,
    )
    ax.set_ylim(*task_axis_limits(values))
    ax.set_title(title, fontsize=11.2, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=8.8)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["model"], rotation=43, ha="right", fontsize=6.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for bar, (_, row) in zip(bars, data.iterrows()):
        score = float(row[score_column])
        label = f"{score:.3f}"
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.018
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=5.8,
            fontweight="bold" if row["group"] == "ours" else "normal",
            color=OURS if row["group"] == "ours" else "#36464c",
        )
    for label, (_, row) in zip(ax.get_xticklabels(), data.iterrows()):
        if row["group"] == "ours":
            label.set_color(OURS)
            label.set_fontweight("bold")
        elif row["group"] == "official":
            label.set_color(OFFICIAL)
            label.set_fontweight("bold")
    ax.text(
        0.99,
        0.97,
        f"shared datasets n={n_datasets}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.1,
        color="#536369",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_taskwise_scores(filename: str) -> pd.DataFrame:
    path = TASKWISE_SOURCE_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    required = {"model", "score", "group"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    data["score"] = pd.to_numeric(data["score"], errors="raise")
    return data


def plot_archived_taskwise_bars(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    ylabel: str,
    n_datasets: int,
    higher_is_better: bool = True,
    show_xlabels: bool = True,
) -> None:
    data = data.sort_values(["score", "model"], ascending=[not higher_is_better, True]).reset_index(drop=True)
    values = data["score"].to_numpy(dtype=float)
    colors = [TASKWISE_COLORS.get(str(model), "#a9b1b4") for model in data["model"]]
    bars = ax.bar(np.arange(len(data)), values, color=colors, edgecolor="#6d777a", linewidth=0.7)
    ax.set_ylim(*task_axis_limits(values))
    ax.set_title(title, fontsize=11.2, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=8.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(np.arange(len(data)))
    if show_xlabels:
        ax.set_xticklabels(data["model"], rotation=43, ha="right", fontsize=6.8)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.018
    for bar, (_, row) in zip(bars, data.iterrows()):
        score = float(row["score"])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + offset,
            f"{score:.3f}" if abs(score) < 10 else f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.8,
            fontweight="bold" if row["model"] == "BioDINOv3" else "normal",
            color=OURS if row["model"] == "BioDINOv3" else "#36464c",
        )
    if show_xlabels:
        for label, (_, row) in zip(ax.get_xticklabels(), data.iterrows()):
            if row["model"] == "BioDINOv3":
                label.set_color(OURS)
                label.set_fontweight("bold")
    ax.text(
        0.99,
        0.97,
        f"datasets n={n_datasets}; models n={len(data)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.1,
        color="#536369",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_panel_c(out_dir: Path) -> dict[str, Any]:
    source_specs = [
        ("a", "id_classification_balanced_accuracy.csv", "ID Classification: Balanced Accuracy", "balanced accuracy", 14, True),
        ("b", "id_segmentation_mdice.csv", "ID Segmentation: mDice", "mDice", 7, True),
        ("c", "id_retrieval_recall_at_1.csv", "ID Retrieval: Recall@1", "recall@1", 3, True),
        ("d", "id_clustering_nmi.csv", "ID Clustering: NMI", "NMI", 3, True),
    ]
    loaded = {filename: load_taskwise_scores(filename) for _, filename, *_ in source_specs}
    regression_r2 = load_taskwise_scores("id_regression_r2.csv")
    regression_mae = load_taskwise_scores("id_regression_mae.csv")

    fig = plt.figure(figsize=(14.2, 17.2), dpi=300)
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 1.36],
        hspace=0.66,
        wspace=0.30,
        left=0.07,
        right=0.985,
        top=0.93,
        bottom=0.055,
    )
    for panel_index, (letter, filename, title, ylabel, n_datasets, higher) in enumerate(source_specs):
        axis = fig.add_subplot(grid[panel_index // 2, panel_index % 2])
        plot_archived_taskwise_bars(
            axis, loaded[filename], title, ylabel, n_datasets, higher_is_better=higher
        )
        axis.text(
            -0.11,
            1.14,
            letter,
            transform=axis.transAxes,
            fontsize=13,
            fontweight="bold",
            color=TEXT,
        )
    # Reserve room for the R2 model labels before the MAE subplot title.
    regression_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=grid[2, :], hspace=0.70)
    r2_axis = fig.add_subplot(regression_grid[0])
    mae_axis = fig.add_subplot(regression_grid[1])
    plot_archived_taskwise_bars(
        r2_axis,
        regression_r2,
        "ID Regression: BBBC005 R2",
        "R2 (higher is better)",
        1,
        show_xlabels=True,
    )
    plot_archived_taskwise_bars(
        mae_axis,
        regression_mae,
        "ID Regression: BBBC005 MAE",
        "MAE (lower is better)",
        1,
        higher_is_better=False,
    )
    r2_axis.text(
        -0.045,
        1.17,
        "e",
        transform=r2_axis.transAxes,
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )
    fig.suptitle(
        "C  Task-wise frozen evaluation across foundation models",
        x=0.07,
        y=0.987,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.016,
        "Archived all-FM taskwise benchmark summaries. Bars show raw task metrics, not ranks; "
        "the available model set is shown in each panel. BioDINOv3 is the archived taskwise "
        "BioDINO reference; the newer H+ S6 alpha=1 checkpoint is reported separately.",
        fontsize=7.8,
        color="#4d5b60",
    )
    save_figure(fig, out_dir, "fig3_panel_c_taskwise_fm_montage_white")
    return {
        "source_dir": str(TASKWISE_SOURCE_DIR),
        "sources": [str(TASKWISE_SOURCE_DIR / filename) for _, filename, *_ in source_specs]
        + [
            str(TASKWISE_SOURCE_DIR / "id_regression_r2.csv"),
            str(TASKWISE_SOURCE_DIR / "id_regression_mae.csv"),
        ],
        "tasks": {
            "classification": {"metric": "balanced_accuracy", "n_datasets": 14},
            "segmentation": {"metric": "mDice", "n_datasets": 7},
            "retrieval": {"metric": "recall_at_1", "n_datasets": 3},
            "clustering": {"metric": "nmi", "n_datasets": 3},
            "regression": {"metrics": ["r2", "mae"], "n_datasets": 1},
        },
    }


def make_panel_d(out_dir: Path) -> dict[str, Any]:
    fewshot_path = out_dir / "panel_d_fewshot_summary.csv"
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=300)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.15)
    status = plot_panel_d(ax, fewshot_path)
    ax.set_facecolor("white")
    save_figure(fig, out_dir, "fig3_panel_d_annotation_efficiency_white")
    return status


def make_panel_e(out_dir: Path) -> dict[str, Any]:
    fig = plt.figure(figsize=(11.6, 8.2), dpi=300)
    grid = fig.add_gridspec(1, 1, left=0.10, right=0.97, top=0.94, bottom=0.10)
    status = plot_panel_e(grid[0], out_dir)
    for axis in fig.axes:
        axis.set_facecolor("white")
    save_figure(fig, out_dir, "fig3_panel_e_spatial_representation_white")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    statuses = {
        "a": make_panel_a(args.output_dir),
        "b": make_panel_b(args.output_dir),
        "c": make_panel_c(args.output_dir),
        "d": make_panel_d(args.output_dir),
        "e": make_panel_e(args.output_dir),
    }
    output = args.output_dir / "fig3_separate_panels_white_summary.json"
    output.write_text(json.dumps(statuses, indent=2) + "\n")
    print(f"[fig3-separate] wrote A-E white panels under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
