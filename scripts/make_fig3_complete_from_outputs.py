#!/usr/bin/env python3
"""Assemble a multi-panel Fig. 3 from cached representation outputs."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/04_figures/fig3_representation_20260812"
DEFAULT_ALPHA_JSON = (
    REPO_ROOT
    / "outputs/00_reports/s6_sweetspot_scaling_e15_20260809/remote_summaries/hplus_nosigreg/alpha.json"
)
PALETTE = [
    "#006d77",
    "#e29578",
    "#2a9d8f",
    "#e76f51",
    "#8d6b94",
    "#b08900",
    "#3a86ff",
    "#6c757d",
    "#588157",
    "#d62828",
    "#9c6644",
    "#118ab2",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def remap_top_labels(labels: pd.Series, max_labels: int, other: str = "other") -> tuple[list[str], list[str]]:
    labels = labels.astype(str).fillna("unknown")
    order = labels.value_counts().index.tolist()
    keep = set(order[:max_labels])
    mapped = [x if x in keep else other for x in labels]
    legend_order = [x for x in order[:max_labels] if x in set(mapped)]
    if other in mapped:
        legend_order.append(other)
    return mapped, legend_order


def style_scatter_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#27343a")


def scatter_umap(ax: plt.Axes, df: pd.DataFrame, label_col: str, title: str, max_labels: int) -> None:
    mapped, order = remap_top_labels(df[label_col], max_labels=max_labels)
    color_map = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(order)}
    colors = [color_map[label] for label in mapped]
    ax.scatter(df["umap_1"], df["umap_2"], c=colors, s=13, alpha=0.84, linewidths=0)
    style_scatter_axis(ax)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=7)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[label], label=label, markersize=4.3)
        for label in order
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.02),
        frameon=False,
        fontsize=6.2,
        handletextpad=0.25,
        borderaxespad=0.0,
        labelspacing=0.38,
    )


def plot_panel_b(
    ax: plt.Axes,
    pair_scores: pd.DataFrame,
    metrics: dict[str, Any],
    model_alignment: pd.DataFrame | None = None,
) -> None:
    if model_alignment is not None and not model_alignment.empty:
        label_map = {
            "imagenet_resnet50": "ImageNet\nResNet50",
            "dinov2_local": "DINOv2\nViT-B/14",
            "dinov3_official_vitl16": "DINOv3-L/16\noriginal",
            "Biodino H+": "Biodino H+\nours",
        }
        order = ["imagenet_resnet50", "dinov2_local", "dinov3_official_vitl16", "Biodino H+"]
        rows = []
        for key in order:
            hit = model_alignment[model_alignment["model"].astype(str).eq(key)]
            if len(hit):
                rows.append(hit.iloc[0].to_dict())
        if not rows:
            rows = model_alignment.to_dict(orient="records")
        labels = [label_map.get(str(row["model"]), str(row["model"])) for row in rows]
        scores = np.asarray([float(row["alignment_score"]) for row in rows], dtype=float)
        lows = np.asarray([float(row["ci_low"]) for row in rows], dtype=float)
        highs = np.asarray([float(row["ci_high"]) for row in rows], dtype=float)
        y = np.arange(len(rows))
        colors = ["#b8b8aa", "#9db5c7", "#89a2a6", "#0a9396"][: len(rows)]
        ax.axvline(0.0, color="#6c6c62", lw=0.9, ls=(0, (3, 3)), zorder=0)
        ax.barh(y, scores, color=colors, edgecolor="#27343a", linewidth=0.8, height=0.62)
        ax.errorbar(
            scores,
            y,
            xerr=np.vstack([scores - lows, highs - scores]),
            fmt="none",
            ecolor="#27343a",
            elinewidth=1.1,
            capsize=3,
        )
        for yy, score in zip(y, scores):
            ax.text(score + 0.006, yy, f"{score:.3f}", va="center", ha="left", fontsize=8.2, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.1)
        ax.invert_yaxis()
        ax.set_xlabel("alignment = same-biology cross-modality sim - different-biology same-modality sim", fontsize=8.1)
        ax.set_title("B  Cross-modal biological correspondence | same pair mask", loc="left", fontsize=10.5, fontweight="bold", pad=7)
        ax.grid(axis="x", color="#dbcdb7", linewidth=0.7, alpha=0.85)
        return

    pos = pair_scores[pair_scores["pair_type"].eq("same_biology_cross_modality")]["cosine_similarity"].to_numpy()
    neg = pair_scores[pair_scores["pair_type"].eq("different_biology_same_modality")]["cosine_similarity"].to_numpy()
    parts = ax.violinplot([pos, neg], positions=[1, 2], widths=0.68, showmeans=False, showextrema=False)
    for body, color in zip(parts["bodies"], ["#0a9396", "#ca6702"]):
        body.set_facecolor(color)
        body.set_alpha(0.25)
        body.set_edgecolor("none")
    rng = np.random.default_rng(20260812)
    for xpos, values, color in [(1, pos, "#0a9396"), (2, neg, "#ca6702")]:
        draw = values if len(values) <= 700 else rng.choice(values, size=700, replace=False)
        ax.scatter(
            np.full(len(draw), xpos) + rng.normal(0.0, 0.055, len(draw)),
            draw,
            s=8,
            color=color,
            alpha=0.20,
            linewidths=0,
        )
        mean = float(np.mean(values))
        ax.plot([xpos - 0.23, xpos + 0.23], [mean, mean], color=color, lw=2.6)
        ax.text(xpos, mean + 0.022, f"{mean:.3f}", ha="center", va="bottom", fontsize=7.8, color=color)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["same biology\ncross modality", "different biology\nsame modality"], fontsize=8)
    ax.set_ylabel("cosine similarity", fontsize=9)
    ax.set_title("B  Cross-modal biological correspondence", loc="left", fontsize=10.5, fontweight="bold", pad=7)
    ax.grid(axis="y", color="#dbcdb7", linewidth=0.7, alpha=0.85)
    score = metrics["panel_b"]["biological_alignment_score"]
    ax.text(
        0.03,
        0.96,
        f"alignment = {score['score']:.3f}\n95% CI [{score['ci_low']:.3f}, {score['ci_high']:.3f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.30", fc="#fff8e8", ec="#c9b17d", lw=0.9),
    )


def plot_panel_c(ax: plt.Axes, matrix: pd.DataFrame) -> dict[str, Any]:
    required = {
        "model",
        "group",
        "classification_percentile",
        "regression_percentile",
        "retrieval_percentile",
        "clustering_percentile",
        "detection_percentile",
    }
    missing = sorted(required - set(matrix.columns))
    if missing:
        raise ValueError(f"Panel C matrix is missing columns: {missing}")
    matrix = matrix.copy()
    task_columns = [
        ("classification_percentile", "Class.\nmacro F1", 25),
        ("regression_percentile", "Reg.\nR2", 1),
        ("retrieval_percentile", "Retr.\nmAP@5", 4),
        ("clustering_percentile", "Clust.\nNMI", 4),
        ("detection_percentile", "Detect.\nF1", 1),
    ]
    values = matrix[[column for column, _, _ in task_columns]].to_numpy(dtype=float)
    ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap="YlGnBu", interpolation="nearest")
    ax.set_yticks(np.arange(len(matrix)))
    labels = matrix["model"].astype(str).tolist()
    ax.set_yticklabels(labels, fontsize=6.7)
    for tick, group in zip(ax.get_yticklabels(), matrix["group"].astype(str)):
        if group == "ours":
            tick.set_color("#007f78")
            tick.set_fontweight("bold")
        elif group == "official":
            tick.set_color("#43545c")
            tick.set_fontweight("bold")
    ax.set_xticks(range(len(task_columns)))
    ax.set_xticklabels(
        [f"{label}\n(n={count})" for _, label, count in task_columns],
        fontsize=6.7,
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=2)
    ax.set_title(
        "C  Frozen transfer across biological tasks",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        pad=31,
    )
    for row_index, row in matrix.reset_index(drop=True).iterrows():
        if row["group"] == "ours":
            color = "#007f78"
            linewidth = 1.9
        elif row["group"] == "official":
            color = "#43545c"
            linewidth = 1.3
        else:
            continue
        ax.add_patch(
            Rectangle(
                (-0.5, row_index - 0.5),
                len(task_columns),
                1.0,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                clip_on=False,
            )
        )
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            text_color = "#f8f4eb" if value >= 0.70 else "#17313b"
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.8,
                color=text_color,
                fontweight="bold" if matrix.iloc[row_index]["group"] == "ours" else "normal",
            )
    ax.set_xlabel(
        "Mean within-dataset rank percentile (higher is better; 16 frozen encoders)",
        fontsize=7.3,
        labelpad=8,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return {
        "n_models": int(len(matrix)),
        "task_columns": [column for column, _, _ in task_columns],
        "models": matrix["model"].astype(str).tolist(),
    }


def plot_panel_d(ax: plt.Axes, fewshot_csv: Path | None) -> dict[str, Any]:
    ax.set_title("D  Annotation efficiency | BBBC048 cell-cycle", loc="left", fontsize=10.5, fontweight="bold", pad=7)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 120)
    ax.set_xticks([1, 5, 10, 25, 100])
    ax.set_xticklabels(["1", "5", "10", "25", "100"], fontsize=8)
    ax.set_xlabel("training labels used (%)", fontsize=8.5)
    ax.set_ylabel("macro F1 (fixed group test fold)", fontsize=8.5)
    ax.grid(True, color="#dbcdb7", linewidth=0.7, alpha=0.85)
    status: dict[str, Any] = {"status": "placeholder", "fewshot_csv": None}
    if fewshot_csv is not None and fewshot_csv.exists():
        df = pd.read_csv(fewshot_csv)
        required = {"model", "label_percent", "score"}
        if required.issubset(df.columns):
            df = df[np.isfinite(df["score"].astype(float))].copy()
            if df.empty:
                raise ValueError(f"No finite few-shot scores in {fewshot_csv}")
            expected_labels = {1.0, 5.0, 10.0, 25.0, 100.0}
            complete_models = (
                df.groupby("model")["label_percent"]
                .apply(lambda values: expected_labels.issubset(set(np.asarray(values, dtype=float))))
            )
            df = df[df["model"].isin(complete_models[complete_models].index)].copy()
            if df.empty:
                raise ValueError(f"No models with all 1/5/10/25/100% points in {fewshot_csv}")
            low = max(0.0, float(df["score"].min()) - 0.07)
            high = min(1.0, float(df["score"].max()) + 0.08)
            ax.set_ylim(low, max(high, low + 0.10))
            for i, (model, sub) in enumerate(df.groupby("model")):
                sub = sub.sort_values("label_percent")
                ax.plot(
                    sub["label_percent"],
                    sub["score"],
                    marker="o",
                    lw=2.2,
                    color=PALETTE[i % len(PALETTE)],
                    label=str(model),
                )
                if "score_std" in sub.columns:
                    std = sub["score_std"].fillna(0.0).to_numpy(dtype=float)
                    ax.fill_between(
                        sub["label_percent"].to_numpy(dtype=float),
                        sub["score"].to_numpy(dtype=float) - std,
                        sub["score"].to_numpy(dtype=float) + std,
                        color=PALETTE[i % len(PALETTE)],
                        alpha=0.10,
                        linewidth=0,
                    )
            ax.legend(frameon=False, fontsize=6.5, loc="lower right", handlelength=1.6, labelspacing=0.3)
            status = {
                "status": "plotted",
                "fewshot_csv": str(fewshot_csv),
                "n": int(len(df)),
                "models": sorted(df["model"].astype(str).unique().tolist()),
            }
            return status
    ax.set_ylim(0.15, 0.75)
    xs = np.array([1, 5, 10, 25, 100])
    for i, label in enumerate(["Random init", "ImageNet", "DINOv2", "Biodino"]):
        ax.plot(xs, np.full_like(xs, 0.55 + 0.06 * i, dtype=float), "--", lw=1.4, color=PALETTE[i], alpha=0.42)
        ax.text(102, 0.55 + 0.06 * i, label, fontsize=7, va="center", color=PALETTE[i])
    ax.text(
        0.04,
        0.93,
        "Protocol fixed; true few-shot runs not found locally.\nPopulate with 1/5/10/25/100% labels before final submission.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.32", fc="#fff8e8", ec="#c9b17d", lw=0.9),
    )
    return status


def plot_panel_e(container: Any, out_dir: Path) -> dict[str, Any]:
    qualitative = [
        (out_dir / "panel_e_dense_tissuenet_input.png", "Input"),
        (out_dir / "panel_e_dense_tissuenet_ground_truth.png", "Ground truth"),
        (out_dir / "panel_e_dense_tissuenet_official_prediction.png", "Original DINOv3 H+"),
        (out_dir / "panel_e_dense_tissuenet_biodino_prediction.png", "BioDINO H+"),
    ]
    metrics_path = out_dir / "panel_e_dense_transfer_metrics.csv"
    metadata_path = out_dir / "panel_e_dense_transfer_metadata.json"
    missing = [str(path) for path, _ in qualitative if not path.is_file()]
    if not metrics_path.is_file():
        missing.append(str(metrics_path))
    if not metadata_path.is_file():
        missing.append(str(metadata_path))
    if missing:
        raise FileNotFoundError(
            "Panel E requires supervised dense-transfer assets; the old unsupervised "
            "cluster image is not a valid fallback. Missing: " + ", ".join(missing)
        )

    metadata = load_json(metadata_path)
    example = metadata["qualitative"]
    metrics = pd.read_csv(metrics_path)
    expected_models = ["dinov3_hplus_official", "biodino_hplus"]
    paired = (
        metrics.pivot(index="dataset_label", columns="model", values="mDice")
        .dropna(subset=expected_models)
        .reset_index()
    )
    if paired.empty:
        raise ValueError(f"No paired test mDice rows in {metrics_path}")

    gs = GridSpecFromSubplotSpec(
        2,
        4,
        subplot_spec=container,
        height_ratios=[1.0, 0.93],
        hspace=0.36,
        wspace=0.08,
    )
    image_axes = [plt.subplot(gs[0, index]) for index in range(4)]
    for ax, (path, title) in zip(image_axes, qualitative):
        ax.imshow(Image.open(path))
        ax.set_title(title, fontsize=8.2, fontweight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
            spine.set_color("#647176")
    image_axes[0].set_title("E  Frozen dense transfer\nInput", loc="left", fontsize=10.2, fontweight="bold", pad=4)

    image_axes[0].text(
        0.03,
        0.04,
        "TissueNet test",
        transform=image_axes[0].transAxes,
        fontsize=6.6,
        color="white",
        bbox=dict(fc="#172a31", alpha=0.76, ec="none", pad=1.8),
    )
    image_axes[2].text(
        0.03,
        0.04,
        f"cell Dice {float(example['official_foreground_dice']):.3f}",
        transform=image_axes[2].transAxes,
        fontsize=6.5,
        color="white",
        bbox=dict(fc="#172a31", alpha=0.76, ec="none", pad=1.8),
    )
    image_axes[3].text(
        0.03,
        0.04,
        f"cell Dice {float(example['biodino_foreground_dice']):.3f}",
        transform=image_axes[3].transAxes,
        fontsize=6.5,
        color="white",
        bbox=dict(fc="#172a31", alpha=0.76, ec="none", pad=1.8),
    )

    metric_ax = plt.subplot(gs[1, :])
    dataset_order = metrics["dataset_label"].drop_duplicates().tolist()
    paired = paired.set_index("dataset_label").loc[
        [dataset for dataset in dataset_order if dataset in set(paired["dataset_label"])]
    ]
    y = np.arange(len(paired))
    official = paired["dinov3_hplus_official"].to_numpy(dtype=float)
    biodino = paired["biodino_hplus"].to_numpy(dtype=float)
    metric_ax.hlines(y, official, biodino, color="#b5bec1", linewidth=2.1, zorder=1)
    metric_ax.scatter(official, y, s=42, color="#345c8c", edgecolor="white", linewidth=0.7, zorder=2)
    metric_ax.scatter(biodino, y, s=48, color="#ef3b33", edgecolor="white", linewidth=0.7, zorder=3)
    for row, (base, bio) in enumerate(zip(official, biodino)):
        metric_ax.text(
            max(base, bio) + 0.012,
            row,
            f"{(bio - base) * 100:+.1f}",
            va="center",
            ha="left",
            fontsize=6.5,
            fontweight="bold",
            color="#b82d28",
        )
    low = max(0.0, float(min(official.min(), biodino.min())) - 0.06)
    high = min(1.0, float(max(official.max(), biodino.max())) + 0.10)
    metric_ax.set_xlim(low, high)
    metric_ax.set_yticks(y)
    metric_ax.set_yticklabels(paired.index, fontsize=7.2)
    metric_ax.invert_yaxis()
    metric_ax.set_xlabel("Test mDice", fontsize=7.7)
    metric_ax.set_title(
        "Frozen final-layer patch tokens + 1x1 linear head (20 epochs)",
        loc="left",
        fontsize=8.2,
        fontweight="bold",
        pad=5,
    )
    metric_ax.grid(axis="x", color="#e2e6e7", linewidth=0.8)
    metric_ax.grid(axis="y", visible=False)
    metric_ax.spines["top"].set_visible(False)
    metric_ax.spines["right"].set_visible(False)
    metric_ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color="#345c8c", label="Original DINOv3 H+", markersize=5.5),
            Line2D([0], [0], marker="o", linestyle="", color="#ef3b33", label="BioDINO H+", markersize=5.5),
            Patch(facecolor="#f07c21", label="false positive"),
            Patch(facecolor="#cf296e", label="false negative"),
        ],
        ncol=4,
        frameon=False,
        fontsize=6.4,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.005),
        handletextpad=0.35,
        columnspacing=0.8,
    )
    return {
        "status": "plotted",
        "task": "cell and nucleus semantic segmentation",
        "qualitative_assets": [str(path) for path, _ in qualitative],
        "metrics_csv": str(metrics_path),
        "metadata_json": str(metadata_path),
        "paired_datasets": paired.index.tolist(),
        "protocol": metadata["protocol"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--alpha-json", type=Path, default=DEFAULT_ALPHA_JSON)
    parser.add_argument("--panel-a-csv", type=Path, default=None)
    parser.add_argument("--panel-c-csv", type=Path, default=None)
    parser.add_argument("--fewshot-csv", type=Path, default=None)
    parser.add_argument("--prefix", default="fig3_complete")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    metrics_path = out_dir / "metrics.json"
    umap_path = args.panel_a_csv if args.panel_a_csv is not None else out_dir / "panel_a_umap.csv"
    pair_path = out_dir / "panel_b_pair_scores.csv"
    for path in (metrics_path, umap_path, pair_path, args.alpha_json):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
    metrics = load_json(metrics_path)
    alpha = load_json(args.alpha_json)
    umap = pd.read_csv(umap_path)
    pair_scores = pd.read_csv(pair_path)
    model_alignment_path = out_dir / "panel_b_model_alignment.csv"
    model_alignment = pd.read_csv(model_alignment_path) if model_alignment_path.exists() else None
    panel_c_path = args.panel_c_csv if args.panel_c_csv is not None else out_dir / "panel_c_foundation_matrix.csv"
    if not panel_c_path.exists():
        raise FileNotFoundError(
            f"Missing Panel C cross-model matrix: {panel_c_path}. "
            "Run scripts/build_fig3_panel_c_foundation_matrix.py first."
        )
    panel_c_matrix = pd.read_csv(panel_c_path)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#f8f4eb",
            "figure.facecolor": "#f8f4eb",
            "axes.edgecolor": "#27343a",
            "axes.labelcolor": "#27343a",
            "xtick.color": "#27343a",
            "ytick.color": "#27343a",
        }
    )
    fig = plt.figure(figsize=(13.6, 11.4), dpi=220)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.82, 0.82], wspace=0.45, hspace=0.42)
    scatter_umap(
        fig.add_subplot(gs[0, 0]),
        umap,
        "modality",
        "A  Biological embedding landscape | modality",
        max_labels=9,
    )
    scatter_umap(
        fig.add_subplot(gs[0, 1]),
        umap,
        "biology_id",
        "A  Same embedding | biology identity",
        max_labels=10,
    )
    plot_panel_b(fig.add_subplot(gs[1, 0]), pair_scores, metrics, model_alignment=model_alignment)
    panel_c_status = plot_panel_c(fig.add_subplot(gs[1, 1]), panel_c_matrix)
    fewshot_path = args.fewshot_csv if args.fewshot_csv is not None else out_dir / "panel_d_fewshot_summary.csv"
    fewshot_status = plot_panel_d(fig.add_subplot(gs[2, 0]), fewshot_path)
    spatial_status = plot_panel_e(gs[2, 1], out_dir)

    fig.suptitle(
        "Fig. 3 - Biodino learns a universal biological representation",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
        color="#1d2b30",
    )
    panel_d_note = (
        "Panel D uses fixed-test-fold, frozen-feature linear probes."
        if fewshot_status["status"] == "plotted"
        else "Panel D is a protocol placeholder until few-shot results are available."
    )
    fig.text(
        0.02,
        0.012,
        "H+ S6 nosigreg alpha=1.0, checkpoint 100. Panel C ranks 16 encoders within each shared dataset; "
        "Panel A/B use deterministic held-out masks; " + panel_d_note,
        fontsize=7.6,
        color="#4a4a4a",
    )
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"{args.prefix}.{ext}", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "figure_prefix": args.prefix,
        "output_dir": str(out_dir),
        "metrics_json": str(metrics_path),
        "alpha_json": str(args.alpha_json),
        "fewshot": fewshot_status,
        "spatial": spatial_status,
        "panel_a_csv": str(umap_path),
        "panel_a_plotted_n": int(len(umap)),
        "panel_a_full_n": int(metrics["panel_a"]["n"]),
        "panel_b_n_pairs": int(metrics["panel_b"]["n_pairs"]),
        "panel_b_alignment": metrics["panel_b"]["biological_alignment_score"],
        "panel_b_model_alignment": str(model_alignment_path) if model_alignment_path.exists() else None,
        "panel_c_cross_model_matrix": str(panel_c_path),
        "panel_c": panel_c_status,
    }
    (out_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[fig3-complete] wrote {out_dir / (args.prefix + '.png')}", flush=True)


if __name__ == "__main__":
    main()
