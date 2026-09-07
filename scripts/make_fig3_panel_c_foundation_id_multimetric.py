#!/usr/bin/env python3
"""Deprecated Fig. 3C renderer retained only for audit provenance.

This renderer produced an invalid main-result segmentation panel by using a
six-dataset 224 final-layer ablation rather than the project-selected
multi-layer spatial protocol. It is intentionally disabled. Use
``scripts/audit_fig3_data.py`` to inspect separate comparable cohorts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "outputs/04_figures/fig3_representation_20260812"
EXTERNAL_ROOT = REPO_ROOT / "outputs/02_eval_runs/external_fm_fair_protocol_20260721"
GAPFILL_ROOT = REPO_ROOT / "outputs/02_eval_runs/external_fm_hplus_protocol_gapfill_20260811"
DENSE_ROOT = REPO_ROOT / "outputs/02_eval_runs/fig3_hplus_external_dense_protocol_20260813_v2"
HPLUS_DETAILS = FIG_DIR / "sources/hplus_s6_alpha1_full_details.json"
HPLUS_SCALARS = FIG_DIR / "sources/hplus_s6_scalar_metrics_complete.json"

TEXT = "#1d2b30"
GRID = "#e5e8e8"
OURS = "#e33d32"
OFFICIAL = "#315d89"
EXTERNAL_COLORS = [
    "#6db6ad", "#a88ac0", "#8caed0", "#83b67e", "#df9e6d", "#7fbfd0", "#cfba5d",
    "#be8cac", "#af927e", "#86a69f", "#c5af63", "#b796c5", "#91a9c5", "#a7a79d",
]
DISPLAY = {
    "bioclip": "BioCLIP", "conch": "CONCH", "cytoimagenet": "CytoImageNet",
    "cytoself": "CytoSelf", "dinov2": "DINOv2", "gigapath": "GigaPath",
    "hoptimus0": "H-optimus-0", "jump_cp": "JUMP-CP", "mae": "MAE", "pe": "PE",
    "phikon2": "Phikon-v2", "siglip2": "SigLIP2", "uni": "UNI", "virchow2": "Virchow2",
}
MODEL_META = {
    "biodino_hplus": ("BioDINO H+/16", "ours"),
    "dinov3_hplus_official": ("DINOv3 H+/16", "official"),
    **{key: (value, "external") for key, value in DISPLAY.items()},
}
SEG_DATASETS = ["bbbc038", "conic", "livecell", "monuseg", "pannuke", "tissuenet"]
RETRIEVAL_METRICS = ("recall_at_1", "map_at_5", "mrr")
CLUSTERING_METRICS = ("cluster_accuracy", "ari", "nmi")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, Any], field: str, source: Path) -> float:
    try:
        result = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} in {source}") from exc
    if not np.isfinite(result):
        raise ValueError(f"Non-finite {field} in {source}")
    return result


def common_datasets(scores: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    common: set[str] | None = None
    for model, rows in scores.items():
        datasets = set(rows)
        if not datasets:
            raise ValueError(f"No task scores for {model}")
        common = datasets if common is None else common & datasets
    if not common:
        raise ValueError("No shared datasets")
    return sorted(common)


def load_hplus_details() -> tuple[dict[str, dict[str, dict[str, float]]], list[Path]]:
    payload = json.loads(HPLUS_DETAILS.read_text())
    by_label = {item["label"]: item for item in payload}
    entries = {
        "biodino_hplus": by_label["alpha_1.00"],
        "dinov3_hplus_official": by_label["alpha_0.00"],
    }
    tasks: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for model, entry in entries.items():
        for row in entry["dataset_rows"]:
            task = str(row["task"])
            if task in {"classification", "detection"}:
                tasks[task][model][str(row["dataset"])] = float(row["value"])
    return dict(tasks), [HPLUS_DETAILS]


def load_external_classification_detection() -> tuple[dict[str, dict[str, dict[str, float]]], list[Path]]:
    tasks: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    sources: list[Path] = []
    for model in DISPLAY:
        scalar = EXTERNAL_ROOT / "classification" / model / "summary.csv"
        sources.append(scalar)
        for row in read_csv(scalar):
            if row.get("task") in {"classification", "multilabel_classification"}:
                tasks["classification"][model][row["dataset"]] = value(row, "macro_f1", scalar)
        detection = GAPFILL_ROOT / "detection" / model / "results_bio_detection.json"
        sources.append(detection)
        score = value(json.loads(detection.read_text()), "test_patch_f1", detection)
        tasks["detection"][model]["livecell"] = score / 100.0 if score > 1 else score
    return dict(tasks), sources


def load_retrieval_regression() -> tuple[dict[str, dict[str, dict[str, dict[str, float]]]], list[Path]]:
    payload = json.loads(HPLUS_SCALARS.read_text())
    result: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        "retrieval": defaultdict(dict), "clustering": defaultdict(dict), "regression": defaultdict(dict)
    }
    sources: list[Path] = [HPLUS_SCALARS]
    for model, item in payload["models"].items():
        for dataset, row in item["retrieval_clustering"].items():
            result["retrieval"][model][dataset] = {metric: float(row[metric]) for metric in RETRIEVAL_METRICS}
            result["clustering"][model][dataset] = {metric: float(row[metric]) for metric in CLUSTERING_METRICS}
        result["regression"][model]["bbbc005"] = {
            "r2": float(item["regression"]["r2"]), "mae": float(item["regression"]["mae"])
        }
    for model in DISPLAY:
        scalar = EXTERNAL_ROOT / "classification" / model / "summary.csv"
        sources.append(scalar)
        for row in read_csv(scalar):
            if row.get("task") == "regression" and row.get("dataset") == "bbbc005":
                result["regression"][model]["bbbc005"] = {
                    "r2": value(row, "r2", scalar), "mae": value(row, "mae", scalar)
                }
        retrieval = GAPFILL_ROOT / "retrieval_clustering" / model / "summary.csv"
        sources.append(retrieval)
        for row in read_csv(retrieval):
            if row.get("error"):
                raise ValueError(f"Retrieval error for {model}: {row['error']}")
            result["retrieval"][model][row["dataset"]] = {
                metric: value(row, metric, retrieval) for metric in RETRIEVAL_METRICS
            }
            result["clustering"][model][row["dataset"]] = {
                metric: value(row, metric, retrieval) for metric in CLUSTERING_METRICS
            }
    return {key: dict(rows) for key, rows in result.items()}, sources


def load_segmentation() -> tuple[dict[str, dict[str, dict[str, float]]], list[Path]]:
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    sources: list[Path] = []
    for model in MODEL_META:
        root = DENSE_ROOT / model if model in {"biodino_hplus", "dinov3_hplus_official"} else EXTERNAL_ROOT / "segmentation"
        for dataset in SEG_DATASETS:
            if model in {"biodino_hplus", "dinov3_hplus_official"}:
                result = root / "linear_probe" / dataset / model / "results.json"
            else:
                result = root / "linear_probe" / dataset / model / "results.json"
            if not result.is_file():
                raise FileNotFoundError(
                    f"Missing matched 20-epoch segmentation result: {result}. "
                    "Do not generate the Fig. 3C comparison until it exists."
                )
            payload = json.loads(result.read_text())
            scores[model][dataset] = {"mdice": value(payload["test"], "mDice", result)}
            sources.append(result)
    return dict(scores), sources


def merge_scalar_task(
    hplus: dict[str, dict[str, dict[str, float]]],
    external: dict[str, dict[str, dict[str, float]]],
    task: str,
) -> dict[str, dict[str, float]]:
    merged = {**hplus.get(task, {}), **external.get(task, {})}
    missing = set(MODEL_META) - set(merged)
    if missing:
        raise ValueError(f"{task} missing models: {sorted(missing)}")
    return merged


def aggregate_single(scores: dict[str, dict[str, float]], metric: str) -> tuple[pd.DataFrame, list[str]]:
    datasets = common_datasets(scores)
    rows = []
    for model, items in scores.items():
        rows.append({"model_key": model, "score": mean(items[dataset] for dataset in datasets)})
    return pd.DataFrame(rows), datasets


def aggregate_multi(
    scores: dict[str, dict[str, dict[str, float]]], metrics: tuple[str, ...]
) -> tuple[pd.DataFrame, list[str]]:
    datasets = common_datasets({model: {dataset: 0.0 for dataset in values} for model, values in scores.items()})
    rows = []
    for model, items in scores.items():
        row: dict[str, Any] = {"model_key": model}
        for metric in metrics:
            row[metric] = mean(items[dataset][metric] for dataset in datasets)
        rows.append(row)
    return pd.DataFrame(rows), datasets


def colors() -> dict[str, str]:
    result = {"biodino_hplus": OURS, "dinov3_hplus_official": OFFICIAL}
    for color, model in zip(EXTERNAL_COLORS, DISPLAY):
        result[model] = color
    return result


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "figure.facecolor": "white",
        "savefig.facecolor": "white", "axes.facecolor": "white", "axes.labelcolor": TEXT,
        "xtick.color": "#3e4c51", "ytick.color": "#3e4c51", "text.color": TEXT,
    })


def ordered(df: pd.DataFrame, key: str, ascending: bool = False) -> pd.DataFrame:
    return df.assign(
        model=df["model_key"].map(lambda item: MODEL_META[item][0]),
        group=df["model_key"].map(lambda item: MODEL_META[item][1]),
    ).sort_values([key, "model"], ascending=[ascending, True]).reset_index(drop=True)


def set_model_labels(ax: plt.Axes, data: pd.DataFrame, color_map: dict[str, str]) -> None:
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(data["model"], fontsize=6.8)
    for label, (_, row) in zip(ax.get_yticklabels(), data.iterrows()):
        if row["group"] != "external":
            label.set_color(color_map[row["model_key"]])
            label.set_fontweight("bold")


def base_axis(ax: plt.Axes, title: str, n: int, panel: str, *, show_n: bool = True) -> None:
    ax.set_title(f"{panel}  {title}", loc="left", fontsize=10.5, fontweight="bold", pad=5)
    if show_n:
        ax.text(0.99, 1.015, f"shared ID datasets n={n}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.9, color="#536369")
    ax.grid(axis="x", color=GRID, linewidth=0.75)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_bar(ax: plt.Axes, frame: pd.DataFrame, score: str, title: str, xlabel: str, n: int, panel: str, *, lower: float = 0.0) -> None:
    cmap = colors()
    data = ordered(frame, score)
    y = np.arange(len(data))
    ax.barh(y, data[score], color=[cmap[x] for x in data["model_key"]], height=0.72, edgecolor="#657176", linewidth=0.35)
    set_model_labels(ax, data, cmap)
    ax.invert_yaxis()
    high = float(data[score].max())
    ax.set_xlim(lower, max(1.0 if high <= 1 else high * 1.05, high + max((high - lower) * 0.15, 0.02)))
    ax.set_xlabel(xlabel, fontsize=7.6)
    base_axis(ax, title, n, panel)
    for idx, row in data.iterrows():
        if row["group"] != "external":
            ax.text(float(row[score]) + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012, idx, f"{float(row[score]):.3f}", va="center", fontsize=6.4, color=cmap[row["model_key"]], fontweight="bold")


def plot_multi(ax: plt.Axes, frame: pd.DataFrame, primary: str, overlays: tuple[str, ...], title: str, n: int, panel: str, labels: dict[str, str]) -> None:
    cmap = colors()
    data = ordered(frame, primary)
    y = np.arange(len(data))
    ax.barh(y, data[primary], color=[cmap[x] for x in data["model_key"]], height=0.66, alpha=0.92, edgecolor="#657176", linewidth=0.35, label=labels[primary])
    marker_specs = [("o", "#20282b"), ("D", "#7f4f24"), ("^", "#456f77")]
    for metric, (marker, color) in zip(overlays, marker_specs):
        ax.scatter(data[metric], y, marker=marker, s=20, color=color, edgecolors="white", linewidths=0.35, zorder=3, label=labels[metric])
    set_model_labels(ax, data, cmap)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("score", fontsize=7.6)
    base_axis(ax, title, n, panel)
    ax.legend(loc="lower right", fontsize=6.3, frameon=False, ncol=len(overlays) + 1, handlelength=1.0, columnspacing=0.8)
    for idx, row in data.iterrows():
        if row["group"] != "external":
            text = " / ".join(f"{labels[metric]}={float(row[metric]):.3f}" for metric in (primary, *overlays))
            ax.text(0.01, idx, text, va="center", fontsize=5.8, color="white", fontweight="bold")


def plot_regression(ax: plt.Axes, frame: pd.DataFrame, n: int, panel: str) -> None:
    cmap = colors()
    data = ordered(frame, "r2")
    y = np.arange(len(data))
    ax.barh(y, data["r2"], color=[cmap[x] for x in data["model_key"]], height=0.68, edgecolor="#657176", linewidth=0.35)
    ax.scatter(data["r2"], y, s=16, color="#26363d", zorder=3)
    set_model_labels(ax, data, cmap)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("R2 (higher is better)", fontsize=7.6)
    # The BBBC005 title already identifies the sole shared dataset; the top MAE axis needs the space.
    base_axis(ax, "Regression: BBBC005", n, panel, show_n=False)
    right = ax.twiny()
    right.scatter(data["mae"], y, marker="D", s=20, color="#7f4f24", edgecolors="white", linewidths=0.35, zorder=4)
    right.set_xlim(max(0, float(data["mae"].min()) - 0.5), float(data["mae"].max()) + 0.5)
    right.set_xlabel("MAE (lower is better)", fontsize=7.6, color="#7f4f24", labelpad=3)
    right.tick_params(axis="x", colors="#7f4f24", labelsize=6.7)
    right.spines[["bottom", "left", "right"]].set_visible(False)
    for idx, row in data.iterrows():
        if row["group"] != "external":
            ax.text(float(row["r2"]) - 0.005, idx, f"R2={float(row['r2']):.3f}; MAE={float(row['mae']):.2f}", ha="right", va="center", fontsize=5.9, color="white", fontweight="bold")


def save(fig: plt.Figure, output: Path) -> None:
    for extension in ("png", "svg", "pdf"):
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = 300
        fig.savefig(output.with_suffix("." + extension), **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()
    raise RuntimeError(
        "Fig. 3C rendering is frozen: this script mixed incompatible segmentation protocols. "
        "Run `python3 scripts/audit_fig3_data.py` and resolve data_audit/fig3_data_issues.csv first."
    )
    if not DENSE_ROOT.is_dir():
        raise FileNotFoundError(
            f"Missing v2 matched DINOv3 segmentation run: {DENSE_ROOT}. "
            "The prior unnormalized smoke run is deliberately excluded."
        )
    style()

    hplus_scalar, detail_sources = load_hplus_details()
    external_scalar, external_sources = load_external_classification_detection()
    classification = merge_scalar_task(hplus_scalar, external_scalar, "classification")
    detection = merge_scalar_task(hplus_scalar, external_scalar, "detection")
    classification_df, classification_datasets = aggregate_single(classification, "score")
    detection_df, detection_datasets = aggregate_single(detection, "score")

    multi, multi_sources = load_retrieval_regression()
    retrieval_df, retrieval_datasets = aggregate_multi(multi["retrieval"], RETRIEVAL_METRICS)
    clustering_df, clustering_datasets = aggregate_multi(multi["clustering"], CLUSTERING_METRICS)
    regression_df, regression_datasets = aggregate_multi(multi["regression"], ("r2", "mae"))

    segmentation, segmentation_sources = load_segmentation()
    segmentation_scalar = {
        model: {dataset: values["mdice"] for dataset, values in rows.items()}
        for model, rows in segmentation.items()
    }
    segmentation_df, segmentation_datasets = aggregate_single(segmentation_scalar, "score")

    fig = plt.figure(figsize=(16.2, 17.1), dpi=300)
    grid = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.38, top=0.915, bottom=0.06, left=0.14, right=0.985)
    plot_bar(fig.add_subplot(grid[0, 0]), classification_df, "score", "Classification", "macro F1", len(classification_datasets), "a")
    plot_bar(fig.add_subplot(grid[0, 1]), segmentation_df, "score", "Semantic segmentation", "test mDice", len(segmentation_datasets), "b")
    plot_multi(fig.add_subplot(grid[1, 0]), retrieval_df, "map_at_5", ("recall_at_1", "mrr"), "Retrieval", len(retrieval_datasets), "c", {"map_at_5": "mAP@5", "recall_at_1": "Recall@1", "mrr": "MRR"})
    plot_multi(fig.add_subplot(grid[1, 1]), clustering_df, "nmi", ("cluster_accuracy", "ari"), "Clustering", len(clustering_datasets), "d", {"nmi": "NMI", "cluster_accuracy": "Cluster acc.", "ari": "ARI"})
    plot_regression(fig.add_subplot(grid[2, 0]), regression_df, len(regression_datasets), "e")
    plot_bar(fig.add_subplot(grid[2, 1]), detection_df, "score", "Cell detection", "LIVECell test patch F1", len(detection_datasets), "f")
    fig.suptitle("C  A single frozen BioDINO representation supports diverse ID biological tasks", x=0.14, ha="left", y=0.985, fontsize=15, fontweight="bold")
    fig.text(0.14, 0.018, "All evaluations are ID test folds only. Segmentation: matched one-last-layer frozen probe, 224 input, 20 epochs, seed 0. Bars are raw scores; red = BioDINO H+/16, blue = original DINOv3 H+/16.", fontsize=7.2, color="#4e5d62")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save(fig, args.output_dir / "fig3_panel_c_taskwise_foundation_montage_white")

    tables = []
    for task, frame, datasets in [
        ("classification_macro_f1", classification_df, classification_datasets),
        ("segmentation_mdice", segmentation_df, segmentation_datasets),
        ("retrieval", retrieval_df, retrieval_datasets),
        ("clustering", clustering_df, clustering_datasets),
        ("regression", regression_df, regression_datasets),
        ("detection_f1", detection_df, detection_datasets),
    ]:
        table = frame.copy()
        table.insert(0, "task", task)
        table["datasets"] = ";".join(datasets)
        tables.append(table)
    pd.concat(tables, ignore_index=True, sort=False).to_csv(args.output_dir / "fig3_panel_c_foundation_id_scores.csv", index=False)
    sources = [*detail_sources, *external_sources, *multi_sources, *segmentation_sources]
    provenance = {
        "description": "Raw ID-only frozen evaluation for Fig. 3C.",
        "models": {key: {"display": name, "group": group} for key, (name, group) in MODEL_META.items()},
        "tasks": {
            "classification": {"metric": "macro_f1", "datasets": classification_datasets},
            "segmentation": {"metric": "test mDice", "datasets": segmentation_datasets, "protocol": "one last layer; input 224; 20 epochs; seed 0"},
            "retrieval": {"metrics": list(RETRIEVAL_METRICS), "datasets": retrieval_datasets},
            "clustering": {"metrics": list(CLUSTERING_METRICS), "datasets": clustering_datasets},
            "regression": {"metrics": ["r2", "mae"], "datasets": regression_datasets},
            "detection": {"metric": "test_patch_f1", "datasets": detection_datasets},
        },
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in sorted(set(sources))],
    }
    (args.output_dir / "fig3_panel_c_foundation_id_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
