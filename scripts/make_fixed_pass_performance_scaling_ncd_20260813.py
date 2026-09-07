#!/usr/bin/env python3
"""Build task-wise N/C/D performance scaling plots from the fixed-pass matrix."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
SBL_ROOT = ROOT / "outputs/02_eval_runs/SBL_splus_datafp_alpha075_full_20260810"
H_ROOT = ROOT / "outputs/02_eval_runs/Hplus_nosigreg_datafp_full_20260813/H"
OUT = ROOT / "outputs/00_reports/fixed_pass_performance_scaling_ncd_20260813"

MODELS = ("S+", "B", "L", "H+")
PARAMS_M = {"S+": 22.0, "B": 86.0, "L": 300.0, "H+": 840.0}
MODEL_DIR = {"S+": "S", "B": "B", "L": "L"}
COLORS = {"S+": "#117A76", "B": "#D17C19", "L": "#B33A3A", "H+": "#275DAD"}
MARKERS = {8: "o", 15: "s"}
IMAGES = {10: 104_877, 20: 209_754, 50: 524_385, 100: 1_048_771}
H_CKPT = {
    (10, 8): "823", (10, 15): "1544",
    (20, 8): "1639", (20, 15): "3074",
    (50, 8): "4103", (50, 15): "7694",
    (100, 8): "8199", (100, 15): "15374",
}

CLASSIFICATION = {
    "bloodmnist", "pathmnist", "tissuemnist", "breastmnist", "organamnist",
    "organcmnist", "organsmnist", "dermamnist", "octmnist", "pneumoniamnist",
    "retinamnist", "chestmnist", "bbbc048-cellcycle", "cyclops-protein-loc",
    "midog25-atypical", "pcam", "nct-crc-he", "lc25000", "chammi-allen-task1",
    "chammi-allen-task2", "chammi-cp-task1", "chammi-cp-task2", "chammi-cp-task3",
    "chammi-hpa-task1", "chammi-hpa-task2",
}
REGRESSION = {"bbbc005", "bbbc013"}
RETRIEVAL = {"lc25000", "nct-crc-he-100", "nct-crc-he-1k", "crc-val-he-7k"}
SEGMENTATION = {
    "bbbc038", "conic", "monuseg", "pannuke", "tissuenet", "livecell",
    "multimodal_cellseg", "cellpose",
}

AUDIT_FAMILIES = (
    ("classification25_macro_f1", "Classification-25", "Macro F1"),
    ("regression2_spearman", "Regression-2", "Spearman"),
    ("retrieval4_map_at_5", "Retrieval-4", "mAP@5"),
    ("clustering4_nmi", "Clustering-4", "NMI"),
    ("segmentation8_mdice", "Segmentation-8", "mDice"),
    ("detection_livecell_patch_f1", "Detection", "LiveCell patch F1"),
    ("ood_xray2_composite", "OOD X-ray-2", "Mean score"),
)
AUDIT_FAMILY_KEYS = tuple(item[0] for item in AUDIT_FAMILIES)
ID_FAMILIES = (
    ("classification25_macro_f1", "(a) Classification (25 sets)", "macro-F1", "#3E6D9C"),
    ("regression2_spearman", "(b) Regression (2 sets)", r"Spearman $\rho$", "#C6A9CE"),
    ("retrieval4_map_at_5", "(c) Retrieval / clustering (4)", "mAP@5", "#8FB8DC"),
    ("segmentation8_mdice", "(d) Segmentation (8 sets)", "mDice", "#6FBF8B"),
)
ID_KEYS = tuple(item[0] for item in ID_FAMILIES)
PANEL_META = {key: (title, metric, color) for key, title, metric, color in ID_FAMILIES}

PAPER = "white"
INK = "#22272B"
MUTED = "#667078"
GRID = "#D8D1C4"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.1,
    "svg.fonttype": "none",
})


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {label}: {value}")
    return number


def mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot average an empty collection")
    return float(np.mean(values))


def point_roots(model: str, ratio: int, passes: int) -> tuple[Path, Path, str]:
    if model != "H+":
        root = SBL_ROOT / MODEL_DIR[model] / f"random_{ratio}" / f"pass_{passes}"
        return root, root, "75"
    checkpoint = H_CKPT[(ratio, passes)]
    if ratio < 100:
        root = H_ROOT / f"random_{ratio}" / "raw"
        return root, root, checkpoint
    return (
        H_ROOT / "random_100" / "scalar",
        H_ROOT / "random_100" / "dense",
        checkpoint,
    )


def unique_task_results(
    root: Path,
    task_dir: str,
    checkpoint: str,
    expected: set[str],
    metric_keys: tuple[str, ...],
) -> dict[str, tuple[dict, Path]]:
    found: dict[str, tuple[dict, Path]] = {}
    for path in sorted((root / task_dir).glob("**/last_result.json")):
        # Gap-fill jobs store shards as .../<checkpoint>/<dataset>/last_result.json,
        # while the main evaluator stores .../<dataset>/<checkpoint>/last_result.json.
        if checkpoint not in {path.parent.name, path.parents[1].name}:
            continue
        result = read_json(path)
        dataset = str(result.get("dataset", ""))
        if dataset not in expected:
            continue
        for key in metric_keys:
            finite(result[key], f"{task_dir}/{dataset}/{key}")
        if dataset in found:
            old = found[dataset][0]
            if any(not math.isclose(float(old[key]), float(result[key]), abs_tol=1e-12)
                   for key in metric_keys):
                raise ValueError(f"Conflicting duplicate for {task_dir}/{dataset}: {path}")
            continue
        found[dataset] = (result, path)
    missing = sorted(expected - set(found))
    if missing:
        raise ValueError(
            f"Incomplete {task_dir} at {root} checkpoint {checkpoint}; missing={missing}"
        )
    return found


def segmentation_results(root: Path, checkpoint: str) -> dict[str, tuple[dict, Path]]:
    found: dict[str, tuple[dict, Path]] = {}
    for path in sorted((root / "bio_segmentation").glob("**/results.json")):
        if path.parent.name != checkpoint:
            continue
        dataset = path.parents[1].name
        if dataset not in SEGMENTATION:
            continue
        result = read_json(path)
        value = finite(result["test"]["mDice"], f"segmentation/{dataset}/test.mDice")
        if dataset in found:
            old = float(found[dataset][0]["test"]["mDice"])
            if not math.isclose(old, value, abs_tol=1e-12):
                raise ValueError(f"Conflicting duplicate segmentation/{dataset}: {path}")
            continue
        found[dataset] = (result, path)
    missing = sorted(SEGMENTATION - set(found))
    if missing:
        raise ValueError(
            f"Incomplete bio_segmentation at {root} checkpoint {checkpoint}; missing={missing}"
        )
    return found


def detection_result(root: Path, checkpoint: str) -> tuple[dict, Path]:
    paths = [
        path for path in (root / "bio_detection").glob("**/results_bio_detection.json")
        if path.parent.name == checkpoint
    ]
    if len(paths) != 1:
        raise ValueError(f"Expected one detection result at {root}/ck{checkpoint}; got {paths}")
    result = read_json(paths[0])
    finite(result["test_patch_f1"], "detection/test_patch_f1")
    return result, paths[0]


def ood_result(root: Path, checkpoint: str) -> tuple[dict, Path]:
    paths = [
        path for path in (root / "ood").glob("**/last_result.json")
        if path.parent.name == checkpoint
    ]
    if len(paths) != 1:
        raise ValueError(f"Expected one X-ray OOD result at {root}/ck{checkpoint}; got {paths}")
    result = read_json(paths[0])
    for key in ("xray_pair_recall_at_1", "xray_dose_r2"):
        finite(result[key], f"OOD/{key}")
    return result, paths[0]


def load_point(model: str, ratio: int, passes: int) -> tuple[dict, list[dict]]:
    scalar_root, dense_root, checkpoint = point_roots(model, ratio, passes)
    classification = unique_task_results(
        scalar_root, "bio_classification", checkpoint, CLASSIFICATION, ("macro_f1",)
    )
    regression = unique_task_results(
        scalar_root, "bio_regression", checkpoint, REGRESSION, ("spearman",)
    )
    retrieval = unique_task_results(
        scalar_root, "bio_retrieval", checkpoint, RETRIEVAL, ("map_at_5", "nmi")
    )
    segmentation = segmentation_results(dense_root, checkpoint)
    detection, detection_path = detection_result(dense_root, checkpoint)
    ood, ood_path = ood_result(dense_root, checkpoint)

    detection_f1 = float(detection["test_patch_f1"])
    if detection_f1 > 1.0:
        detection_f1 /= 100.0
    metrics = {
        "classification25_macro_f1": mean(v[0]["macro_f1"] for v in classification.values()),
        "regression2_spearman": mean(v[0]["spearman"] for v in regression.values()),
        "retrieval4_map_at_5": mean(v[0]["map_at_5"] for v in retrieval.values()),
        "clustering4_nmi": mean(v[0]["nmi"] for v in retrieval.values()),
        "segmentation8_mdice": mean(v[0]["test"]["mDice"] for v in segmentation.values()),
        "detection_livecell_patch_f1": detection_f1,
        "ood_xray2_composite": mean(
            (ood["xray_pair_recall_at_1"], ood["xray_dose_r2"])
        ),
    }
    row = {
        "model": model,
        "parameters_millions": PARAMS_M[model],
        "data_ratio_percent": ratio,
        "unique_images": IMAGES[ratio],
        "passes": passes,
        "image_visits": IMAGES[ratio] * passes,
        "compute_proxy_param_images": PARAMS_M[model] * 1e6 * IMAGES[ratio] * passes,
        "checkpoint": checkpoint,
        "sigreg": 0.0 if model == "H+" else 0.05,
        "alpha": 1.0 if model == "H+" else 0.75,
        **metrics,
        "id4_overall": mean(metrics[key] for key in ID_KEYS),
        "family6_id_equal_mean": mean(metrics[key] for key in AUDIT_FAMILY_KEYS[:-1]),
        "family7_equal_mean": mean(metrics[key] for key in AUDIT_FAMILY_KEYS),
        "scalar_root": str(scalar_root),
        "dense_root": str(dense_root),
    }

    details: list[dict] = []
    collections = (
        ("classification", classification, ("macro_f1",)),
        ("regression", regression, ("spearman",)),
        ("retrieval_clustering", retrieval, ("map_at_5", "nmi")),
        ("segmentation", segmentation, ("test.mDice",)),
    )
    for family, values, metric_names in collections:
        for dataset, (result, path) in sorted(values.items()):
            for metric_name in metric_names:
                value = result["test"]["mDice"] if metric_name == "test.mDice" else result[metric_name]
                details.append({
                    "model": model, "data_ratio_percent": ratio, "passes": passes,
                    "checkpoint": checkpoint, "family": family, "dataset": dataset,
                    "metric": metric_name, "value": float(value), "source": str(path),
                })
    details.extend((
        {
            "model": model, "data_ratio_percent": ratio, "passes": passes,
            "checkpoint": checkpoint, "family": "detection", "dataset": "livecell",
            "metric": "test_patch_f1", "value": detection_f1, "source": str(detection_path),
        },
        {
            "model": model, "data_ratio_percent": ratio, "passes": passes,
            "checkpoint": checkpoint, "family": "ood", "dataset": "xray",
            "metric": "xray_pair_recall_at_1", "value": float(ood["xray_pair_recall_at_1"]),
            "source": str(ood_path),
        },
        {
            "model": model, "data_ratio_percent": ratio, "passes": passes,
            "checkpoint": checkpoint, "family": "ood", "dataset": "xray",
            "metric": "xray_dose_r2", "value": float(ood["xray_dose_r2"]),
            "source": str(ood_path),
        },
    ))
    return row, details


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    return float(slope), float(intercept), r2


def bubble_sizes(values, low: float = 110.0, high: float = 520.0) -> np.ndarray:
    values = np.log10(np.asarray(values, dtype=float))
    if np.ptp(values) == 0:
        return np.full_like(values, (low + high) / 2.0)
    return np.interp(values, (values.min(), values.max()), (low, high))


def style_axis(axis, x, labels, xlabel: str) -> None:
    axis.set_facecolor("white")
    axis.set_xscale("log")
    axis.set_xlim(min(x) * 0.70, max(x) * 1.45)
    axis.set_xticks(x, labels)
    axis.grid(True, which="major", color="#DADDE2", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=10, color="#333333", length=0)
    axis.set_xlabel(xlabel, fontsize=11)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#333333")
    axis.plot(1, 0, ">", ms=7, color="#333333", transform=axis.get_yaxis_transform(), clip_on=False)
    axis.plot(0, 1, "^", ms=7, color="#333333", transform=axis.get_xaxis_transform(), clip_on=False)


def set_metric_ylim(axis, values) -> None:
    values = np.asarray(values, dtype=float)
    span = float(np.ptp(values))
    pad = max(span * 0.22, 0.004)
    axis.set_ylim(float(values.min()) - pad, float(values.max()) + pad * 1.65)


def finish_figure(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor=PAPER)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def task_axes(title: str):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), dpi=180)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=15.5, fontweight="bold", y=0.985)
    return fig, axes.ravel()


def plot_n_tasks(rows: list[dict]) -> None:
    selected = [row for row in rows if row["data_ratio_percent"] == 100 and row["passes"] == 15]
    selected.sort(key=lambda row: row["parameters_millions"])
    fig, axes = task_axes(r"BioDINOv3 model scaling $N$ — by in-domain task ($D$=1M, 15 passes)")
    x = np.array([row["parameters_millions"] for row in selected])
    sizes = bubble_sizes(x)
    labels = [f"{row['model']}\n{row['parameters_millions']:.0f}M" for row in selected]
    for axis, (key, title, metric, _) in zip(axes, ID_FAMILIES):
        y = np.array([row[key] for row in selected])
        style_axis(axis, x, labels, "model parameters $N$ (log)")
        set_metric_ylim(axis, y)
        axis.plot(x, y, color="#4A4A4A", linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
        for row, size in zip(selected, sizes):
            axis.scatter(row["parameters_millions"], row[key], s=size, color=COLORS[row["model"]],
                         edgecolor="#2B2B2B", linewidth=1.1, zorder=4)
            axis.annotate(row["model"], (row["parameters_millions"], row[key]),
                          xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9,
                          color=COLORS[row["model"]], fontweight="bold")
        axis.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
        axis.set_ylabel(metric, fontsize=11.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.075, wspace=0.25, hspace=0.36)
    finish_figure(fig, "performance_scaling_n_tasks")


def plot_d_tasks(rows: list[dict]) -> None:
    selected = [row for row in rows if row["passes"] == 15]
    fig, axes = task_axes(r"BioDINOv3 data scaling $D$ — by in-domain task (15 passes)")
    x = np.array([IMAGES[r] for r in (10, 20, 50, 100)], dtype=float)
    labels = ["0.1M", "0.2M", "0.5M", "1M"]
    sizes = bubble_sizes(x, 90, 390)
    for axis, (key, title, metric, _) in zip(axes, ID_FAMILIES):
        all_y = [row[key] for row in selected]
        style_axis(axis, x, labels, "# training images $D$ (log)")
        set_metric_ylim(axis, all_y)
        for model in MODELS:
            points = sorted((row for row in selected if row["model"] == model),
                            key=lambda row: row["unique_images"])
            y = np.array([row[key] for row in points])
            axis.plot(x, y, color=COLORS[model], linewidth=1.35,
                      linestyle=(0, (5, 4)), label=model, zorder=2)
            axis.scatter(x, y, s=sizes, facecolor=COLORS[model], edgecolor="#2B2B2B",
                         linewidth=1.0, zorder=4)
        axis.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
        axis.set_ylabel(metric, fontsize=11.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.105, wspace=0.25, hspace=0.36)
    finish_figure(fig, "performance_scaling_d_tasks")


def plot_c_tasks(rows: list[dict]) -> None:
    fig, axes = task_axes(r"BioDINOv3 compute scaling $C$ — by in-domain task (32 measured points)")
    all_x = np.array([row["compute_proxy_param_images"] for row in rows])
    tick_x = np.array([1e14, 1e15, 1e16])
    tick_labels = [r"$10^{14}$", r"$10^{15}$", r"$10^{16}$"]
    for axis, (key, title, metric, _) in zip(axes, ID_FAMILIES):
        style_axis(axis, tick_x, tick_labels, r"compute proxy $C=N\times D\times$ passes")
        axis.set_xlim(float(all_x.min()) * 0.70, float(all_x.max()) * 1.45)
        set_metric_ylim(axis, [row[key] for row in rows])
        for model in MODELS:
            for ratio in (10, 20, 50, 100):
                pair = sorted(
                    (row for row in rows if row["model"] == model
                     and row["data_ratio_percent"] == ratio),
                    key=lambda row: row["passes"],
                )
                axis.plot([row["compute_proxy_param_images"] for row in pair],
                          [row[key] for row in pair], color=COLORS[model], alpha=0.42,
                          linewidth=1.1, linestyle=(0, (5, 4)))
                for row in pair:
                    axis.scatter(row["compute_proxy_param_images"], row[key],
                                 color=COLORS[model], marker=MARKERS[row["passes"]],
                                 s=float(bubble_sizes([all_x.min(), row["compute_proxy_param_images"], all_x.max()], 85, 330)[1]),
                                 edgecolor="#2B2B2B", linewidth=0.8, zorder=4)
        axis.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
        axis.set_ylabel(metric, fontsize=11.5)
    model_handles = [Line2D([0], [0], color=COLORS[m], marker="o", lw=1.5, label=m) for m in MODELS]
    pass_handles = [
        Line2D([0], [0], color="#555", marker=MARKERS[p], lw=0, label=f"{p} passes")
        for p in (8, 15)
    ]
    fig.legend(handles=model_handles + pass_handles, frameon=False, ncol=6, fontsize=9,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.105, wspace=0.25, hspace=0.36)
    finish_figure(fig, "performance_scaling_c_tasks")


def plot_overall(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.4), dpi=190)
    fig.patch.set_facecolor("white")

    n_rows = sorted(
        (row for row in rows if row["data_ratio_percent"] == 100 and row["passes"] == 15),
        key=lambda row: row["parameters_millions"],
    )
    n_x = np.array([r["parameters_millions"] for r in n_rows])
    n_y = np.array([r["id4_overall"] for r in n_rows])
    style_axis(axes[0], n_x, [r["model"] for r in n_rows], "model parameters $N$ (log)")
    set_metric_ylim(axes[0], n_y)
    axes[0].plot(n_x, n_y, color="#4A4A4A", linewidth=1.4, linestyle=(0, (5, 4)))
    for row, size in zip(n_rows, bubble_sizes(n_x, 100, 390)):
        axes[0].scatter(row["parameters_millions"], row["id4_overall"], s=size,
                        color=COLORS[row["model"]], edgecolor="#2B2B2B", linewidth=1.0, zorder=3)
        axes[0].annotate(row["model"], (row["parameters_millions"], row["id4_overall"]),
                         xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9,
                         color=COLORS[row["model"]], fontweight="bold")
    axes[0].set_title(r"(a) Model scaling $N$" + "\n" + r"$D$=1M, 15 passes", fontweight="bold")

    d_x = np.array([IMAGES[r] for r in (10, 20, 50, 100)], dtype=float)
    style_axis(axes[1], d_x, ["0.1M", "0.2M", "0.5M", "1M"], "# training images $D$ (log)")
    d_selected = [row for row in rows if row["passes"] == 15]
    set_metric_ylim(axes[1], [row["id4_overall"] for row in d_selected])
    for model in MODELS:
        points = sorted((row for row in rows if row["model"] == model and row["passes"] == 15),
                        key=lambda row: row["unique_images"])
        axes[1].plot(d_x, [r["id4_overall"] for r in points], color=COLORS[model],
                     linewidth=1.35, linestyle=(0, (5, 4)), label=model)
        axes[1].scatter(d_x, [r["id4_overall"] for r in points],
                        s=bubble_sizes(d_x, 65, 240), color=COLORS[model],
                        edgecolor="#2B2B2B", linewidth=0.8, zorder=3)
    axes[1].set_title(r"(b) Data scaling $D$" + "\nfixed 15 passes", fontweight="bold")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)

    all_c = np.array([row["compute_proxy_param_images"] for row in rows])
    style_axis(axes[2], np.array([1e14, 1e15, 1e16]),
               [r"$10^{14}$", r"$10^{15}$", r"$10^{16}$"],
               r"compute proxy $C=N\times D\times$ passes")
    axes[2].set_xlim(float(all_c.min()) * 0.70, float(all_c.max()) * 1.45)
    set_metric_ylim(axes[2], [row["id4_overall"] for row in rows])
    for model in MODELS:
        for ratio in (10, 20, 50, 100):
            pair = sorted((row for row in rows if row["model"] == model
                           and row["data_ratio_percent"] == ratio),
                          key=lambda row: row["passes"])
            axes[2].plot([r["compute_proxy_param_images"] for r in pair],
                         [r["id4_overall"] for r in pair], color=COLORS[model],
                         alpha=0.42, linewidth=1.0, linestyle=(0, (5, 4)))
            for row in pair:
                axes[2].scatter(row["compute_proxy_param_images"], row["id4_overall"],
                                color=COLORS[model], marker=MARKERS[row["passes"]], s=55,
                                edgecolor="#2B2B2B", linewidth=0.7)
    axes[2].set_title(r"(c) Compute scaling $C$" + "\n32 measured points", fontweight="bold")
    axes[2].legend(handles=[Line2D([0], [0], color="#555", marker=MARKERS[p], lw=0,
                                   label=f"{p} passes") for p in (8, 15)],
                   frameon=False, fontsize=8)

    axes[0].set_ylabel("ID-4 overall mean")
    fig.suptitle("BioDINOv3 in-domain downstream performance scaling", fontsize=16,
                 fontweight="bold", y=0.985)
    fig.text(0.5, -0.015,
             "ID-4 = equal mean of Classification-25, Regression-2, Retrieval-4, and Segmentation-8. OOD excluded.",
             ha="center", color=MUTED, fontsize=8.5)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.16, wspace=0.27)
    finish_figure(fig, "performance_scaling_ncd_overall")


def fit_rows(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    n_rows = [row for row in rows if row["data_ratio_percent"] == 100 and row["passes"] == 15]
    for key in (*ID_KEYS, "id4_overall"):
        x = np.log10([row["parameters_millions"] for row in n_rows])
        y = np.array([row[key] for row in n_rows])
        slope, intercept, r2 = linear_fit(x, y)
        output.append({"axis": "N", "series": "all_models_D100_pass15", "metric": key,
                       "n_points": len(x), "log10_slope": slope, "intercept": intercept, "r2": r2})
    for model in MODELS:
        d_rows = [row for row in rows if row["model"] == model and row["passes"] == 15]
        for key in (*ID_KEYS, "id4_overall"):
            x = np.log10([row["unique_images"] for row in d_rows])
            y = np.array([row[key] for row in d_rows])
            slope, intercept, r2 = linear_fit(x, y)
            output.append({"axis": "D", "series": f"{model}_pass15", "metric": key,
                           "n_points": len(x), "log10_slope": slope, "intercept": intercept, "r2": r2})
    for model in MODELS:
        for ratio in (10, 20, 50, 100):
            c_rows = [row for row in rows if row["model"] == model and row["data_ratio_percent"] == ratio]
            for key in (*ID_KEYS, "id4_overall"):
                x = np.log10([row["compute_proxy_param_images"] for row in c_rows])
                y = np.array([row[key] for row in c_rows])
                slope, intercept, r2 = linear_fit(x, y)
                output.append({"axis": "C_two_point", "series": f"{model}_D{ratio}", "metric": key,
                               "n_points": len(x), "log10_slope": slope, "intercept": intercept,
                               "r2": r2})
    return output


def write_readme(rows: list[dict]) -> None:
    n_rows = sorted((r for r in rows if r["data_ratio_percent"] == 100 and r["passes"] == 15),
                    key=lambda r: r["parameters_millions"])
    lines = [
        "# Fixed-pass downstream performance scaling: N, C, D",
        "",
        "This report uses the complete 4-model x 4-data-pool x 2-pass matrix (32 points).",
        "Every point passes a strict 25 classification / 2 regression / 4 retrieval-clustering /",
        "8 segmentation / 1 detection / 1 X-ray OOD result audit before plotting.",
        "",
        "## Primary slices",
        "",
        "- N: 100% data (1,048,771 images), 15 passes; S+=22M, B=86M, L=300M, H+=840M.",
        "- D: nested random 10/20/50/100% pools, fixed 15 passes; one curve per model.",
        "- C: all 32 measured points with C = parameters x unique images x passes.",
        "- Overall: equal-weight mean of seven task families, not a dataset-weighted mean.",
        "",
        "## Metrics",
        "",
        "Classification-25 macro F1; Regression-2 Spearman; Retrieval-4 mAP@5;",
        "Clustering-4 NMI; Segmentation-8 test mDice; LiveCell detection patch F1;",
        "and X-ray OOD mean of pair R@1 plus dose R2.",
        "",
        "## N slice (100% data, 15 passes)",
        "",
        "| Model | N (M) | Family-7 score | Classification | Regression | Retrieval | Clustering | Segmentation | Detection | OOD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in n_rows:
        lines.append(
            f"| {row['model']} | {row['parameters_millions']:.0f} | {row['family7_equal_mean']:.6f} | "
            f"{row['classification25_macro_f1']:.6f} | {row['regression2_spearman']:.6f} | "
            f"{row['retrieval4_map_at_5']:.6f} | {row['clustering4_nmi']:.6f} | "
            f"{row['segmentation8_mdice']:.6f} | {row['detection_livecell_patch_f1']:.6f} | "
            f"{row['ood_xray2_composite']:.6f} |"
        )
    d_rows = {
        model: {
            int(row["data_ratio_percent"]): row
            for row in rows if row["model"] == model and row["passes"] == 15
        }
        for model in MODELS
    }
    c_rows = {
        model: {
            int(row["passes"]): row
            for row in rows if row["model"] == model and row["data_ratio_percent"] == 100
        }
        for model in MODELS
    }
    lines.extend((
        "",
        "## D slice: family-7 score (15 passes)",
        "",
        "| Model | 10% / 0.10M | 20% / 0.21M | 50% / 0.52M | 100% / 1.05M | Delta 10->100 |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for model in MODELS:
        values = d_rows[model]
        lines.append(
            f"| {model} | {values[10]['family7_equal_mean']:.6f} | "
            f"{values[20]['family7_equal_mean']:.6f} | {values[50]['family7_equal_mean']:.6f} | "
            f"{values[100]['family7_equal_mean']:.6f} | "
            f"{values[100]['family7_equal_mean'] - values[10]['family7_equal_mean']:+.6f} |"
        )
    lines.extend((
        "",
        "## C slice: 100% data, 8 vs 15 passes",
        "",
        "| Model | 8 passes | 15 passes | Delta 8->15 |",
        "|---|---:|---:|---:|",
    ))
    for model in MODELS:
        values = c_rows[model]
        lines.append(
            f"| {model} | {values[8]['family7_equal_mean']:.6f} | "
            f"{values[15]['family7_equal_mean']:.6f} | "
            f"{values[15]['family7_equal_mean'] - values[8]['family7_equal_mean']:+.6f} |"
        )
    n_scores = [row["family7_equal_mean"] for row in n_rows]
    lines.extend((
        "",
        "## Main observations",
        "",
        f"- N is a narrow, non-monotonic plateau: family-7 spread is only "
        f"{max(n_scores) - min(n_scores):.6f}; B is highest on this slice, not H+.",
        "- D is the clearest scaling direction: S+ rises monotonically; B is nearly flat until 100%; L and H+ peak at 50% then soften slightly at 100%.",
        "- At 100% data, 15 passes beats 8 passes only for B; S+, L, and H+ are already on an 8-pass plateau within a few thousandths.",
        "- Model leadership swaps by task: H+ leads classification/regression/retrieval/detection, B leads clustering, and L leads segmentation.",
        "- Therefore the result supports task-dependent downstream scaling and a stronger D trend, not one universal monotonic N/C law.",
        "",
        "## Interpretation limits",
        "",
        "- These are downstream performance trends, not training-loss scaling laws.",
        "- The D curve is the cleanest controlled comparison because N and pass count stay fixed.",
        "- N is descriptive: S+/B/L use SigReg=0.05 and alpha=0.75, while H+ uses no SigReg and alpha=1.0.",
        "- C is a parameter-image proxy, not measured FLOPs. It mixes N, D, and 8/15-pass allocation.",
        "- C has only two pass values per fixed (N,D); two-point slopes are saved but are not reliable exponents.",
        "- This is seed 0. Individual task curves need not be monotonic even when the balanced aggregate improves.",
        "",
        "## Files",
        "",
        "- `performance_points.csv`: all 32 aggregate points and provenance roots.",
        "- `performance_per_dataset.csv`: every underlying task result and source JSON.",
        "- `performance_scaling_fits.csv`: descriptive log-linear slopes and R2 values.",
        "- `performance_scaling_{n,c,d}_tasks.{png,pdf,svg}`: task-wise panels.",
        "- `performance_scaling_ncd_overall.{png,pdf,svg}`: final three-axis summary.",
        "",
    ))
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    details: list[dict] = []
    for model in MODELS:
        for ratio in (10, 20, 50, 100):
            for passes in (8, 15):
                row, point_details = load_point(model, ratio, passes)
                rows.append(row)
                details.extend(point_details)
    if len(rows) != 32:
        raise ValueError(f"Expected 32 points, got {len(rows)}")
    rows.sort(key=lambda row: (MODELS.index(row["model"]), row["data_ratio_percent"], row["passes"]))
    details.sort(key=lambda row: (MODELS.index(row["model"]), row["data_ratio_percent"],
                                  row["passes"], row["family"], row["dataset"], row["metric"]))
    write_csv(OUT / "performance_points.csv", rows)
    write_csv(OUT / "performance_per_dataset.csv", details)
    write_csv(OUT / "performance_scaling_fits.csv", fit_rows(rows))
    plot_n_tasks(rows)
    plot_d_tasks(rows)
    plot_c_tasks(rows)
    plot_overall(rows)
    write_readme(rows)
    print(f"Wrote {len(rows)} complete points and {len(details)} source metrics to {OUT}")


if __name__ == "__main__":
    main()
