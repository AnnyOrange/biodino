#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "uint8_vs_16bit_dense_fair_20260603"
OUT = RUN / "merged"

CKPTS = (1024, 2049, 3074, 4099, 5124, 6149, 7174, 8199, 9224, 10249, 11274, 12299, 13324, 14349, 15374)
SEG_DATASETS = ("bbbc038", "conic", "monuseg", "pannuke", "tissuenet")
PRECISIONS = ("16-bit", "uint8")
MODELS = ("B", "L")
MODEL_LABELS = {"B": "ViT-B", "L": "ViT-L"}
COLORS = {"16-bit": "#1f5f9e", "uint8": "#e8682a"}

SOURCES = {
    ("B", "16-bit"): RUN / "vitb16_16bit" / "eval_full",
    ("B", "uint8"): ROOT / "outputs" / "uint8_vitb16_b1024" / "eval_full",
    ("L", "16-bit"): RUN / "vitl16_16bit" / "eval_full",
    ("L", "uint8"): ROOT / "outputs" / "uint8_vitl16_b1024" / "eval_full",
}

DET_METRICS = ("patch_f1", "patch_accuracy", "patch_precision", "patch_recall")
SEG_METRICS = ("mIoU", "mDice", "AJI", "AP", "AP50", "AP75", "bPQ")
SEG_PRIMARY_METRICS = ("mIoU", "mDice", "AJI", "AP50", "bPQ")
METRIC_LABELS = {
    "patch_f1": "F1",
    "patch_accuracy": "Accuracy",
    "patch_precision": "Precision",
    "patch_recall": "Recall",
    "mIoU": "mIoU",
    "mDice": "mDice",
    "AJI": "AJI",
    "AP": "AP",
    "AP50": "AP50",
    "AP75": "AP75",
    "bPQ": "bPQ",
}


def as_float(value: object) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def fmt(value: float, ndigits: int = 2) -> str:
    if value is None or math.isnan(float(value)):
        return "nan"
    return f"{float(value):.{ndigits}f}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_detection_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (model, precision), base in SOURCES.items():
        for ckpt in CKPTS:
            path = base / "bio_detection" / "livecell" / str(ckpt) / "results_bio_detection.json"
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
            row: dict[str, object] = {
                "task": "detection",
                "model_size": model,
                "model": MODEL_LABELS[model],
                "precision": precision,
                "dataset": "livecell",
                "ckpt": ckpt,
                "json_path": str(path.relative_to(ROOT)),
            }
            for key, value in data.items():
                if key in {"dataset", "checkpoint"}:
                    continue
                row[key] = value
            rows.append(row)
    return sorted(rows, key=lambda r: (str(r["model_size"]), str(r["precision"]), str(r["dataset"]), int(r["ckpt"])))


def read_segmentation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (model, precision), base in SOURCES.items():
        for dataset in SEG_DATASETS:
            for ckpt in CKPTS:
                path = base / "bio_segmentation" / "bio_eval" / dataset / str(ckpt) / "results.json"
                if not path.is_file():
                    continue
                data = json.loads(path.read_text())
                row: dict[str, object] = {
                    "task": "segmentation",
                    "model_size": model,
                    "model": MODEL_LABELS[model],
                    "precision": precision,
                    "dataset": dataset,
                    "ckpt": ckpt,
                    "json_path": str(path.relative_to(ROOT)),
                }
                meta = data.get("_meta", {})
                for key in ("full_train_samples", "used_train_samples", "train_fraction", "seed"):
                    if key in meta:
                        row[key] = meta[key]
                for split in ("val", "test"):
                    for key, value in data.get(split, {}).items():
                        # Store segmentation scores as percentages for readability
                        # and consistency with the detection probe outputs.
                        row[f"{split}_{key}"] = float(value) * 100.0
                rows.append(row)
    return sorted(rows, key=lambda r: (str(r["model_size"]), str(r["precision"]), str(r["dataset"]), int(r["ckpt"])))


def best_scores(
    rows: list[dict[str, object]],
    metrics: tuple[str, ...],
    *,
    prefix: str = "test_",
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        for metric in metrics:
            grouped[(str(row["model_size"]), str(row["precision"]), str(row["dataset"]), metric)].append(row)

    out: list[dict[str, object]] = []
    for (model, precision, dataset, metric), group in sorted(grouped.items()):
        key = f"{prefix}{metric}"
        valid = [row for row in group if key in row and not math.isnan(as_float(row[key]))]
        if not valid:
            continue
        best = max(valid, key=lambda row: as_float(row[key]))
        out.append(
            {
                "model_size": model,
                "model": MODEL_LABELS[model],
                "precision": precision,
                "dataset": dataset,
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "best_score": as_float(best[key]),
                "best_ckpt": int(best["ckpt"]),
                "json_path": best["json_path"],
            }
        )
    return out


def average_best_scores(best_rows: list[dict[str, object]], metrics: tuple[str, ...]) -> list[dict[str, object]]:
    by_key = {(r["model_size"], r["precision"], r["dataset"], r["metric"]): r for r in best_rows}
    out: list[dict[str, object]] = []
    for model in MODELS:
        for metric in metrics:
            row: dict[str, object] = {"model_size": model, "model": MODEL_LABELS[model], "metric": metric, "metric_label": METRIC_LABELS[metric]}
            for precision in PRECISIONS:
                vals = [
                    as_float(by_key[(model, precision, dataset, metric)]["best_score"])
                    for dataset in SEG_DATASETS
                    if (model, precision, dataset, metric) in by_key
                ]
                row[precision] = float(np.mean(vals)) if vals else float("nan")
            row["uint8_minus_16bit"] = as_float(row.get("uint8")) - as_float(row.get("16-bit"))
            out.append(row)
    return out


def detection_delta_rows(best_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(r["model_size"], r["precision"], r["metric"]): r for r in best_rows}
    out: list[dict[str, object]] = []
    for model in MODELS:
        for metric in DET_METRICS:
            r16 = by_key[(model, "16-bit", metric)]
            r8 = by_key[(model, "uint8", metric)]
            out.append(
                {
                    "task": "detection",
                    "model_size": model,
                    "model": MODEL_LABELS[model],
                    "dataset": "livecell",
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "16-bit": r16["best_score"],
                    "16-bit_ckpt": r16["best_ckpt"],
                    "uint8": r8["best_score"],
                    "uint8_ckpt": r8["best_ckpt"],
                    "uint8_minus_16bit": as_float(r8["best_score"]) - as_float(r16["best_score"]),
                }
            )
    return out


def segmentation_delta_rows(best_rows: list[dict[str, object]], metrics: tuple[str, ...]) -> list[dict[str, object]]:
    by_key = {(r["model_size"], r["precision"], r["dataset"], r["metric"]): r for r in best_rows}
    out: list[dict[str, object]] = []
    for model in MODELS:
        for dataset in SEG_DATASETS:
            for metric in metrics:
                r16 = by_key[(model, "16-bit", dataset, metric)]
                r8 = by_key[(model, "uint8", dataset, metric)]
                out.append(
                    {
                        "task": "segmentation",
                        "model_size": model,
                        "model": MODEL_LABELS[model],
                        "dataset": dataset,
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "16-bit": r16["best_score"],
                        "16-bit_ckpt": r16["best_ckpt"],
                        "uint8": r8["best_score"],
                        "uint8_ckpt": r8["best_ckpt"],
                        "uint8_minus_16bit": as_float(r8["best_score"]) - as_float(r16["best_score"]),
                    }
                )
    return out


def savefig(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def plot_detection_f1_curves(rows: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9), sharey=True)
    fig.suptitle("LIVECell detection probe: test patch-F1 over checkpoints", fontsize=17, fontweight="bold")
    for ax, model in zip(axes, MODELS):
        for precision in PRECISIONS:
            series = [r for r in rows if r["model_size"] == model and r["precision"] == precision]
            series = sorted(series, key=lambda r: int(r["ckpt"]))
            xs = [int(r["ckpt"]) for r in series]
            ys = [as_float(r["test_patch_f1"]) for r in series]
            ax.plot(xs, ys, marker="o", linewidth=2.3, markersize=4.2, color=COLORS[precision], label=precision)
            if ys:
                idx = int(np.nanargmax(ys))
                ax.scatter([xs[idx]], [ys[idx]], marker="*", s=145, color=COLORS[precision], edgecolor="white", linewidth=0.8, zorder=5)
        ax.set_title(MODEL_LABELS[model], fontsize=13, fontweight="bold")
        ax.set_xlabel("Checkpoint iteration")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test patch-F1")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return savefig(OUT / "detection_livecell_f1_curves.png")


def plot_detection_best_bars(delta_rows: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)
    fig.suptitle("LIVECell detection: best test scores", fontsize=17, fontweight="bold")
    width = 0.36
    x = np.arange(len(DET_METRICS))
    for ax, model in zip(axes, MODELS):
        subset = [r for r in delta_rows if r["model_size"] == model]
        vals16 = [as_float(r["16-bit"]) for r in subset]
        vals8 = [as_float(r["uint8"]) for r in subset]
        ax.bar(x - width / 2, vals16, width, color=COLORS["16-bit"], label="16-bit")
        ax.bar(x + width / 2, vals8, width, color=COLORS["uint8"], label="uint8")
        for i, (v16, v8) in enumerate(zip(vals16, vals8)):
            ax.text(x[i], max(v16, v8) + 0.35, f"{v8 - v16:+.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(MODEL_LABELS[model], fontsize=13, fontweight="bold")
        ax.set_xticks(x, [METRIC_LABELS[m] for m in DET_METRICS], rotation=18)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Best test score")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return savefig(OUT / "detection_best_scores_bar.png")


def plot_segmentation_miou_curves(rows: list[dict[str, object]], model: str) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2), sharex=True)
    fig.suptitle(f"{MODEL_LABELS[model]} segmentation: test mIoU over checkpoints", fontsize=17, fontweight="bold")
    axes_flat = axes.ravel()
    for ax, dataset in zip(axes_flat, SEG_DATASETS):
        for precision in PRECISIONS:
            series = [r for r in rows if r["model_size"] == model and r["precision"] == precision and r["dataset"] == dataset]
            series = sorted(series, key=lambda r: int(r["ckpt"]))
            xs = [int(r["ckpt"]) for r in series]
            ys = [as_float(r["test_mIoU"]) for r in series]
            ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=3.8, color=COLORS[precision], label=precision)
            if ys:
                idx = int(np.nanargmax(ys))
                ax.scatter([xs[idx]], [ys[idx]], marker="*", s=120, color=COLORS[precision], edgecolor="white", linewidth=0.7, zorder=5)
        ax.set_title(dataset, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes_flat[-1].axis("off")
    axes_flat[0].set_ylabel("Test mIoU")
    axes_flat[3].set_ylabel("Test mIoU")
    axes_flat[0].legend(frameon=False, loc="lower right")
    for ax in axes_flat[3:5]:
        ax.set_xlabel("Checkpoint iteration")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return savefig(OUT / f"segmentation_vit{model.lower()}_miou_curves.png")


def plot_segmentation_best_miou(delta_rows: list[dict[str, object]]) -> Path:
    rows = [r for r in delta_rows if r["metric"] == "mIoU"]
    by_key = {(r["model_size"], r["dataset"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.3), sharey=True)
    fig.suptitle("Segmentation: best test mIoU by dataset", fontsize=17, fontweight="bold")
    width = 0.36
    x = np.arange(len(SEG_DATASETS))
    for ax, model in zip(axes, MODELS):
        vals16 = [as_float(by_key[(model, ds)]["16-bit"]) for ds in SEG_DATASETS]
        vals8 = [as_float(by_key[(model, ds)]["uint8"]) for ds in SEG_DATASETS]
        ax.bar(x - width / 2, vals16, width, color=COLORS["16-bit"], label="16-bit")
        ax.bar(x + width / 2, vals8, width, color=COLORS["uint8"], label="uint8")
        for i, (v16, v8) in enumerate(zip(vals16, vals8)):
            ax.text(x[i], max(v16, v8) + 0.45, f"{v8 - v16:+.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(MODEL_LABELS[model], fontsize=13, fontweight="bold")
        ax.set_xticks(x, SEG_DATASETS, rotation=20)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Best test mIoU")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return savefig(OUT / "segmentation_best_miou_by_dataset.png")


def plot_segmentation_avg_bars(avg_rows: list[dict[str, object]]) -> Path:
    metrics = SEG_PRIMARY_METRICS
    by_key = {(r["model_size"], r["metric"]): r for r in avg_rows}
    fig, axes = plt.subplots(1, len(metrics), figsize=(18.5, 4.8), sharey=False)
    fig.suptitle("Segmentation: average best test scores across 5 datasets", fontsize=17, fontweight="bold")
    width = 0.34
    x = np.arange(len(MODELS))
    for ax, metric in zip(axes, metrics):
        vals16 = [as_float(by_key[(m, metric)]["16-bit"]) for m in MODELS]
        vals8 = [as_float(by_key[(m, metric)]["uint8"]) for m in MODELS]
        ax.bar(x - width / 2, vals16, width, color=COLORS["16-bit"], label="16-bit")
        ax.bar(x + width / 2, vals8, width, color=COLORS["uint8"], label="uint8")
        for i, (v16, v8) in enumerate(zip(vals16, vals8)):
            ax.text(x[i], max(v16, v8) + 0.35, f"{v8 - v16:+.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
        ax.set_xticks(x, [MODEL_LABELS[m] for m in MODELS], rotation=12)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Average best test score")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return savefig(OUT / "segmentation_avg_best_scores_bar.png")


def plot_delta_heatmap(delta_rows: list[dict[str, object]], *, task: str, metrics: tuple[str, ...], path: Path) -> Path:
    if task == "detection":
        row_labels = [f"{MODEL_LABELS[m]} livecell" for m in MODELS]
        rows = []
        for model in MODELS:
            vals = []
            for metric in metrics:
                hit = next(r for r in delta_rows if r["model_size"] == model and r["metric"] == metric)
                vals.append(as_float(hit["uint8_minus_16bit"]))
            rows.append(vals)
    else:
        row_labels = []
        rows = []
        for model in MODELS:
            for dataset in SEG_DATASETS:
                row_labels.append(f"{MODEL_LABELS[model]} {dataset}")
                vals = []
                for metric in metrics:
                    hit = next(r for r in delta_rows if r["model_size"] == model and r["dataset"] == dataset and r["metric"] == metric)
                    vals.append(as_float(hit["uint8_minus_16bit"]))
                rows.append(vals)
    arr = np.asarray(rows, dtype=float)
    vmax = max(0.5, float(np.nanmax(np.abs(arr))))
    fig_h = 3.6 if task == "detection" else 7.4
    fig, ax = plt.subplots(figsize=(1.35 * len(metrics) + 4.3, fig_h))
    im = ax.imshow(arr, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    ax.set_title(f"{task.capitalize()} delta: uint8 - 16-bit", fontsize=16, fontweight="bold", pad=12)
    ax.set_xticks(np.arange(len(metrics)), [METRIC_LABELS[m] for m in metrics])
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:+.2f}", ha="center", va="center", fontsize=9, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Delta score")
    fig.tight_layout()
    return savefig(path)


def plot_primary_summary(det_delta: list[dict[str, object]], seg_avg: list[dict[str, object]]) -> Path:
    det_f1 = {r["model_size"]: r for r in det_delta if r["metric"] == "patch_f1"}
    seg_miou = {r["model_size"]: r for r in seg_avg if r["metric"] == "mIoU"}
    tasks = ("Detection F1", "Seg. avg mIoU")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), sharey=False)
    fig.suptitle("Primary dense-probe comparison", fontsize=17, fontweight="bold")
    width = 0.34
    x = np.arange(len(MODELS))
    for ax, task in zip(axes, tasks):
        if task == "Detection F1":
            vals16 = [as_float(det_f1[m]["16-bit"]) for m in MODELS]
            vals8 = [as_float(det_f1[m]["uint8"]) for m in MODELS]
        else:
            vals16 = [as_float(seg_miou[m]["16-bit"]) for m in MODELS]
            vals8 = [as_float(seg_miou[m]["uint8"]) for m in MODELS]
        ax.bar(x - width / 2, vals16, width, color=COLORS["16-bit"], label="16-bit")
        ax.bar(x + width / 2, vals8, width, color=COLORS["uint8"], label="uint8")
        for i, (v16, v8) in enumerate(zip(vals16, vals8)):
            ax.text(x[i], max(v16, v8) + 0.35, f"{v8 - v16:+.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(task, fontsize=13, fontweight="bold")
        ax.set_xticks(x, [MODEL_LABELS[m] for m in MODELS])
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Best test score")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return savefig(OUT / "dense_primary_summary_bar.png")


def markdown_table(headers: list[str], rows: list[list[object]], *, numeric_from: int = 0) -> str:
    align = ["---"] * len(headers)
    for i in range(numeric_from, len(headers)):
        align[i] = "---:"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def build_summary(
    det_rows: list[dict[str, object]],
    seg_rows: list[dict[str, object]],
    det_delta: list[dict[str, object]],
    seg_delta: list[dict[str, object]],
    seg_avg: list[dict[str, object]],
    figures: list[Path],
) -> None:
    det_expected = len(SOURCES) * len(CKPTS)
    seg_expected = len(SOURCES) * len(CKPTS) * len(SEG_DATASETS)
    det_primary = {r["model_size"]: r for r in det_delta if r["metric"] == "patch_f1"}
    seg_primary = {r["model_size"]: r for r in seg_avg if r["metric"] == "mIoU"}

    lines: list[str] = [
        "# uint8 vs 16-bit dense fair comparison",
        "",
        "Protocol: same DINOv3 repo evaluator for both sides. Detection is the LIVECell center-to-patch frozen-backbone linear probe, not full COCO mAP. Segmentation is cached frozen-feature linear probe with `layer-preset=last1`, `probe_epochs=50`, datasets `bbbc038 conic monuseg pannuke tissuenet`. Segmentation scores below are multiplied by 100.",
        "",
        f"Missing detection rows: `{det_expected - len(det_rows)}`",
        f"Missing segmentation rows: `{seg_expected - len(seg_rows)}`",
        "",
        "## Primary best scores",
        "",
    ]
    primary_rows = []
    for model in MODELS:
        d = det_primary[model]
        s = seg_primary[model]
        primary_rows.append(
            [
                MODEL_LABELS[model],
                fmt(as_float(d["16-bit"])),
                int(d["16-bit_ckpt"]),
                fmt(as_float(d["uint8"])),
                int(d["uint8_ckpt"]),
                fmt(as_float(d["uint8_minus_16bit"])),
                fmt(as_float(s["16-bit"])),
                fmt(as_float(s["uint8"])),
                fmt(as_float(s["uint8_minus_16bit"])),
            ]
        )
    lines.append(
        markdown_table(
            [
                "model",
                "det F1 16-bit",
                "det ckpt",
                "det F1 uint8",
                "det ckpt",
                "det Δ",
                "seg avg mIoU 16-bit",
                "seg avg mIoU uint8",
                "seg Δ",
            ],
            primary_rows,
            numeric_from=1,
        )
    )

    lines.extend(["", "## Detection best LIVECell scores", ""])
    det_table = []
    for row in det_delta:
        if row["metric"] != "patch_f1":
            continue
        det_table.append(
            [
                row["model"],
                row["metric_label"],
                fmt(as_float(row["16-bit"])),
                int(row["16-bit_ckpt"]),
                fmt(as_float(row["uint8"])),
                int(row["uint8_ckpt"]),
                fmt(as_float(row["uint8_minus_16bit"])),
            ]
        )
    lines.append(markdown_table(["model", "metric", "16-bit best", "16-bit ckpt", "uint8 best", "uint8 ckpt", "uint8 - 16-bit"], det_table, numeric_from=2))

    lines.extend(["", "## Segmentation best mIoU by dataset", ""])
    seg_table = []
    for row in seg_delta:
        if row["metric"] != "mIoU":
            continue
        seg_table.append(
            [
                row["model"],
                row["dataset"],
                fmt(as_float(row["16-bit"])),
                int(row["16-bit_ckpt"]),
                fmt(as_float(row["uint8"])),
                int(row["uint8_ckpt"]),
                fmt(as_float(row["uint8_minus_16bit"])),
            ]
        )
    lines.append(markdown_table(["model", "dataset", "16-bit best", "16-bit ckpt", "uint8 best", "uint8 ckpt", "uint8 - 16-bit"], seg_table, numeric_from=2))

    lines.extend(["", "## Segmentation average best scores", ""])
    avg_table = []
    for row in seg_avg:
        if row["metric"] not in SEG_PRIMARY_METRICS:
            continue
        avg_table.append([row["model"], row["metric_label"], fmt(as_float(row["16-bit"])), fmt(as_float(row["uint8"])), fmt(as_float(row["uint8_minus_16bit"]))])
    lines.append(markdown_table(["model", "metric", "16-bit", "uint8", "uint8 - 16-bit"], avg_table, numeric_from=2))

    lines.extend(["", "## Figures", ""])
    for figure in figures:
        lines.append(f"- `{figure.name}`")

    lines.extend(
        [
            "",
            "## CSV files",
            "",
            "- `dense_detection_all_rows.csv`",
            "- `dense_detection_best_scores.csv`",
            "- `dense_detection_delta_scores.csv`",
            "- `dense_segmentation_all_rows.csv`",
            "- `dense_segmentation_best_scores.csv`",
            "- `dense_segmentation_delta_scores.csv`",
            "- `dense_segmentation_avg_best_scores.csv`",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    det_rows = read_detection_rows()
    seg_rows = read_segmentation_rows()
    det_best = best_scores(det_rows, DET_METRICS)
    seg_best = best_scores(seg_rows, SEG_METRICS)
    det_delta = detection_delta_rows(det_best)
    seg_delta = segmentation_delta_rows(seg_best, SEG_METRICS)
    seg_avg = average_best_scores(seg_best, SEG_METRICS)

    write_csv(OUT / "dense_detection_all_rows.csv", det_rows)
    write_csv(OUT / "dense_segmentation_all_rows.csv", seg_rows)
    write_csv(OUT / "dense_detection_best_scores.csv", det_best)
    write_csv(OUT / "dense_segmentation_best_scores.csv", seg_best)
    write_csv(OUT / "dense_detection_delta_scores.csv", det_delta)
    write_csv(OUT / "dense_segmentation_delta_scores.csv", seg_delta)
    write_csv(OUT / "dense_segmentation_avg_best_scores.csv", seg_avg)

    figures = [
        plot_primary_summary(det_delta, seg_avg),
        plot_detection_f1_curves(det_rows),
        plot_detection_best_bars(det_delta),
        plot_delta_heatmap(det_delta, task="detection", metrics=DET_METRICS, path=OUT / "detection_delta_heatmap.png"),
        plot_segmentation_miou_curves(seg_rows, "B"),
        plot_segmentation_miou_curves(seg_rows, "L"),
        plot_segmentation_best_miou(seg_delta),
        plot_segmentation_avg_bars(seg_avg),
        plot_delta_heatmap(seg_delta, task="segmentation", metrics=SEG_PRIMARY_METRICS, path=OUT / "segmentation_delta_heatmap.png"),
    ]
    build_summary(det_rows, seg_rows, det_delta, seg_delta, seg_avg, figures)

    print(OUT / "summary.md")
    for figure in figures:
        print(figure)


if __name__ == "__main__":
    main()
