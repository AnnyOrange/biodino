#!/usr/bin/env python3
"""Build and plot the fair uint8-vs-16-bit classification comparison.

The output directory name still contains "fairB" for backward compatibility with
existing links, but this script now uses the corrected sklearn reruns for both
ViT-B and ViT-L 16-bit references.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = Path("/mnt/huawei_deepcad/benchmark_model/benchmark_runs")
# Historical path kept so previous report links now resolve to corrected data.
OUT = ROOT / "outputs" / "uint8_vs_16bit_sklearn_fairB_20260602"

PRECISIONS = ("16-bit", "uint8")
MODELS = ("B", "L")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Acc.",
    "macro_f1": "Macro-F1",
}
DATASET_ORDER = ("bloodmnist", "bbbc048", "cyclops", "midog25")
DATASET_TO_BENCH = {
    "bloodmnist": "bloodmnist",
    "bbbc048": "bbbc048-cellcycle",
    "cyclops": "cyclops-protein-loc",
    "midog25": "midog25-atypical",
}
BENCH_TO_LABEL = {v: k for k, v in DATASET_TO_BENCH.items()}
BENCH_DATASET_ORDER = tuple(DATASET_TO_BENCH[d] for d in DATASET_ORDER)
REQUIRED_CKPTS = (
    1024,
    2049,
    3074,
    4099,
    5124,
    6149,
    7174,
    8199,
    9224,
    10249,
    11274,
    12299,
    13324,
    14349,
    15374,
)
MODEL_LABELS = {"B": "ViT-B", "L": "ViT-L"}
COLORS = {"16-bit": "#1f5f9e", "uint8": "#e8682a"}
SOURCE_ROOTS = {
    ("B", "16-bit"): [BENCH_ROOT / "dinov3_b_ckpts_sklearn_20260602"],
    ("L", "16-bit"): [BENCH_ROOT / "dinov3_l_ckpts_sklearn_20260602"],
    ("B", "uint8"): sorted(BENCH_ROOT.glob("uint8_vitb16_sklearn*")),
    ("L", "uint8"): sorted(BENCH_ROOT.glob("uint8_vitl16_sklearn*")),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str | float | int | None) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def sort_key(row: dict[str, object]) -> tuple[int, int, int, int]:
    return (
        MODELS.index(str(row["model_size"])),
        PRECISIONS.index(str(row["precision"])),
        REQUIRED_CKPTS.index(int(row["ckpt"])),
        BENCH_DATASET_ORDER.index(str(row["dataset"])),
    )


def collect_classification_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Read benchmark summaries and deduplicate repeated uint8 fill/extra runs."""
    rows_by_key: dict[tuple[str, str, int, str], dict[str, object]] = {}
    duplicate_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []

    for model in MODELS:
        for precision in PRECISIONS:
            roots = SOURCE_ROOTS[(model, precision)]
            if not roots:
                continue
            for root in roots:
                for summary in sorted(root.glob("*/summary.csv")):
                    try:
                        ckpt = int(summary.parent.name)
                    except ValueError:
                        continue
                    if ckpt not in REQUIRED_CKPTS:
                        continue
                    with summary.open(newline="") as f:
                        for raw in csv.DictReader(f):
                            if raw.get("task") != "classification":
                                continue
                            dataset = raw.get("dataset", "")
                            if dataset not in BENCH_TO_LABEL:
                                continue
                            item = {
                                "model_size": model,
                                "precision": precision,
                                "ckpt": ckpt,
                                "dataset": dataset,
                                "dataset_label": BENCH_TO_LABEL[dataset],
                                "accuracy": to_float(raw.get("accuracy")) * 100.0,
                                "balanced_accuracy": to_float(raw.get("balanced_accuracy")) * 100.0,
                                "macro_f1": to_float(raw.get("macro_f1")) * 100.0,
                                "source_file": str(summary),
                            }
                            key = (model, precision, ckpt, dataset)
                            prev = rows_by_key.get(key)
                            if prev is None:
                                rows_by_key[key] = item
                                continue
                            duplicate_rows.append(item)
                            if any(abs(float(prev[m]) - float(item[m])) > 1e-9 for m in METRICS):
                                conflict = dict(item)
                                conflict["previous_source_file"] = prev["source_file"]
                                conflict_rows.append(conflict)

    all_rows = sorted(rows_by_key.values(), key=sort_key)
    return all_rows, duplicate_rows, conflict_rows


def expected_missing(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = {
        (str(r["model_size"]), str(r["precision"]), int(r["ckpt"]), str(r["dataset"]))
        for r in rows
    }
    missing: list[dict[str, object]] = []
    for model in MODELS:
        for precision in PRECISIONS:
            for ckpt in REQUIRED_CKPTS:
                for dataset in BENCH_DATASET_ORDER:
                    key = (model, precision, ckpt, dataset)
                    if key not in seen:
                        missing.append(
                            {
                                "model_size": model,
                                "precision": precision,
                                "ckpt": ckpt,
                                "dataset": dataset,
                            }
                        )
    return missing


def classification_best_scores(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_rows: list[dict[str, object]] = []
    for model in MODELS:
        for precision in PRECISIONS:
            for dataset in BENCH_DATASET_ORDER:
                subset = [
                    r
                    for r in rows
                    if r["model_size"] == model
                    and r["precision"] == precision
                    and r["dataset"] == dataset
                ]
                for metric in METRICS:
                    best = max(subset, key=lambda r: (float(r[metric]), int(r["ckpt"])))
                    best_rows.append(
                        {
                            "model_size": model,
                            "precision": precision,
                            "dataset": dataset,
                            "dataset_label": BENCH_TO_LABEL[dataset],
                            "metric": metric,
                            "best_score": best[metric],
                            "best_ckpt": best["ckpt"],
                        }
                    )
    return best_rows


def classification_avg_best_scores(best_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    avg_rows: list[dict[str, object]] = []
    for model in MODELS:
        for metric in METRICS:
            vals = {}
            for precision in PRECISIONS:
                scores = [
                    float(r["best_score"])
                    for r in best_rows
                    if r["model_size"] == model and r["precision"] == precision and r["metric"] == metric
                ]
                vals[precision] = sum(scores) / len(scores)
            avg_rows.append(
                {
                    "model_size": model,
                    "metric": metric,
                    "16-bit": vals["16-bit"],
                    "uint8": vals["uint8"],
                    "uint8_minus_16bit": vals["uint8"] - vals["16-bit"],
                }
            )
    return avg_rows


def best_lookup(best_rows: list[dict[str, object]]) -> dict[tuple[str, str, str, str], tuple[float, int]]:
    out: dict[tuple[str, str, str, str], tuple[float, int]] = {}
    for r in best_rows:
        key = (str(r["model_size"]), str(r["precision"]), str(r["dataset_label"]), str(r["metric"]))
        out[key] = (to_float(r["best_score"]), int(r["best_ckpt"]))
    return out


def plot_accuracy_curves(rows: list[dict[str, object]], model: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharex=True)
    fig.suptitle(
        f"{MODEL_LABELS[model]} fair classification accuracy over checkpoints",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )
    axes = axes.ravel()
    for ax, dataset in zip(axes, DATASET_ORDER):
        for precision in PRECISIONS:
            series = [
                r
                for r in rows
                if r["model_size"] == model
                and r["precision"] == precision
                and r["dataset_label"] == dataset
            ]
            series = sorted(series, key=lambda r: int(r["ckpt"]))
            xs = [int(r["ckpt"]) for r in series]
            ys = [to_float(r["accuracy"]) for r in series]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.2,
                markersize=4.2,
                label=precision,
                color=COLORS[precision],
            )
            if ys:
                best_idx = int(np.nanargmax(ys))
                ax.scatter(
                    [xs[best_idx]],
                    [ys[best_idx]],
                    color=COLORS[precision],
                    marker="*",
                    s=145,
                    zorder=5,
                    edgecolor="white",
                    linewidth=0.8,
                )
        ax.set_title(dataset, fontsize=13, fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=35)
    axes[0].legend(frameon=False, loc="lower right")
    for ax in axes[-2:]:
        ax.set_xlabel("Checkpoint iteration")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUT / f"classification_vit{model.lower()}_accuracy_curves.png"
    savefig(path)
    return path


def plot_best_dataset_bars(best_rows: list[dict[str, object]]) -> Path:
    best = best_lookup(best_rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    fig.suptitle("Best classification accuracy by dataset (fair B+L)", fontsize=18, fontweight="bold")
    width = 0.36
    x = np.arange(len(DATASET_ORDER))
    for ax, model in zip(axes, MODELS):
        vals16 = [best[(model, "16-bit", ds, "accuracy")][0] for ds in DATASET_ORDER]
        vals8 = [best[(model, "uint8", ds, "accuracy")][0] for ds in DATASET_ORDER]
        b1 = ax.bar(x - width / 2, vals16, width, label="16-bit", color=COLORS["16-bit"])
        b2 = ax.bar(x + width / 2, vals8, width, label="uint8", color=COLORS["uint8"])
        for i, (v16, v8) in enumerate(zip(vals16, vals8)):
            delta = v8 - v16
            ax.text(
                x[i],
                max(v16, v8) + 1.0,
                fmt_signed(delta),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#1f1f1f",
            )
        ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
        ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=2)
        ax.set_title(MODEL_LABELS[model], fontsize=14, fontweight="bold")
        ax.set_xticks(x, DATASET_ORDER, rotation=18)
        ax.set_ylim(45, 102)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Best accuracy")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUT / "classification_best_accuracy_by_dataset.png"
    savefig(path)
    return path


def plot_avg_bars(avg_rows: list[dict[str, object]]) -> Path:
    by_key = {(str(r["model_size"]), str(r["metric"])): r for r in avg_rows}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    fig.suptitle("Average best classification scores (fair B+L)", fontsize=18, fontweight="bold")
    width = 0.34
    x = np.arange(len(MODELS))
    for ax, metric in zip(axes, METRICS):
        vals16 = [to_float(by_key[(m, metric)]["16-bit"]) for m in MODELS]
        vals8 = [to_float(by_key[(m, metric)]["uint8"]) for m in MODELS]
        ax.bar(x - width / 2, vals16, width, label="16-bit", color=COLORS["16-bit"])
        ax.bar(x + width / 2, vals8, width, label="uint8", color=COLORS["uint8"])
        for i, (v16, v8) in enumerate(zip(vals16, vals8)):
            ax.text(
                x[i],
                max(v16, v8) + 0.7,
                fmt_signed(v8 - v16),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight="bold")
        ax.set_xticks(x, [MODEL_LABELS[m] for m in MODELS])
        ax.set_ylim(55, 82)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Score")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    path = OUT / "classification_avg_best_scores_bar.png"
    savefig(path)
    return path


def plot_delta_heatmap(best_rows: list[dict[str, object]]) -> Path:
    best = best_lookup(best_rows)
    row_labels = []
    values = []
    for model in MODELS:
        for ds in DATASET_ORDER:
            row_labels.append(f"{MODEL_LABELS[model]} {ds}")
            vals = []
            for metric in METRICS:
                v16 = best[(model, "16-bit", ds, metric)][0]
                v8 = best[(model, "uint8", ds, metric)][0]
                vals.append(v8 - v16)
            values.append(vals)
    arr = np.asarray(values)
    vmax = max(0.75, float(np.nanmax(np.abs(arr))))
    fig, ax = plt.subplots(figsize=(8.8, 6.7))
    im = ax.imshow(arr, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    ax.set_title("uint8 - 16-bit best-score delta (fair B+L)", fontsize=17, fontweight="bold", pad=14)
    ax.set_xticks(np.arange(len(METRICS)), [METRIC_LABELS[m] for m in METRICS])
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            ax.text(j, i, fmt_signed(val), ha="center", va="center", fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Delta score")
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    fig.tight_layout()
    path = OUT / "classification_best_delta_heatmap.png"
    savefig(path)
    return path



def fmt_signed(value: float) -> str:
    if abs(value) < 0.005:
        return "+0.00"
    return f"{value:+.2f}"

def markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_summary(
    avg_rows: list[dict[str, object]],
    best_rows: list[dict[str, object]],
    missing_rows: list[dict[str, object]],
    duplicate_rows: list[dict[str, object]],
    conflict_rows: list[dict[str, object]],
    figure_paths: list[Path],
) -> None:
    avg_table = []
    for model in MODELS:
        for metric in METRICS:
            row = next(r for r in avg_rows if r["model_size"] == model and r["metric"] == metric)
            avg_table.append(
                [
                    MODEL_LABELS[model],
                    metric,
                    f"{float(row['16-bit']):.2f}",
                    f"{float(row['uint8']):.2f}",
                    fmt_signed(float(row['uint8_minus_16bit'])),
                ]
            )

    best = best_lookup(best_rows)
    dataset_table = []
    for model in MODELS:
        for ds in DATASET_ORDER:
            v16, c16 = best[(model, "16-bit", ds, "accuracy")]
            v8, c8 = best[(model, "uint8", ds, "accuracy")]
            dataset_table.append(
                [
                    MODEL_LABELS[model],
                    ds,
                    f"{v16:.2f}",
                    c16,
                    f"{v8:.2f}",
                    c8,
                    fmt_signed(v8 - v16),
                ]
            )

    lines = [
        "# uint8 vs 16-bit sklearn linear probe comparison (fair ViT-B + ViT-L)",
        "",
        "Protocol: `run_dinov3_ckpt_benchmark.py` with sklearn `StandardScaler + LogisticRegression(class_weight=\"balanced\", max_iter=10000)`, train_fraction 0.8, seed 0, n_last_blocks=1, use_avgpool=True, bf16, resize256+centercrop224.",
        "",
        "NOTE: both 16-bit references now use the corrected sklearn reruns: `dinov3_b_ckpts_sklearn_20260602` and `dinov3_l_ckpts_sklearn_20260602`. The old `dinov3_b_ckpts` torch-contaminated rows and the old `dinov3_l_ckpts` bloodmnist outlier are not used.",
        "",
        f"Missing classification rows: `{len(missing_rows)}`",
        f"Duplicate uint8 fill/extra rows ignored after deduplication: `{len(duplicate_rows)}`",
        f"Conflicting duplicate rows: `{len(conflict_rows)}`",
        "",
        "## Average best scores",
        "",
        markdown_table(avg_table, ["model", "metric", "16-bit", "uint8", "uint8 - 16-bit"]),
        "",
        "## Best accuracy by dataset",
        "",
        markdown_table(
            dataset_table,
            ["model", "dataset", "16-bit best", "16-bit ckpt", "uint8 best", "uint8 ckpt", "uint8 - 16-bit"],
        ),
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in figure_paths)
    (OUT / "summary.md").write_text("\n".join(lines).rstrip() + "\n")


def build_tables_and_plots() -> list[Path]:
    rows, duplicate_rows, conflict_rows = collect_classification_rows()
    missing_rows = expected_missing(rows)
    best_rows = classification_best_scores(rows)
    avg_rows = classification_avg_best_scores(best_rows)

    write_csv(
        OUT / "classification_all_rows.csv",
        rows,
        ["model_size", "precision", "ckpt", "dataset", "dataset_label", *METRICS, "source_file"],
    )
    write_csv(
        OUT / "classification_best_scores.csv",
        best_rows,
        ["model_size", "precision", "dataset", "dataset_label", "metric", "best_score", "best_ckpt"],
    )
    write_csv(
        OUT / "classification_avg_best_scores.csv",
        avg_rows,
        ["model_size", "metric", "16-bit", "uint8", "uint8_minus_16bit"],
    )
    write_csv(OUT / "missing_rows.csv", missing_rows, ["model_size", "precision", "ckpt", "dataset"])
    write_csv(
        OUT / "duplicate_rows_ignored.csv",
        duplicate_rows,
        ["model_size", "precision", "ckpt", "dataset", "dataset_label", *METRICS, "source_file"],
    )
    write_csv(
        OUT / "conflicting_duplicate_rows.csv",
        conflict_rows,
        [
            "model_size",
            "precision",
            "ckpt",
            "dataset",
            "dataset_label",
            *METRICS,
            "source_file",
            "previous_source_file",
        ],
    )

    paths = [
        plot_accuracy_curves(rows, "B"),
        plot_accuracy_curves(rows, "L"),
        plot_best_dataset_bars(best_rows),
        plot_avg_bars(avg_rows),
        plot_delta_heatmap(best_rows),
    ]
    write_summary(avg_rows, best_rows, missing_rows, duplicate_rows, conflict_rows, paths)
    return paths


def main() -> None:
    paths = build_tables_and_plots()
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
