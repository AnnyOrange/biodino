#!/usr/bin/env python3
"""Merge and plot fair uint8 vs 16-bit retrieval/clustering results."""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path("/mnt/huawei_deepcad/benchmark_model/benchmark_runs")
RUN = ROOT / "outputs" / "uint8_vs_16bit_retrieval_fair_20260605"
OUT = RUN / "merged"

MODELS = ("B", "L")
PRECISIONS = ("16-bit", "uint8")
MODEL_LABELS = {"B": "ViT-B", "L": "ViT-L"}
COLORS = {"16-bit": "#1f5f9e", "uint8": "#e8682a"}
CKPTS = (1024, 2049, 3074, 4099, 5124, 6149, 7174, 8199, 9224, 10249, 11274, 12299, 13324, 14349, 15374)
DATASETS = ("lc25000", "nct-crc-he-1k", "crc-val-he-7k")
METRICS = ("recall_at_1", "map_at_10", "mrr", "cluster_accuracy", "ari", "nmi")
PLOT_METRICS = ("recall_at_1", "map_at_10", "nmi")
METRIC_LABELS = {
    "recall_at_1": "Recall@1",
    "recall_at_5": "Recall@5",
    "recall_at_10": "Recall@10",
    "map_at_1": "mAP@1",
    "map_at_5": "mAP@5",
    "map_at_10": "mAP@10",
    "mrr": "MRR",
    "cluster_accuracy": "Cluster Acc.",
    "ari": "ARI",
    "nmi": "NMI",
    "silhouette_cosine": "Silhouette",
}
SOURCES = {
    ("B", "16-bit"): BENCH / "retrieval_clustering_dinov3_b_ckpts",
    ("L", "16-bit"): BENCH / "retrieval_clustering_dinov3_l_ckpts",
    ("B", "uint8"): RUN / "uint8_b",
    ("L", "uint8"): RUN / "uint8_l",
}
PREFIX = {
    ("B", "16-bit"): "dinov3-b",
    ("L", "16-bit"): "dinov3-l",
    ("B", "uint8"): "uint8-b",
    ("L", "uint8"): "uint8-l",
}


def as_float(v: object) -> float:
    if v in (None, ""):
        return float("nan")
    return float(v)


def fmt(v: float) -> str:
    return "nan" if np.isnan(v) else f"{v:.4f}"


def fmt_delta(v: float) -> str:
    if np.isnan(v):
        return "nan"
    if abs(v) < 0.00005:
        return "+0.0000"
    return f"{v:+.4f}"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def collect_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: dict[tuple[str, str, int, str], dict[str, object]] = {}
    duplicate_rows: list[dict[str, object]] = []
    for model in MODELS:
        for precision in PRECISIONS:
            root = SOURCES[(model, precision)]
            prefix = PREFIX[(model, precision)]
            for ckpt in CKPTS:
                summary = root / str(ckpt) / "summary.csv"
                if not summary.exists():
                    continue
                with summary.open(newline="") as f:
                    for raw in csv.DictReader(f):
                        if raw.get("task") != "retrieval_clustering" or raw.get("error"):
                            continue
                        dataset = raw.get("dataset", "")
                        if dataset not in DATASETS:
                            continue
                        expected_model = f"{prefix}-{ckpt}"
                        if raw.get("model") != expected_model:
                            continue
                        item = {
                            "model_size": model,
                            "precision": precision,
                            "ckpt": ckpt,
                            "dataset": dataset,
                            "source_file": str(summary),
                        }
                        for metric in METRICS:
                            item[metric] = as_float(raw.get(metric))
                        key = (model, precision, ckpt, dataset)
                        if key in rows:
                            duplicate_rows.append(item)
                        else:
                            rows[key] = item
    order = {ds: i for i, ds in enumerate(DATASETS)}
    out = sorted(rows.values(), key=lambda r: (MODELS.index(str(r["model_size"])), PRECISIONS.index(str(r["precision"])), int(r["ckpt"]), order[str(r["dataset"])]))
    return out, duplicate_rows


def missing_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = {(r["model_size"], r["precision"], int(r["ckpt"]), r["dataset"]) for r in rows}
    missing = []
    for model in MODELS:
        for precision in PRECISIONS:
            for ckpt in CKPTS:
                for dataset in DATASETS:
                    if (model, precision, ckpt, dataset) not in seen:
                        missing.append({"model_size": model, "precision": precision, "ckpt": ckpt, "dataset": dataset})
    return missing


def best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for model in MODELS:
        for precision in PRECISIONS:
            for dataset in DATASETS:
                subset = [r for r in rows if r["model_size"] == model and r["precision"] == precision and r["dataset"] == dataset]
                if not subset:
                    continue
                for metric in METRICS:
                    best = max(subset, key=lambda r: (as_float(r[metric]), int(r["ckpt"])))
                    out.append({
                        "model_size": model,
                        "precision": precision,
                        "dataset": dataset,
                        "metric": metric,
                        "best_score": best[metric],
                        "best_ckpt": best["ckpt"],
                    })
    return out


def avg_best_rows(best: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for model in MODELS:
        for metric in METRICS:
            vals = {}
            for precision in PRECISIONS:
                xs = [as_float(r["best_score"]) for r in best if r["model_size"] == model and r["precision"] == precision and r["metric"] == metric]
                vals[precision] = float(np.mean(xs)) if xs else float("nan")
            out.append({"model_size": model, "metric": metric, "16-bit": vals["16-bit"], "uint8": vals["uint8"], "uint8_minus_16bit": vals["uint8"] - vals["16-bit"]})
    return out


def lookup(best: list[dict[str, object]]) -> dict[tuple[str, str, str, str], tuple[float, int]]:
    return {(str(r["model_size"]), str(r["precision"]), str(r["dataset"]), str(r["metric"])): (as_float(r["best_score"]), int(r["best_ckpt"])) for r in best}


def savefig(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def plot_avg(avg: list[dict[str, object]]) -> Path:
    by = {(r["model_size"], r["metric"]): r for r in avg}
    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(14, 4.8), sharey=True)
    fig.suptitle("Average best retrieval/clustering scores", fontsize=17, fontweight="bold")
    width = 0.34
    x = np.arange(len(MODELS))
    for ax, metric in zip(axes, PLOT_METRICS):
        vals16 = [as_float(by[(m, metric)]["16-bit"]) for m in MODELS]
        vals8 = [as_float(by[(m, metric)]["uint8"]) for m in MODELS]
        ax.bar(x - width / 2, vals16, width, label="16-bit", color=COLORS["16-bit"])
        ax.bar(x + width / 2, vals8, width, label="uint8", color=COLORS["uint8"])
        for i, (a, b) in enumerate(zip(vals16, vals8)):
            if not np.isnan(a) and not np.isnan(b):
                ax.text(x[i], max(a, b) + 0.02, fmt_delta(b - a), ha="center", fontsize=9, fontweight="bold")
        ax.set_title(METRIC_LABELS[metric], fontweight="bold")
        ax.set_xticks(x, [MODEL_LABELS[m] for m in MODELS])
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("Score")
    axes[-1].legend(frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return savefig(OUT / "retrieval_avg_best_scores_bar.png")


def plot_recall_dataset(best: list[dict[str, object]]) -> Path:
    lk = lookup(best)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    fig.suptitle("Best retrieval Recall@1 by dataset", fontsize=17, fontweight="bold")
    x = np.arange(len(DATASETS))
    width = 0.36
    for ax, model in zip(axes, MODELS):
        vals16 = [lk.get((model, "16-bit", ds, "recall_at_1"), (float("nan"), 0))[0] for ds in DATASETS]
        vals8 = [lk.get((model, "uint8", ds, "recall_at_1"), (float("nan"), 0))[0] for ds in DATASETS]
        ax.bar(x - width / 2, vals16, width, label="16-bit", color=COLORS["16-bit"])
        ax.bar(x + width / 2, vals8, width, label="uint8", color=COLORS["uint8"])
        for i, (a, b) in enumerate(zip(vals16, vals8)):
            if not np.isnan(a) and not np.isnan(b):
                ax.text(x[i], max(a, b) + 0.02, fmt_delta(b - a), ha="center", fontsize=9, fontweight="bold")
        ax.set_title(MODEL_LABELS[model], fontweight="bold")
        ax.set_xticks(x, DATASETS, rotation=18)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("Recall@1")
    axes[-1].legend(frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return savefig(OUT / "retrieval_best_recall_at_1_by_dataset.png")


def plot_delta_heatmap(best: list[dict[str, object]]) -> Path:
    lk = lookup(best)
    labels, values = [], []
    for model in MODELS:
        for ds in DATASETS:
            labels.append(f"{MODEL_LABELS[model]} {ds}")
            row = []
            for metric in PLOT_METRICS:
                a = lk.get((model, "16-bit", ds, metric), (float("nan"), 0))[0]
                b = lk.get((model, "uint8", ds, metric), (float("nan"), 0))[0]
                row.append(b - a)
            values.append(row)
    arr = np.asarray(values, dtype=float)
    finite = np.abs(arr[np.isfinite(arr)])
    vmax = max(0.02, float(finite.max()) if finite.size else 0.02)
    fig, ax = plt.subplots(figsize=(8.8, 5.9))
    im = ax.imshow(arr, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    ax.set_title("uint8 - 16-bit retrieval/clustering delta", fontsize=16, fontweight="bold")
    ax.set_xticks(np.arange(len(PLOT_METRICS)), [METRIC_LABELS[m] for m in PLOT_METRICS])
    ax.set_yticks(np.arange(len(labels)), labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, fmt_delta(float(arr[i, j])), ha="center", va="center", fontsize=9, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Delta")
    fig.tight_layout()
    return savefig(OUT / "retrieval_delta_heatmap.png")


def md_table(rows: list[list[str]], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def write_summary(avg: list[dict[str, object]], best: list[dict[str, object]], missing: list[dict[str, object]], figs: list[Path]) -> None:
    avg_table = []
    for model in MODELS:
        for metric in PLOT_METRICS:
            row = next(r for r in avg if r["model_size"] == model and r["metric"] == metric)
            avg_table.append([MODEL_LABELS[model], METRIC_LABELS[metric], fmt(row["16-bit"]), fmt(row["uint8"]), fmt_delta(row["uint8_minus_16bit"])])
    lk = lookup(best)
    ds_table = []
    for model in MODELS:
        for ds in DATASETS:
            a, ca = lk.get((model, "16-bit", ds, "recall_at_1"), (float("nan"), 0))
            b, cb = lk.get((model, "uint8", ds, "recall_at_1"), (float("nan"), 0))
            ds_table.append([MODEL_LABELS[model], ds, fmt(a), str(ca), fmt(b), str(cb), fmt_delta(b - a)])
    lines = [
        "# uint8 vs 16-bit retrieval/clustering fair comparison",
        "",
        "Protocol: `run_dinov3_retrieval_clustering_benchmark.py`, DINOv3 frozen features, resize256+centercrop224, n_last_blocks=1, avgpool=True, bf16. 16-bit rows come from existing `retrieval_clustering_dinov3_{b,l}_ckpts`; uint8 rows come from `outputs/uint8_vs_16bit_retrieval_fair_20260605`.",
        "",
        f"Missing rows: `{len(missing)}`",
        "",
        "## Average best scores",
        "",
        md_table(avg_table, ["model", "metric", "16-bit", "uint8", "uint8 - 16-bit"]),
        "",
        "## Best Recall@1 by dataset",
        "",
        md_table(ds_table, ["model", "dataset", "16-bit best", "16-bit ckpt", "uint8 best", "uint8 ckpt", "uint8 - 16-bit"]),
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{p.name}`" for p in figs)
    (OUT / "summary.md").write_text("\n".join(lines).rstrip() + "\n")


def main() -> int:
    rows, duplicates = collect_rows()
    missing = missing_rows(rows)
    best = best_rows(rows)
    avg = avg_best_rows(best)
    write_csv(OUT / "retrieval_all_rows.csv", rows, ["model_size", "precision", "ckpt", "dataset", *METRICS, "source_file"])
    write_csv(OUT / "retrieval_missing_rows.csv", missing, ["model_size", "precision", "ckpt", "dataset"])
    write_csv(OUT / "retrieval_duplicate_rows.csv", duplicates, ["model_size", "precision", "ckpt", "dataset", *METRICS, "source_file"])
    write_csv(OUT / "retrieval_best_scores.csv", best, ["model_size", "precision", "dataset", "metric", "best_score", "best_ckpt"])
    write_csv(OUT / "retrieval_avg_best_scores.csv", avg, ["model_size", "metric", "16-bit", "uint8", "uint8_minus_16bit"])
    figs = [plot_avg(avg), plot_recall_dataset(best), plot_delta_heatmap(best)]
    write_summary(avg, best, missing, figs)
    print(f"rows={len(rows)} missing={len(missing)} duplicates={len(duplicates)}")
    for p in figs:
        print(p)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
