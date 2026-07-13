#!/usr/bin/env python3
"""Generate presentation figures for the BioDINOv3 July 2026 report.

The figures intentionally use task-family labels only and exclude BBBC013 from
the main regression claims.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("outputs/00_reports/20260707_talk_figures_no_bbbc013")

COLORS = {
    "ours": "#1B6F5F",
    "external": "#C95C32",
    "muted": "#9FA7A0",
    "accent": "#E0A72E",
    "blue": "#3B6FA8",
    "purple": "#7C5A9E",
    "bg": "#F7F4ED",
    "grid": "#D8D0C2",
    "text": "#1D2321",
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["bg"],
            "savefig.facecolor": COLORS["bg"],
            "axes.edgecolor": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.75,
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:.3f}", dy: float = 0.006) -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fig_id_task_family() -> None:
    tasks = ["Class.", "Reg.", "Retr.", "Clust.", "Seg."]
    ours = np.array([0.7703, 0.9693, 0.9873, 0.8677, 0.7574])
    ext = np.array([0.7279, 0.9688, 0.9960, 0.9292, 0.6848])
    winners = ["Ours", "Tie", "External", "External", "Ours"]

    x = np.arange(len(tasks))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    b1 = ax.bar(x - w / 2, ours, w, label="BioDINOv3 task-best", color=COLORS["ours"])
    b2 = ax.bar(x + w / 2, ext, w, label="Best external", color=COLORS["external"])
    add_bar_labels(ax, b1)
    add_bar_labels(ax, b2)
    for i, winner in enumerate(winners):
        color = COLORS["ours"] if winner == "Ours" else COLORS["external"] if winner == "External" else COLORS["accent"]
        ax.text(i, 1.045, winner, ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    ax.axhline(0.8704, color=COLORS["ours"], lw=1.8, ls="--", alpha=0.8)
    ax.axhline(0.8613, color=COLORS["external"], lw=1.8, ls="--", alpha=0.8)
    ax.text(4.55, 0.8704, " ours mean 0.870", va="center", fontsize=9, color=COLORS["ours"])
    ax.text(4.55, 0.8613, " external mean 0.861", va="center", fontsize=9, color=COLORS["external"])
    ax.set_title("ID Task-Family Comparison")
    ax.set_ylabel("Score (higher is better)")
    ax.set_xticks(x, tasks)
    ax.set_ylim(0.62, 1.08)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", visible=False)
    savefig(fig, "01_id_task_family_bars")

    write_csv(
        "01_id_task_family_values.csv",
        [
            {"task": t, "ours": o, "best_external": e, "winner": w}
            for t, o, e, w in zip(tasks, ours, ext, winners)
        ],
    )


def fig_ood_internal() -> None:
    rows = [
        ("X-ray comp.", 0.6785, "L 1TB"),
        ("Cryo comp.", 0.3526, "L dualroute"),
        ("Combined", 0.4849, "L 1TB"),
        ("X-ray pair", 0.4508, "L 1TB"),
        ("X-ray dose", 0.9062, "L 1TB"),
        ("Cryo class", 0.1012, "H+ 5TB"),
        ("Cryo clust.", 0.3394, "S+ 1TB"),
        ("Cryo quality", 0.9348, "L gram"),
        ("Cryo retr.", 0.0392, "L 1TB"),
    ]
    labels, scores, models = zip(*rows)
    colors = [
        COLORS["ours"],
        COLORS["purple"],
        COLORS["ours"],
        COLORS["ours"],
        COLORS["ours"],
        COLORS["blue"],
        COLORS["accent"],
        COLORS["purple"],
        COLORS["ours"],
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    y = np.arange(len(labels))
    bars = ax.barh(y, scores, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title("OOD Task-Family Results (BioDINOv3 Internal; External FM Not Run)")
    for yi, bar, model in zip(y, bars, models):
        v = bar.get_width()
        ax.text(v + 0.012, yi, f"{v:.4f}  {model}", va="center", fontsize=9)
    ax.grid(axis="y", visible=False)
    savefig(fig, "02_ood_internal_task_family")

    write_csv(
        "02_ood_internal_values.csv",
        [{"task": a, "score": b, "best_current_model": c} for a, b, c in rows],
    )


def fig_regression_metrics() -> None:
    metrics = ["R2 ↑", "Spearman ↑", "MAE ↓"]
    ours = [0.9693, 0.9838, 3.6720]
    virchow = [0.9688, 0.9844, 3.9584]
    ylims = [(0.966, 0.971), (0.9825, 0.9850), (3.4, 4.15)]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for ax, metric, o, v, ylim in zip(axes, metrics, ours, virchow, ylims):
        bars = ax.bar([0, 1], [o, v], color=[COLORS["ours"], COLORS["external"]], width=0.58)
        ax.set_xticks([0, 1], ["BioDINOv3", "Virchow2"], rotation=12)
        ax.set_title(metric)
        ax.set_ylim(*ylim)
        add_bar_labels(ax, bars, "{:.4f}", dy=(ylim[1] - ylim[0]) * 0.025)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Regression Detail", y=1.04, fontsize=15, fontweight="bold")
    savefig(fig, "03_regression_r2_mae_spearman")

    write_csv(
        "03_regression_values.csv",
        [
            {"model": "BioDINOv3 H+ 5TB", "R2": 0.9693, "Spearman": 0.9838, "MAE": 3.6720},
            {"model": "Virchow2", "R2": 0.9688, "Spearman": 0.9844, "MAE": 3.9584},
        ],
    )


def fig_segmentation() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [0.9, 1.4]})

    ax = axes[0]
    bars = ax.bar([0, 1], [0.7574, 0.6848], color=[COLORS["ours"], COLORS["external"]], width=0.6)
    ax.set_xticks([0, 1], ["BioDINOv3", "Virchow2"])
    ax.set_ylim(0.60, 0.80)
    ax.set_ylabel("Mean mDice")
    ax.set_title("FM Dense-Probe Setting")
    add_bar_labels(ax, bars)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    labels = ["Cellpose", "PanNuke"]
    x = np.arange(len(labels))
    w = 0.35
    biodino = [0.8674, 0.5936]
    cpsam = [0.9183, 0.7943]
    b1 = ax.bar(x - w / 2, biodino, w, label="BioDINOv3 + decoder", color=COLORS["ours"])
    b2 = ax.bar(x + w / 2, cpsam, w, label="CPSAM", color=COLORS["accent"])
    add_bar_labels(ax, b1)
    add_bar_labels(ax, b2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Dice / mDice")
    ax.set_title("Specialized-System Reference")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", visible=False)

    fig.suptitle("Segmentation Comparisons: FM Probe vs Specialized System", y=1.03, fontsize=15, fontweight="bold")
    savefig(fig, "04_segmentation_fm_and_cpsam")

    write_csv(
        "04_segmentation_values.csv",
        [
            {"comparison": "fm_dense_probe", "method": "BioDINOv3", "task": "Segmentation", "score": 0.7574},
            {"comparison": "fm_dense_probe", "method": "Virchow2", "task": "Segmentation", "score": 0.6848},
            {"comparison": "specialized_reference", "method": "BioDINOv3 + decoder", "task": "Cellpose", "score": 0.8674},
            {"comparison": "specialized_reference", "method": "CPSAM", "task": "Cellpose", "score": 0.9183},
            {"comparison": "specialized_reference", "method": "BioDINOv3 + decoder", "task": "PanNuke", "score": 0.5936},
            {"comparison": "specialized_reference", "method": "CPSAM", "task": "PanNuke", "score": 0.7943},
        ],
    )


def fig_method_ablation() -> None:
    methods = ["Original-L", "Robust", "Gram+HiRes", "DualRoute", "RC"]
    tasks = ["Class. BA", "Reg. R2", "Retr. R@1", "Clust. NMI", "Seg. mDice", "OOD comp."]
    values = np.array(
        [
            [0.7703, 0.7690, 0.7627, 0.7517, 0.7318],
            [0.9568, 0.9550, 0.9518, 0.9507, 0.9572],
            [0.9830, 0.9853, 0.9873, 0.9868, 0.9852],
            [0.8396, 0.8538, 0.8456, 0.8374, 0.8549],
            [0.7518, 0.7542, 0.7510, 0.7416, 0.7509],
            [0.4849, 0.4741, 0.4675, 0.4361, 0.4743],
        ]
    )
    delta = values - values[:, [0]]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    im = ax.imshow(delta * 100, cmap="BrBG", vmin=-5, vmax=5, aspect="auto")
    ax.set_xticks(np.arange(len(methods)), methods, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(tasks)), tasks)
    ax.set_title("ViT-L Method Ablation: Delta vs Original-L (pp)")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            text = "0.0" if j == 0 else f"{delta[i, j] * 100:+.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=COLORS["text"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Delta vs Original-L (percentage points)")
    ax.grid(False)
    savefig(fig, "05_method_ablation_delta_heatmap")

    rows = []
    for task, row in zip(tasks, values):
        for method, score in zip(methods, row):
            rows.append({"task": task, "method": method, "score": score, "delta_pp_vs_original": (score - row[0]) * 100})
    write_csv("05_method_ablation_values.csv", rows)


def fig_scaling_1tb() -> None:
    tasks = ["Class.", "Reg.", "Retr.", "Clust.", "Seg."]
    series = {
        "S+": [0.7667, 0.7145, 0.9766, 0.8352, 0.7319],
        "B": [0.7287, 0.7377, 0.9789, 0.8354, 0.7436],
        "L": [0.7703, 0.7600, 0.9830, 0.8396, 0.7518],
        "H+": [0.7362, 0.7507, 0.9836, 0.8604, 0.7574],
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(tasks))
    for name, vals in series.items():
        ax.plot(x, vals, marker="o", lw=2.2, label=name)
    ax.set_xticks(x, tasks)
    ax.set_ylim(0.68, 1.0)
    ax.set_ylabel("Task score")
    ax.set_title("Model-Size Scaling at 1TB (Task-Family Scores)")
    ax.legend(frameon=False, ncol=4, loc="lower right")
    ax.grid(axis="x", visible=False)
    savefig(fig, "06_scaling_model_size_1tb")


def fig_scaling_7b() -> None:
    tasks = ["Class.", "Retr.", "Clust.", "Seg."]
    series = {
        "S+": [0.7000, 0.9766, 0.8292, 0.6714],
        "B": [0.7044, 0.9789, 0.8269, 0.6943],
        "L": [0.7162, 0.9830, 0.8396, 0.6924],
        "H+": [0.7189, 0.9836, 0.8700, 0.7121],
        "7B": [0.7222, 0.9867, 0.8531, 0.7153],
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(tasks))
    for name, vals in series.items():
        ax.plot(x, vals, marker="o", lw=2.2, label=name)
    ax.set_xticks(x, tasks)
    ax.set_ylim(0.64, 1.0)
    ax.set_ylabel("Task score")
    ax.set_title("Model-Size Scaling with 7B Common Subset")
    ax.legend(frameon=False, ncol=5, loc="lower right")
    ax.grid(axis="x", visible=False)
    savefig(fig, "07_scaling_model_size_7b_common")


def fig_scaling_data() -> None:
    metrics = [
        "No-reg overall",
        "Class.",
        "Retr.",
        "Detect.",
        "Seg.",
        "Reg.",
    ]
    data = np.array(
        [
            [0.8644, 0.8562, 0.8552],
            [0.7016, 0.6937, 0.6953],
            [0.9642, 0.9421, 0.9304],
            [0.9084, 0.9061, 0.9092],
            [0.8833, 0.8827, 0.8861],
            [0.9586, 0.9486, 0.9571],
        ]
    )
    labels = ["1TB", "5TB", "10TB"]
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    x = np.arange(len(metrics))
    w = 0.25
    for j, label in enumerate(labels):
        bars = ax.bar(x + (j - 1) * w, data[:, j], w, label=label)
        if j == 0:
            pass
    ax.set_xticks(x, metrics, rotation=15, ha="right")
    ax.set_ylim(0.66, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("ViT-L Data Scaling: More Data Is Not Monotonic")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="x", visible=False)
    for i in range(data.shape[0]):
        best = int(np.argmax(data[i]))
        ax.text(x[i] + (best - 1) * w, data[i, best] + 0.006, "best", ha="center", va="bottom", fontsize=8, fontweight="bold")
    savefig(fig, "08_scaling_data_vitl")

    rows = []
    for metric, vals in zip(metrics, data):
        for label, score in zip(labels, vals):
            rows.append({"metric": metric, "data_scale": label, "score": score})
    write_csv("08_scaling_data_values.csv", rows)


def write_readme() -> None:
    (OUT / "README.md").write_text(
        """# BioDINOv3 talk figures, BBBC013 excluded

Generated for the July 2026 report. Task plots use task-family labels only and
do not list the underlying datasets. BBBC013 is excluded from all main
regression and scaling figures.

## Figures

- `01_id_task_family_bars.png`: ID task-family comparison against best external baselines.
- `02_ood_internal_task_family.png`: OOD results among BioDINOv3/DINOv3-family variants only.
- `03_regression_r2_mae_spearman.png`: regression R2, Spearman, and MAE detail.
- `04_segmentation_fm_and_cpsam.png`: FM dense-probe comparison plus CPSAM system reference.
- `05_method_ablation_delta_heatmap.png`: ViT-L method deltas vs Original-L, excluding BBBC013.
- `06_scaling_model_size_1tb.png`: 1TB model-size scaling.
- `07_scaling_model_size_7b_common.png`: common-subset model-size scaling including 7B.
- `08_scaling_data_vitl.png`: ViT-L data scaling, excluding BBBC013.

## OOD caveat

The OOD figure is an internal BioDINOv3/DINOv3-family comparison. I did not find
external foundation-model xray/cryo OOD results under the same protocol.
""",
        encoding="utf-8",
    )


def main() -> None:
    setup()
    fig_id_task_family()
    fig_ood_internal()
    fig_regression_metrics()
    fig_segmentation()
    fig_method_ablation()
    fig_scaling_1tb()
    fig_scaling_7b()
    fig_scaling_data()
    write_readme()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
