#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ppt_friendly_svg import configure_matplotlib, sanitize_svg_file

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plot/fig2/data/bloodmnist_k10_l_splus_epochs_20260910.csv"
OUT = ROOT / "plot/fig2/scalingvit_fig2match"
COLORS = {"S+": "#465ECF", "L": "#E97A5F"}
INK, GRID = "#2C3338", "#E4E8EB"


def load():
    grouped = defaultdict(list)
    with DATA.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["model"], row["series"], int(row["epoch"]))].append(
                100.0 * float(row["balanced_accuracy"])
            )
    return {
        key: (float(np.mean(values)), float(np.std(values, ddof=1)))
        for key, values in grouped.items()
    }


def main() -> None:
    configure_matplotlib()
    points = load()
    fig, ax = plt.subplots(figsize=(4.35, 3.35))
    for model in ("S+", "L"):
        epochs = sorted(e for m, series, e in points if m == model and series == "current")
        means = [points[(model, "current", e)][0] for e in epochs]
        stds = [points[(model, "current", e)][1] for e in epochs]
        ax.plot(epochs, means, color=COLORS[model], lw=2.0, marker="o", ms=5.5, label=model, zorder=3)
        ax.errorbar(epochs, means, yerr=stds, fmt="none", ecolor=COLORS[model], elinewidth=1.0,
                    capsize=2.5, alpha=0.8, zorder=2)

    old_mean, old_std = points[("L", "old_e4", 4)]
    ax.errorbar([4], [old_mean], yerr=[old_std], fmt="o", ms=6, mfc="white",
                mec=COLORS["L"], mew=1.4, ecolor=COLORS["L"], capsize=2.5,
                alpha=0.8, zorder=4)
    ax.annotate("old L e4", (4, old_mean), xytext=(5, 8), textcoords="offset points",
                fontsize=7.5, color=COLORS["L"])

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], labels=["1", "2", "4", "8"])
    ax.set_xlim(0.82, 9.3)
    ax.set_ylim(60, 82)
    ax.set_xlabel("Training epochs", color=INK)
    ax.set_ylabel("10-shot balanced accuracy (%)", color=INK)
    ax.set_title("BloodMNIST · 1M images", loc="left", fontsize=11, color=INK, pad=8)
    ax.text(8, 61.0, "L e8\npending", ha="center", va="bottom", fontsize=7.5,
            color=COLORS["L"])
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.tick_params(colors=INK, labelsize=8.5)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5, ncol=2)
    fig.tight_layout(pad=1.0)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "bloodmnist_k10_l_splus_epochs_20260910"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    sanitize_svg_file(stem.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
