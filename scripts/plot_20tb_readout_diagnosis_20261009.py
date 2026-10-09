#!/usr/bin/env python3
"""Plot exploratory readout diagnostics from the paired CSV artifacts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "outputs/00_reports/20tb_readout_diagnosis_20261009"
EARLY = "5tb_ck29279"
TWENTY = "20tb_ck29279"
MID = "20tb_ck47823"
LATE = "20tb_ck65391"
COLORS = {EARLY: "#338e96", TWENTY: "#D58166", MID: "#338e96", LATE: "#D58166"}


def seed_panel(ax, directory, first, second, title):
    frame = pd.read_csv(ROOT / directory / "benchmark_kmeans_seeds.csv")
    paired = frame.pivot(index="seed", columns="model", values="nmi")
    x = np.arange(2)
    for i, model in enumerate((first, second)):
        values = paired[model].to_numpy() * 100
        ax.scatter(np.full(len(values), x[i]) + np.linspace(-.11, .11, len(values)),
                   values, s=8, alpha=.38, color=COLORS[model], linewidths=0)
        ax.plot(x[i], values.mean(), marker="D", ms=6, color=COLORS[model])
        ax.plot(x[i], values[0], marker="o", ms=9, markerfacecolor="white",
                markeredgewidth=2, color=COLORS[model])
    ax.plot(x, [paired[first].mean() * 100, paired[second].mean() * 100],
            color="#555555", lw=1, alpha=.7)
    ax.set_xticks(x, [first.replace("_", " "), second.replace("_", " ")])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("NCT-1K NMI (%)")
    ax.grid(axis="y", alpha=.2)


def probe_panel(ax, dataset, title):
    frame = pd.read_csv(ROOT / "analysis_5tb_20tb/probe_learning_curves.csv")
    frame = frame[(frame.dataset == dataset) & (frame.representation == "full")]
    for model in (EARLY, TWENTY):
        subset = frame[frame.model == model]
        values = subset.groupby("fraction").balanced_accuracy.agg(["mean", "std"])
        ax.errorbar(values.index * 100, values["mean"] * 100,
                    yerr=values["std"].fillna(0) * 100, marker="o", lw=1.5,
                    color=COLORS[model], label=model.replace("_", " "), capsize=2)
    ax.set(xlabel="Labeled training images used (%)", ylabel="Balanced accuracy (%)", title=title)
    ax.grid(alpha=.2)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    seed_panel(axes[0, 0], "analysis_5tb_20tb", EARLY, TWENTY,
               "Same step, different data mixture")
    seed_panel(axes[0, 1], "analysis_late", MID, LATE,
               "20TB continuation: seed-0 versus seed mean")
    probe_panel(axes[1, 0], "nct-crc-he", "NCT training-pool validation")
    probe_panel(axes[1, 1], "chammi-allen-task2", "Allen Train, FOV-disjoint validation")
    axes[1, 1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("20TB representation diagnosis (exploratory; not official v4)", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"readout_diagnosis.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
