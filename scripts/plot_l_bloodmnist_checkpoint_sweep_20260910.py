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
DATA = ROOT / "plot/fig2/data/l_bloodmnist_checkpoint_sweep_20260910.csv"
OUT = ROOT / "plot/fig2/scalingvit_fig2match/l_bloodmnist_checkpoint_diagnostic_20260910"
STYLE = {
    "L-original-e4-lr1e4": ("Original e4 run", "#2C3338", "D"),
    "L-retrained-e4-lr5e5": ("Retrained e4 · lr 5e-5", "#35A187", "o"),
    "L-e8-lr1e4": ("Current e8 trajectory · lr 1e-4", "#E97A5F", "o"),
}


def grouped():
    values = defaultdict(list)
    with DATA.open(newline="") as handle:
        for row in csv.DictReader(handle):
            values[(row["run"], int(row["epoch"]), row["metric"])].append(100 * float(row["value"]))
    return values


def main() -> None:
    configure_matplotlib()
    values = grouped()
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.25))
    panels = (("full_macro_f1", "Full linear Macro-F1 (%)"),
              ("k10_balanced_accuracy", "10-shot balanced accuracy (%)"))
    for ax, (metric, ylabel) in zip(axes, panels):
        for run, (label, color, marker) in STYLE.items():
            epochs = sorted(e for r, e, m in values if r == run and m == metric)
            if not epochs:
                continue
            means = [np.mean(values[(run, e, metric)]) for e in epochs]
            stds = [np.std(values[(run, e, metric)], ddof=1) if len(values[(run, e, metric)]) > 1 else 0
                    for e in epochs]
            line = "none" if len(epochs) == 1 else "-"
            ax.errorbar(epochs, means, yerr=stds, color=color, marker=marker, linestyle=line,
                        linewidth=1.8, markersize=5.5, capsize=2.5, label=label, zorder=3)
        ax.set_xlabel("Checkpoint epoch")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 8))
        ax.grid(axis="y", color="#E4E8EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.5, colors="#2C3338")
    axes[0].set_title("BloodMNIST · L checkpoint trajectories", loc="left", fontsize=11)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.57, 1.02),
               ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    sanitize_svg_file(OUT.with_suffix(".svg"))


if __name__ == "__main__":
    main()
