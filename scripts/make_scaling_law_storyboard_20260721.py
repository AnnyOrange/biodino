#!/usr/bin/env python3
"""Create a deck-ready storyboard for the BioDINOv3 scaling-law study.

The figure deliberately separates measured S+ results from the future model x
data grid. It transfers the Chinchilla experimental framing to vision without
claiming that language-model exponents or constants already hold here.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/03_comparisons/splus_random_data_scaling_e15/CANONICAL_verified_table.csv"
OUT = ROOT / "outputs/00_reports/scaling_law_storyboard_20260721"

MODELS = ["S+", "B", "L", "H+", "7B"]
PARAMS_M = {"S+": 22, "B": 86, "L": 300, "H+": 840, "7B": 7000}
DATA_M = [0.1, 0.2, 0.5, 1.0, 5.0, 10.0]
DATA_LABELS = ["0.1M", "0.2M", "0.5M", "1M", "5M", "10M"]

BG = "#F3F0E8"
PANEL = "#FCFBF7"
INK = "#19313A"
MUTED = "#65747A"
GRID = "#DCE1DE"
TEAL = "#0B7A75"
TEAL_DARK = "#075955"
CORAL = "#DF6B4F"
GOLD = "#D6A327"
PLANNED = "#E8E8E2"
WHITE = "#FFFFFF"


def load_splus() -> tuple[float, list[dict[str, float | str]]]:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing verified S+ table: {SOURCE}")

    rows = list(csv.DictReader(SOURCE.open(newline="")))
    by_scale = {row["scale"]: row for row in rows}
    base_score = float(by_scale["base (0)"]["5fam_mean"])
    observed = []
    for label in ["0.1M", "0.2M", "0.5M", "1.0M"]:
        row = by_scale[label]
        observed.append(
            {
                "label": label,
                "images": float(row["images"]),
                "data_m": float(row["images"]) / 1e6,
                "score": float(row["5fam_mean"]),
            }
        )
    return base_score, observed


def rounded_panel(ax) -> None:
    ax.set_facecolor("none")
    card = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=PANEL,
        edgecolor="#D8DBD5",
        linewidth=1.0,
        clip_on=False,
        zorder=-20,
    )
    ax.add_patch(card)


def stage_header(ax, number: str, title: str, subtitle: str) -> None:
    ax.text(
        0.035,
        0.955,
        number,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=CORAL,
        fontsize=10,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    ax.text(
        0.115,
        0.955,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=INK,
        fontsize=14,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    ax.text(
        0.035,
        0.895,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=8.8,
        linespacing=1.25,
        fontfamily="DejaVu Sans",
    )


def pill(ax, x: float, y: float, text: str, color: str, *, width: float = 0.22) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        0.042,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="none",
        clip_on=False,
        zorder=20,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + 0.021,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=WHITE,
        fontsize=7.5,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        zorder=21,
    )


def plot_principle(ax) -> None:
    rounded_panel(ax)
    stage_header(
        ax,
        "01",
        "Compute-optimal principle",
        "Balance model capacity N and training data D\nunder a fixed compute budget C.",
    )

    chart = ax.inset_axes([0.13, 0.19, 0.79, 0.55])
    chart.set_xscale("log")
    chart.set_yscale("log")
    chart.set_xlim(0.07, 17)
    chart.set_ylim(15, 11000)
    chart.set_facecolor("none")

    d = np.logspace(math.log10(0.07), math.log10(17), 300)
    compute_levels = [20, 100, 500, 2500, 12500]
    for i, c in enumerate(compute_levels):
        n = c / d
        chart.plot(d, n, color="#C7CFCC", linewidth=0.85, zorder=1)
        valid = (n > 20) & (n < 9000)
        if valid.any() and i in {1, 3}:
            idx = np.flatnonzero(valid)[-1]
            chart.text(
                d[idx] * 0.92,
                n[idx] * 1.05,
                "fixed C",
                color="#9AA6A4",
                fontsize=6.4,
                rotation=-38,
                ha="right",
                va="bottom",
            )

    candidate_points = [(0.1, "S+"), (0.45, "B"), (1.2, "L"), (3.6, "H+"), (10, "7B")]
    candidate_x = np.array([point[0] for point in candidate_points])
    candidate_y = np.array([PARAMS_M[point[1]] for point in candidate_points])
    frontier_d = np.logspace(-1, 1, 200)
    frontier_n = 10 ** np.interp(np.log10(frontier_d), np.log10(candidate_x), np.log10(candidate_y))
    chart.plot(frontier_d, frontier_n, color=GOLD, linewidth=2.6, zorder=4)
    chart.fill_between(
        frontier_d,
        frontier_n / 1.45,
        frontier_n * 1.45,
        color=GOLD,
        alpha=0.12,
        linewidth=0,
        zorder=2,
    )
    chart.text(
        0.19,
        84,
        "compute-balanced\nfrontier",
        color="#8A6714",
        fontsize=7.2,
        fontweight="bold",
        rotation=31,
        ha="left",
        va="bottom",
    )

    for x, model in candidate_points:
        y = PARAMS_M[model]
        chart.scatter(x, y, s=30, facecolor=PANEL, edgecolor=GOLD, linewidth=1.5, zorder=5)
        chart.text(x * 1.12, y * 0.88, model, color=INK, fontsize=6.6, fontweight="bold")

    chart.set_xticks([0.1, 1, 10], ["0.1M", "1M", "10M"])
    chart.set_yticks([22, 86, 300, 840, 7000], ["22M", "86M", "300M", "840M", "7B"])
    chart.tick_params(axis="both", which="both", colors=MUTED, labelsize=7.2, length=0, pad=3)
    chart.grid(False)
    for spine in chart.spines.values():
        spine.set_visible(False)
    chart.set_xlabel("unique microscopy images  D", fontsize=7.8, color=INK, labelpad=5)
    chart.set_ylabel("model parameters  N", fontsize=7.8, color=INK, labelpad=5)

    ax.text(
        0.50,
        0.105,
        r"$L(N,D)=E+AN^{-\alpha}+BD^{-\beta}$     $C\;\propto\;N\,D_{seen}$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=INK,
        fontsize=10.5,
        fontfamily="STIXGeneral",
    )
    ax.text(
        0.50,
        0.052,
        "Use the framing; re-estimate the exponents for microscopy.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=7.3,
        fontfamily="DejaVu Sans",
    )
    ax.set_axis_off()


def plot_splus(ax, base_score: float, observed: list[dict[str, float | str]]) -> None:
    rounded_panel(ax)
    stage_header(
        ax,
        "02",
        "Current evidence: fixed S+",
        "22M parameters; nested-random microscopy subsets;\n15 passes; identical downstream protocol.",
    )
    pill(ax, 0.755, 0.815, "OBSERVED", TEAL, width=0.20)

    chart = ax.inset_axes([0.14, 0.21, 0.80, 0.55])
    x = np.array([float(r["data_m"]) for r in observed])
    y = np.array([float(r["score"]) for r in observed])
    chart.set_xscale("log")
    chart.set_xlim(0.077, 1.38)
    chart.set_ylim(0.672, 0.718)
    chart.set_facecolor("none")
    chart.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)

    chart.axhline(base_score, color="#9DA9A8", linewidth=1.0, linestyle=(0, (4, 4)), zorder=1)
    chart.text(
        0.082,
        base_score + 0.0006,
        f"official S+ (no microscopy adaptation)  {base_score:.3f}",
        color="#879392",
        fontsize=6.7,
        ha="left",
        va="bottom",
    )
    chart.plot(x, y, color=TEAL_DARK, linewidth=2.2, zorder=3)
    sizes = np.array([88, 125, 190, 270])
    colors = ["#A6D5CA", "#65B7A9", "#299589", TEAL]
    chart.scatter(x, y, s=sizes, color=colors, edgecolor=PANEL, linewidth=1.8, zorder=4)
    value_offsets = [(0, 10), (0, 11), (0, -13), (0, -14)]
    for point_idx, (xx, yy, row) in enumerate(zip(x, y, observed)):
        label = str(row["label"]).replace("1.0M", "1M")
        chart.annotate(
            f"{yy:.3f}",
            xy=(xx, yy),
            xytext=value_offsets[point_idx],
            textcoords="offset points",
            ha="center",
            va="center",
            color=INK,
            fontsize=7.1,
            fontweight="bold",
            zorder=5,
        )
        chart.text(xx, 0.6735, label, ha="center", va="bottom", color=INK, fontsize=7.2, fontweight="bold")

    chart.annotate(
        f"+{y[-1] - base_score:.3f} vs base",
        xy=(x[-1], y[-1]),
        xytext=(0.43, 0.7136),
        color=TEAL_DARK,
        fontsize=8.0,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2, connectionstyle="arc3,rad=-0.12"),
    )
    chart.set_xticks(x, [""] * len(x))
    chart.set_yticks([0.68, 0.69, 0.70, 0.71], [".68", ".69", ".70", ".71"])
    chart.tick_params(axis="both", which="both", colors=MUTED, labelsize=7.4, length=0)
    for spine in chart.spines.values():
        spine.set_visible(False)
    chart.set_xlabel("unique microscopy images  D  (log scale)", fontsize=8.0, color=INK, labelpad=7)
    chart.set_ylabel("diagnostic 5-family mean", fontsize=8.0, color=INK, labelpad=7)

    ax.text(
        0.055,
        0.115,
        "What this already says",
        transform=ax.transAxes,
        color=CORAL,
        fontsize=7.5,
        fontweight="bold",
    )
    ax.text(
        0.055,
        0.075,
        "More unique microscopy data improves the broad aggregate,\nwhile individual task families remain non-monotonic.",
        transform=ax.transAxes,
        color=INK,
        fontsize=8.0,
        linespacing=1.3,
    )
    ax.set_axis_off()


def observed_by_column(observed: list[dict[str, float | str]]) -> dict[int, float]:
    result = {}
    for row in observed:
        data_m = float(row["data_m"])
        col = min(range(len(DATA_M)), key=lambda i: abs(math.log(DATA_M[i]) - math.log(data_m)))
        result[col] = float(row["score"])
    return result


def plot_target_matrix(ax, observed: list[dict[str, float | str]]) -> None:
    rounded_panel(ax)
    stage_header(
        ax,
        "03",
        "Target result: performance map",
        "Jointly vary model size and unique data; then fit the\nperformance surface and locate the efficient frontier.",
    )
    pill(ax, 0.755, 0.915, "MEASURE + FIT", CORAL, width=0.205)

    matrix = ax.inset_axes([0.17, 0.25, 0.78, 0.50])
    matrix.set_xlim(-0.5, len(DATA_M) - 0.5)
    matrix.set_ylim(len(MODELS) - 0.5, -0.5)
    matrix.set_facecolor("none")
    observed_cols = observed_by_column(observed)
    norm = Normalize(vmin=0.675, vmax=0.715)
    cmap = LinearSegmentedColormap.from_list("observed_teal", ["#D8ECE6", "#6FBCAF", TEAL])

    for row_idx, model in enumerate(MODELS):
        for col_idx, _data in enumerate(DATA_M):
            is_observed = model == "S+" and col_idx in observed_cols
            face = cmap(norm(observed_cols[col_idx])) if is_observed else PLANNED
            rect = Rectangle(
                (col_idx - 0.43, row_idx - 0.36),
                0.86,
                0.72,
                facecolor=face,
                edgecolor=PANEL,
                linewidth=2.3,
                zorder=2,
            )
            matrix.add_patch(rect)
            if is_observed:
                matrix.text(
                    col_idx,
                    row_idx,
                    f"{observed_cols[col_idx]:.3f}",
                    ha="center",
                    va="center",
                    color=WHITE if observed_cols[col_idx] > 0.695 else INK,
                    fontsize=7.0,
                    fontweight="bold",
                    zorder=4,
                )
            else:
                matrix.text(col_idx, row_idx, "·", ha="center", va="center", color="#B7BCB8", fontsize=12, zorder=3)

    # A candidate diagonal is a study-design cue, not a fitted conclusion.
    candidate = [(3, 0), (3, 1), (4, 2), (4, 3), (5, 4)]
    for col_idx, row_idx in candidate:
        rect = Rectangle(
            (col_idx - 0.43, row_idx - 0.36),
            0.86,
            0.72,
            fill=False,
            edgecolor=GOLD,
            linewidth=1.9,
            linestyle=(0, (3, 2)),
            zorder=5,
        )
        matrix.add_patch(rect)

    matrix.set_xticks(range(len(DATA_M)), DATA_LABELS)
    matrix.set_yticks(
        range(len(MODELS)),
        [f"{m}   {PARAMS_M[m]}M" if m != "7B" else "7B   7,000M" for m in MODELS],
    )
    matrix.xaxis.tick_top()
    matrix.tick_params(axis="x", colors=INK, labelsize=7.3, length=0, pad=5)
    matrix.tick_params(axis="y", colors=INK, labelsize=7.3, length=0, pad=5)
    for label in matrix.get_yticklabels():
        label.set_horizontalalignment("right")
    for spine in matrix.spines.values():
        spine.set_visible(False)
    matrix.set_xlabel("unique images  D", fontsize=7.8, color=MUTED, labelpad=4)
    matrix.xaxis.set_label_position("top")
    matrix.set_ylabel("model size  N", fontsize=7.8, color=MUTED, labelpad=8)

    ax.scatter([0.075], [0.165], transform=ax.transAxes, s=54, color=TEAL, edgecolor="none", clip_on=False)
    ax.text(0.105, 0.165, "measured S+", transform=ax.transAxes, ha="left", va="center", color=INK, fontsize=7.2)
    ax.scatter([0.335], [0.165], transform=ax.transAxes, s=54, color=PLANNED, edgecolor="#D1D2CC", linewidth=0.7, clip_on=False)
    ax.text(0.365, 0.165, "planned cell", transform=ax.transAxes, ha="left", va="center", color=INK, fontsize=7.2)
    legend_box = Rectangle(
        (0.595, 0.153),
        0.035,
        0.024,
        transform=ax.transAxes,
        fill=False,
        edgecolor=GOLD,
        linewidth=1.5,
        linestyle=(0, (3, 2)),
        clip_on=False,
    )
    ax.add_patch(legend_box)
    ax.text(0.646, 0.165, "candidate compute-balanced path", transform=ax.transAxes, ha="left", va="center", color=INK, fontsize=7.2)

    ax.text(
        0.50,
        0.074,
        "Final color = held-out performance; fit only after recipe and image-visits are controlled.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=7.5,
    )
    ax.set_axis_off()


def connector(fig, left_ax, right_ax) -> None:
    left = left_ax.get_position()
    right = right_ax.get_position()
    arrow = FancyArrowPatch(
        (left.x1 + 0.006, (left.y0 + left.y1) / 2),
        (right.x0 - 0.006, (right.y0 + right.y1) / 2),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.2,
        color="#AAB4B1",
        zorder=30,
    )
    fig.add_artist(arrow)


def write_figure_data(base_score: float, observed: list[dict[str, float | str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "scaling_law_storyboard_data.csv"
    fields = ["status", "model", "parameters_m", "data_m_images", "score", "metric", "source"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "status": "reference",
                "model": "S+",
                "parameters_m": PARAMS_M["S+"],
                "data_m_images": 0,
                "score": base_score,
                "metric": "diagnostic_5_family_mean",
                "source": SOURCE.relative_to(ROOT),
            }
        )
        observed_lookup = {round(float(row["data_m"]), 1): float(row["score"]) for row in observed}
        for model in MODELS:
            for data_m in DATA_M:
                score = observed_lookup.get(data_m) if model == "S+" and data_m <= 1.0 else ""
                writer.writerow(
                    {
                        "status": "observed" if score != "" else "planned",
                        "model": model,
                        "parameters_m": PARAMS_M[model],
                        "data_m_images": data_m,
                        "score": score,
                        "metric": "diagnostic_5_family_mean" if score != "" else "",
                        "source": SOURCE.relative_to(ROOT) if score != "" else "",
                    }
                )


def main() -> None:
    base_score, observed = load_splus()
    OUT.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    grid = fig.add_gridspec(
        1,
        3,
        left=0.035,
        right=0.965,
        bottom=0.135,
        top=0.82,
        wspace=0.075,
        width_ratios=[0.92, 1.07, 1.32],
    )
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]

    fig.text(
        0.04,
        0.935,
        "FROM CHINCHILLA TO BIODINOV3",
        ha="left",
        va="top",
        color=CORAL,
        fontsize=9.5,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.04,
        0.902,
        "Scaling is a surface, not a single curve",
        ha="left",
        va="top",
        color=INK,
        fontsize=28,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.96,
        0.898,
        "Current evidence  →  next experiment",
        ha="right",
        va="top",
        color=MUTED,
        fontsize=10.5,
        fontfamily="DejaVu Sans",
    )

    plot_principle(axes[0])
    plot_splus(axes[1], base_score, observed)
    plot_target_matrix(axes[2], observed)
    connector(fig, axes[0], axes[1])
    connector(fig, axes[1], axes[2])

    fig.text(
        0.04,
        0.065,
        "READOUT",
        ha="left",
        va="center",
        color=CORAL,
        fontsize=8.0,
        fontweight="bold",
    )
    fig.text(
        0.105,
        0.065,
        "S+ establishes the data axis. The full model × data grid is required before claiming a microscopy scaling law.",
        ha="left",
        va="center",
        color=INK,
        fontsize=10.0,
        fontweight="bold",
    )
    fig.text(
        0.96,
        0.042,
        "Observed metric is a diagnostic raw five-family mean, not the final scaling-law objective.",
        ha="right",
        va="center",
        color=MUTED,
        fontsize=7.2,
    )

    base = OUT / "scaling_law_storyboard"
    for extension in ("png", "svg", "pdf"):
        kwargs = {"dpi": 220} if extension == "png" else {}
        fig.savefig(base.with_suffix(f".{extension}"), **kwargs)
    plt.close(fig)
    write_figure_data(base_score, observed)
    print(f"wrote {base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
