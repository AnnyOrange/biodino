#!/usr/bin/env python3
"""BloodMNIST 10-shot C-scale plots in the Fig.2 lineB 1M-only style."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np

from ppt_friendly_svg import configure_matplotlib, sanitize_svg_file

ROOT = Path(__file__).resolve().parents[1]
K10_DATA = ROOT / "plot/fig2/data/bloodmnist_cscale_k10_prop15_20260908.csv"
FULL_DATA = ROOT / "plot/fig2/data/bloodmnist_cscale_full_linear_20260908.csv"
DSCALE_RAW = (
    ROOT / "outputs/auto_eval_logs/hs6_kshot_feat_20260901/kshot.csv",
    ROOT / "plot/fig2/kshot/kshot_merged.csv",
    ROOT / "plot/fig2/data/bloodmnist_hplus_dscale_k10_allckpt_20260908.csv",
)
OUT = ROOT / "plot/fig2/scalingvit_fig2match"

TOKENS = 514
GBS = 1024
C0 = 1e18
MODELS = ("S+", "B", "L", "H+")
PARAMS = {"S+": 21e6, "B": 86e6, "L": 300e6, "H+": 840e6}
COLORS = {"S+": "#465ECF", "B": "#88ABFD", "L": "#E97A5F", "H+": "#B40426"}
POOLS = ("0.1M", "0.2M", "0.5M", "1M")
POOL_AREA = {"0.1M": 26.0, "0.2M": 42.0, "0.5M": 68.0, "1M": 104.0}
POOL_SAMPLES = {"0.1M": 104877, "0.2M": 209754, "0.5M": 524385, "1M": 1048771}
INK, GRID = "#2C3338", "#E4E8EB"
_SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


def compute_flops(model: str, ckpt: int) -> float:
    return 6.0 * PARAMS[model] * (ckpt * GBS) * TOKENS


def load_k10_points() -> list[dict]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    with K10_DATA.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], row["endpoint"], int(row["ckpt"]))
            grouped[key].append(100.0 * (1.0 - float(row["balanced_accuracy"])))

    points = []
    for (model, endpoint, ckpt), values in grouped.items():
        if len(values) != 3:
            raise ValueError(f"Expected 3 seeds for {(model, endpoint, ckpt)}, got {len(values)}")
        points.append(
            {
                "model": model,
                "endpoint": endpoint,
                "ckpt": ckpt,
                "compute": compute_flops(model, ckpt),
                "error": float(np.mean(values)),
            }
        )
    if len(points) != 16:
        raise ValueError(f"Expected 16 endpoint means, got {len(points)}")
    # The available H+ e15 result is a late, strongly degraded point from the
    # older e15 run.  Keep it in the source table for provenance, but omit it
    # from this focused figure while the matched prop15 e8 audit is pending.
    return [p for p in points if not (p["model"] == "H+" and p["endpoint"] == "e15")]


def load_full_points() -> list[dict]:
    points = []
    with FULL_DATA.open(newline="") as handle:
        for row in csv.DictReader(handle):
            points.append(
                {
                    "model": row["model"],
                    "endpoint": row["endpoint"],
                    "ckpt": int(row["ckpt"]),
                    "compute": compute_flops(row["model"], int(row["ckpt"])),
                    "error": 100.0 * (1.0 - float(row["macro_f1"])),
                }
            )
    if len(points) != 16:
        raise ValueError(f"Expected 16 full-linear endpoints, got {len(points)}")
    return [p for p in points if not (p["model"] == "H+" and p["endpoint"] == "e15")]


def load_dscale_points() -> list[dict]:
    # N/D use the complete D-scale checkpoint cloud (e1--e8), not only e8 last.
    # The source files overlap at e8; key by seed so duplicates cannot bias means.
    grouped: dict[tuple[str, str, int, int, int], dict[str, float]] = defaultdict(dict)
    for path in DSCALE_RAW:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if not (
                    row.get("axis") == "data"
                    and row.get("dataset") == "bloodmnist"
                    and row.get("k") == "10"
                    and row.get("model") in MODELS
                    and row.get("pool") in POOLS
                ):
                    continue
                key = (
                    row["model"], row["pool"], int(row["samples"]),
                    int(row["ckpt"]), int(row["epoch"]),
                )
                grouped[key][row["seed"]] = 100.0 * float(row["error_balanced"])

    points = []
    for (model, pool, samples, ckpt, epoch), seeds in grouped.items():
        values = [seeds[str(seed)] for seed in (0, 1, 2) if str(seed) in seeds]
        if len(values) != 3:
            continue
        points.append(
            {
                "model": model,
                "pool": pool,
                "samples": samples,
                "ckpt": ckpt,
                "epoch": epoch,
                "error": float(np.mean(values)),
            }
        )
    if len(points) != 128:
        raise ValueError(f"Expected 128 D-scale checkpoint means, got {len(points)}")
    return points


def pareto_front(points: list[dict]) -> list[dict]:
    best = np.inf
    front = []
    for point in sorted(points, key=lambda p: p["compute"]):
        if point["error"] < best - 1e-12:
            front.append(point)
            best = point["error"]
    return front


def power_error(compute, a: float, b: float):
    """Pure power law, which is exactly linear when both axes are logarithmic."""
    return a * np.power(np.asarray(compute, dtype=float) / C0, -b)


def fit_power_law(points: list[dict]) -> tuple[float, float]:
    front = pareto_front(points)
    compute = np.asarray([p["compute"] for p in front], dtype=float)
    error = np.asarray([p["error"] for p in front], dtype=float)
    slope, intercept = np.polyfit(np.log(compute / C0), np.log(error), 1)
    return float(np.exp(intercept)), -float(slope)


def superscript(value: str) -> str:
    return "".join("·" if ch == "." else ch.translate(_SUP) for ch in value)


def fit_label(a: float, b: float) -> str:
    return f"E={a:.2f}(C/10{superscript('18')}){superscript(f'-{b:.2f}')}"


def pow10_tick(value, _position=None) -> str:
    if not np.isfinite(value) or value <= 0:
        return ""
    exponent = int(round(np.log10(value)))
    if abs(np.log10(value) - exponent) > 1e-6:
        return f"{value:g}"
    return "10" + superscript(str(exponent))


def style_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=8.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def set_error_scale(ax, errors: list[float], *, log_y: bool) -> None:
    low, high = min(errors), max(errors)
    if log_y:
        # Match ScalingViT Fig. 2: logarithmic geometry with the original
        # percentage values printed on the ticks (20, 24, ...), rather than
        # rewriting those values as fractional powers such as 10^1.3.
        ax.set_yscale("log")
        if low >= 19.5:
            bottom = 20.0
            top = max(36.0, 4.0 * np.ceil(high / 4.0))
            ticks = np.arange(bottom, top + 0.01, 4.0)
        else:
            bottom = max(1.0, np.floor(low))
            top = np.ceil(high)
            ticks = np.arange(bottom, top + 0.01, 1.0)
        ax.set_ylim(bottom, top)
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
    else:
        padding = 0.10 * (high - low)
        ax.set_ylim(low - padding, high + padding)


def draw_panel(ax, points: list[dict], *, title: str, ylabel: str, log_y: bool) -> None:
    style_axes(ax)

    for model in MODELS:
        model_points = sorted((p for p in points if p["model"] == model), key=lambda p: p["ckpt"])
        x = [p["compute"] for p in model_points]
        y = [p["error"] for p in model_points]
        ax.plot(x, y, color=COLORS[model], lw=0.9, alpha=0.28, zorder=2, solid_capstyle="round")
        ax.scatter(x, y, s=104.0, c=COLORS[model], marker="o", zorder=4, edgecolors="white", linewidths=0.45)

    fit = fit_power_law(points)
    all_x = np.asarray([p["compute"] for p in points], dtype=float)
    all_y = np.asarray([p["error"] for p in points], dtype=float)
    grid = np.logspace(np.log10(all_x.min() * 0.85), np.log10(all_x.max() * 1.15), 300)
    ax.plot(grid, power_error(grid, *fit), color=INK, ls="--", lw=1.65, zorder=3)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(pow10_tick))
    ax.set_xlabel("pretrain FLOPs", fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=6)

    set_error_scale(ax, all_y.tolist(), log_y=log_y)

    ax.text(0.03, 0.045, fit_label(*fit), transform=ax.transAxes, fontsize=7.5, color=INK)


def draw_n_panel(ax, k10_points: list[dict], dscale_points: list[dict], *, log_y: bool) -> None:
    style_axes(ax)
    cloud = []
    for point in dscale_points:
        cloud.append(point["error"])
        ax.scatter(
            [PARAMS[point["model"]]], [point["error"]], s=POOL_AREA[point["pool"]],
            c="#D5DBE5", marker="o", linewidths=0, alpha=0.58, zorder=2,
        )
    for point in k10_points:
        cloud.append(point["error"])
        ax.scatter([PARAMS[point["model"]]], [point["error"]], s=104.0, c="#D5DBE5", marker="o", linewidths=0, alpha=0.40, zorder=2)

    best_x, best_y = [], []
    for model in MODELS:
        model_points = [p for p in [*k10_points, *dscale_points] if p["model"] == model]
        best = min(model_points, key=lambda p: p["error"])
        best_x.append(PARAMS[model])
        best_y.append(best["error"])
        ax.scatter([PARAMS[model]], [best["error"]], s=104.0, c=COLORS[model], marker="o", edgecolors="white", linewidths=0.5, zorder=4)
    ax.plot(best_x, best_y, color=INK, ls="--", lw=1.0, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([PARAMS[m] for m in MODELS])
    ax.set_xticklabels(MODELS)
    ax.set_xlabel("model size", fontsize=9, color=INK)
    ax.set_ylabel("10-shot balanced error [%]", fontsize=9, color=INK)
    ax.set_title("N", loc="left", fontsize=11, color=INK, pad=6)
    set_error_scale(ax, cloud, log_y=log_y)


def draw_d_panel(ax, dscale_points: list[dict], *, log_y: bool) -> None:
    style_axes(ax)
    errors = [p["error"] for p in dscale_points]
    for point in dscale_points:
        ax.scatter(
            [point["samples"]], [point["error"]], s=POOL_AREA[point["pool"]],
            c="#D5DBE5", marker="o", linewidths=0, alpha=0.58, zorder=2,
        )
    best_x, best_y = [], []
    for pool in POOLS:
        best = min((p for p in dscale_points if p["pool"] == pool), key=lambda p: p["error"])
        best_x.append(best["samples"])
        best_y.append(best["error"])
        ax.scatter(
            [best["samples"]], [best["error"]], s=POOL_AREA[pool], c=COLORS[best["model"]],
            marker="o", edgecolors="white", linewidths=0.5, zorder=4,
        )
    ax.plot(best_x, best_y, color=INK, ls="--", lw=1.0, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([POOL_SAMPLES[p] for p in POOLS])
    ax.set_xticklabels(POOLS)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("unique images", fontsize=9, color=INK)
    ax.set_ylabel("10-shot balanced error [%]", fontsize=9, color=INK)
    ax.set_title("D", loc="left", fontsize=11, color=INK, pad=6)
    set_error_scale(ax, errors, log_y=log_y)


def draw(full_points: list[dict], k10_points: list[dict], dscale_points: list[dict], *, log_y: bool) -> tuple[Path, Path]:
    configure_matplotlib()
    plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 4, figsize=(20.4, 5.35))
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.13, top=0.75, wspace=0.30)
    draw_panel(axes[0], full_points, title="C   full linear", ylabel="linear error [%]", log_y=log_y)
    draw_panel(axes[1], k10_points, title="C   10-shot", ylabel="10-shot balanced error [%]", log_y=log_y)
    draw_n_panel(axes[2], k10_points, dscale_points, log_y=log_y)
    draw_d_panel(axes[3], dscale_points, log_y=log_y)

    suffix = "logy" if log_y else "lineary"
    scale_note = "log y-axis" if log_y else "linear y-axis"

    fig.text(0.045, 0.965, "BloodMNIST", fontsize=13, color=INK, ha="left", va="top")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[m], markeredgecolor="white", markersize=8, label=m)
        for m in MODELS
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.15, 0.972), ncol=4, frameon=False, fontsize=9)
    size_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#8A9096", markeredgecolor="white", markersize=max(4.5, np.sqrt(POOL_AREA[p]) * 0.7), label=p)
        for p in POOLS
    ]
    fig.legend(handles=size_handles, loc="upper left", bbox_to_anchor=(0.42, 0.972), ncol=4, frameon=False, fontsize=8.5)
    fig.text(
        0.045,
        0.855,
        f"1M unique-D; e1→e2→e4→e15; H+ e15 omitted; frozen full linear + 3-seed 10-shot; {scale_note}.",
        fontsize=7.4,
        color="#5B656C",
        ha="left",
        va="top",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"bloodmnist_k10_cscale_fig2_{suffix}.png"
    svg = OUT / f"bloodmnist_k10_cscale_fig2_{suffix}.svg"
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    plt.close(fig)
    sanitize_svg_file(svg)
    return png, svg


def main() -> None:
    full_points = load_full_points()
    k10_points = load_k10_points()
    dscale_points = load_dscale_points()
    for log_y in (False, True):
        png, svg = draw(full_points, k10_points, dscale_points, log_y=log_y)
        print(png)
        print(svg)


if __name__ == "__main__":
    main()
