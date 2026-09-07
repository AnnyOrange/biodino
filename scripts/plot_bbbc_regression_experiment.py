#!/usr/bin/env python3
"""Plot BBBC005/BBBC013 examples and held-out regression predictions."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dinov3.eval.bio_frozen_eval.make_group_splits import group_split_indices
from dinov3.eval.bio_frozen_eval.registry import build_dataset


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = Path("/mnt/huawei_deepcad/benchmark")
RUN = (
    ROOT
    / "outputs/01_training_runs"
    / "HS6_Dscale_Splus_robust_biosafe256_gb1024_lr2e4_wu3_tw30_nosig_e8_random20_seed0_20260820"
    / "eval/e8_full_20260820/bio_regression"
)
FEATURES = {
    "bbbc005": RUN / "bbbc005/1639/features/bbbc005/dinov3-1639.npz",
    "bbbc013": RUN / "bbbc013/1639/features/bbbc013/dinov3-1639.npz",
}
OUT = ROOT / "plot/fig2/data"

INK = "#24343A"
MUTED = "#68777D"
GRID = "#DCE2E4"
TEAL = "#16706D"
GOLD = "#D09531"
RED = "#BF4A3C"
PAPER = "#FFFFFF"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_feature_file(name: str) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(FEATURES[name], allow_pickle=True)
    return payload["features"], payload["labels"].astype(float)


def bbbc005_predictions() -> dict[str, np.ndarray | float]:
    dataset, _ = build_dataset("bbbc005", "train", None, None, benchmark_root=BENCHMARK)
    features, targets = load_feature_file("bbbc005")
    train_idx, test_idx = group_split_indices("bbbc005", dataset, BENCHMARK)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    reg.fit(features[train_idx], targets[train_idx])
    pred = reg.predict(features[test_idx])
    return {
        "target": targets[test_idx],
        "prediction": pred,
        "spearman": float(spearmanr(targets[test_idx], pred).correlation),
        "indices": test_idx,
    }


def bbbc013_predictions() -> dict[str, np.ndarray | float]:
    dataset, _ = build_dataset("bbbc013", "train", None, None, benchmark_root=BENCHMARK)
    features, targets = load_feature_file("bbbc013")
    paths = [str(sample.image_path) for sample in dataset.samples]
    rows = np.asarray(
        [re.search(r"Channel\d+-\d+-([A-H])-\d+\.BMP$", path, re.I).group(1).upper() for path in paths]
    )
    compounds = {
        "Wortmannin": np.asarray([row in "ABCD" for row in rows]),
        "LY294002": np.asarray([row in "EFGH" for row in rows]),
    }
    log_target = np.log1p(targets)
    pred = np.full(len(targets), np.nan, dtype=float)
    rho: dict[str, float] = {}
    for compound, compound_mask in compounds.items():
        compound_rows = sorted(set(rows[compound_mask]))
        for held_out_row in compound_rows:
            test_mask = rows == held_out_row
            train_mask = compound_mask & ~test_mask
            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            reg.fit(features[train_mask], log_target[train_mask])
            pred[test_mask] = reg.predict(features[test_mask])
        rho[compound] = float(spearmanr(log_target[compound_mask], pred[compound_mask]).correlation)
    compound_name = np.where(compounds["Wortmannin"], "Wortmannin", "LY294002")
    return {
        "target": log_target,
        "raw_target": targets,
        "prediction": pred,
        "compound": compound_name,
        "replicate_row": rows,
        "wortmannin_spearman": rho["Wortmannin"],
        "ly294002_spearman": rho["LY294002"],
        "spearman": float(np.mean(list(rho.values()))),
    }


def image_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=float)
    lo, hi = np.percentile(arr, [1, 99.7])
    return np.clip((arr - lo) / max(hi - lo, 1e-8), 0, 1)


def sample_strip(parent, paths: list[Path], labels: list[str], cmap: str) -> None:
    grid = parent.subgridspec(1, len(paths), wspace=0.06)
    for i, (path, label) in enumerate(zip(paths, labels)):
        ax = plt.gcf().add_subplot(grid[0, i])
        ax.imshow(image_array(path), cmap=cmap, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#FFFFFF")
            spine.set_linewidth(2)
        ax.text(
            0.5,
            -0.08,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color=INK,
        )


def style_prediction_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def draw_bbbc005(ax: plt.Axes, result: dict[str, np.ndarray | float]) -> None:
    target = np.asarray(result["target"])
    pred = np.asarray(result["prediction"])
    rng = np.random.default_rng(7)
    ax.scatter(target + rng.normal(0, 0.24, len(target)), pred, s=10, alpha=0.16, color=TEAL, linewidth=0)
    edges = np.arange(0, 111, 10)
    bin_id = np.clip(np.digitize(target, edges) - 1, 0, len(edges) - 2)
    populated = [i for i in range(len(edges) - 1) if np.any(bin_id == i)]
    centers = np.asarray([np.median(target[bin_id == i]) for i in populated])
    medians = np.asarray([np.median(pred[bin_id == i]) for i in populated])
    q25 = np.asarray([np.quantile(pred[bin_id == i], 0.25) for i in populated])
    q75 = np.asarray([np.quantile(pred[bin_id == i], 0.75) for i in populated])
    ax.fill_between(centers, q25, q75, color=TEAL, alpha=0.14, linewidth=0, label="IQR")
    ax.plot(centers, medians, color=TEAL, linewidth=2.4, marker="o", markersize=4, label="Binned median")
    ax.plot([0, 105], [0, 105], color=MUTED, linestyle="--", linewidth=1, alpha=0.75)
    ax.set(xlim=(0, 105), ylim=(-2, 108), xlabel="True cell count", ylabel="Held-out predicted count")
    ax.text(
        0.04,
        0.94,
        rf"Spearman $\rho$ = {float(result['spearman']):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=TEAL,
    )
    ax.text(0.04, 0.84, "1,925 held-out images", transform=ax.transAxes, color=MUTED)
    style_prediction_axis(ax)


def draw_bbbc013(ax: plt.Axes, result: dict[str, np.ndarray | float]) -> None:
    target = np.asarray(result["target"])
    pred = np.asarray(result["prediction"])
    compound = np.asarray(result["compound"])
    specs = [("Wortmannin", RED), ("LY294002", GOLD)]
    for name, color in specs:
        mask = compound == name
        ax.scatter(target[mask], pred[mask], s=25, alpha=0.42, color=color, edgecolor="white", linewidth=0.35)
        levels = np.unique(target[mask])
        medians = np.asarray([np.median(pred[mask & np.isclose(target, level)]) for level in levels])
        ax.plot(levels, medians, color=color, linewidth=2.2, marker="o", markersize=4, label=name)
    limit = max(float(np.max(target)), float(np.max(pred))) + 0.2
    ax.plot([0, limit], [0, limit], color=MUTED, linestyle="--", linewidth=1, alpha=0.75)
    ax.set(
        xlim=(-0.15, limit),
        ylim=(-0.35, limit),
        xlabel=r"True dose, $\log(1+x)$",
        ylabel=r"OOF predicted dose, $\log(1+x)$",
    )
    ax.text(
        0.04,
        0.95,
        rf"Macro Spearman $\rho$ = {float(result['spearman']):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.04,
        0.84,
        rf"Wort. {float(result['wortmannin_spearman']):.3f}  |  LY294002 {float(result['ly294002_spearman']):.3f}",
        transform=ax.transAxes,
        color=MUTED,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    style_prediction_axis(ax)


def write_predictions(bbbc005: dict, bbbc013: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "bbbc_regression_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "sample",
                "compound",
                "replicate_row",
                "true_target",
                "predicted_target",
                "target_scale",
                "protocol",
            ]
        )
        for i, (target, pred) in enumerate(zip(bbbc005["target"], bbbc005["prediction"])):
            writer.writerow(["BBBC005", i, "", "", target, pred, "cell_count", "group-held-out"])
        for i, (target, pred, compound, replicate_row) in enumerate(
            zip(
                bbbc013["target"],
                bbbc013["prediction"],
                bbbc013["compound"],
                bbbc013["replicate_row"],
            )
        ):
            writer.writerow(
                [
                    "BBBC013",
                    i,
                    compound,
                    replicate_row,
                    target,
                    pred,
                    "log1p_dose",
                    "compound-specific-replicate-row-OOF",
                ]
            )


def save_formats(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_single_dataset_panel(
    letter: str,
    title: str,
    paths: list[Path],
    labels: list[str],
    cmap: str,
    result: dict,
    draw_prediction,
    stem: Path,
) -> None:
    fig = plt.figure(figsize=(14.2, 4.5), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        left=0.055,
        right=0.975,
        bottom=0.16,
        top=0.76,
        width_ratios=(1.15, 1.0),
        wspace=0.18,
    )
    sample_strip(grid[0, 0], paths, labels, cmap)
    ax = fig.add_subplot(grid[0, 1])
    draw_prediction(ax, result)
    fig.text(0.055, 0.91, f"{letter}   {title}", fontsize=20, fontweight="bold", color=INK)
    fig.text(
        0.055,
        0.835,
        "Frozen S+ representation (0.2M pretraining pool, checkpoint 1639)  +  linear Ridge probe",
        fontsize=11.5,
        color=MUTED,
    )
    save_formats(fig, stem)


def make_bbbc013_oof_figure(result: dict, stem: Path) -> None:
    target = np.asarray(result["target"])
    pred = np.asarray(result["prediction"])
    compound = np.asarray(result["compound"])
    rows = np.asarray(result["replicate_row"])
    specs = [
        ("Wortmannin", RED, "ABCD", float(result["wortmannin_spearman"])),
        ("LY294002", GOLD, "EFGH", float(result["ly294002_spearman"])),
    ]
    markers = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), facecolor="white", sharex=True, sharey=True)
    for ax, (name, color, compound_rows, rho) in zip(axes, specs):
        mask = compound == name
        for marker, row in zip(markers, compound_rows):
            row_mask = mask & (rows == row)
            ax.scatter(
                target[row_mask],
                pred[row_mask],
                s=42,
                marker=marker,
                color=color,
                alpha=0.70,
                edgecolor="white",
                linewidth=0.5,
                label=f"held-out row {row}",
            )
        levels = np.unique(target[mask])
        medians = np.asarray([np.median(pred[mask & np.isclose(target, level)]) for level in levels])
        ax.plot(levels, medians, color=color, linewidth=2.5, marker="o", markersize=4, label="dose median")
        limit = max(float(np.max(target)), float(np.max(pred))) + 0.2
        ax.plot([0, limit], [0, limit], color=MUTED, linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set(xlim=(-0.15, limit), ylim=(-0.35, limit), title=name, xlabel=r"True dose, $\log(1+x)$")
        ax.text(
            0.04,
            0.94,
            rf"Spearman $\rho$ = {rho:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=13,
            fontweight="bold",
            color=color,
        )
        ax.text(0.04, 0.855, "4 folds | 48 OOF predictions", transform=ax.transAxes, color=MUTED)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        style_prediction_axis(ax)
    axes[0].set_ylabel(r"OOF predicted dose, $\log(1+x)$")
    fig.suptitle("BBBC013 compound-specific dose regression", x=0.07, y=0.985, ha="left", fontsize=20, fontweight="bold", color=INK)
    fig.text(
        0.07,
        0.90,
        "Eight replicate-row folds in total: train on three rows and predict the held-out row within each compound",
        fontsize=11,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.80, wspace=0.16)
    save_formats(fig, stem)


def main() -> None:
    configure()
    bbbc005 = bbbc005_predictions()
    bbbc013 = bbbc013_predictions()
    write_predictions(bbbc005, bbbc013)

    fig = plt.figure(figsize=(14.2, 7.6), facecolor=PAPER)
    outer = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.975,
        bottom=0.105,
        top=0.855,
        width_ratios=(1.15, 1.0),
        hspace=0.40,
        wspace=0.18,
    )

    bbbc005_root = BENCHMARK / "Regression/BBBC005/extracted/BBBC005_v1_images"
    counts = [1, 48, 100]
    bbbc005_paths = [sorted(bbbc005_root.glob(f"*_C{count}_F1_*_w1.TIF"))[0] for count in counts]
    sample_strip(outer[0, 0], bbbc005_paths, ["1 cell", "48 cells", "100 cells"], "magma")
    ax005 = fig.add_subplot(outer[0, 1])
    draw_bbbc005(ax005, bbbc005)

    bbbc013_root = BENCHMARK / "Regression/BBBC013/BBBC013_v1_images_bmp"
    wells = ["03", "07", "11"]
    doses = ["0.98 nM", "15.63 nM", "250 nM"]
    bbbc013_paths = [bbbc013_root / f"Channel1-{well}-A-{well}.BMP" for well in wells]
    sample_strip(outer[1, 0], bbbc013_paths, doses, "Greens")
    ax013 = fig.add_subplot(outer[1, 1])
    draw_bbbc013(ax013, bbbc013)

    fig.text(0.055, 0.948, "Continuous biological prediction from microscopy", fontsize=23, fontweight="bold", color=INK)
    fig.text(
        0.055,
        0.903,
        "Frozen S+ representation (0.2M pretraining pool, checkpoint 1639)  +  linear Ridge probe",
        fontsize=12,
        color=MUTED,
    )
    fig.text(0.055, 0.852, "A   BBBC005  |  Synthetic cell-count regression", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.055, 0.458, "B   BBBC013  |  Compound-specific dose regression", fontsize=14, fontweight="bold", color=INK)
    fig.text(
        0.055,
        0.035,
        "Points are held-out predictions. Curves show median predictions across target levels/bins; shading denotes the interquartile range.",
        fontsize=10,
        color=MUTED,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "bbbc005_bbbc013_regression_overview"
    save_formats(fig, stem)

    make_single_dataset_panel(
        "A",
        "BBBC005  |  Synthetic cell-count regression",
        bbbc005_paths,
        ["1 cell", "48 cells", "100 cells"],
        "magma",
        bbbc005,
        draw_bbbc005,
        OUT / "bbbc005_regression_panel_A_white",
    )
    make_single_dataset_panel(
        "B",
        "BBBC013  |  Compound-specific dose regression",
        bbbc013_paths,
        doses,
        "Greens",
        bbbc013,
        draw_bbbc013,
        OUT / "bbbc013_regression_panel_B_white",
    )
    make_bbbc013_oof_figure(bbbc013, OUT / "bbbc013_oof_predicted_dose_white")

    print(f"Wrote white-background figures under {OUT}")
    print(f"BBBC005 Spearman: {bbbc005['spearman']:.6f}")
    print(f"BBBC013 macro Spearman: {bbbc013['spearman']:.6f}")


if __name__ == "__main__":
    main()
