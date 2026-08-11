#!/usr/bin/env python3
"""Build loss-scaling diagnostics for the final ViT-S S+ C/D experiment."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/00_reports/splus_final_alpha_cd_seed0_20260724/loss"
WINDOW = 200
FIXED_COMPUTE_CHECKPOINT = 8199
CHECKPOINTS = (1024, 2049, 4099, 6149, 8199, 10249, 12299, 15374)
PASSES = (1, 2, 4, 6, 8, 10, 12, 15)

RUNS = (
    (
        "0.1M",
        104_877,
        ROOT
        / "outputs/01_training_runs/"
        "DscaleFinal_splus_sigreg005_random10_fixed15M_b1024_seed0_qi4gbs64acc4",
    ),
    (
        "0.2M",
        209_754,
        ROOT
        / "outputs/01_training_runs/"
        "DscaleFinal_splus_sigreg005_random20_fixed15M_b1024_seed0_qi4gbs64acc4",
    ),
    (
        "0.5M",
        524_385,
        ROOT
        / "outputs/01_training_runs/"
        "DscaleFinal_splus_sigreg005_random50_fixed15M_b1024_seed0_local8gbs64acc2",
    ),
    (
        "1.0M",
        1_048_771,
        ROOT
        / "outputs/01_training_runs/"
        "S6sigreg005_rgb_robust_biosafe256_b1024_officialprec_lr1e-4_wu2_e15_local5090_20260716",
    ),
)

INK = "#24343A"
MUTED = "#69777C"
GRID = "#D8DEE0"
TEAL = "#146B68"
ORANGE = "#C76535"
RED = "#BF3B35"
POOL_COLORS = {
    "0.1M": "#D8A33B",
    "0.2M": "#CE7541",
    "0.5M": "#4D8E85",
    "1.0M": TEAL,
}
COMPONENTS = (
    ("dino_global_crops_loss", "DINO global-crop loss", "#245B78"),
    ("dino_local_crops_loss", "DINO local-crop loss", "#3E8A81"),
    ("ibot_loss", "iBOT loss", "#C76535"),
    ("sigreg_loss", "SIGReg loss (unweighted)", "#B89B32"),
)


def load_run(path: Path) -> tuple[dict[int, dict], int]:
    raw_path = path / "raw_loss_metrics.jsonl"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    rows: dict[int, dict] = {}
    line_count = 0
    with raw_path.open() as handle:
        for line in handle:
            line_count += 1
            row = json.loads(line)
            rows[int(row["optimizer_update"])] = row
    expected = list(range(CHECKPOINTS[-1] + 1))
    if sorted(rows) != expected:
        raise ValueError(f"Non-contiguous optimizer updates in {raw_path}")
    for checkpoint in CHECKPOINTS:
        row = rows[checkpoint]
        checks = {
            "effective_global_batch_size": int(row["effective_global_batch_size"]) == 1024,
            "image_visits": int(row["image_visits"]) == (checkpoint + 1) * 1024,
            "epoch_float": math.isclose(
                float(row["epoch_float"]), (checkpoint + 1) / 1025, abs_tol=1e-12
            ),
            "arch": row["arch"] == "vit_small",
            "augmentation_policy": row["augmentation_policy"] == "bio_safe",
            "sigreg_loss_weight": math.isclose(
                float(row["sigreg_loss_weight"]), 0.05, abs_tol=1e-7
            ),
        }
        failures = [name for name, passed in checks.items() if not passed]
        if failures:
            raise ValueError(f"Protocol mismatch in {raw_path} ck{checkpoint}: {failures}")
    return rows, line_count - len(rows)


def trailing_stats(rows: dict[int, dict], checkpoint: int, key: str) -> dict[str, float]:
    values = np.array(
        [float(rows[index][key]) for index in range(checkpoint - WINDOW + 1, checkpoint + 1)],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite {key} near ck{checkpoint}")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def smooth_series(rows: dict[int, dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    updates = np.arange(CHECKPOINTS[-1] + 1)
    values = np.array([float(rows[int(index)][key]) for index in updates], dtype=float)
    visits = np.array([int(rows[int(index)]["image_visits"]) for index in updates], dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    means = (cumulative[WINDOW:] - cumulative[:-WINDOW]) / WINDOW
    return visits[WINDOW - 1 :] / 1e6, means


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def power_law(x: np.ndarray, floor: float, amplitude: float, exponent: float) -> np.ndarray:
    return floor + amplitude * np.power(x, -exponent)


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=0, colors=INK)


def save_figure(figure: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUT / f"{stem}.{suffix}",
            dpi=240 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(figure)


def build_tables(
    loaded: dict[str, dict[int, dict]], duplicates: dict[str, int]
) -> tuple[list[dict], list[dict], list[dict], dict]:
    compute_rows = []
    one_m = loaded["1.0M"]
    for checkpoint, pass_count in zip(CHECKPOINTS, PASSES):
        row = one_m[checkpoint]
        stats = trailing_stats(one_m, checkpoint, "total_loss")
        compute_rows.append(
            {
                "checkpoint": checkpoint,
                "reference_passes": pass_count,
                "optimizer_updates_completed": checkpoint + 1,
                "image_visits": int(row["image_visits"]),
                "image_visits_m": int(row["image_visits"]) / 1e6,
                "exact_total_loss": float(row["total_loss"]),
                "trailing_window_updates": WINDOW,
                "total_loss_mean": stats["mean"],
                "total_loss_std_within_window": stats["std"],
                "total_loss_min_within_window": stats["min"],
                "total_loss_max_within_window": stats["max"],
            }
        )

    data_rows = []
    for label, samples, path in RUNS:
        rows = loaded[label]
        row = rows[FIXED_COMPUTE_CHECKPOINT]
        stats = trailing_stats(rows, FIXED_COMPUTE_CHECKPOINT, "total_loss")
        data_rows.append(
            {
                "pool": label,
                "unique_images": samples,
                "checkpoint": FIXED_COMPUTE_CHECKPOINT,
                "optimizer_updates_completed": FIXED_COMPUTE_CHECKPOINT + 1,
                "image_visits": int(row["image_visits"]),
                "dataset_equivalent_passes": int(row["image_visits"]) / samples,
                "exact_total_loss": float(row["total_loss"]),
                "trailing_window_updates": WINDOW,
                "total_loss_mean": stats["mean"],
                "total_loss_std_within_window": stats["std"],
                "total_loss_min_within_window": stats["min"],
                "total_loss_max_within_window": stats["max"],
                "duplicate_log_rows_deduplicated": duplicates[label],
                "loss_log": str(path / "raw_loss_metrics.jsonl"),
            }
        )

    trajectory_rows = []
    for label, samples, _ in RUNS:
        rows = loaded[label]
        visits, means = smooth_series(rows, "total_loss")
        for index in range(0, len(means), 20):
            update = index + WINDOW - 1
            trajectory_rows.append(
                {
                    "pool": label,
                    "unique_images": samples,
                    "optimizer_update": update,
                    "image_visits": int(rows[update]["image_visits"]),
                    "image_visits_m": visits[index],
                    "total_loss_mean_trailing_200": means[index],
                }
            )
        if trajectory_rows[-1]["optimizer_update"] != CHECKPOINTS[-1]:
            stats = trailing_stats(rows, CHECKPOINTS[-1], "total_loss")
            trajectory_rows.append(
                {
                    "pool": label,
                    "unique_images": samples,
                    "optimizer_update": CHECKPOINTS[-1],
                    "image_visits": int(rows[CHECKPOINTS[-1]]["image_visits"]),
                    "image_visits_m": int(rows[CHECKPOINTS[-1]]["image_visits"]) / 1e6,
                    "total_loss_mean_trailing_200": stats["mean"],
                }
            )

    fit_compute = [row for row in compute_rows if int(row["reference_passes"]) >= 4]
    compute_x = np.array([float(row["image_visits_m"]) for row in fit_compute])
    compute_y = np.array([float(row["total_loss_mean"]) for row in fit_compute])
    parameters, covariance = curve_fit(
        power_law,
        compute_x,
        compute_y,
        p0=(10.0, 5.0, 0.5),
        bounds=([0.0, 0.0, 0.01], [float(compute_y.min()), 100.0, 5.0]),
        maxfev=100_000,
    )
    compute_prediction = power_law(compute_x, *parameters)
    compute_r2 = 1.0 - float(np.sum((compute_y - compute_prediction) ** 2)) / float(
        np.sum((compute_y - compute_y.mean()) ** 2)
    )

    data_x = np.array([float(row["unique_images"]) / 1e6 for row in data_rows])
    data_y = np.array([float(row["total_loss_mean"]) for row in data_rows])
    data_slope, data_intercept = np.polyfit(np.log10(data_x), data_y, 1)
    data_prediction = data_intercept + data_slope * np.log10(data_x)
    data_r2 = 1.0 - float(np.sum((data_y - data_prediction) ** 2)) / float(
        np.sum((data_y - data_y.mean()) ** 2)
    )
    fit = {
        "compute_floor": float(parameters[0]),
        "compute_amplitude": float(parameters[1]),
        "compute_exponent": float(parameters[2]),
        "compute_floor_std": float(np.sqrt(covariance[0, 0])),
        "compute_amplitude_std": float(np.sqrt(covariance[1, 1])),
        "compute_exponent_std": float(np.sqrt(covariance[2, 2])),
        "compute_r2": compute_r2,
        "compute_fit_min_passes": 4,
        "compute_fit_points": len(compute_x),
        "data_log10_slope": float(data_slope),
        "data_log10_intercept": float(data_intercept),
        "data_r2": data_r2,
        "data_fit_points": len(data_x),
    }
    return compute_rows, data_rows, trajectory_rows, fit


def plot_main(compute_rows: list[dict], data_rows: list[dict], fit: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.7))
    for axis in axes:
        style_axis(axis)

    compute_x = np.array([float(row["image_visits_m"]) for row in compute_rows])
    compute_y = np.array([float(row["total_loss_mean"]) for row in compute_rows])
    compute_std = np.array(
        [float(row["total_loss_std_within_window"]) for row in compute_rows]
    )
    axes[0].set_xscale("log")
    axes[0].errorbar(
        compute_x,
        compute_y,
        yerr=compute_std,
        color=TEAL,
        linewidth=2.4,
        marker="o",
        markersize=7,
        capsize=3,
        label=f"trailing-{WINDOW}-update mean",
    )
    fit_x = np.geomspace(compute_x[2], compute_x[-1], 240)
    axes[0].plot(
        fit_x,
        power_law(
            fit_x,
            fit["compute_floor"],
            fit["compute_amplitude"],
            fit["compute_exponent"],
        ),
        color=INK,
        linestyle=(0, (5, 4)),
        linewidth=1.5,
        label=rf"post-warmup fit: $\beta={fit['compute_exponent']:.2f}$",
    )
    axes[0].scatter(
        [compute_x[4]], [compute_y[4]], marker="*", s=220, color=RED, edgecolor="white", zorder=5
    )
    tick_indices = (0, 1, 2, 4, 7)
    axes[0].set_xticks(
        compute_x[list(tick_indices)], [str(PASSES[index]) for index in tick_indices]
    )
    axes[0].set_xlabel("Compute C (1M-reference passes, log)")
    axes[0].set_ylabel("SSL training total_loss (lower is better)")
    axes[0].set_title("(a) Loss vs compute | fixed ViT-S + 1M", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=9)

    data_x = np.array([float(row["unique_images"]) / 1e6 for row in data_rows])
    data_y = np.array([float(row["total_loss_mean"]) for row in data_rows])
    data_std = np.array([float(row["total_loss_std_within_window"]) for row in data_rows])
    axes[1].set_xscale("log")
    axes[1].errorbar(
        data_x,
        data_y,
        yerr=data_std,
        color=ORANGE,
        linewidth=2.4,
        marker="o",
        markersize=8,
        capsize=3,
        label=f"fixed 8.397M visits; trailing-{WINDOW} mean",
    )
    fit_x = np.geomspace(data_x[0], data_x[-1], 240)
    axes[1].plot(
        fit_x,
        fit["data_log10_intercept"] + fit["data_log10_slope"] * np.log10(fit_x),
        color=INK,
        linestyle=(0, (5, 4)),
        linewidth=1.5,
        label=rf"descriptive log slope: {fit['data_log10_slope']:.3f}/decade",
    )
    axes[1].scatter(
        [data_x[-1]], [data_y[-1]], marker="*", s=220, color=RED, edgecolor="white", zorder=5
    )
    for x_value, y_value in zip(data_x, data_y):
        axes[1].annotate(
            f"{y_value:.3f}",
            (x_value, y_value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )
    axes[1].set_xticks(data_x, [row["pool"] for row in data_rows])
    axes[1].set_xlabel("Unique microscopy images $D_u$ (log)")
    axes[1].set_ylabel("SSL training total_loss (lower is better)")
    axes[1].set_title("(b) Loss vs unique data | fixed compute", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=9)

    figure.suptitle(
        r"Underlying S6 + SIGReg training-loss scaling diagnostics (seed 0)",
        fontsize=16.5,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.018,
        "These are online training-objective diagnostics for the trajectory used to construct S+; not held-out S+ likelihood.",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.17, wspace=0.25)
    save_figure(figure, "splus_loss_cd_scaling_main")


def plot_trajectories(loaded: dict[str, dict[int, dict]]) -> None:
    figure, axis = plt.subplots(figsize=(9.6, 6.3))
    style_axis(axis)
    for label, _, _ in RUNS:
        x, y = smooth_series(loaded[label], "total_loss")
        axis.plot(x, y, color=POOL_COLORS[label], linewidth=2.0, label=label)
    axis.axvline(8.3968, color=RED, linestyle=(0, (4, 4)), linewidth=1.4)
    axis.text(8.3968, 16.15, "fixed-compute readout", color=RED, ha="center", fontsize=9)
    axis.set_xlabel("Image visits (millions)")
    axis.set_ylabel(f"SSL total_loss (trailing-{WINDOW}-update mean)")
    axis.set_title(
        "Loss trajectories at matched update schedule",
        fontsize=15,
        fontweight="bold",
    )
    axis.legend(title="unique-image pool", frameon=False, ncol=2)
    figure.text(
        0.5,
        0.02,
        "All runs: ViT-S, GBS 1024, SIGReg 0.05, identical 15.744M-visit scheduler; duplicate restart rows use the last record.",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    figure.subplots_adjust(left=0.10, right=0.98, top=0.89, bottom=0.14)
    save_figure(figure, "splus_loss_all_pools_trajectories")


def plot_components(rows: dict[int, dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.0))
    checkpoint_x = np.array([int(rows[checkpoint]["image_visits"]) / 1e6 for checkpoint in CHECKPOINTS])
    for axis, (key, title, color) in zip(axes.flat, COMPONENTS):
        style_axis(axis)
        x, y = smooth_series(rows, key)
        axis.plot(x, y, color=color, linewidth=2.0)
        checkpoint_y = np.array([trailing_stats(rows, checkpoint, key)["mean"] for checkpoint in CHECKPOINTS])
        axis.scatter(checkpoint_x, checkpoint_y, color=color, s=28, edgecolor="white", zorder=4)
        axis.axvline(8.3968, color=RED, linestyle=(0, (4, 4)), linewidth=1.0, alpha=0.8)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Image visits (millions)")
        axis.set_ylabel(f"trailing-{WINDOW} mean")
    figure.suptitle(
        "Fixed-1M training-loss components",
        fontsize=16,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.075, right=0.98, top=0.91, bottom=0.08, wspace=0.25, hspace=0.34)
    save_figure(figure, "splus_loss_fixed1m_components")


def write_readme(compute_rows: list[dict], data_rows: list[dict], fit: dict) -> None:
    endpoint_delta = float(data_rows[-1]["total_loss_mean"]) - float(
        data_rows[0]["total_loss_mean"]
    )
    lines = [
        "# S+ C/D loss-scaling diagnostics (seed0)",
        "",
        "These plots use the online SSL `total_loss` from the underlying S6 +",
        "SIGReg 0.05 training trajectories. S+ evaluation checkpoints are later",
        "constructed as `0.25 * official + 0.75 * EMA teacher`; consequently this",
        "logged loss is not a held-out loss measured on the interpolated S+ weights.",
        "",
        "## Readout rule",
        "",
        f"Each checkpoint value is the mean of the {WINDOW} optimizer updates ending",
        "at that checkpoint. Error bars are the within-window standard deviation,",
        "not seed uncertainty. Every update has effective GBS 1024. Restart duplicates",
        "are deduplicated by retaining the final record for an optimizer update.",
        "",
        "## Fixed-compute data slice",
        "",
        "| pool | visits | actual pool traversals | total_loss mean | window std |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in data_rows:
        lines.append(
            f"| {row['pool']} | {int(row['image_visits']):,} | "
            f"{float(row['dataset_equivalent_passes']):.3f} | "
            f"{float(row['total_loss_mean']):.6f} | "
            f"{float(row['total_loss_std_within_window']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"The 1.0M - 0.1M endpoint change is {endpoint_delta:+.6f} (lower is better).",
            "The 0.1M -> 0.2M step is not monotonic and is comparable to the local",
            "window fluctuation; the large-pool endpoint trend should not be presented",
            "as a four-point strictly monotonic law before additional seeds.",
            "",
            "## Diagnostic fits",
            "",
            f"- Post-warmup compute fit (4-15 passes): `L(C) = {fit['compute_floor']:.6f} + "
            f"{fit['compute_amplitude']:.6f} * C^(-{fit['compute_exponent']:.6f})`, where C is",
            f"  millions of image visits; R2 = {fit['compute_r2']:.4f}.",
            f"- Fixed-compute data trend: `L(Du) = {fit['data_log10_intercept']:.6f} "
            f"{fit['data_log10_slope']:+.6f} * log10(Du_M)`; R2 = {fit['data_r2']:.4f}.",
            "- These seed0 fits are descriptive. `total_loss` is a moving-target SSL",
            "  training objective, not a calibrated held-out likelihood or a substitute",
            "  for the full-suite transfer scaling curves.",
            "",
            "## Files",
            "",
            "- `splus_loss_cd_scaling_main.{png,pdf,svg}`",
            "- `splus_loss_all_pools_trajectories.{png,pdf,svg}`",
            "- `splus_loss_fixed1m_components.{png,pdf,svg}`",
            "- `loss_compute_points.csv`",
            "- `loss_data_fixed_compute.csv`",
            "- `loss_trajectories_downsampled.csv`",
            "- `loss_fit_diagnostics.csv`",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.color": INK,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    loaded: dict[str, dict[int, dict]] = {}
    duplicates: dict[str, int] = {}
    for label, _, path in RUNS:
        loaded[label], duplicates[label] = load_run(path)
    compute_rows, data_rows, trajectory_rows, fit = build_tables(loaded, duplicates)
    write_csv(OUT / "loss_compute_points.csv", compute_rows)
    write_csv(OUT / "loss_data_fixed_compute.csv", data_rows)
    write_csv(OUT / "loss_trajectories_downsampled.csv", trajectory_rows)
    write_csv(OUT / "loss_fit_diagnostics.csv", [fit])
    plot_main(compute_rows, data_rows, fit)
    plot_trajectories(loaded)
    plot_components(loaded["1.0M"])
    write_readme(compute_rows, data_rows, fit)
    (OUT / "._complete").touch()
    print(f"Wrote loss-scaling report to {OUT}")


if __name__ == "__main__":
    main()
