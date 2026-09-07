#!/usr/bin/env python3
"""Build audited H+ training-objective loss scaling tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = (
    "total_loss",
    "dino_local_crops_loss",
    "dino_global_crops_loss",
    "ibot_loss",
    "koleo_loss",
    "sigreg_loss",
)
COMPUTE_PASSES = (1, 2, 4, 6, 8, 10, 12, 15)
DATA_RUNS = {
    "0.1M": {
        "unique_images": 104_877,
        "paths": (
            "/data_2/suxin/runs/"
            "Hplus_s6recipe_nosigreg_datafp_random10_e15_gb1024_seed0_2xH100_20260811/"
            "raw_loss_metrics.jsonl",
        ),
    },
    "0.2M": {
        "unique_images": 209_754,
        "paths": (
            "/data_2/suxin/runs/"
            "Hplus_s6recipe_nosigreg_datafp_random20_e15_gb1024_seed0_2xH100_20260811/"
            "raw_loss_metrics.jsonl",
        ),
    },
    "0.5M": {
        "unique_images": 524_385,
        "paths": (
            "/data_2/suxin/runs/"
            "Hplus_s6recipe_nosigreg_datafp_random50_e15_gb1024_seed0_2xH100_20260811/"
            "raw_loss_metrics.jsonl",
            "/tmp/suxin/runs/"
            "Hplus_s6recipe_nosigreg_datafp_random50_e15_gb1024_seed0_2xH100_20260811_resume_tmp/"
            "raw_loss_metrics.jsonl",
        ),
    },
    "1M": {
        "unique_images": 1_048_770,
        "paths": (
            "/data_2/suxin/runs/h100_hplus_7b_sigreg_ab_tuning_20260725/"
            "hplus_nosigreg/full_e15_2gpu/raw_loss_metrics.jsonl",
        ),
    },
}
COMPUTE_RUN = (
    "/data_2/suxin/runs/h100_hplus_7b_sigreg_ab_tuning_20260725/"
    "hplus_nosigreg/full_e15_2gpu/raw_loss_metrics.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compute-run", type=Path, default=Path(COMPUTE_RUN))
    parser.add_argument(
        "--data-run",
        action="append",
        default=[],
        metavar="LABEL:UNIQUE_IMAGES:PATH[,PATH...]",
        help="Override the default H+ data runs; repeat once per data pool.",
    )
    parser.add_argument("--recipe", default="H+ S6 no-SIGReg (raw teacher, alpha=1.0)")
    parser.add_argument("--figure-title", default="H+ S6 no-SIGReg")
    parser.add_argument("--output-prefix", default="hplus")
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def resolve_data_runs(specs: list[str]) -> dict[str, dict[str, Any]]:
    if not specs:
        return DATA_RUNS
    runs: dict[str, dict[str, Any]] = {}
    for spec in specs:
        try:
            label, unique_images, paths = spec.split(":", 2)
        except ValueError as error:
            raise ValueError(f"Invalid --data-run value: {spec!r}") from error
        runs[label] = {
            "unique_images": int(unique_images),
            "paths": tuple(path for path in paths.split(",") if path),
        }
    return runs


def load_rows(paths: list[Path] | tuple[Path, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_update: dict[int, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for priority, path in enumerate(paths):
        source = {"path": str(path), "exists": path.is_file(), "valid_rows": 0, "invalid_rows": 0}
        if not path.is_file():
            sources.append(source)
            continue
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                    update = int(row["optimizer_update"])
                    value = float(row["total_loss"])
                    if not math.isfinite(value):
                        raise ValueError("non-finite total_loss")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    source["invalid_rows"] += 1
                    continue
                row["_source_priority"] = priority
                row["_source_line"] = line_number
                by_update[update] = row
                source["valid_rows"] += 1
        sources.append(source)
    rows = [by_update[key] for key in sorted(by_update)]
    if not rows:
        raise RuntimeError(f"No valid raw loss rows in {[str(path) for path in paths]}")
    return rows, {"sources": sources, "unique_optimizer_updates": len(rows)}


def trailing_mean(rows: list[dict[str, Any]], end_update: int, window: int, key: str) -> tuple[float, float, int]:
    values = [
        float(row[key])
        for row in rows
        if key in row
        and math.isfinite(float(row[key]))
        and end_update - window + 1 <= int(row["optimizer_update"]) <= end_update
    ]
    if not values:
        return math.nan, math.nan, 0
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, sem, len(values)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(int(window), len(values)))
    if window == 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    if len(x) < 3 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None


def fit_log_linear(x: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    if len(x) < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    log_x = np.log2(x)
    slope, intercept = np.polyfit(log_x, y, 1)
    predicted = intercept + slope * log_x
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "slope_per_data_doubling": float(slope),
        "intercept": float(intercept),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan,
    }
    try:
        from scipy.optimize import curve_fit

        def law(z, floor, amplitude, exponent):
            return floor + amplitude * np.power(z / 1e6, -exponent)

        floor_upper = float(np.min(y) - 1e-5)
        params, _ = curve_fit(
            law,
            x,
            y,
            p0=(max(0.0, floor_upper - 1), max(0.01, float(np.ptp(y))), 0.5),
            bounds=([0.0, 0.0, 0.001], [floor_upper, 100.0, 5.0]),
            maxfev=100_000,
        )
        predicted = law(x, *params)
        ss_res = float(np.sum((y - predicted) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return {
            "floor": float(params[0]),
            "amplitude": float(params[1]),
            "exponent": float(params[2]),
            "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan,
        }
    except (ImportError, RuntimeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    data_runs = resolve_data_runs(args.data_run)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
        }
    )

    compute_rows, compute_audit = load_rows([args.compute_run.resolve()])
    compute_epoch_length = int(compute_rows[-1]["official_epoch_length"])
    compute_table: list[dict[str, Any]] = []
    for passes in COMPUTE_PASSES:
        checkpoint = passes * compute_epoch_length - 1
        window = max(5, round(0.01 * checkpoint))
        row: dict[str, Any] = {
            "passes": passes,
            "checkpoint": checkpoint,
            "optimizer_updates_completed": checkpoint + 1,
            "image_visits": (checkpoint + 1) * int(compute_rows[-1]["effective_global_batch_size"]),
            "complete": int(checkpoint <= int(compute_rows[-1]["optimizer_update"])),
        }
        for component in COMPONENTS:
            mean, sem, n = trailing_mean(compute_rows, checkpoint, window, component)
            row[f"{component}_mean"] = mean
            row[f"{component}_sem"] = sem
            row[f"{component}_n"] = n
        compute_table.append(row)
    compute_fields = list(compute_table[0])
    write_csv(output_dir / f"{args.output_prefix}_compute_c_loss_scaling.csv", compute_table, compute_fields)

    data_rows: dict[str, list[dict[str, Any]]] = {}
    data_audits: dict[str, Any] = {}
    data_table: list[dict[str, Any]] = []
    for label, spec in data_runs.items():
        paths = tuple(Path(path) for path in spec["paths"])
        rows, audit = load_rows(paths)
        data_rows[label] = rows
        data_audits[label] = audit
        epoch_length = int(rows[-1]["official_epoch_length"])
        final_checkpoint = 15 * epoch_length - 1
        last_update = int(rows[-1]["optimizer_update"])
        window = max(5, round(0.01 * (final_checkpoint + 1)))
        row = {
            "data_label": label,
            "unique_images": int(spec["unique_images"]),
            "official_epoch_length": epoch_length,
            "final_checkpoint": final_checkpoint,
            "last_observed_update": last_update,
            "protocol_passes_observed": (last_update + 1) / epoch_length,
            "image_visits_observed": (last_update + 1) * int(rows[-1]["effective_global_batch_size"]),
            "complete": int(last_update >= final_checkpoint),
        }
        for component in COMPONENTS:
            if last_update >= final_checkpoint:
                mean, sem, n = trailing_mean(rows, final_checkpoint, window, component)
            else:
                mean, sem, n = math.nan, math.nan, 0
            row[f"{component}_mean"] = mean
            row[f"{component}_sem"] = sem
            row[f"{component}_n"] = n
        data_table.append(row)
    data_fields = list(data_table[0])
    write_csv(output_dir / f"{args.output_prefix}_data_d_loss_scaling.csv", data_table, data_fields)

    colors = {"0.1M": "#C8553D", "0.2M": "#E6A33D", "0.5M": "#3E7C78", "1M": "#173F5F"}
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.35))

    compute_updates = np.asarray([row["optimizer_updates_completed"] for row in compute_rows], dtype=float)
    compute_losses = np.asarray([row["total_loss"] for row in compute_rows], dtype=float)
    compute_window = max(5, round(0.01 * len(compute_losses)))
    axes[0].plot(
        compute_updates * int(compute_rows[-1]["effective_global_batch_size"]) / 1e6,
        rolling_mean(compute_losses, compute_window),
        color="#173F5F",
        linewidth=2.2,
    )
    complete_compute = [row for row in compute_table if row["complete"]]
    axes[0].scatter(
        [row["image_visits"] / 1e6 for row in complete_compute],
        [row["total_loss_mean"] for row in complete_compute],
        color="#C8553D",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    axes[0].set_title("A  Compute C: 1M-image pool")
    axes[0].set_xlabel("Image visits (millions)")
    axes[0].set_ylabel("Smoothed training objective")

    for label in data_runs:
        rows = data_rows[label]
        epoch_length = int(rows[-1]["official_epoch_length"])
        x = np.asarray([row["optimizer_updates_completed"] / epoch_length for row in rows])
        y = np.asarray([row["total_loss"] for row in rows])
        window = max(5, round(0.01 * 15 * epoch_length))
        axes[1].plot(x, rolling_mean(y, window), color=colors[label], linewidth=2, label=label)
        if data_table[list(data_runs).index(label)]["complete"]:
            axes[1].scatter([15], [data_table[list(data_runs).index(label)]["total_loss_mean"]],
                            color=colors[label], edgecolor="white", linewidth=0.7, zorder=3)
    axes[1].set_xlim(0, 15.25)
    axes[1].set_title("B  Data D: fixed 15 passes")
    axes[1].set_xlabel("Training passes (protocol)")
    axes[1].set_ylabel("Smoothed training objective")
    axes[1].legend(frameon=False, ncol=2)

    complete_data = [row for row in data_table if row["complete"]]
    x_data = np.asarray([row["unique_images"] for row in complete_data], dtype=float)
    y_data = np.asarray([row["total_loss_mean"] for row in complete_data], dtype=float)
    y_err = np.asarray([row["total_loss_sem"] for row in complete_data], dtype=float)
    axes[2].errorbar(
        x_data,
        y_data,
        yerr=y_err,
        color="#173F5F",
        marker="o",
        markersize=6,
        linewidth=2,
        capsize=3,
    )
    data_fit = fit_power_law(x_data, y_data) if len(complete_data) == len(data_runs) else None
    data_log_linear_fit = fit_log_linear(x_data, y_data) if len(complete_data) == len(data_runs) else None
    if data_fit:
        grid = np.geomspace(x_data.min(), x_data.max(), 200)
        fitted = data_fit["floor"] + data_fit["amplitude"] * np.power(grid / 1e6, -data_fit["exponent"])
        axes[2].plot(grid, fitted, color="#C8553D", linestyle="--", linewidth=1.6,
                     label=rf"$L(D)=E+A D^{{-{data_fit['exponent']:.2f}}}$")
        axes[2].legend(frameon=False)
    elif len(complete_data) < len(data_runs):
        axes[2].text(0.04, 0.06, "0.5M final point pending\nfit intentionally withheld",
                     transform=axes[2].transAxes, color="#7A4E2D", fontsize=9)
    axes[2].set_xscale("log")
    axes[2].set_title("C  Terminal loss vs. unique data")
    axes[2].set_xlabel("Unique training images")
    axes[2].set_ylabel("15-pass terminal objective")
    fig.suptitle(f"{args.figure_title} — training-objective scaling", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, f"{args.output_prefix}_loss_scaling_c_d", args.dpi)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), sharex=True)
    component_styles = {
        "dino_local_crops_loss": ("DINO local", "#173F5F"),
        "dino_global_crops_loss": ("DINO global", "#3E7C78"),
        "ibot_loss": ("iBOT", "#C8553D"),
        "koleo_loss": ("KoLeo", "#E6A33D"),
        "sigreg_loss": ("SIGReg", "#E6A33D"),
    }
    available_components = [
        component for component in component_styles if all(component in row for row in compute_rows)
    ][:4]
    for axis, component in zip(axes.flat, available_components):
        title, color = component_styles[component]
        values = np.asarray([row[component] for row in compute_rows])
        axis.plot(compute_updates / compute_epoch_length, rolling_mean(values, compute_window),
                  color=color, linewidth=2)
        axis.set_title(title)
        axis.set_ylabel("Loss component")
        axis.set_xlabel("Passes through 1M pool")
    for axis in axes.flat[len(available_components):]:
        axis.set_visible(False)
    fig.suptitle(f"{args.figure_title} Compute C — objective components", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, f"{args.output_prefix}_compute_c_loss_components", args.dpi)

    manifest = {
        "status": "complete" if all(row["complete"] for row in data_table) else "partial",
        "recipe": args.recipe,
        "protocol": {
            "compute_c": "1M unique-image pool; 1/2/4/6/8/10/12/15 passes",
            "data_d": "0.1/0.2/0.5/1M unique-image pools; fixed 15 passes",
            "terminal_loss": "mean over final 1% of optimizer updates; SEM is temporal, not seed uncertainty",
            "warning": "These are self-supervised training-objective losses, not held-out downstream losses.",
        },
        "compute_audit": compute_audit,
        "data_audits": data_audits,
        "data_fit": data_fit,
        "data_log_linear_fit": data_log_linear_fit,
        "available_components": [
            component for component in COMPONENTS if all(component in row for row in compute_rows)
        ],
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / "manifest.json")
    print(json.dumps({"output_dir": str(output_dir), "status": manifest["status"], "data_fit": data_fit}))


if __name__ == "__main__":
    main()
