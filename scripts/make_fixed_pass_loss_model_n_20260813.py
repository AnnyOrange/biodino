#!/usr/bin/env python3
"""Build a descriptive 1M/15-pass model-size loss comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    ("S+", 22, "SIGReg=0.05", "S/splus_data_d_loss_scaling.csv", "S/manifest.json"),
    ("B", 86, "SIGReg=0.05", "B/b_data_d_loss_scaling.csv", "B/manifest.json"),
    ("L", 300, "SIGReg=0.05", "L/l_data_d_loss_scaling.csv", "L/manifest.json"),
    ("H+", 840, "no-SIGReg", "Hplus/hplus_data_d_loss_scaling.csv", "Hplus/manifest.json"),
)
COMPONENTS = (
    "total_loss_mean",
    "dino_local_crops_loss_mean",
    "dino_global_crops_loss_mean",
    "ibot_loss_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, params_m, objective, csv_name, manifest_name in MODELS:
        with (args.input_root / csv_name).open(newline="") as handle:
            candidates = list(csv.DictReader(handle))
        row = next(item for item in candidates if item["data_label"] in {"1M", "1.0M"})
        manifest = json.loads((args.input_root / manifest_name).read_text())
        output = {
            "model": model,
            "params_m": params_m,
            "data_images": int(row["unique_images"]),
            "passes": float(row["protocol_passes_observed"]),
            "objective_variant": objective,
            "recipe": manifest["recipe"],
        }
        for component in COMPONENTS:
            output[component] = float(row[component])
        rows.append(output)

    output_csv = args.output_dir / "model_n_terminal_loss_1m_15pass.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = ("#C8553D", "#E6A33D", "#3E7C78", "#173F5F")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    panels = (
        ("dino_local_crops_loss_mean", "DINO local"),
        ("dino_global_crops_loss_mean", "DINO global"),
        ("ibot_loss_mean", "iBOT"),
    )
    for axis, (key, title) in zip(axes, panels):
        for row, color in zip(rows, colors):
            axis.scatter(row["params_m"], row[key], s=62, color=color, label=row["model"])
            axis.annotate(row["model"], (row["params_m"], row[key]), xytext=(4, 3),
                          textcoords="offset points", fontsize=9)
        axis.set_xscale("log")
        axis.set_xlabel("Parameters (millions)")
        axis.set_ylabel("15-pass terminal loss")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    fig.suptitle("Model N: shared objective components at 1M images / 15 passes", fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.output_dir / "model_n_shared_loss_components.png", dpi=240, bbox_inches="tight")
    fig.savefig(args.output_dir / "model_n_shared_loss_components.pdf", bbox_inches="tight")
    plt.close(fig)

    # Total loss is still useful as a selected-recipe diagnostic even though
    # H+ omits the SIGReg term used by S+/B/L.
    x = np.log10([row["params_m"] for row in rows])
    y = np.asarray([row["total_loss_mean"] for row in rows], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    prediction = intercept + slope * x
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else float("nan")

    fig, axis = plt.subplots(figsize=(7.4, 4.6))
    axis.plot(x, y, color="#173F5F", linewidth=2.2, marker="o", markersize=7.5)
    axis.plot(x, prediction, color="#756B5E", linewidth=1.2, linestyle=(0, (4, 3)),
              label=f"descriptive log-linear fit (R2={r_squared:.3f})")
    for xx, yy, row, color in zip(x, y, rows, colors):
        axis.scatter(xx, yy, s=72, color=color, zorder=3)
        axis.annotate(
            f'{row["model"]}\n{yy:.3f}',
            (xx, yy),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.set_xticks(x, [f'{row["model"]}\n{row["params_m"]}M' for row in rows])
    axis.set_xlabel("Model parameters (log scale)")
    axis.set_ylabel("Terminal total training loss")
    axis.set_title("Model N: selected-recipe total loss at 1M images / 15 passes", fontweight="bold")
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=9)
    fig.text(
        0.5,
        -0.015,
        "Descriptive only: S+/B/L use SIGReg=0.05; H+ uses no-SIGReg.",
        ha="center",
        fontsize=8.5,
        color="#5E574F",
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "model_n_selected_recipe_total_loss.png", dpi=240, bbox_inches="tight")
    fig.savefig(args.output_dir / "model_n_selected_recipe_total_loss.pdf", bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "status": "complete",
        "protocol": "1M unique images, 15 passes",
        "warning": (
            "Total loss is descriptive only: S+/B/L include SIGReg=0.05 while H+ uses no-SIGReg. "
            "The displayed log-linear fit summarizes these four selected recipes; it is not a "
            "controlled or causal scaling exponent."
        ),
        "total_loss_descriptive_fit": {
            "slope_per_log10_parameters": float(slope),
            "intercept": float(intercept),
            "r_squared": r_squared,
        },
        "outputs": [
            output_csv.name,
            "model_n_shared_loss_components.png",
            "model_n_shared_loss_components.pdf",
            "model_n_selected_recipe_total_loss.png",
            "model_n_selected_recipe_total_loss.pdf",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
