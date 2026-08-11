#!/usr/bin/env python3
"""Build S6-lineage data, model, and compute scaling figures from saved summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("family6_equal_mean", "Six-family overall"),
    ("c25_macro_f1", "Classification"),
    ("bbbc005_r2", "Regression"),
    ("retrieval4_map_at_5", "Retrieval"),
    ("clustering4_nmi", "Clustering"),
    ("segmentation8_mdice", "Segmentation"),
    ("livecell_detection_f1", "Detection"),
)
COLORS = {"sigreg": "#D34A2C", "nosigreg": "#087E8B"}
OBJECTIVE_LABELS = {"sigreg": "SigReg", "nosigreg": "No SigReg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-csv", type=Path)
    parser.add_argument("--compute-csv", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def as_float(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return number


def load_csv_selection(path: Path, selector: str) -> dict[str, object]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if selector in next(iter(row.values()), "")]
    if len(matches) != 1:
        raise ValueError(f"Expected one row containing {selector!r} in {path}; got {len(matches)}")
    return dict(matches[0])


def load_source(row: dict[str, str]) -> dict[str, object]:
    path = Path(row["source_path"])
    source_type = row.get("source_type", "json")
    if source_type == "json":
        with path.open() as handle:
            return json.load(handle)
    if source_type == "csv":
        return load_csv_selection(path, row.get("selector", ""))
    raise ValueError(f"Unsupported source_type={source_type!r}")


def metric_values(source: dict[str, object]) -> dict[str, float]:
    values = {metric: as_float(source[metric], metric) for metric, _ in METRICS[1:]}
    values["family6_equal_mean"] = as_float(
        source.get("family6_equal_mean", np.mean(list(values.values()))),
        "family6_equal_mean",
    )
    return values


def load_rows(manifest: Path, allow_partial: bool) -> tuple[list[dict[str, object]], list[str]]:
    with manifest.open(newline="") as handle:
        specs = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for spec in specs:
        path = Path(spec["source_path"])
        if not path.is_file():
            missing.append(f'{spec["model"]}/{spec["objective"]}/{spec["stage"]}: {path}')
            continue
        source = load_source(spec)
        checkpoint_value = spec.get("checkpoint_override") or source.get("checkpoint", "")
        alpha_value = spec.get("alpha_override") or source.get("alpha", "")
        checkpoint = int(float(checkpoint_value)) if checkpoint_value != "" else None
        alpha = as_float(alpha_value, "alpha") if alpha_value != "" else None
        row: dict[str, object] = {
            "model": spec["model"],
            "parameters_millions": as_float(spec["parameters_millions"], "parameters_millions"),
            "objective": spec["objective"],
            "stage": spec["stage"],
            "checkpoint": checkpoint,
            "alpha": alpha,
            "alpha_checkpoint": source.get("alpha_checkpoint", ""),
            "source": str(path),
        }
        row.update(metric_values(source))
        rows.append(row)

    raw_checkpoints = {
        (str(row["model"]), str(row["objective"])): row["checkpoint"]
        for row in rows
        if row["stage"] == "raw" and row["checkpoint"] is not None
    }
    for row in rows:
        if row["checkpoint"] is None:
            row["checkpoint"] = raw_checkpoints.get((str(row["model"]), str(row["objective"])))
        checkpoint = row["checkpoint"]
        if checkpoint is not None:
            visits = (int(checkpoint) + 1) * 1024
            row["image_visits"] = visits
            row["sweetspot_epochs"] = (int(checkpoint) + 1) / 1025
            row["parameter_images"] = float(row["parameters_millions"]) * 1e6 * visits
        else:
            row["image_visits"] = ""
            row["sweetspot_epochs"] = ""
            row["parameter_images"] = ""

    if missing and not allow_partial:
        raise SystemExit("Incomplete manifest:\n" + "\n".join(f"- {item}" for item in missing))
    return rows, missing


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Serif",
            "axes.edgecolor": "#26231F",
            "axes.labelcolor": "#26231F",
            "text.color": "#26231F",
            "xtick.color": "#26231F",
            "ytick.color": "#26231F",
            "axes.grid": True,
            "grid.color": "#DDD8CF",
            "grid.alpha": 0.72,
            "grid.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, base: Path) -> None:
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(base.with_suffix(f".{suffix}"), dpi=280 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)


def selected_rows(rows: list[dict[str, object]], objective: str, stage: str) -> list[dict[str, object]]:
    return sorted(
        (row for row in rows if row["objective"] == objective and row["stage"] == stage),
        key=lambda row: float(row["parameters_millions"]),
    )


def plot_metric(axis: plt.Axes, rows: list[dict[str, object]], metric: str, title: str) -> None:
    for objective in ("sigreg", "nosigreg"):
        for stage in ("raw", "alpha"):
            selected = selected_rows(rows, objective, stage)
            if not selected:
                continue
            x = np.log10([float(row["parameters_millions"]) for row in selected])
            y = [float(row[metric]) for row in selected]
            axis.plot(
                x,
                y,
                color=COLORS[objective],
                linestyle="-" if stage == "alpha" else (0, (4, 3)),
                marker="o" if stage == "alpha" else "s",
                markerfacecolor=COLORS[objective] if stage == "alpha" else "white",
                markeredgecolor=COLORS[objective],
                linewidth=2.1 if stage == "alpha" else 1.2,
                markersize=6.0,
                label=f'{OBJECTIVE_LABELS[objective]} - {"alpha" if stage == "alpha" else "raw"}',
            )
            for xx, yy, row in zip(x, y, selected):
                axis.annotate(str(row["model"]), (xx, yy), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5)
    axis.set_title(title, fontsize=11)
    axis.set_xlabel("Model parameters (log scale)")
    axis.spines[["top", "right"]].set_visible(False)


def make_model_figures(rows: list[dict[str, object]], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    plot_metric(axis, rows, "family6_equal_mean", "S6-lineage model scaling at each downstream sweet spot")
    axis.set_ylabel("Equal-weight six-family score")
    axis.legend(frameon=False, ncol=2, fontsize=8.5)
    figure.tight_layout()
    save_figure(figure, output_dir / "s6_model_scaling_overall")

    figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.1))
    for axis, (metric, title) in zip(axes.flat, METRICS[1:]):
        plot_metric(axis, rows, metric, title)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    figure.subplots_adjust(left=0.065, right=0.985, top=0.95, bottom=0.105, wspace=0.30, hspace=0.38)
    save_figure(figure, output_dir / "s6_model_scaling_tasks")


def best_sweetspot_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for row in rows:
        model = str(row["model"])
        if model not in best or float(row["family6_equal_mean"]) > float(best[model]["family6_equal_mean"]):
            best[model] = row
    return sorted(best.values(), key=lambda row: float(row["parameters_millions"]))


def make_best_sweetspot_figure(rows: list[dict[str, object]], output_dir: Path) -> None:
    selected = best_sweetspot_rows(rows)
    if not selected:
        return
    x = np.log10([float(row["parameters_millions"]) for row in selected])
    y = np.array([float(row["family6_equal_mean"]) for row in selected])
    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    axis.plot(x, y, color="#B6402A", marker="o", linewidth=2.4, markersize=7.2, label="Best validated sweet spot")
    if len(selected) >= 3:
        slope, intercept = np.polyfit(x, y, 1)
        prediction = intercept + slope * x
        ss_res = float(np.sum((y - prediction) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        axis.plot(x, prediction, color="#55504A", linestyle=(0, (4, 3)), linewidth=1.2, label=f"log-linear fit (R2={r2:.2f})")
    for xx, yy, row in zip(x, y, selected):
        alpha = row["alpha"]
        suffix = f"alpha={float(alpha):.2f}" if alpha not in (None, "") else "raw"
        objective = OBJECTIVE_LABELS[str(row["objective"])]
        axis.annotate(
            f'{row["model"]}\n{objective}, {suffix}\n{yy:.4f}',
            (xx, yy),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.2,
        )
    span = max(float(np.ptp(y)), 0.002)
    axis.set_ylim(float(np.min(y)) - 0.15 * span, float(np.max(y)) + 0.38 * span)
    axis.set_title("Best validated S6-lineage sweet spot by model size", fontsize=12, pad=12)
    axis.set_xlabel("Model parameters (log scale)")
    axis.set_ylabel("Equal-weight six-family score")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8.8)
    figure.tight_layout()
    save_figure(figure, output_dir / "s6_model_scaling_best_sweetspots")

    fields = [
        "model",
        "parameters_millions",
        "objective",
        "stage",
        "checkpoint",
        "alpha",
        "family6_equal_mean",
        "source",
    ]
    with (output_dir / "s6_model_scaling_best_sweetspots.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)


def load_numeric_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make_three_axis_figure(
    rows: list[dict[str, object]], data_csv: Path | None, compute_csv: Path | None, output_dir: Path
) -> None:
    if not data_csv or not compute_csv or not data_csv.is_file() or not compute_csv.is_file():
        return
    data = load_numeric_csv(data_csv)
    compute = load_numeric_csv(compute_csv)
    alpha_model = selected_rows(rows, "sigreg", "alpha")
    raw_model = selected_rows(rows, "sigreg", "raw")
    model = alpha_model if len(alpha_model) >= len(raw_model) else raw_model

    figure, axes = plt.subplots(1, 4, figsize=(17.0, 4.4))
    dx = [as_float(row["samples"], "samples") for row in data]
    dy = [as_float(row["five_family_mean"], "five_family_mean") for row in data]
    axes[0].plot(dx, dy, color="#C95A31", marker="o", linewidth=2)
    axes[0].set_xscale("log")
    axes[0].set_title("Data scaling D")
    axes[0].set_xlabel("Unique microscopy images")
    axes[0].set_ylabel("Five-family mean")

    cx = [as_float(row["image_visits_m"], "image_visits_m") for row in compute]
    cy = [as_float(row["five_family_mean"], "five_family_mean") for row in compute]
    axes[1].plot(cx, cy, color="#D19A19", marker="o", linewidth=2)
    axes[1].set_title("Compute scaling C | S+")
    axes[1].set_xlabel("Image visits (millions)")
    axes[1].set_ylabel("Five-family mean")

    if model:
        nx = [float(row["parameters_millions"]) for row in model]
        ny = [float(row["family6_equal_mean"]) for row in model]
        axes[2].plot(nx, ny, color=COLORS["sigreg"], marker="o", linewidth=2)
        axes[2].set_xscale("log")
        for xx, yy, row in zip(nx, ny, model):
            axes[2].annotate(str(row["model"]), (xx, yy), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        axes[2].set_title("Model scaling N")
        axes[2].set_xlabel("Parameters (millions)")
        axes[2].set_ylabel("Six-family mean")

        compute_model = [row for row in model if row["parameter_images"] != ""]
        mx = [float(row["parameter_images"]) for row in compute_model]
        my = [float(row["family6_equal_mean"]) for row in compute_model]
        axes[3].plot(mx, my, color="#216869", marker="o", linewidth=2)
        axes[3].set_xscale("log")
        for xx, yy, row in zip(mx, my, compute_model):
            axes[3].annotate(str(row["model"]), (xx, yy), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        axes[3].set_title("Sweet-spot compute N x visits")
        axes[3].set_xlabel("Parameter-images")
        axes[3].set_ylabel("Six-family mean")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("BioDINOv3 S6 scaling: data, optimization compute, and model size", fontsize=13, y=1.02)
    figure.tight_layout()
    save_figure(figure, output_dir / "s6_data_model_compute_scaling")


def write_rows(rows: list[dict[str, object]], output: Path) -> None:
    fields = [
        "model",
        "parameters_millions",
        "objective",
        "stage",
        "checkpoint",
        "sweetspot_epochs",
        "image_visits",
        "parameter_images",
        "alpha",
        "alpha_checkpoint",
        *(metric for metric, _ in METRICS),
        "source",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_fits(rows: list[dict[str, object]], output: Path) -> None:
    records: list[dict[str, object]] = []
    for x_name in ("parameters_millions", "parameter_images"):
        for objective in ("sigreg", "nosigreg"):
            for stage in ("raw", "alpha"):
                selected = [row for row in selected_rows(rows, objective, stage) if row[x_name] != ""]
                if len(selected) < 3:
                    continue
                x = np.log10([float(row[x_name]) for row in selected])
                for metric, _ in METRICS:
                    y = np.array([float(row[metric]) for row in selected])
                    slope, intercept = np.polyfit(x, y, 1)
                    pred = intercept + slope * x
                    ss_res = float(np.sum((y - pred) ** 2))
                    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                    records.append(
                        {
                            "x": x_name,
                            "objective": objective,
                            "stage": stage,
                            "metric": metric,
                            "points": len(selected),
                            "intercept": intercept,
                            "slope_per_log10": slope,
                            "r2": r2,
                        }
                    )
    fields = ["x", "objective", "stage", "metric", "points", "intercept", "slope_per_log10", "r2"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    rows, missing = load_rows(args.manifest, args.allow_partial)
    if not rows:
        raise SystemExit("No available summary rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    write_rows(rows, args.output_dir / "s6_scaling_values.csv")
    write_fits(rows, args.output_dir / "s6_scaling_fits.csv")
    make_model_figures(rows, args.output_dir)
    make_best_sweetspot_figure(rows, args.output_dir)
    make_three_axis_figure(rows, args.data_csv, args.compute_csv, args.output_dir)
    status = {
        "complete": not missing,
        "available_rows": len(rows),
        "missing_rows": len(missing),
        "missing": missing,
        "note": "Model and parameter-image fits use downstream-selected S6 sweet spots; they are descriptive, not fixed-compute causal exponents.",
    }
    (args.output_dir / "manifest_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
