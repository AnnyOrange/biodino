#!/usr/bin/env python3
"""Plot the matched H100 model-size scaling campaign from saved summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = ("B", "L", "H+", "7B")
PARAMETERS_M = {"B": 86, "L": 300, "H+": 840, "7B": 7000}
OBJECTIVES = ("sigreg", "nosigreg")
STAGES = ("raw", "alpha")
COLORS = {"sigreg": "#D64B32", "nosigreg": "#178C86"}
LABELS = {"sigreg": "SigReg", "nosigreg": "No SigReg"}
METRICS = (
    ("family6_equal_mean", "Family-6 overall"),
    ("c25_macro_f1", "Classification (macro-F1)"),
    ("bbbc005_r2", "Regression (R2)"),
    ("retrieval4_map_at_5", "Retrieval (mAP@5)"),
    ("clustering4_nmi", "Clustering (NMI)"),
    ("segmentation8_mdice", "Segmentation (mDice)"),
    ("livecell_detection_f1", "Detection (F1)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("/data_2/suxin/runs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/00_reports/h100_sigreg_scaling_law_20260803"),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write a progress preview even when some model/objective summaries are missing.",
    )
    return parser.parse_args()


def alpha_root(run_root: Path, model: str, objective: str) -> Path:
    if model == "B":
        names = {
            "sigreg": "Binterp_official_sigreg_auto_best_alpha_tune_20260725",
            "nosigreg": "Binterp_official_nosigreg_bestck11274_alpha_sweep_20260725",
        }
        return run_root / names[objective]
    if model == "L":
        names = {
            "sigreg": "Linterp_official_sigreg_auto_best_alpha_tune_20260725",
            "nosigreg": "Linterp_official_nosigreg_auto_best_alpha_tune_20260725",
        }
        return run_root / names[objective]
    campaign = run_root / "h100_hplus_7b_sigreg_ab_tuning_20260725"
    campaign_model = "hplus" if model == "H+" else "7b"
    return campaign / f"{campaign_model}_{objective}" / "alpha_tune"


def summary_path(run_root: Path, model: str, objective: str, stage: str) -> Path:
    root = alpha_root(run_root, model, objective)
    relative = "raw_sweep/best.json" if stage == "raw" else "full_summary/best.json"
    return root / relative


def load_rows(run_root: Path) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    required_metrics = {metric for metric, _ in METRICS}
    for model in MODELS:
        for objective in OBJECTIVES:
            for stage in STAGES:
                path = summary_path(run_root, model, objective, stage)
                if not path.is_file():
                    missing.append(f"{model}/{objective}/{stage}: {path}")
                    continue
                with path.open() as handle:
                    summary = json.load(handle)
                absent = sorted(required_metrics - summary.keys())
                if absent:
                    raise KeyError(f"{path} is missing metrics: {', '.join(absent)}")
                metric_values: dict[str, float] = {}
                for metric in required_metrics:
                    try:
                        value = float(summary[metric])
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"{path} has a non-numeric {metric}: {summary[metric]!r}") from error
                    if not math.isfinite(value):
                        raise ValueError(f"{path} has a non-finite {metric}: {value}")
                    metric_values[metric] = value
                if stage == "alpha":
                    try:
                        alpha = float(summary["alpha"])
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(f"{path} is missing a numeric alpha") from error
                    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
                        raise ValueError(f"{path} has an invalid alpha outside [0, 1]: {alpha}")
                row: dict[str, object] = {
                    "model": model,
                    "parameters_millions": PARAMETERS_M[model],
                    "log10_parameters_millions": math.log10(PARAMETERS_M[model]),
                    "objective": objective,
                    "stage": stage,
                    "checkpoint": summary.get("checkpoint", ""),
                    "alpha": summary.get("alpha", ""),
                    "alpha_checkpoint": summary.get("alpha_checkpoint", ""),
                    "source": str(path),
                }
                row.update(metric_values)
                rows.append(row)
    return rows, missing


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "model",
        "parameters_millions",
        "log10_parameters_millions",
        "objective",
        "stage",
        "checkpoint",
        "alpha",
        "alpha_checkpoint",
        *(metric for metric, _ in METRICS),
        "source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Serif",
            "axes.edgecolor": "#1A1A1A",
            "axes.labelcolor": "#1A1A1A",
            "text.color": "#1A1A1A",
            "xtick.color": "#1A1A1A",
            "ytick.color": "#1A1A1A",
            "axes.grid": True,
            "grid.color": "#D9D5CD",
            "grid.alpha": 0.65,
            "grid.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot_metric(axis: plt.Axes, rows: list[dict[str, object]], metric: str, title: str) -> None:
    for objective in OBJECTIVES:
        for stage in STAGES:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["objective"] == objective and row["stage"] == stage
                ),
                key=lambda row: float(row["parameters_millions"]),
            )
            if not selected:
                continue
            x = [math.log10(float(row["parameters_millions"])) for row in selected]
            y = [float(row[metric]) for row in selected]
            label = f"{LABELS[objective]} - {'alpha-tuned' if stage == 'alpha' else 'raw'}"
            axis.plot(
                x,
                y,
                color=COLORS[objective],
                linestyle="-" if stage == "alpha" else (0, (4, 3)),
                linewidth=2.0 if stage == "alpha" else 1.2,
                marker="o" if stage == "alpha" else "s",
                markersize=6.2 if stage == "alpha" else 4.8,
                markerfacecolor=COLORS[objective] if stage == "alpha" else "white",
                markeredgecolor=COLORS[objective],
                label=label,
                zorder=3 if stage == "alpha" else 2,
            )
    positions = [math.log10(PARAMETERS_M[model]) for model in MODELS]
    axis.set_xticks(positions, MODELS)
    axis.set_xlim(positions[0] - 0.13, positions[-1] + 0.13)
    axis.set_title(title, fontsize=11.5, pad=7)
    axis.set_xlabel("Model size")
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(figure: plt.Figure, base: Path) -> None:
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(
            base.with_suffix(f".{suffix}"),
            dpi=280 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def make_figures(rows: list[dict[str, object]], output_dir: Path) -> None:
    configure_style()
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    plot_metric(axis, rows, METRICS[0][0], METRICS[0][1])
    axis.set_ylabel("Equal-weight mean")
    axis.legend(frameon=False, ncol=2, fontsize=8.6, loc="best")
    figure.tight_layout()
    save_figure(figure, output_dir / "h100_sigreg_scaling_overall")

    figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.0))
    for axis, (metric, title) in zip(axes.flat, METRICS[1:]):
        plot_metric(axis, rows, metric, title)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    figure.subplots_adjust(left=0.065, right=0.985, top=0.95, bottom=0.105, wspace=0.30, hspace=0.38)
    save_figure(figure, output_dir / "h100_sigreg_scaling_tasks")


def main() -> None:
    args = parse_args()
    rows, missing = load_rows(args.run_root)
    if missing and not args.allow_partial:
        raise SystemExit("Incomplete campaign:\n" + "\n".join(f"- {item}" for item in missing))
    if not rows:
        raise SystemExit("No campaign summaries were found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "h100_sigreg_scaling_values.csv")
    manifest = {
        "complete": not missing,
        "expected_rows": len(MODELS) * len(OBJECTIVES) * len(STAGES),
        "available_rows": len(rows),
        "missing": missing,
    }
    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    make_figures(rows, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
