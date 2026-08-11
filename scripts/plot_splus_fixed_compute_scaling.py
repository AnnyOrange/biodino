#!/usr/bin/env python3
"""Build the final S+ fixed-compute dataset and checkpoint scaling curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHECKPOINT_PASSES = {
    1024: 1,
    2049: 2,
    4099: 4,
    6149: 6,
    8199: 8,
    10249: 10,
    12299: 12,
    15374: 15,
}
CLASSIFICATION_DATASETS = (
    "chammi-allen-task1",
    "chammi-allen-task2",
    "chammi-cp-task1",
    "chammi-cp-task2",
    "chammi-cp-task3",
    "chammi-hpa-task1",
    "chammi-hpa-task2",
)
EXPECTED_KEYS = {
    *(('classification', dataset) for dataset in CLASSIFICATION_DATASETS),
    ("regression", "bbbc005"),
    ("retrieval_clustering", "lc25000"),
}
COLORS = {
    "0.1M": "#164E63",
    "0.2M": "#0F766E",
    "0.5M": "#D97706",
    "1.0M": "#B42318",
}


@dataclass(frozen=True)
class RunSpec:
    label: str
    pool_m: float
    root: Path


def parse_run(value: str) -> RunSpec:
    try:
        label, pool, root = value.split("=", 2)
        return RunSpec(label=label, pool_m=float(pool), root=Path(root))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected LABEL=POOL_M=EVAL_DIR, for example 0.1M=0.1=outputs/run"
        ) from exc


def checkpoint_id(result: dict, path: Path) -> int:
    checkpoint = str(result.get("checkpoint", ""))
    match = re.search(r"/ckpt/(\d+)(?:/checkpoint\.pth)?$", checkpoint)
    if match:
        return int(match.group(1))
    model = str(result.get("model", ""))
    match = re.search(r"(\d+)$", model)
    if match:
        return int(match.group(1))
    for part in reversed(path.parts):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot infer checkpoint from {path}")


def metric_for(result: dict) -> tuple[str, float] | None:
    task = str(result.get("task", ""))
    if task == "classification" and "balanced_accuracy" in result:
        return "balanced_accuracy", float(result["balanced_accuracy"])
    if task == "regression" and "r2" in result:
        return "r2", float(result["r2"])
    if task == "retrieval_clustering" and "nmi" in result:
        return "nmi", float(result["nmi"])
    return None


def load_run(spec: RunSpec) -> list[dict]:
    if not spec.root.is_dir():
        raise FileNotFoundError(spec.root)
    rows: list[dict] = []
    seen: set[tuple[int, str, str]] = set()
    for path in sorted(spec.root.rglob("last_result.json")):
        result = json.loads(path.read_text())
        metric = metric_for(result)
        if metric is None:
            continue
        checkpoint = checkpoint_id(result, path)
        if checkpoint not in CHECKPOINT_PASSES:
            continue
        task = str(result["task"])
        dataset = str(result["dataset"])
        if (task, dataset) not in EXPECTED_KEYS:
            continue
        key = (checkpoint, task, dataset)
        if key in seen:
            raise ValueError(f"Duplicate result for {spec.label} {key}: {path}")
        seen.add(key)
        metric_name, value = metric
        rows.append(
            {
                "pool": spec.label,
                "pool_m": spec.pool_m,
                "checkpoint": checkpoint,
                "passes": CHECKPOINT_PASSES[checkpoint],
                "task": task,
                "dataset": dataset,
                "metric": metric_name,
                "value": value,
                "result_path": str(path),
            }
        )

    missing = []
    for checkpoint in CHECKPOINT_PASSES:
        for task, dataset in EXPECTED_KEYS:
            if (checkpoint, task, dataset) not in seen:
                missing.append(f"ck{checkpoint}:{task}/{dataset}")
    if missing:
        preview = ", ".join(missing[:12])
        raise ValueError(
            f"{spec.label} has {len(rows)}/72 expected results; missing {preview}"
        )
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float, int, int], list[dict]] = {}
    for row in rows:
        key = (row["pool"], row["pool_m"], row["checkpoint"], row["passes"])
        grouped.setdefault(key, []).append(row)

    output = []
    for (pool, pool_m, checkpoint, passes), values in grouped.items():
        classification = [
            row["value"] for row in values if row["task"] == "classification"
        ]
        regression = [row["value"] for row in values if row["task"] == "regression"]
        retrieval = [
            row["value"] for row in values if row["task"] == "retrieval_clustering"
        ]
        if len(classification) != 7 or len(regression) != 1 or len(retrieval) != 1:
            raise ValueError(f"Unexpected task coverage for {pool} checkpoint {checkpoint}")
        all_values = classification + regression + retrieval
        output.append(
            {
                "pool": pool,
                "pool_m": pool_m,
                "checkpoint": checkpoint,
                "passes": passes,
                "classification7_balanced_accuracy": sum(classification) / 7,
                "bbbc005_r2": regression[0],
                "lc25000_nmi": retrieval[0],
                "proxy9_mean": sum(all_values) / 9,
            }
        )
    return sorted(output, key=lambda row: (row["pool_m"], row["passes"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#D7D2C8", linewidth=0.7, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors="#2F3437")


def plot_curves(aggregate: list[dict], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#202528",
            "text.color": "#202528",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    figure.patch.set_facecolor("#F6F1E7")
    for axis in axes.flat:
        axis.set_facecolor("#FBF8F1")
        style_axis(axis)

    by_pool: dict[str, list[dict]] = {}
    for row in aggregate:
        by_pool.setdefault(row["pool"], []).append(row)
    for label, values in sorted(by_pool.items(), key=lambda item: item[1][0]["pool_m"]):
        values.sort(key=lambda row: row["passes"])
        axes[0, 0].plot(
            [row["passes"] for row in values],
            [row["proxy9_mean"] for row in values],
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=COLORS.get(label),
            label=label,
        )
    axes[0, 0].set_title("Compute scaling within each data pool")
    axes[0, 0].set_xlabel("Training passes")
    axes[0, 0].set_ylabel("9-task proxy mean")
    axes[0, 0].set_xticks(sorted(set(CHECKPOINT_PASSES.values())))
    axes[0, 0].legend(frameon=False, ncol=2)

    selected_passes = (1, 4, 8, 15)
    for passes in selected_passes:
        values = sorted(
            (row for row in aggregate if row["passes"] == passes),
            key=lambda row: row["pool_m"],
        )
        axes[0, 1].plot(
            [row["pool_m"] for row in values],
            [row["proxy9_mean"] for row in values],
            marker="o",
            linewidth=2,
            label=f"{passes} pass" if passes == 1 else f"{passes} passes",
        )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xticks([0.1, 0.2, 0.5, 1.0], ["0.1M", "0.2M", "0.5M", "1.0M"])
    axes[0, 1].set_title("Dataset scaling at fixed training passes")
    axes[0, 1].set_xlabel("Nested microscopy data pool")
    axes[0, 1].set_ylabel("9-task proxy mean")
    axes[0, 1].legend(frameon=False, ncol=2)

    metric_specs = (
        ("proxy9_mean", "Proxy-9 mean", "#111827"),
        ("classification7_balanced_accuracy", "Classification-7 BA", "#0F766E"),
        ("bbbc005_r2", "BBBC005 R2", "#D97706"),
        ("lc25000_nmi", "LC25000 NMI", "#B42318"),
    )
    for axis, passes in zip(axes[1], (8, 15)):
        values = sorted(
            (row for row in aggregate if row["passes"] == passes),
            key=lambda row: row["pool_m"],
        )
        for key, label, color in metric_specs:
            axis.plot(
                [row["pool_m"] for row in values],
                [row[key] for row in values],
                marker="o",
                linewidth=2,
                color=color,
                label=label,
            )
        axis.set_xscale("log")
        axis.set_xticks([0.1, 0.2, 0.5, 1.0], ["0.1M", "0.2M", "0.5M", "1.0M"])
        axis.set_title(f"Task-family scaling at {passes} passes")
        axis.set_xlabel("Nested microscopy data pool")
        axis.set_ylabel("Raw downstream metric")
        axis.legend(frameon=False, fontsize=8)

    figure.suptitle(
        "S+ fixed-compute empirical scaling curves",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(output / "splus_fixed_compute_scaling.png", dpi=220)
    figure.savefig(output / "splus_fixed_compute_scaling.pdf")
    plt.close(figure)


def write_readme(output: Path, specs: list[RunSpec], aggregate: list[dict]) -> None:
    by_pass = {row["passes"]: [] for row in aggregate}
    for row in aggregate:
        by_pass[row["passes"]].append(row)
    best_rows = []
    for passes in sorted(by_pass):
        best = max(by_pass[passes], key=lambda row: row["proxy9_mean"])
        best_rows.append((passes, best["pool"], best["proxy9_mean"]))
    lines = [
        "# S+ fixed-compute scaling",
        "",
        "All four nested data pools use the locked S+ recipe, the same 15.744M-image",
        "schedule, and the same 8-checkpoint / 9-task frozen proxy protocol.",
        "These are empirical downstream scaling curves, not a fitted power-law claim.",
        "",
        "## Coverage",
        "",
        "| pool | result root | coverage |",
        "|---|---|---:|",
    ]
    for spec in sorted(specs, key=lambda item: item.pool_m):
        lines.append(f"| {spec.label} | `{spec.root}` | 72/72 |")
    lines.extend(
        [
            "",
            "## Best pool by checkpoint",
            "",
            "| passes | best pool | proxy-9 mean |",
            "|---:|---:|---:|",
        ]
    )
    for passes, pool, value in best_rows:
        lines.append(f"| {passes} | {pool} | {value:.6f} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `splus_fixed_compute_scaling.png` and `.pdf`: empirical compute and data scaling curves.",
            "- `aggregate.csv`: checkpoint-level Classification-7, BBBC005, LC25000, and Proxy-9 metrics.",
            "- `per_dataset.csv`: all source metrics and result paths.",
            "",
            "The arithmetic Proxy-9 mean combines seven balanced-accuracy values, BBBC005 R2,",
            "and LC25000 NMI. Inspect task-family panels before making a single-number claim.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="LABEL=POOL_M=EVAL_DIR",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    specs: list[RunSpec] = args.run
    if len(specs) < 2:
        parser.error("At least two --run entries are required")
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        parser.error("Run labels must be unique")

    rows = []
    for spec in specs:
        rows.extend(load_run(spec))
    aggregate = aggregate_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_dataset.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", aggregate)
    plot_curves(aggregate, args.output_dir)
    write_readme(args.output_dir, specs, aggregate)
    print(f"Wrote S+ scaling report to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
