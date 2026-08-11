#!/usr/bin/env python3
"""Build the strict fixed-1M S6b full-suite compute-scaling report."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = (
    ROOT
    / "outputs/01_training_runs/"
    "S6b_rgb_robust_biosafe256_b4096_lr2e-4_wu2_e30"
)
EVAL_ROOT = ROOT / "outputs/02_eval_runs"
GAP_ROOT = EVAL_ROOT / "s6b_fixed1m_compute_full_gapfill_20260723"
OUT = ROOT / "outputs/00_reports/s6b_fixed1m_full_compute_scaling_20260723"
OLD_TABLE = ROOT / "outputs/03_comparisons/splus_random_data_scaling_e15/CANONICAL_verified_table.csv"

CHECKPOINTS = ((3899, 15), (5199, 20), (6499, 25), (7799, 30))
CLASSIFICATION_DATASETS = {
    "bloodmnist",
    "pathmnist",
    "tissuemnist",
    "breastmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "chestmnist",
    "bbbc048-cellcycle",
    "cyclops-protein-loc",
    "midog25-atypical",
    "pcam",
    "nct-crc-he",
    "lc25000",
    "chammi-allen-task1",
    "chammi-allen-task2",
    "chammi-cp-task1",
    "chammi-cp-task2",
    "chammi-cp-task3",
    "chammi-hpa-task1",
    "chammi-hpa-task2",
}
REGRESSION_DATASETS = {"bbbc005", "bbbc013"}
RETRIEVAL_DATASETS = {"lc25000", "nct-crc-he-100", "nct-crc-he-1k", "crc-val-he-7k"}
SEGMENTATION_DATASETS = {
    "bbbc038",
    "conic",
    "monuseg",
    "pannuke",
    "tissuenet",
    "livecell",
    "multimodal_cellseg",
    "cellpose",
}
OOD_KEYS = (
    "xray_pair_recall_at_1",
    "xray_dose_r2",
    "cryo_class_accuracy",
    "cryo_quality_auroc",
    "cryo_retrieval_map_at_10",
)

COLORS = {
    "classification25_macro_f1": "#3E6D9C",
    "regression2_spearman": "#C6A9CE",
    "retrieval4_map_at_5": "#8FB8DC",
    "segmentation8_mdice": "#6FBF8B",
    "ood_composite": "#F0A868",
    "id4_overall": "#2F6E8F",
}
INK = "#2B2B2B"
GRID = "#DADDE2"
MUTED = "#8A9099"
SELECTED = "#E4572E"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text())
    if not isinstance(result, dict) or result.get("error"):
        raise ValueError(f"Invalid result: {path}")
    return result


def checkpoint_id(result: dict, path: Path) -> int:
    match = re.search(r"/ckpt/(\d+)(?:/checkpoint\.pth)?$", str(result.get("checkpoint", "")))
    if match:
        return int(match.group(1))
    for part in reversed(path.parts):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot infer checkpoint from {path}")


def classification_roots(checkpoint: int) -> tuple[Path, ...]:
    prefix = "S6b_rgb_robust_biosafe256_b4096_lr2e-4_wu2_e30"
    if checkpoint == 3899:
        return (EVAL_ROOT / f"{prefix}__Dscale_core_first3_ck3899_qi",)
    return (
        EVAL_ROOT / f"{prefix}__ckpt{checkpoint}_cls_med",
        EVAL_ROOT / f"{prefix}__ckpt{checkpoint}_cls_bio",
        EVAL_ROOT / f"{prefix}__ckpt{checkpoint}_cls_chammi_cp",
        EVAL_ROOT / f"{prefix}__ckpt{checkpoint}_cls_chammi_hpa",
        GAP_ROOT / f"ckpt_{checkpoint}/classification",
    )


def core_roots(checkpoint: int) -> tuple[Path, ...]:
    prefix = "S6b_rgb_robust_biosafe256_b4096_lr2e-4_wu2_e30"
    if checkpoint == 3899:
        return (EVAL_ROOT / f"{prefix}__Dscale_core_first3_ck3899_qi",)
    return (
        EVAL_ROOT / f"{prefix}__S6_ck{checkpoint}_regret_fill3090_singlecard_20260712_1616",
    )


def load_family(
    roots: tuple[Path, ...],
    expected: set[str],
    tasks: set[str],
    checkpoint: int,
) -> dict[str, tuple[dict, Path]]:
    found: dict[str, tuple[dict, Path]] = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("last_result.json")):
            result = json.loads(path.read_text())
            if not isinstance(result, dict) or result.get("error"):
                continue
            dataset = str(result.get("dataset", ""))
            if dataset not in expected or result.get("task") not in tasks:
                continue
            if checkpoint_id(result, path) != checkpoint:
                continue
            policy = result.get("channel_policy")
            if policy not in {"auto", "first3"}:
                raise ValueError(f"Unexpected channel policy in {path}: {policy}")
            if dataset in found:
                raise ValueError(f"Duplicate result for ck{checkpoint} {dataset}: {found[dataset][1]} and {path}")
            found[dataset] = (result, path)
    missing = sorted(expected - set(found))
    if missing:
        raise ValueError(f"Incomplete ck{checkpoint} coverage: missing={missing}")
    return found


def segmentation_roots(checkpoint: int) -> tuple[Path, ...]:
    prefix = "S6b_rgb_robust_biosafe256_b4096_lr2e-4_wu2_e30"
    return (
        EVAL_ROOT / f"{prefix}__ckpt{checkpoint}_dense",
        EVAL_ROOT / f"{prefix}__ckpt3899_5199_6499_7799_seg_monuseg_cellpose_retry_cpu12",
        EVAL_ROOT / f"{prefix}__ckpt3899_5199_6499_7799_seg_multimodal_livecell_deepcad7",
    )


def load_segmentation(checkpoint: int) -> dict[str, tuple[dict, Path]]:
    found: dict[str, tuple[dict, Path]] = {}
    for root in segmentation_roots(checkpoint):
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob(f"{checkpoint}/results.json")):
            dataset = path.parents[1].name
            if dataset not in SEGMENTATION_DATASETS:
                continue
            result = load_json(path)
            if dataset in found:
                old = float(found[dataset][0]["test"]["mDice"])
                new = float(result["test"]["mDice"])
                if not math.isclose(old, new, abs_tol=1e-12):
                    raise ValueError(f"Conflicting segmentation result for ck{checkpoint} {dataset}")
                continue
            found[dataset] = (result, path)
    missing = sorted(SEGMENTATION_DATASETS - set(found))
    if missing:
        raise ValueError(f"Incomplete ck{checkpoint} segmentation: missing={missing}")
    return found


def load_ood(checkpoint: int) -> tuple[dict, Path]:
    if checkpoint == 3899:
        path = (
            EVAL_ROOT
            / "splus_random_data_scaling_seg_ood/random_100/ood/"
            "splus_random_100/3899/last_result.json"
        )
    else:
        path = (
            GAP_ROOT
            / f"ckpt_{checkpoint}/ood/"
            "S6b_rgb_robust_biosafe256_b4096_lr2e-4_wu2_e30"
            / str(checkpoint)
            / "last_result.json"
        )
    result = load_json(path)
    missing = [key for key in OOD_KEYS if result.get(key) is None]
    if missing:
        raise ValueError(f"Incomplete OOD result {path}: missing={missing}")
    if result.get("channel_policy") != "first3":
        raise ValueError(f"OOD result does not use first3: {path}")
    return result, path


def load_training() -> dict[int, dict]:
    wanted = {checkpoint for checkpoint, _ in CHECKPOINTS}
    found = {}
    for line in (TRAIN_ROOT / "raw_loss_metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        checkpoint = int(row.get("optimizer_update", -1))
        if checkpoint in wanted:
            if int(row["effective_global_batch_size"]) != 4096:
                raise ValueError(f"Unexpected GBS at ck{checkpoint}")
            found[checkpoint] = row
    if set(found) != wanted:
        raise ValueError(f"Missing training metadata: {sorted(wanted - set(found))}")
    return found


def summarize_checkpoint(checkpoint: int, passes: int, training: dict) -> tuple[dict, list[dict]]:
    classification = load_family(
        classification_roots(checkpoint),
        CLASSIFICATION_DATASETS,
        {"classification", "multilabel_classification"},
        checkpoint,
    )
    regression = load_family(core_roots(checkpoint), REGRESSION_DATASETS, {"regression"}, checkpoint)
    retrieval = load_family(
        core_roots(checkpoint), RETRIEVAL_DATASETS, {"retrieval_clustering"}, checkpoint
    )
    segmentation = load_segmentation(checkpoint)
    ood, ood_path = load_ood(checkpoint)
    values = {
        "classification25_macro_f1": mean(float(result["macro_f1"]) for result, _ in classification.values()),
        "regression2_spearman": mean(float(result["spearman"]) for result, _ in regression.values()),
        "retrieval4_map_at_5": mean(float(result["map_at_5"]) for result, _ in retrieval.values()),
        "segmentation8_mdice": mean(float(result["test"]["mDice"]) for result, _ in segmentation.values()),
        "ood_composite": mean(float(ood[key]) for key in OOD_KEYS),
    }
    row = {
        "checkpoint": checkpoint,
        "passes": passes,
        "unique_images": 1_048_771,
        "image_visits": int(training["image_visits"]),
        "image_visits_m": float(training["image_visits"]) / 1e6,
        "patch_tokens_seen": int(training["patch_tokens_seen_estimate"]),
        "training_total_loss": float(training["total_loss"]),
        **values,
        "id4_overall": mean(
            values[key]
            for key in (
                "classification25_macro_f1",
                "regression2_spearman",
                "retrieval4_map_at_5",
                "segmentation8_mdice",
            )
        ),
        "five_family_mean": mean(values.values()),
    }
    details = []
    for family, results, metric in (
        ("classification", classification, "macro_f1"),
        ("regression", regression, "spearman"),
        ("retrieval_clustering", retrieval, "map_at_5"),
        ("segmentation", segmentation, "test.mDice"),
    ):
        for dataset, (result, path) in sorted(results.items()):
            value = result["test"]["mDice"] if family == "segmentation" else result[metric]
            details.append(
                {
                    "checkpoint": checkpoint,
                    "passes": passes,
                    "family": family,
                    "dataset": dataset,
                    "metric": metric,
                    "value": value,
                    "result_path": str(path),
                }
            )
    for key in OOD_KEYS:
        details.append(
            {
                "checkpoint": checkpoint,
                "passes": passes,
                "family": "ood",
                "dataset": key,
                "metric": key,
                "value": ood[key],
                "result_path": str(ood_path),
            }
        )
    return row, details


def load_baseline() -> dict:
    with OLD_TABLE.open(newline="") as handle:
        row = next(row for row in csv.DictReader(handle) if row["scale"] == "base (0)")
    baseline = {
        "classification25_macro_f1": float(row["cls_macroF1"]),
        "regression2_spearman": float(row["reg_spearman"]),
        "retrieval4_map_at_5": float(row["ret_mAP@5"]),
        "segmentation8_mdice": float(row["seg_mDice"]),
        "ood_composite": float(row["ood_composite"]),
        "id4_overall": float(row["ID4_overall"]),
    }
    expected = mean(
        baseline[key]
        for key in (
            "classification25_macro_f1",
            "regression2_spearman",
            "retrieval4_map_at_5",
            "segmentation8_mdice",
        )
    )
    if not math.isclose(baseline["id4_overall"], expected, abs_tol=1e-12):
        raise ValueError("Official ID-4 baseline is inconsistent")
    return baseline


def save_all(figure: plt.Figure, stem: str) -> None:
    figure.savefig(OUT / f"{stem}.png", dpi=220, facecolor="white")
    figure.savefig(OUT / f"{stem}.pdf", facecolor="white")
    figure.savefig(OUT / f"{stem}.svg", facecolor="white")


def style_axis(axis: plt.Axes, rows: list[dict]) -> None:
    x = np.array([float(row["image_visits_m"]) for row in rows])
    axis.set_xscale("log")
    axis.set_xlim(x[0] * 0.94, x[-1] * 1.08)
    axis.set_xticks(x, [f"{value:.1f}M" for value in x])
    axis.minorticks_off()
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(INK)
    axis.tick_params(length=0, labelsize=8.8, colors=INK)
    axis.set_xlabel("cumulative compute C (image-visits, log)", fontsize=9.7)
    for value, row in zip(x, rows):
        axis.text(
            value,
            0.985,
            f"{row['passes']}p\nck{row['checkpoint']}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.5,
            color=MUTED,
            linespacing=0.9,
        )


def limits(values: list[float], baseline: float) -> tuple[float, float]:
    low = min([*values, baseline])
    high = max([*values, baseline])
    span = max(high - low, 0.005)
    return low - 0.22 * span, high + 0.42 * span


def draw_curve(axis: plt.Axes, rows: list[dict], baseline: dict, key: str) -> None:
    x = np.array([float(row["image_visits_m"]) for row in rows])
    y = np.array([float(row[key]) for row in rows])
    color = COLORS[key]
    axis.axhline(baseline[key], color="#AAB0B8", linewidth=1.1, linestyle=(0, (4, 4)))
    axis.plot(x, y, color="#4A4A4A", linewidth=1.45, linestyle=(0, (5, 4)), zorder=2)
    axis.scatter(x, y, s=145, color=color, edgecolor=INK, linewidth=1.1, zorder=3)
    axis.scatter(
        [x[0]], [y[0]], s=235, facecolor="none", edgecolor=SELECTED, linewidth=2.0, zorder=4
    )
    axis.annotate(
        "S6b",
        (x[-1], y[-1]),
        xytext=(0, 16),
        textcoords="offset points",
        ha="center",
        color="white",
        fontsize=9.8,
        bbox=dict(boxstyle="round,pad=0.32", facecolor=color, edgecolor="none"),
    )
    axis.annotate(
        "official base",
        (x[0] * 0.97, baseline[key]),
        fontsize=7.6,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    axis.set_ylim(*limits(y.tolist(), baseline[key]))
    style_axis(axis, rows)


def plot_tasks(rows: list[dict], baseline: dict) -> None:
    panels = (
        ("classification25_macro_f1", "(a) Classification (25 sets)", "macro-F1"),
        ("regression2_spearman", "(b) Regression (2 sets)", r"Spearman $\rho$"),
        ("retrieval4_map_at_5", "(c) Retrieval / clustering (4)", "mAP@5"),
        ("segmentation8_mdice", "(d) Segmentation (8 sets)", "mDice"),
        ("ood_composite", "(e) OOD (X-ray + cryo-EM)", "composite"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.8, 8.6))
    for axis, (key, title, ylabel) in zip(axes.flat, panels):
        draw_curve(axis, rows, baseline, key)
        axis.set_title(title, fontsize=12.2, fontweight="bold", pad=7)
        axis.set_ylabel(ylabel, fontsize=10.7)
    axes.flat[5].axis("off")
    axes.flat[5].text(
        0.04,
        0.77,
        "PURE COMPUTE SCALING",
        color=COLORS["id4_overall"],
        fontsize=13,
        fontweight="bold",
        transform=axes.flat[5].transAxes,
    )
    axes.flat[5].text(
        0.04,
        0.58,
        "D fixed at 1,048,771 images\nSame S6b training run\nOnly checkpoint / passes change",
        color=INK,
        fontsize=11,
        linespacing=1.5,
        transform=axes.flat[5].transAxes,
    )
    figure.suptitle(
        r"$\mathrm{S6b}$ continual pre-training - compute scaling by task (fixed 1M)",
        fontsize=15.8,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.014,
        "Constant marker size: data volume is fixed. Coral ring marks the 15-pass point shared with the data-scaling endpoint.",
        ha="center",
        fontsize=8.8,
        color="#5C6770",
    )
    figure.subplots_adjust(left=0.06, right=0.985, top=0.89, bottom=0.08, wspace=0.28, hspace=0.43)
    save_all(figure, "s6b_full_compute_scaling_tasks")
    plt.close(figure)


def plot_overall(rows: list[dict], baseline: dict) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 6.25))
    draw_curve(axis, rows, baseline, "id4_overall")
    axis.set_ylabel("ID-4 family-balanced mean", fontsize=11.5)
    axis.set_title(
        r"$\mathrm{S6b}$ compute scaling - ID overall (fixed 1M)" + "\n"
        + "Classification + regression + retrieval/clustering + segmentation",
        fontsize=14.1,
        fontweight="bold",
        pad=11,
    )
    axis.text(
        0.985,
        0.025,
        "OOD excluded from overall",
        transform=axis.transAxes,
        ha="right",
        color="#C6803B",
        fontsize=9.2,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.018,
        "D = 1,048,771 throughout; C increases from 15.97M to 31.95M image-visits by extending the same training run.",
        ha="center",
        fontsize=8.7,
        color="#5C6770",
    )
    figure.subplots_adjust(left=0.12, right=0.97, top=0.83, bottom=0.15)
    save_all(figure, "s6b_full_compute_scaling_overall")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(rows: list[dict]) -> None:
    lines = [
        "# S6b fixed-1M full-suite compute scaling",
        "",
        "This is a strict compute-scaling curve: all points come from the same S6b training run",
        "and the same 1,048,771-image pool. Only checkpoint, passes, and cumulative compute change.",
        "",
        "Coverage at every point is Classification-25 macro-F1, Regression-2 Spearman,",
        "Retrieval/Clustering-4 mAP@5, Segmentation-8 mDice, and the five-component",
        "non-saturated X-ray + cryo-EM OOD composite. ID-4 excludes OOD.",
        "",
        "| checkpoint | passes | image-visits | C25 | Reg-2 | Ret-4 | Seg-8 | OOD | ID-4 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {row['passes']} | {row['image_visits']:,} | "
            f"{row['classification25_macro_f1']:.6f} | {row['regression2_spearman']:.6f} | "
            f"{row['retrieval4_map_at_5']:.6f} | {row['segmentation8_mdice']:.6f} | "
            f"{row['ood_composite']:.6f} | {row['id4_overall']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The 15-pass OOD result is the established S6b data-scaling endpoint. The 20/25/30-pass",
            "OOD and missing C25 datasets were locally gap-filled with the same first3 protocol.",
            "",
            "## Outputs",
            "",
            "- `s6b_full_compute_scaling_tasks.{png,pdf,svg}`",
            "- `s6b_full_compute_scaling_overall.{png,pdf,svg}`",
            "- `s6b_full_compute_summary.csv`",
            "- `s6b_full_compute_per_dataset.csv`",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    training = load_training()
    rows = []
    details = []
    for checkpoint, passes in CHECKPOINTS:
        row, source_rows = summarize_checkpoint(checkpoint, passes, training[checkpoint])
        rows.append(row)
        details.extend(source_rows)
    baseline = load_baseline()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.0,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    plot_tasks(rows, baseline)
    plot_overall(rows, baseline)
    write_csv(OUT / "s6b_full_compute_summary.csv", rows)
    write_csv(OUT / "s6b_full_compute_per_dataset.csv", details)
    write_csv(OUT / "official_baseline.csv", [baseline])
    write_readme(rows)
    print(f"Wrote S6b fixed-1M full compute-scaling report to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
