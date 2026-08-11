#!/usr/bin/env python3
"""Build the strict full-suite compute curve for the locked 1M S+ run."""

from __future__ import annotations

import csv
import json
import math
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
    "S6sigreg005_rgb_robust_biosafe256_b1024_officialprec_lr1e-4_wu2_e15_local5090_20260716"
)
FULL_ROOT = (
    ROOT / "outputs/02_eval_runs/splus_fixed1m_full_compute_clean_bf16_auto_20260723"
)
CK8199_ROOT = (
    ROOT
    / "outputs/02_eval_runs/"
    "S6sigreg005_raw_ck8199__full_dense_local_bf16_b64_clean_20260723"
)
CK8199_OOD_ROOT = (
    ROOT / "outputs/02_eval_runs/splus_fixed_compute_full_ck8199_20260723/random_100"
)
S6_REPORT = ROOT / "outputs/00_reports/s6b_fixed1m_full_compute_scaling_20260723"
OUT = ROOT / "outputs/00_reports/splus_fixed1m_full_compute_scaling_20260723"

CHECKPOINTS = (1024, 2049, 4099, 6149, 8199, 10249, 12299, 15374)
SELECTED_CHECKPOINT = 8199
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

FAMILY_KEYS = (
    "classification25_macro_f1",
    "regression2_spearman",
    "retrieval4_map_at_5",
    "segmentation8_mdice",
    "ood_composite",
)
ID4_KEYS = FAMILY_KEYS[:4]
COLORS = {
    "classification25_macro_f1": "#35698F",
    "regression2_spearman": "#B783BA",
    "retrieval4_map_at_5": "#72A8CF",
    "segmentation8_mdice": "#559B70",
    "ood_composite": "#D18A45",
    "id4_overall": "#25647E",
}
INK = "#273238"
GRID = "#D9DEE2"
MUTED = "#68747A"
S6_COLOR = "#999FA4"
SELECTED = "#D7543D"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("error"):
        raise ValueError(f"Invalid result: {path}")
    return payload


def finite_float(value: object, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite metric {label}={value!r}")
    return result


def resolve_recorded_path(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def expected_checkpoint(checkpoint: int) -> Path:
    return (TRAIN_ROOT / "ckpt" / str(checkpoint) / "checkpoint.pth").resolve()


def checkpoint_roots(checkpoint: int) -> tuple[Path, Path]:
    if checkpoint == 8199:
        return CK8199_ROOT, CK8199_OOD_ROOT
    root = FULL_ROOT / f"ckpt_{checkpoint}"
    return root, root


def unique_results(
    checkpoint: int,
    root: Path,
    task: str,
    expected: set[str],
) -> dict[str, tuple[dict, Path]]:
    found: dict[str, tuple[dict, Path]] = {}
    wanted_checkpoint = expected_checkpoint(checkpoint)
    for path in sorted(root.rglob("last_result.json")):
        result = load_json(path)
        result_task = result.get("task")
        task_matches = result_task == task or (
            task == "classification" and result_task == "multilabel_classification"
        )
        if not task_matches or result.get("dataset") not in expected:
            continue
        if resolve_recorded_path(result.get("checkpoint")) != wanted_checkpoint:
            continue
        if result.get("channel_policy") != "auto" or result.get("channel_tta_samples") != 8:
            raise ValueError(f"Unexpected channel protocol: {path}")
        if task in {"classification", "regression"} and result.get("resolution_protocol") != "best":
            raise ValueError(f"Unexpected resolution protocol: {path}")
        dataset = str(result["dataset"])
        if dataset in found:
            raise ValueError(
                f"Duplicate ck{checkpoint} {task}/{dataset}: {found[dataset][1]} and {path}"
            )
        found[dataset] = (result, path)
    missing = sorted(expected - set(found))
    if missing:
        raise ValueError(f"Incomplete ck{checkpoint} {task}: missing={missing}; root={root}")
    return found


def segmentation_results(checkpoint: int, root: Path) -> dict[str, tuple[dict, Path]]:
    found: dict[str, tuple[dict, Path]] = {}
    for path in sorted(root.glob(f"bio_segmentation/**/{checkpoint}/results.json")):
        if "seed1" in path.parts or "seed2" in path.parts:
            continue
        dataset = path.parents[1].name
        if dataset not in SEGMENTATION_DATASETS:
            continue
        result = load_json(path)
        if dataset in found:
            old = float(found[dataset][0]["test"]["mDice"])
            new = float(result["test"]["mDice"])
            if not math.isclose(old, new, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Conflicting ck{checkpoint} segmentation result for {dataset}")
            continue
        found[dataset] = (result, path)
    missing = sorted(SEGMENTATION_DATASETS - set(found))
    if missing:
        raise ValueError(f"Incomplete ck{checkpoint} segmentation: missing={missing}; root={root}")
    return found


def ood_result(checkpoint: int, root: Path) -> tuple[dict, Path]:
    candidates = sorted(root.glob(f"ood/*/{checkpoint}/last_result.json"))
    candidates = [path for path in candidates if path.parents[1].name not in {"xray", "cryo"}]
    if len(candidates) != 1:
        raise ValueError(f"Expected one combined OOD result for ck{checkpoint}, found {candidates}")
    path = candidates[0]
    result = load_json(path)
    if resolve_recorded_path(result.get("checkpoint")) != expected_checkpoint(checkpoint):
        raise ValueError(f"Wrong OOD checkpoint: {path}")
    if result.get("channel_policy") != "auto" or result.get("channel_tta_samples") != 8:
        raise ValueError(f"Unexpected OOD channel protocol: {path}")
    missing = [key for key in OOD_KEYS if result.get(key) is None]
    if missing:
        raise ValueError(f"Incomplete ck{checkpoint} OOD metrics: {missing}")
    return result, path


def load_training() -> dict[int, dict]:
    wanted = set(CHECKPOINTS)
    found = {}
    for line in (TRAIN_ROOT / "raw_loss_metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        checkpoint = int(row.get("optimizer_update", -1))
        if checkpoint not in wanted:
            continue
        if int(row["effective_global_batch_size"]) != 1024:
            raise ValueError(f"Unexpected global batch at ck{checkpoint}")
        found[checkpoint] = row
    if set(found) != wanted:
        raise ValueError(f"Missing training metadata: {sorted(wanted - set(found))}")
    return found


def summarize_checkpoint(checkpoint: int, training: dict) -> tuple[dict, list[dict]]:
    root, ood_root = checkpoint_roots(checkpoint)
    classification = unique_results(checkpoint, root, "classification", CLASSIFICATION_DATASETS)
    regression = unique_results(checkpoint, root, "regression", REGRESSION_DATASETS)
    retrieval = unique_results(checkpoint, root, "retrieval_clustering", RETRIEVAL_DATASETS)
    segmentation = segmentation_results(checkpoint, root)
    ood, ood_path = ood_result(checkpoint, ood_root)
    metrics = {
        "classification25_macro_f1": mean(
            finite_float(result["macro_f1"], label=f"classification/{dataset}")
            for dataset, (result, _) in classification.items()
        ),
        "regression2_spearman": mean(
            finite_float(result["spearman"], label=f"regression/{dataset}")
            for dataset, (result, _) in regression.items()
        ),
        "retrieval4_map_at_5": mean(
            finite_float(result["map_at_5"], label=f"retrieval/{dataset}")
            for dataset, (result, _) in retrieval.items()
        ),
        "segmentation8_mdice": mean(
            finite_float(result["test"]["mDice"], label=f"segmentation/{dataset}")
            for dataset, (result, _) in segmentation.items()
        ),
        "ood_composite": mean(
            finite_float(ood[key], label=f"OOD/{key}") for key in OOD_KEYS
        ),
    }
    training_row = training[checkpoint]
    row = {
        "checkpoint": checkpoint,
        "passes": float(training_row["epoch_float"]),
        "unique_images": 1_048_771,
        "image_visits": int(training_row["image_visits"]),
        "image_visits_m": float(training_row["image_visits"]) / 1e6,
        "patch_tokens_seen": int(training_row["patch_tokens_seen_estimate"]),
        "training_total_loss": float(training_row["total_loss"]),
        "evaluation_autocast_dtype": "bf16",
        "evaluation_batch_size": 64,
        "evaluation_channel_policy": "auto",
        **metrics,
        "id4_overall": mean(metrics[key] for key in ID4_KEYS),
        "five_family_mean": mean(metrics[key] for key in FAMILY_KEYS),
    }

    details = []
    for family, values, metric in (
        ("classification", classification, "macro_f1"),
        ("regression", regression, "spearman"),
        ("retrieval_clustering", retrieval, "map_at_5"),
        ("segmentation", segmentation, "test.mDice"),
    ):
        for dataset, (result, path) in values.items():
            value = result["test"]["mDice"] if family == "segmentation" else result[metric]
            details.append(
                {
                    "checkpoint": checkpoint,
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
                "family": "ood",
                "dataset": "xray+cryo",
                "metric": key,
                "value": ood[key],
                "result_path": str(ood_path),
            }
        )
    if len(details) != 44:
        raise ValueError(f"Expected 44 provenance rows at ck{checkpoint}, found {len(details)}")
    return row, details


def load_s6_summary() -> list[dict]:
    path = S6_REPORT / "s6b_full_compute_summary.csv"
    with path.open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    numeric = {
        "checkpoint": int,
        "passes": float,
        "image_visits": int,
        "image_visits_m": float,
        **{key: float for key in FAMILY_KEYS},
        "id4_overall": float,
        "five_family_mean": float,
    }
    for row in rows:
        for key, converter in numeric.items():
            row[key] = converter(row[key])
    return rows


def build_matched_comparison(current: list[dict], s6: list[dict]) -> list[dict]:
    new = next(row for row in current if math.isclose(float(row["passes"]), 15.0))
    old = next(row for row in s6 if math.isclose(float(row["passes"]), 15.0))
    rows = []
    for metric in (*FAMILY_KEYS, "id4_overall", "five_family_mean"):
        rows.append(
            {
                "metric": metric,
                "new_checkpoint": new["checkpoint"],
                "new_passes": new["passes"],
                "new_image_visits": new["image_visits"],
                "new_value": new[metric],
                "s6_checkpoint": old["checkpoint"],
                "s6_passes": old["passes"],
                "s6_image_visits": old["image_visits"],
                "s6_value": old[metric],
                "new_minus_s6": float(new[metric]) - float(old[metric]),
                "training_budget_near_matched": True,
                "strict_eval_precision_matched": False,
            }
        )
    return rows


def style_axis(axis: plt.Axes) -> None:
    axis.set_xscale("log")
    axis.set_xlim(0.85, 37.0)
    axis.set_xticks(
        [1.05, 2.10, 4.20, 8.40, 15.74, 31.95],
        ["1.05", "2.10", "4.20", "8.40", "15.7", "31.9"],
    )
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(INK)
    axis.tick_params(length=0, colors=INK)
    axis.set_xlabel("training image visits (millions, log)")


def plot_tasks(current: list[dict], s6: list[dict]) -> None:
    new_x = np.array([float(row["image_visits_m"]) for row in current])
    old_x = np.array([float(row["image_visits_m"]) for row in s6])
    panels = (
        ("classification25_macro_f1", "(a) Classification (25 datasets)", "macro-F1"),
        ("regression2_spearman", "(b) Regression (2 datasets)", r"Spearman $\rho$"),
        ("retrieval4_map_at_5", "(c) Retrieval (4 datasets)", "mAP@5"),
        ("segmentation8_mdice", "(d) Segmentation (8 datasets)", "mDice"),
        ("ood_composite", "(e) OOD (X-ray + cryo-EM)", "non-saturated composite"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.5))
    for axis, (key, title, ylabel) in zip(axes.flat, panels):
        new_y = np.array([float(row[key]) for row in current])
        old_y = np.array([float(row[key]) for row in s6])
        low = min(float(new_y.min()), float(old_y.min()))
        high = max(float(new_y.max()), float(old_y.max()))
        pad = max((high - low) * 0.22, 0.004)
        style_axis(axis)
        axis.set_ylim(low - pad, high + pad)
        axis.plot(
            old_x,
            old_y,
            color=S6_COLOR,
            linewidth=1.8,
            linestyle=(0, (3, 3)),
            marker="o",
            markersize=5,
            label="S6 historical",
        )
        axis.plot(
            new_x,
            new_y,
            color=COLORS[key],
            linewidth=2.2,
            marker="o",
            markersize=7,
            label="S+ new recipe",
        )
        selected_index = CHECKPOINTS.index(SELECTED_CHECKPOINT)
        axis.scatter(
            [new_x[selected_index]],
            [new_y[selected_index]],
            marker="*",
            s=150,
            color=SELECTED,
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_ylabel(ylabel)
    axes.flat[5].axis("off")
    axes.flat[5].text(
        0.04,
        0.84,
        "FIXED 1M DATA POOL",
        color=COLORS["id4_overall"],
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    axes.flat[5].text(
        0.04,
        0.67,
        "solid: new S+ / GBS 1024 / bf16 eval\ndashed: historical S6 / GBS 4096\nstar: selected new ck8199 (8 passes)",
        color=INK,
        fontsize=10.5,
        va="top",
        linespacing=1.35,
    )
    axes.flat[5].text(
        0.04,
        0.30,
        "The 15-pass endpoints have near-matched\nimage visits (15.744M vs 15.974M).\nS6 eval precision is historical, so deltas\nare context rather than strict A/B estimates.",
        color=MUTED,
        fontsize=9.5,
        va="top",
        linespacing=1.3,
    )
    figure.suptitle(
        r"$\mathrm{S^+}$ full-suite compute scaling at fixed data",
        fontsize=16,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.075, wspace=0.28, hspace=0.38)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUT / f"splus_compute_scaling_tasks.{suffix}",
            dpi=220 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(figure)


def plot_overall(current: list[dict], s6: list[dict]) -> None:
    new_x = np.array([float(row["image_visits_m"]) for row in current])
    new_y = np.array([float(row["id4_overall"]) for row in current])
    old_x = np.array([float(row["image_visits_m"]) for row in s6])
    old_y = np.array([float(row["id4_overall"]) for row in s6])
    figure, axis = plt.subplots(figsize=(8.5, 6.4))
    style_axis(axis)
    low = min(float(new_y.min()), float(old_y.min()))
    high = max(float(new_y.max()), float(old_y.max()))
    pad = max((high - low) * 0.18, 0.004)
    axis.set_ylim(low - pad, high + pad)
    axis.plot(
        old_x,
        old_y,
        color=S6_COLOR,
        linewidth=1.9,
        linestyle=(0, (3, 3)),
        marker="o",
        markersize=6,
        label="S6 historical full suite",
    )
    axis.plot(
        new_x,
        new_y,
        color=COLORS["id4_overall"],
        linewidth=2.8,
        marker="o",
        markersize=9,
        label="S+ new recipe (strict bf16 suite)",
    )
    selected_index = CHECKPOINTS.index(SELECTED_CHECKPOINT)
    axis.scatter(
        [new_x[selected_index]],
        [new_y[selected_index]],
        marker="*",
        s=220,
        color=SELECTED,
        edgecolors="white",
        linewidths=0.9,
        zorder=5,
        label="selected ck8199",
    )
    for x, value, row in zip(new_x, new_y, current):
        axis.annotate(
            f"{value:.4f}",
            (x, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )
    delta = new_y[-1] - old_y[0]
    axis.annotate(
        f"near-matched 15-pass context: {delta:+.4f}",
        xy=(new_x[-1], new_y[-1]),
        xytext=(-200, -42),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=COLORS["id4_overall"]),
        color=COLORS["id4_overall"],
        fontsize=10,
        fontweight="bold",
    )
    axis.set_ylabel("ID-4 family-balanced mean")
    axis.set_title(
        r"$\mathrm{S^+}$ full-suite compute scaling (fixed 1M)",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    axis.legend(frameon=False, loc="lower right")
    figure.text(
        0.5,
        0.025,
        "Solid curve is one bf16/auto-channel protocol. Dashed S6 is historical context; its feature precision was not rerun.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    figure.subplots_adjust(left=0.12, right=0.97, top=0.89, bottom=0.12)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUT / f"splus_compute_scaling_overall.{suffix}",
            dpi=220 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(figure)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(current: list[dict], comparison: list[dict]) -> None:
    lines = [
        "# S+ fixed-1M full-suite compute scaling",
        "",
        "The solid curve uses one locked evaluation protocol at all eight raw checkpoints:",
        "bf16 feature extraction, batch 64, auto channel handling, eight channel-TTA samples,",
        "best classification resolution, seed 0, Seg-8 best protocol, and two-phase OOD.",
        "Proxy-9 is not used in the full-suite aggregates.",
        "",
        "| ckpt | passes | visits (M) | C25 | Reg-2 | Ret-4 | Seg-8 | OOD | ID-4 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in current:
        lines.append(
            f"| {row['checkpoint']} | {float(row['passes']):.0f} | "
            f"{float(row['image_visits_m']):.4f} | "
            f"{float(row['classification25_macro_f1']):.6f} | "
            f"{float(row['regression2_spearman']):.6f} | "
            f"{float(row['retrieval4_map_at_5']):.6f} | "
            f"{float(row['segmentation8_mdice']):.6f} | "
            f"{float(row['ood_composite']):.6f} | {float(row['id4_overall']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## S6 context",
            "",
            "S6 is shown as a dashed historical curve. Both runs use the same 1M pool, and the",
            "new 15-pass point (15.744M visits) nearly matches S6 ck3899 (15.974M visits).",
            "However, the S6 result set was not regenerated under the new bf16 feature-extraction",
            "lane. Therefore the table below is useful context, not a strict evaluation-precision A/B.",
            "",
            "| metric | new 15-pass | S6 15-pass | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in comparison:
        lines.append(
            f"| {row['metric']} | {float(row['new_value']):.6f} | "
            f"{float(row['s6_value']):.6f} | {float(row['new_minus_s6']):+.6f} |"
        )
    lines.extend(
        [
            "",
            "The selected new sweet point remains ck8199 (8 schedule passes). The 15-pass row is",
            "included to make the old/new training budget comparison meaningful; it does not replace",
            "the selected deployment checkpoint.",
            "",
            "## Outputs",
            "",
            "- `splus_compute_scaling_tasks.{png,pdf,svg}`",
            "- `splus_compute_scaling_overall.{png,pdf,svg}`",
            "- `splus_full_compute_summary.csv`",
            "- `splus_full_compute_per_dataset.csv`",
            "- `matched_15pass_vs_s6.csv`",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    training = load_training()
    current = []
    details = []
    for checkpoint in CHECKPOINTS:
        row, checkpoint_details = summarize_checkpoint(checkpoint, training)
        current.append(row)
        details.extend(checkpoint_details)
    s6 = load_s6_summary()
    comparison = build_matched_comparison(current, s6)

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
    plot_tasks(current, s6)
    plot_overall(current, s6)
    write_csv(OUT / "splus_full_compute_summary.csv", current)
    write_csv(OUT / "splus_full_compute_per_dataset.csv", details)
    write_csv(OUT / "matched_15pass_vs_s6.csv", comparison)
    write_readme(current, comparison)
    print(f"wrote strict fixed-1M S+ full compute report to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
