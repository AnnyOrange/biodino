#!/usr/bin/env python3
"""Build the strict fixed-compute S+ full-suite data-scaling report."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickletools
import zipfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = 8199
GLOBAL_BATCH = 1024
OUT = ROOT / "outputs/00_reports/splus_data_scaling_full_current_20260723"
GAP_ROOT = ROOT / "outputs/02_eval_runs/splus_fixed_compute_full_ck8199_20260723"
ONE_M_FRESH_ROOT = (
    ROOT
    / "outputs/02_eval_runs/"
    "S6sigreg005_raw_ck8199__full_dense_local_bf16_b64_clean_20260723"
)
S6_MANIFEST = (
    ROOT / "outputs/03_comparisons/splus_random_data_scaling_e15/run_manifest.csv"
)
S6_TABLE = (
    ROOT / "outputs/03_comparisons/splus_random_data_scaling_e15/CANONICAL_verified_table.csv"
)

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
REGRESSION_DATASETS = {"bbbc013", "bbbc005"}
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
    "classification": "#35698F",
    "regression": "#B783BA",
    "retrieval_clustering": "#72A8CF",
    "segmentation": "#559B70",
    "ood": "#D18A45",
    "id4": "#25647E",
}
INK = "#273238"
GRID = "#D9DEE2"
MUTED = "#68747A"


@dataclass(frozen=True)
class Pool:
    label: str
    samples: int
    result_root: Path
    ood_root: Path
    train_dir: Path


POOLS = (
    Pool(
        "0.1M",
        104_877,
        GAP_ROOT / "random_10",
        GAP_ROOT / "random_10",
        ROOT
        / "outputs/01_training_runs/"
        "DscaleFinal_splus_sigreg005_random10_fixed15M_b1024_seed0_qi4gbs64acc4",
    ),
    Pool(
        "0.2M",
        209_754,
        GAP_ROOT / "random_20",
        GAP_ROOT / "random_20",
        ROOT
        / "outputs/01_training_runs/"
        "DscaleFinal_splus_sigreg005_random20_fixed15M_b1024_seed0_qi4gbs64acc4",
    ),
    Pool(
        "0.5M",
        524_385,
        GAP_ROOT / "random_50",
        GAP_ROOT / "random_50",
        ROOT
        / "outputs/01_training_runs/"
        "DscaleFinal_splus_sigreg005_random50_fixed15M_b1024_seed0_local8gbs64acc2",
    ),
    Pool(
        "1.0M",
        1_048_771,
        ONE_M_FRESH_ROOT,
        GAP_ROOT / "random_100",
        ROOT
        / "outputs/01_training_runs/"
        "S6sigreg005_rgb_robust_biosafe256_b1024_officialprec_lr1e-4_wu2_e15_local5090_20260716",
    ),
)


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


def expected_checkpoint(pool: Pool) -> Path:
    return (pool.train_dir / "ckpt" / str(CHECKPOINT) / "checkpoint.pth").resolve()


def unique_results(
    pool: Pool,
    task: str,
    expected: set[str],
) -> dict[str, tuple[dict, Path]]:
    found: dict[str, tuple[dict, Path]] = {}
    checkpoint = expected_checkpoint(pool)
    for path in sorted(pool.result_root.rglob("last_result.json")):
        result = load_json(path)
        result_task = result.get("task")
        task_matches = result_task == task or (
            task == "classification" and result_task == "multilabel_classification"
        )
        if not task_matches or result.get("dataset") not in expected:
            continue
        if resolve_recorded_path(result.get("checkpoint")) != checkpoint:
            continue
        if result.get("channel_policy") != "auto" or result.get("channel_tta_samples") != 8:
            raise ValueError(f"Unexpected channel protocol: {path}")
        if task in {"classification", "regression"} and result.get("resolution_protocol") != "best":
            raise ValueError(f"Unexpected resolution protocol: {path}")
        dataset = str(result["dataset"])
        if dataset in found:
            raise ValueError(f"Duplicate {task}/{dataset}: {found[dataset][1]} and {path}")
        found[dataset] = (result, path)
    missing = sorted(expected - set(found))
    if missing:
        raise ValueError(
            f"Incomplete {pool.label} {task}: missing={missing}; root={pool.result_root}"
        )
    return found


def segmentation_results(pool: Pool) -> dict[str, tuple[dict, Path]]:
    found: dict[str, tuple[dict, Path]] = {}
    for path in sorted(pool.result_root.glob(f"bio_segmentation/**/{CHECKPOINT}/results.json")):
        if "seed1" in path.parts or "seed2" in path.parts:
            continue
        dataset = path.parents[1].name
        if dataset not in SEGMENTATION_DATASETS:
            continue
        result = load_json(path)
        if dataset in found:
            old_value = float(found[dataset][0]["test"]["mDice"])
            new_value = float(result["test"]["mDice"])
            if not math.isclose(old_value, new_value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Conflicting segmentation results for {pool.label}/{dataset}")
            continue
        found[dataset] = (result, path)
    missing = sorted(SEGMENTATION_DATASETS - set(found))
    if missing:
        raise ValueError(f"Incomplete {pool.label} segmentation: missing={missing}")
    return found


def ood_result(pool: Pool) -> tuple[float, Path]:
    candidates = sorted(pool.ood_root.glob(f"ood/*/{CHECKPOINT}/last_result.json"))
    candidates = [path for path in candidates if path.parents[1].name not in {"xray", "cryo"}]
    if len(candidates) != 1:
        raise ValueError(f"Expected one combined OOD result for {pool.label}, found {candidates}")
    path = candidates[0]
    payload = load_json(path)
    if resolve_recorded_path(payload.get("checkpoint")) != expected_checkpoint(pool):
        raise ValueError(f"Wrong OOD checkpoint for {pool.label}: {path}")
    if payload.get("channel_policy") != "auto" or payload.get("channel_tta_samples") != 8:
        raise ValueError(f"Unexpected OOD channel protocol: {path}")
    missing = [key for key in OOD_KEYS if payload.get(key) is None]
    if missing:
        raise ValueError(f"OOD result is missing non-saturated metrics: {missing}")
    return mean(finite_float(payload[key], label=f"OOD/{key}") for key in OOD_KEYS), path


def training_metadata(pool: Pool) -> dict:
    selected = None
    for line in (pool.train_dir / "raw_loss_metrics.jsonl").read_text().splitlines():
        candidate = json.loads(line)
        if int(candidate.get("optimizer_update", -1)) == CHECKPOINT:
            selected = candidate
            break
    if selected is None:
        raise ValueError(f"Missing training metadata for {pool.label}/ck{CHECKPOINT}")
    if int(selected["effective_global_batch_size"]) != GLOBAL_BATCH:
        raise ValueError(f"Unexpected global batch for {pool.label}")
    return selected


def summarize_pool(pool: Pool) -> tuple[dict[str, float | str | int], list[dict[str, object]]]:
    classification = unique_results(pool, "classification", CLASSIFICATION_DATASETS)
    regression = unique_results(pool, "regression", REGRESSION_DATASETS)
    retrieval = unique_results(pool, "retrieval_clustering", RETRIEVAL_DATASETS)
    segmentation = segmentation_results(pool)
    ood, ood_path = ood_result(pool)
    training = training_metadata(pool)
    image_visits = int(training["image_visits"])

    metrics = {
        "classification": mean(
            finite_float(result["macro_f1"], label=f"classification/{dataset}/macro_f1")
            for dataset, (result, _) in classification.items()
        ),
        "regression": mean(
            finite_float(result["spearman"], label=f"regression/{dataset}/spearman")
            for dataset, (result, _) in regression.items()
        ),
        "retrieval_clustering": mean(
            finite_float(result["map_at_5"], label=f"retrieval/{dataset}/map_at_5")
            for dataset, (result, _) in retrieval.items()
        ),
        "segmentation": mean(
            finite_float(result["test"]["mDice"], label=f"segmentation/{dataset}/mDice")
            for dataset, (result, _) in segmentation.items()
        ),
        "ood": ood,
    }
    row: dict[str, float | str | int] = {
        "pool": pool.label,
        "samples": pool.samples,
        "checkpoint": CHECKPOINT,
        "image_visits": image_visits,
        "schedule_passes": float(training["epoch_float"]),
        "dataset_equivalent_passes": image_visits / pool.samples,
        "evaluation_autocast_dtype": "bf16",
        "evaluation_batch_size": 64,
        "evaluation_channel_policy": "auto",
        **metrics,
        "id4": mean(
            metrics[key]
            for key in ("classification", "regression", "retrieval_clustering", "segmentation")
        ),
        "five_family_mean": mean(metrics.values()),
    }
    details: list[dict[str, object]] = []
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
                    "pool": pool.label,
                    "checkpoint": CHECKPOINT,
                    "family": family,
                    "dataset": dataset,
                    "metric": metric,
                    "value": value,
                    "result_path": str(path),
                }
            )
    details.append(
        {
            "pool": pool.label,
            "checkpoint": CHECKPOINT,
            "family": "ood",
            "dataset": "xray+cryo",
            "metric": "non_saturated_5component_mean",
            "value": ood,
            "result_path": str(ood_path),
        }
    )
    return row, details


def load_s6_protocol() -> dict[str, dict]:
    label_to_pool = {
        "random_10": "0.1M",
        "random_20": "0.2M",
        "random_50": "0.5M",
        "random_100": "1.0M",
    }
    result = {}
    with S6_MANIFEST.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["label"] not in label_to_pool:
                continue
            checkpoint = int(row["checkpoint"])
            train_dir = ROOT / row["train_dir"]
            selected = None
            for line in (train_dir / "raw_loss_metrics.jsonl").read_text().splitlines():
                candidate = json.loads(line)
                if int(candidate.get("optimizer_update", -1)) == checkpoint:
                    selected = candidate
                    break
            if selected is None:
                raise ValueError(f"Missing S6 metadata for {row['label']}/ck{checkpoint}")
            samples = int(row["samples"])
            result[label_to_pool[row["label"]]] = {
                "checkpoint": checkpoint,
                "image_visits": int(selected["image_visits"]),
                "schedule_passes": float(selected["epoch_float"]),
                "dataset_equivalent_passes": int(selected["image_visits"]) / samples,
            }
    if set(result) != {pool.label for pool in POOLS}:
        raise ValueError("Incomplete S6 protocol manifest")
    return result


def build_protocol_comparison(current: list[dict]) -> list[dict]:
    s6 = load_s6_protocol()
    rows = []
    for row in current:
        old = s6[str(row["pool"])]
        rows.append(
            {
                "pool": row["pool"],
                "samples": row["samples"],
                "current_protocol": "fixed_compute",
                "current_checkpoint": row["checkpoint"],
                "current_image_visits": row["image_visits"],
                "current_dataset_equivalent_passes": row["dataset_equivalent_passes"],
                "s6_protocol": "fixed_pass",
                "s6_checkpoint": old["checkpoint"],
                "s6_image_visits": old["image_visits"],
                "s6_dataset_equivalent_passes": old["dataset_equivalent_passes"],
                "direct_performance_delta_valid": False,
                "reason": "data and compute budgets are not matched",
            }
        )
    return rows


def load_s6_id4() -> list[dict]:
    rows = []
    with S6_TABLE.open(newline="") as handle:
        for source in csv.DictReader(handle):
            if int(source["images"]) <= 0:
                continue
            rows.append(
                {
                    "pool": source["scale"],
                    "samples": int(source["images"]),
                    "id4": float(source["ID4_overall"]),
                }
            )
    if len(rows) != 4:
        raise ValueError(f"Expected four positive-data S6 rows, found {len(rows)}")
    return rows


def build_checkpoint_integrity() -> list[dict]:
    rows = []
    for pool in POOLS:
        path = expected_checkpoint(pool)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(64 * 1024 * 1024), b""):
                digest.update(block)
        with zipfile.ZipFile(path) as archive:
            pickle_names = [name for name in archive.namelist() if name.endswith("data.pkl")]
            if len(pickle_names) != 1:
                raise ValueError(f"Expected one data.pkl in {path}, found {pickle_names}")
            operations = list(pickletools.genops(archive.read(pickle_names[0])))
        archive_iteration = None
        for index, (_, argument, _) in enumerate(operations):
            if argument != "iteration":
                continue
            for operation, candidate, _ in operations[index + 1 : index + 8]:
                if operation.name.startswith("BININT") or operation.name == "INT":
                    archive_iteration = int(candidate)
                    break
            break
        if archive_iteration != CHECKPOINT:
            raise ValueError(
                f"Internal checkpoint iteration mismatch for {pool.label}: {archive_iteration}"
            )
        dataset_path = next(
            (
                line.split("dataset_path:", 1)[1].strip()
                for line in (pool.train_dir / "config.yaml").read_text().splitlines()
                if "dataset_path:" in line
            ),
            "",
        )
        rows.append(
            {
                "pool": pool.label,
                "checkpoint": CHECKPOINT,
                "archive_iteration": archive_iteration,
                "checkpoint_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
                "dataset_path": dataset_path,
                "checkpoint_path": str(path),
            }
        )
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Two data-scaling checkpoints have identical SHA256 digests")
    return rows


def style_axis(axis: plt.Axes, samples: np.ndarray) -> None:
    axis.set_xscale("log")
    axis.set_xlim(samples[0] * 0.72, samples[-1] * 1.45)
    axis.set_xticks(samples, ["0.1M", "0.2M", "0.5M", "1M"])
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(INK)
    axis.tick_params(length=0, colors=INK)
    axis.set_xlabel("unique microscopy images (log)")


def plot_tasks(current: list[dict]) -> None:
    samples = np.array([float(row["samples"]) for row in current])
    panels = (
        ("classification", "(a) Classification (25 datasets)", "macro-F1"),
        ("regression", "(b) Regression (2 datasets)", r"Spearman $\rho$"),
        ("retrieval_clustering", "(c) Retrieval (4 datasets)", "mAP@5"),
        ("segmentation", "(d) Segmentation (8 datasets)", "mDice"),
        ("ood", "(e) OOD (X-ray + cryo-EM)", "non-saturated composite"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.5))
    for axis, (key, title, ylabel) in zip(axes.flat, panels):
        values = np.array([float(row[key]) for row in current])
        spread = float(values.max() - values.min())
        pad = max(spread * 0.35, 0.004)
        style_axis(axis, samples)
        axis.set_ylim(float(values.min()) - pad, float(values.max()) + pad * 1.35)
        axis.plot(samples, values, color=COLORS[key], linewidth=2.2, marker="o", markersize=8)
        for x, value in zip(samples, values):
            axis.annotate(
                f"{value:.4f}",
                (x, value),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_ylabel(ylabel)
    axes.flat[5].axis("off")
    axes.flat[5].text(
        0.04,
        0.84,
        "STRICT FIXED-COMPUTE VIEW",
        color=COLORS["id4"],
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    axes.flat[5].text(
        0.04,
        0.68,
        "one raw checkpoint: ck8199\n8.397M image visits per point\nbf16 / auto-channel / batch 64",
        color=INK,
        fontsize=11,
        va="top",
        linespacing=1.35,
    )
    pass_text = "  ".join(
        f"{row['pool']}: {float(row['dataset_equivalent_passes']):.1f}x" for row in current
    )
    axes.flat[5].text(0.04, 0.36, "actual pool passes", color=MUTED, fontsize=9, va="top")
    axes.flat[5].text(0.04, 0.29, pass_text, color=MUTED, fontsize=9, va="top", wrap=True)
    axes.flat[5].text(
        0.04,
        0.10,
        "S6 fixed-pass points are not overlaid:\nthe training budgets are different.",
        color="#A15B3A",
        fontsize=9.5,
        va="top",
    )
    figure.suptitle(
        r"$\mathrm{S^+}$ full-suite data scaling at fixed compute",
        fontsize=16,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.075, wspace=0.28, hspace=0.38)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUT / f"splus_data_scaling_tasks.{suffix}",
            dpi=220 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(figure)


def plot_overall(current: list[dict]) -> None:
    samples = np.array([float(row["samples"]) for row in current])
    values = np.array([float(row["id4"]) for row in current])
    figure, axis = plt.subplots(figsize=(8.2, 6.3))
    style_axis(axis, samples)
    spread = float(values.max() - values.min())
    pad = max(spread * 0.70, 0.0025)
    axis.set_ylim(float(values.min()) - pad, float(values.max()) + pad)
    axis.plot(
        samples,
        values,
        color=COLORS["id4"],
        linewidth=2.8,
        marker="o",
        markersize=10,
        label="Current S+ (ck8199, 8.397M visits)",
    )
    for x, value, row in zip(samples, values, current):
        axis.annotate(
            f"{value:.4f}\n{float(row['dataset_equivalent_passes']):.1f}x pool",
            (x, value),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axis.set_ylabel("ID-4 family-balanced mean")
    axis.set_title(
        r"$\mathrm{S^+}$ fixed-compute data scaling",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    axis.legend(frameon=False, loc="lower right")
    figure.text(
        0.5,
        0.036,
        f"Same 8,396,800 visits; expanded y-axis; max-min ID-4 = {spread:.5f} (one seed).",
        ha="center",
        fontsize=8.8,
        color=MUTED,
    )
    figure.text(
        0.5,
        0.016,
        "ID-4 = equal mean of C25 macro-F1, Reg-2 Spearman, Ret-4 mAP@5, and Seg-8 mDice.",
        ha="center",
        fontsize=8.4,
        color=MUTED,
    )
    figure.subplots_adjust(left=0.12, right=0.97, top=0.89, bottom=0.15)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUT / f"splus_data_scaling_overall.{suffix}",
            dpi=220 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(figure)


def plot_protocol_side_by_side(current: list[dict], protocol: list[dict]) -> None:
    s6 = load_s6_id4()
    samples = np.array([float(row["samples"]) for row in current])
    current_values = np.array([float(row["id4"]) for row in current])
    s6_values = np.array([float(row["id4"]) for row in s6])
    low = min(float(current_values.min()), float(s6_values.min()))
    high = max(float(current_values.max()), float(s6_values.max()))
    pad = max((high - low) * 0.15, 0.004)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)
    panels = (
        (
            axes[0],
            current_values,
            COLORS["id4"],
            "(a) New S+ fixed compute",
            "ck8199 at every pool | 8.397M visits",
        ),
        (
            axes[1],
            s6_values,
            "#9A6A4F",
            "(b) Original S6 fixed pass",
            "ck389/779/1949/3899 | about 15.23 pool passes",
        ),
    )
    for axis, values, color, title, subtitle in panels:
        style_axis(axis, samples)
        axis.set_ylim(low - pad, high + pad)
        axis.plot(samples, values, color=color, linewidth=2.5, marker="o", markersize=8)
        for x, value in zip(samples, values):
            axis.annotate(
                f"{value:.4f}",
                (x, value),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
        axis.set_title(title, fontsize=13, fontweight="bold", pad=18)
        axis.text(0.5, 1.01, subtitle, transform=axis.transAxes, ha="center", color=MUTED, fontsize=9)
    axes[0].set_ylabel("ID-4 family-balanced mean")
    current_visits = " / ".join(f"{float(row['current_image_visits']) / 1e6:.2f}" for row in protocol)
    s6_visits = " / ".join(f"{float(row['s6_image_visits']) / 1e6:.2f}" for row in protocol)
    axes[0].text(0.5, -0.20, f"visits (M): {current_visits}", transform=axes[0].transAxes, ha="center", fontsize=8.5, color=MUTED)
    axes[1].text(0.5, -0.20, f"visits (M): {s6_visits}", transform=axes[1].transAxes, ha="center", fontsize=8.5, color=MUTED)
    figure.suptitle("S+ data curves under two different budget definitions", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.025,
        "Side-by-side for context only: a pointwise performance delta is not valid because compute is not matched.",
        ha="center",
        fontsize=9,
        color="#A15B3A",
    )
    figure.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.24, wspace=0.16)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUT / f"splus_data_scaling_protocol_side_by_side.{suffix}",
            dpi=220 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(figure)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(current: list[dict], protocol: list[dict]) -> None:
    lines = [
        "# Current S+ fixed-compute full-suite data scaling",
        "",
        "This report uses the locked S+ recipe at raw checkpoint 8199 for every pool.",
        "All frozen probes are regenerated or selected under one protocol: bf16 feature",
        "extraction, batch 64, auto channel handling, eight channel-TTA samples, and the",
        "best-resolution table. Proxy-9 fp16 results are not substituted into this report.",
        "",
        "Coverage is strict at every point: C25, Reg-2, Ret-4, Seg-8, and the established",
        "five-component non-saturated OOD composite. The builder fails on missing, duplicate,",
        "wrong-checkpoint, wrong-channel-policy, or non-finite results.",
        "",
        "| pool | visits | pool passes | C25 | Reg-2 | Ret-4 | Seg-8 | OOD | ID-4 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in current:
        lines.append(
            f"| {row['pool']} | {int(row['image_visits']):,} | "
            f"{float(row['dataset_equivalent_passes']):.3f} | "
            f"{float(row['classification']):.6f} | {float(row['regression']):.6f} | "
            f"{float(row['retrieval_clustering']):.6f} | {float(row['segmentation']):.6f} | "
            f"{float(row['ood']):.6f} | {float(row['id4']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Why S6 is not overlaid",
            "",
            "The historical S6 data curve is fixed-pass: its checkpoint and image visits grow",
            "with pool size. The current curve is fixed-compute: ck8199 and 8.397M visits at",
            "every pool size. A pointwise performance delta would combine recipe, data, and",
            "compute changes, so the previous overlay and claimed gains are withdrawn.",
            "The corrected ID-4 range across the four pools is only about 0.00252 with one",
            "training/probe seed. Treat the curve as approximately flat at this resolution;",
            "do not rank 0.1M above 1.0M without seed repeats or paired uncertainty estimates.",
            "",
            "| pool | current ck | current visits | current pool passes | S6 ck | S6 visits | S6 pool passes |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in protocol:
        lines.append(
            f"| {row['pool']} | {row['current_checkpoint']} | "
            f"{int(row['current_image_visits']):,} | "
            f"{float(row['current_dataset_equivalent_passes']):.3f} | "
            f"{row['s6_checkpoint']} | {int(row['s6_image_visits']):,} | "
            f"{float(row['s6_dataset_equivalent_passes']):.3f} |"
        )
    lines.extend(
        [
            "",
            "The legacy `current_vs_previous.csv` filename is retained for downstream users,",
            "but its contents are now a protocol audit with `direct_performance_delta_valid=false`;",
            "it no longer contains misleading performance deltas.",
            "",
            "## Outputs",
            "",
            "- `splus_data_scaling_tasks.{png,pdf,svg}`",
            "- `splus_data_scaling_overall.{png,pdf,svg}`",
            "- `splus_data_scaling_protocol_side_by_side.{png,pdf,svg}`",
            "- `current_full_summary.csv`",
            "- `current_full_per_dataset.csv`",
            "- `checkpoint_protocol_vs_s6.csv`",
            "- `checkpoint_integrity.csv` (archive iteration, SHA256, and training dataset path)",
            "- `current_vs_previous.csv` (legacy alias of the protocol audit)",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    current = []
    details = []
    for pool in POOLS:
        row, pool_details = summarize_pool(pool)
        if len(pool_details) != 40:
            raise ValueError(f"Expected 40 provenance rows for {pool.label}, found {len(pool_details)}")
        current.append(row)
        details.extend(pool_details)
    protocol = build_protocol_comparison(current)
    integrity = build_checkpoint_integrity()

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
    plot_tasks(current)
    plot_overall(current)
    plot_protocol_side_by_side(current, protocol)
    write_csv(OUT / "current_full_summary.csv", current)
    write_csv(OUT / "current_full_per_dataset.csv", details)
    write_csv(OUT / "checkpoint_protocol_vs_s6.csv", protocol)
    write_csv(OUT / "checkpoint_integrity.csv", integrity)
    write_csv(OUT / "current_vs_previous.csv", protocol)
    write_readme(current, protocol)
    print(f"wrote strict fixed-compute S+ data-scaling report to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
