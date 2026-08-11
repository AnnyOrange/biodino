#!/usr/bin/env python3
"""Build strict 8-pass and 15-pass S+ full-suite data-scaling curves."""

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
import yaml


ROOT = Path(__file__).resolve().parents[1]
NEW_EVAL_ROOT = (
    ROOT
    / "outputs/02_eval_runs/splus_fixed_pass_data_scaling_full_bf16_auto_20260723"
)
COMPUTE_EVAL_ROOT = (
    ROOT / "outputs/02_eval_runs/splus_fixed1m_full_compute_clean_bf16_auto_20260723"
)
ONE_M_CK8199_ROOT = (
    ROOT
    / "outputs/02_eval_runs/"
    "S6sigreg005_raw_ck8199__full_dense_local_bf16_b64_clean_20260723"
)
ONE_M_CK8199_OOD_ROOT = (
    ROOT / "outputs/02_eval_runs/splus_fixed_compute_full_ck8199_20260723/random_100"
)
OUT = ROOT / "outputs/00_reports/splus_fixed_pass_data_scaling_20260723"

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
PASS15 = "#A56A4D"
PRETRAINED_TEACHER = (
    "/mnt/huawei_deepcad/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
)
RGB_MEAN = [0.514666, 0.488834, 0.498267]
RGB_STD = [0.338707, 0.339202, 0.336091]


@dataclass(frozen=True)
class Pool:
    label: str
    short_label: str
    samples: int
    official_epoch_length: int
    train_dir: Path
    checkpoint8: int
    checkpoint15: int


POOLS = (
    Pool(
        "0.1M",
        "random_10",
        104_877,
        103,
        ROOT
        / "outputs/01_training_runs/"
        "DscalePass_splus_sigreg005_random10_e15_b1024_seed0_local4gbs64acc4",
        823,
        1544,
    ),
    Pool(
        "0.2M",
        "random_20",
        209_754,
        205,
        ROOT
        / "outputs/01_training_runs/"
        "DscalePass_splus_sigreg005_random20_e15_b1024_seed0_local4gbs64acc4",
        1639,
        3074,
    ),
    Pool(
        "0.5M",
        "random_50",
        524_385,
        513,
        ROOT
        / "outputs/01_training_runs/"
        "DscalePass_splus_sigreg005_random50_e15_b1024_seed0_local4gbs64acc4",
        4103,
        7694,
    ),
    Pool(
        "1.0M",
        "random_100",
        1_048_771,
        1025,
        ROOT
        / "outputs/01_training_runs/"
        "S6sigreg005_rgb_robust_biosafe256_b1024_officialprec_lr1e-4_wu2_e15_local5090_20260716",
        8199,
        15374,
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


def expected_checkpoint(pool: Pool, checkpoint: int) -> Path:
    return (pool.train_dir / "ckpt" / str(checkpoint) / "checkpoint.pth").resolve()


def result_roots(pool: Pool, checkpoint: int) -> tuple[Path, Path]:
    if pool.short_label == "random_100" and checkpoint == pool.checkpoint8:
        return ONE_M_CK8199_ROOT, ONE_M_CK8199_OOD_ROOT
    if pool.short_label == "random_100":
        root = COMPUTE_EVAL_ROOT / f"ckpt_{checkpoint}"
        return root, root
    root = NEW_EVAL_ROOT / pool.short_label / f"ckpt_{checkpoint}"
    return root, root


def unique_results(
    pool: Pool,
    checkpoint: int,
    root: Path,
    task: str,
    expected: set[str],
) -> dict[str, tuple[dict, Path]]:
    found: dict[str, tuple[dict, Path]] = {}
    wanted_checkpoint = expected_checkpoint(pool, checkpoint)
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
                f"Duplicate {pool.label}/ck{checkpoint}/{task}/{dataset}: "
                f"{found[dataset][1]} and {path}"
            )
        found[dataset] = (result, path)
    missing = sorted(expected - set(found))
    if missing:
        raise ValueError(
            f"Incomplete {pool.label}/ck{checkpoint}/{task}: missing={missing}; root={root}"
        )
    return found


def segmentation_results(
    pool: Pool, checkpoint: int, root: Path
) -> dict[str, tuple[dict, Path]]:
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
                raise ValueError(
                    f"Conflicting segmentation result for {pool.label}/ck{checkpoint}/{dataset}"
                )
            continue
        found[dataset] = (result, path)
    missing = sorted(SEGMENTATION_DATASETS - set(found))
    if missing:
        raise ValueError(
            f"Incomplete {pool.label}/ck{checkpoint}/segmentation: missing={missing}; root={root}"
        )
    return found


def ood_result(pool: Pool, checkpoint: int, root: Path) -> tuple[dict, Path]:
    candidates = sorted(root.glob(f"ood/*/{checkpoint}/last_result.json"))
    candidates = [path for path in candidates if path.parents[1].name not in {"xray", "cryo"}]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one combined OOD result for {pool.label}/ck{checkpoint}, found {candidates}"
        )
    path = candidates[0]
    result = load_json(path)
    if resolve_recorded_path(result.get("checkpoint")) != expected_checkpoint(pool, checkpoint):
        raise ValueError(f"Wrong OOD checkpoint: {path}")
    if result.get("channel_policy") != "auto" or result.get("channel_tta_samples") != 8:
        raise ValueError(f"Unexpected OOD channel protocol: {path}")
    missing = [key for key in OOD_KEYS if result.get(key) is None]
    if missing:
        raise ValueError(f"Incomplete OOD metrics for {pool.label}/ck{checkpoint}: {missing}")
    return result, path


def validate_training_recipe(
    pool: Pool,
    training: dict,
    *,
    expected_sigreg: bool,
    expected_lr: float,
) -> None:
    config = yaml.safe_load((pool.train_dir / "config.yaml").read_text())
    train = config["train"]
    optim = config["optim"]
    crops = config["crops"]
    student = config["student"]
    teacher = config["teacher"]
    sigreg_enabled = bool(config.get("sigreg", {}).get("enabled", False))
    channel_subset_enabled = bool(
        config.get("channel_subset", {}).get("enabled", False)
    )
    checks = {
        "train.seed": (train["seed"], 0),
        "train.OFFICIAL_EPOCH_LENGTH": (
            train["OFFICIAL_EPOCH_LENGTH"],
            pool.official_epoch_length,
        ),
        "train.batch_size_per_gpu": (train["batch_size_per_gpu"], 64),
        "optim.epochs": (optim["epochs"], 15),
        "optim.lr": (float(optim["lr"]), expected_lr),
        "optim.warmup_epochs": (optim["warmup_epochs"], 2),
        "optim.min_lr": (float(optim["min_lr"]), 1e-6),
        "optim.freeze_last_layer_epochs": (optim["freeze_last_layer_epochs"], 1),
        "student.arch": (student["arch"], "vit_small"),
        "student.resume_from_teacher_chkpt": (
            student["resume_from_teacher_chkpt"],
            PRETRAINED_TEACHER,
        ),
        "teacher.warmup_teacher_temp_epochs": (
            teacher["warmup_teacher_temp_epochs"],
            5,
        ),
        "crops.global_crops_size": (crops["global_crops_size"], 256),
        "crops.local_crops_size": (crops["local_crops_size"], 112),
        "crops.augmentation_policy": (crops["augmentation_policy"], "bio_safe"),
        "crops.rgb_mean": (crops["rgb_mean"], RGB_MEAN),
        "crops.rgb_std": (crops["rgb_std"], RGB_STD),
        "compute_precision.param_dtype": (
            config["compute_precision"]["param_dtype"],
            "bf16",
        ),
        "sigreg.enabled": (sigreg_enabled, expected_sigreg),
        "channel_subset.enabled": (channel_subset_enabled, False),
        "raw.dataset_path": (training["dataset_path"], train["dataset_path"]),
        "raw.augmentation_policy": (training["augmentation_policy"], "bio_safe"),
        "raw.decoder_rgb_mean": (training["decoder_rgb_mean"], RGB_MEAN),
        "raw.decoder_rgb_std": (training["decoder_rgb_std"], RGB_STD),
    }
    if expected_sigreg:
        if not math.isclose(
            float(training["sigreg_loss_weight"]), 0.05, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(f"Unexpected SIGReg weight for {pool.label}")
    elif "sigreg_loss" in training or "sigreg_loss_weight" in training:
        raise ValueError(f"Unexpected SIGReg metrics for {pool.label}")
    failures = [
        f"{name}: {actual!r} != {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if failures:
        raise ValueError(f"Recipe mismatch for {pool.label}: " + "; ".join(failures))


def training_metadata(
    pool: Pool, checkpoint: int, expected_effective_global_batch_size: int
) -> dict:
    selected = None
    for line in (pool.train_dir / "raw_loss_metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if int(row.get("optimizer_update", -1)) == checkpoint:
            selected = row
            break
    if selected is None:
        raise ValueError(f"Missing training metadata for {pool.label}/ck{checkpoint}")
    if (
        int(selected["effective_global_batch_size"])
        != expected_effective_global_batch_size
    ):
        raise ValueError(f"Unexpected global batch for {pool.label}/ck{checkpoint}")
    expected_passes = (checkpoint + 1) / pool.official_epoch_length
    if not math.isclose(float(selected["epoch_float"]), expected_passes, abs_tol=1e-9):
        raise ValueError(f"Unexpected scheduler pass count for {pool.label}/ck{checkpoint}")
    return selected


def summarize_point(
    pool: Pool,
    checkpoint: int,
    pass_budget: int,
    *,
    roots: tuple[Path, Path] | None = None,
    expected_effective_global_batch_size: int = 1024,
    expected_sigreg: bool = True,
    expected_lr: float = 1e-4,
) -> tuple[dict, list[dict]]:
    root, ood_root = roots if roots is not None else result_roots(pool, checkpoint)
    classification = unique_results(
        pool, checkpoint, root, "classification", CLASSIFICATION_DATASETS
    )
    regression = unique_results(pool, checkpoint, root, "regression", REGRESSION_DATASETS)
    retrieval = unique_results(
        pool, checkpoint, root, "retrieval_clustering", RETRIEVAL_DATASETS
    )
    segmentation = segmentation_results(pool, checkpoint, root)
    ood, ood_path = ood_result(pool, checkpoint, ood_root)
    training = training_metadata(
        pool, checkpoint, expected_effective_global_batch_size
    )
    validate_training_recipe(
        pool,
        training,
        expected_sigreg=expected_sigreg,
        expected_lr=expected_lr,
    )
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
    image_visits = int(training["image_visits"])
    real_global_batch = int(training["real_global_batch_size"])
    local_batch = int(training["local_batch_size"])
    row = {
        "pool": pool.label,
        "samples": pool.samples,
        "pass_budget": pass_budget,
        "checkpoint": checkpoint,
        "official_epoch_length": pool.official_epoch_length,
        "schedule_passes": float(training["epoch_float"]),
        "image_visits": image_visits,
        "dataset_equivalent_passes": image_visits / pool.samples,
        "world_size": real_global_batch // local_batch,
        "local_batch_size": local_batch,
        "real_global_batch_size": real_global_batch,
        "gradient_accumulation_steps": int(training["accum_steps"]),
        "effective_global_batch_size": int(training["effective_global_batch_size"]),
        "base_learning_rate": expected_lr,
        "sigreg_enabled": expected_sigreg,
        "sigreg_loss_weight": 0.05 if expected_sigreg else 0.0,
        "scheduler_horizon_passes": 15,
        "seed": 0,
        "dataset_path": training["dataset_path"],
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
                    "pool": pool.label,
                    "pass_budget": pass_budget,
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
                "pool": pool.label,
                "pass_budget": pass_budget,
                "checkpoint": checkpoint,
                "family": "ood",
                "dataset": "xray+cryo",
                "metric": key,
                "value": ood[key],
                "result_path": str(ood_path),
            }
        )
    if len(details) != 44:
        raise ValueError(
            f"Expected 44 provenance rows for {pool.label}/ck{checkpoint}, found {len(details)}"
        )
    return row, details


def checkpoint_integrity() -> list[dict]:
    rows = []
    for pool in POOLS:
        for pass_budget, checkpoint in (
            (8, pool.checkpoint8),
            (15, pool.checkpoint15),
        ):
            path = expected_checkpoint(pool, checkpoint)
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
            if archive_iteration != checkpoint:
                raise ValueError(
                    f"Internal checkpoint iteration mismatch for {pool.label}: "
                    f"expected={checkpoint} actual={archive_iteration}"
                )
            rows.append(
                {
                    "pool": pool.label,
                    "pass_budget": pass_budget,
                    "checkpoint": checkpoint,
                    "archive_iteration": archive_iteration,
                    "checkpoint_bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                    "checkpoint_path": str(path),
                }
            )
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Two fixed-pass checkpoints have identical SHA256 digests")
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


def split_passes(current: list[dict]) -> tuple[list[dict], list[dict]]:
    return (
        [row for row in current if row["pass_budget"] == 8],
        [row for row in current if row["pass_budget"] == 15],
    )


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def scaling_diagnostics(current: list[dict]) -> list[dict]:
    rows = []
    for pass_budget, points in zip((8, 15), split_passes(current)):
        samples = np.array([float(row["samples"]) for row in points])
        log2_samples = np.log2(samples)
        for key in (*FAMILY_KEYS, "id4_overall", "five_family_mean"):
            values = np.array([float(row[key]) for row in points])
            slope, intercept = np.polyfit(log2_samples, values, 1)
            fitted = slope * log2_samples + intercept
            residual = float(np.sum((values - fitted) ** 2))
            total = float(np.sum((values - values.mean()) ** 2))
            diffs = np.diff(values)
            value_ranks = average_ranks(values)
            rho = (
                float(np.corrcoef(np.arange(len(values)), value_ranks)[0, 1])
                if float(np.std(value_ranks)) > 0.0
                else 0.0
            )
            endpoint_gain = float(values[-1] - values[0])
            positive_steps = int(np.sum(diffs > 0.0))
            if slope > 0.0 and positive_steps == len(diffs):
                evidence = "positive_monotonic"
            elif slope > 0.0 and endpoint_gain > 0.0:
                evidence = "positive_nonmonotonic"
            else:
                evidence = "no_positive_scaling"
            rows.append(
                {
                    "pass_budget": pass_budget,
                    "metric": key,
                    "slope_per_data_doubling": float(slope),
                    "log2_linear_r2": 1.0 - residual / total if total > 0.0 else 1.0,
                    "spearman_rho_data_vs_score": rho,
                    "endpoint_gain_0.1m_to_1m": endpoint_gain,
                    "positive_adjacent_steps": positive_steps,
                    "adjacent_steps": len(diffs),
                    "monotonic_non_decreasing": bool(np.all(diffs >= 0.0)),
                    "evidence": evidence,
                }
            )
    return rows


def plot_tasks(current: list[dict]) -> None:
    pass8, pass15 = split_passes(current)
    samples = np.array([float(row["samples"]) for row in pass8])
    panels = (
        ("classification25_macro_f1", "(a) Classification (25 datasets)", "macro-F1"),
        ("regression2_spearman", "(b) Regression (2 datasets)", r"Spearman $\rho$"),
        ("retrieval4_map_at_5", "(c) Retrieval (4 datasets)", "mAP@5"),
        ("segmentation8_mdice", "(d) Segmentation (8 datasets)", "mDice"),
        ("ood_composite", "(e) OOD (X-ray + cryo-EM)", "non-saturated composite"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.5))
    for axis, (key, title, ylabel) in zip(axes.flat, panels):
        y8 = np.array([float(row[key]) for row in pass8])
        y15 = np.array([float(row[key]) for row in pass15])
        low = min(float(y8.min()), float(y15.min()))
        high = max(float(y8.max()), float(y15.max()))
        pad = max((high - low) * 0.18, 0.004)
        style_axis(axis, samples)
        axis.set_ylim(low - pad, high + pad)
        axis.plot(
            samples,
            y15,
            color=PASS15,
            linewidth=1.9,
            linestyle=(0, (5, 3)),
            marker="o",
            markersize=6,
            label="S+, 15 pass",
        )
        axis.plot(
            samples,
            y8,
            color=COLORS[key],
            linewidth=2.5,
            marker="o",
            markersize=8,
            label="S+ sweet budget, 8 pass",
        )
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_ylabel(ylabel)
    axes.flat[5].axis("off")
    axes.flat[5].text(
        0.04,
        0.86,
        "FIXED-PASS DATA SCALING",
        color=COLORS["id4_overall"],
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    axes.flat[5].text(
        0.04,
        0.68,
        "solid: S+ sweet budget at 8 pool passes\ndashed: S+ 15-pass stability control",
        color=INK,
        fontsize=10.5,
        va="top",
        linespacing=1.4,
    )
    axes.flat[5].text(
        0.04,
        0.34,
        "Every S+ point: GBS 1024, SIGReg 0.05,\nrobust bio-safe recipe, bf16 full suite.\nNo Proxy-9 substitutions.",
        color=MUTED,
        fontsize=9.5,
        va="top",
        linespacing=1.35,
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))
    figure.suptitle(
        r"$\mathrm{S^+}$ full-suite data scaling at matched pool passes",
        fontsize=16,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.10, wspace=0.28, hspace=0.38)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUT / f"splus_data_scaling_tasks.{suffix}",
            dpi=220 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(figure)


def plot_overall(current: list[dict], diagnostics: list[dict]) -> None:
    pass8, pass15 = split_passes(current)
    samples = np.array([float(row["samples"]) for row in pass8])
    y8 = np.array([float(row["id4_overall"]) for row in pass8])
    y15 = np.array([float(row["id4_overall"]) for row in pass15])
    low = min(float(y8.min()), float(y15.min()))
    high = max(float(y8.max()), float(y15.max()))
    pad = max((high - low) * 0.18, 0.004)
    figure, axis = plt.subplots(figsize=(9.0, 6.5))
    style_axis(axis, samples)
    axis.set_ylim(low - pad, high + pad)
    slope, intercept = np.polyfit(np.log2(samples), y8, 1)
    trend_samples = np.geomspace(samples[0], samples[-1], 100)
    axis.plot(
        trend_samples,
        slope * np.log2(trend_samples) + intercept,
        color=COLORS["id4_overall"],
        linewidth=1.4,
        linestyle=(0, (2, 3)),
        alpha=0.65,
        label=f"8-pass log-linear trend ({slope:+.4f}/doubling)",
    )
    axis.plot(
        samples,
        y15,
        color=PASS15,
        linewidth=2.1,
        linestyle=(0, (5, 3)),
        marker="o",
        markersize=7,
        label="S+, 15 pass",
    )
    axis.plot(
        samples,
        y8,
        color=COLORS["id4_overall"],
        linewidth=2.9,
        marker="o",
        markersize=10,
        label="S+ sweet budget, 8 pass",
    )
    for x, value, row in zip(samples, y8, pass8):
        axis.annotate(
            f"{value:.4f}\nck{row['checkpoint']}",
            (x, value),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    axis.set_ylabel("ID-4 family-balanced mean")
    axis.set_title(
        r"$\mathrm{S^+}$ fixed-pass data scaling",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    axis.legend(frameon=False, loc="lower right")
    diagnostic = next(
        row
        for row in diagnostics
        if row["pass_budget"] == 8 and row["metric"] == "id4_overall"
    )
    figure.text(
        0.5,
        0.027,
        "ID-4 equally weights C25, Reg-2, Ret-4, and Seg-8. "
        f"Data-score Spearman={diagnostic['spearman_rho_data_vs_score']:.3f}; "
        f"log-linear R2={diagnostic['log2_linear_r2']:.3f}.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    figure.subplots_adjust(left=0.12, right=0.97, top=0.89, bottom=0.12)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            OUT / f"splus_data_scaling_overall.{suffix}",
            dpi=220 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(figure)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(current: list[dict], diagnostics: list[dict]) -> None:
    pass8, pass15 = split_passes(current)
    id4_diagnostic = next(
        row
        for row in diagnostics
        if row["pass_budget"] == 8 and row["metric"] == "id4_overall"
    )
    metric_names = {
        "classification25_macro_f1": "Classification-25",
        "regression2_spearman": "Regression-2",
        "retrieval4_map_at_5": "Retrieval-4",
        "segmentation8_mdice": "Segmentation-8",
        "ood_composite": "OOD composite",
        "id4_overall": "ID-4 overall",
        "five_family_mean": "Five-family mean",
    }
    lines = [
        "# S+ fixed-pass full-suite data scaling",
        "",
        "This report tests only whether the locked S+ sweet-spot recipe scales with unique",
        "training data. Dataset size and checkpoint are co-scaled so every point sees its own",
        "pool for the same number of passes. The primary curve uses the selected eight-pass",
        "budget; the 15-pass curve is a within-S+ stability control. No S6 comparison is used.",
        "",
        "All S+ results use the complete strict suite: C25, Reg-2, Ret-4, Seg-8, and the",
        "five-component non-saturated OOD composite. Evaluation is bf16, batch 64, auto",
        "channel policy, TTA8, and best resolution. Proxy-9 results are never substituted.",
        "All points use effective global batch 1024. The new small-pool runs use four ranks",
        "with accumulation 4; the existing 1M parent used eight ranks with accumulation 2.",
        "This preserves samples per optimizer update, scheduler length, and image visits; the",
        "topology difference is retained explicitly in the summary CSV.",
        "",
        "| pool | 8-pass ck | 8-pass visits | 8-pass ID-4 | 15-pass ck | 15-pass visits | 15-pass ID-4 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row8, row15 in zip(pass8, pass15):
        lines.append(
            f"| {row8['pool']} | {row8['checkpoint']} | {int(row8['image_visits']):,} | "
            f"{float(row8['id4_overall']):.6f} | {row15['checkpoint']} | "
            f"{int(row15['image_visits']):,} | {float(row15['id4_overall']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Scaling diagnostic",
            "",
            "The log-linear slope is the absolute score change expected for each doubling of",
            "unique data. Positive monotonic evidence requires all three adjacent data-size",
            "steps to improve; positive non-monotonic evidence requires a positive fitted slope",
            "and a positive 0.1M-to-1M endpoint gain.",
            "",
            "| 8-pass metric | slope / doubling | Spearman | R2 | positive steps | assessment |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in diagnostics:
        if row["pass_budget"] != 8:
            continue
        lines.append(
            f"| {metric_names[row['metric']]} | "
            f"{float(row['slope_per_data_doubling']):+.6f} | "
            f"{float(row['spearman_rho_data_vs_score']):.3f} | "
            f"{float(row['log2_linear_r2']):.3f} | "
            f"{int(row['positive_adjacent_steps'])}/{int(row['adjacent_steps'])} | "
            f"{row['evidence']} |"
        )
    lines.extend(
        [
            "",
            f"Primary ID-4 assessment: `{id4_diagnostic['evidence']}`. This is empirical",
            "fixed-pass scaling evidence across four data sizes and one training seed; it is",
            "not presented as a universal asymptotic power-law exponent.",
            "",
            "## Outputs",
            "",
            "- `splus_data_scaling_tasks.{png,pdf,svg}`",
            "- `splus_data_scaling_overall.{png,pdf,svg}`",
            "- `splus_fixed_pass_summary.csv`",
            "- `splus_fixed_pass_per_dataset.csv`",
            "- `splus_data_scaling_diagnostics.csv`",
            "- `checkpoint_integrity.csv`",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    current = []
    details = []
    for pool in POOLS:
        for pass_budget, checkpoint in (
            (8, pool.checkpoint8),
            (15, pool.checkpoint15),
        ):
            row, point_details = summarize_point(pool, checkpoint, pass_budget)
            current.append(row)
            details.extend(point_details)
    current.sort(key=lambda row: (int(row["pass_budget"]), int(row["samples"])))
    details.sort(
        key=lambda row: (
            int(row["pass_budget"]),
            next(pool.samples for pool in POOLS if pool.label == row["pool"]),
            str(row["family"]),
            str(row["dataset"]),
            str(row["metric"]),
        )
    )
    diagnostics = scaling_diagnostics(current)
    integrity = checkpoint_integrity()

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
    plot_overall(current, diagnostics)
    write_csv(OUT / "splus_fixed_pass_summary.csv", current)
    write_csv(OUT / "splus_fixed_pass_per_dataset.csv", details)
    write_csv(OUT / "splus_data_scaling_diagnostics.csv", diagnostics)
    write_csv(OUT / "checkpoint_integrity.csv", integrity)
    write_readme(current, diagnostics)
    print(f"wrote strict fixed-pass S+ data-scaling report to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
