#!/usr/bin/env python3
"""Build separate S6b and current S+ compute-scaling figures."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/00_reports/splus_compute_scaling_separate_20260723"

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
    *(("classification", dataset) for dataset in CLASSIFICATION_DATASETS),
    ("regression", "bbbc005"),
    ("retrieval_clustering", "lc25000"),
}
NMI_SEEDS = tuple(range(10))

BASE_ROOT = ROOT / "outputs/02_eval_runs/S7z_splus_official_objective_proxy_cpu1"
OLD_TRAIN_ROOT = (
    ROOT
    / "outputs/01_training_runs/"
    "S6b_rgb_robust_biosafe256_b4096_lr2e-4_wu2_e30"
)
NEW_TRAIN_ROOT = (
    ROOT
    / "outputs/01_training_runs/"
    "S6sigreg005_rgb_robust_biosafe256_b1024_officialprec_lr1e-4_wu2_e15_local5090_20260716"
)
NEW_EVAL_ROOT = (
    ROOT
    / "outputs/02_eval_runs/"
    "S6sigreg005_rgb_robust_biosafe256_b1024__compute_proxy_curve_20260721"
)
S6B_DATA_TABLE = (
    ROOT / "outputs/03_comparisons/splus_random_data_scaling_e15/CANONICAL_verified_table.csv"
)
S6B_DATA_MANIFEST = (
    ROOT / "outputs/03_comparisons/splus_random_data_scaling_e15/run_manifest.csv"
)

COLORS = {
    "classification7_balanced_accuracy": "#3E6D9C",
    "bbbc005_r2": "#C6A9CE",
    "lc25000_map_at_5": "#8FB8DC",
    "lc25000_nmi": "#F0A868",
    "family_balanced_mean": "#2F6E8F",
    "proxy9_mean": "#C6803B",
    "classification25_macro_f1": "#3E6D9C",
    "regression2_spearman": "#C6A9CE",
    "retrieval4_map_at_5": "#8FB8DC",
    "segmentation8_mdice": "#6FBF8B",
    "ood_composite": "#F0A868",
    "id4_overall": "#2F6E8F",
}
GRID = "#DADDE2"
INK = "#2B2B2B"
MUTED = "#8A9099"
SELECTED = "#E4572E"


@dataclass(frozen=True)
class Recipe:
    key: str
    title: str
    subtitle: str
    curve_label: str
    checkpoints: tuple[tuple[int, int], ...]
    selected_checkpoint: int
    global_batch: int
    train_root: Path


RECIPES = (
    Recipe(
        key="s6b",
        title=r"$\mathrm{S6b}$ Proxy-9 compute scaling by task (fixed 1M pool)",
        subtitle="no SIGReg | nominal GBS 4096 | 4-30 passes",
        curve_label="S6b",
        checkpoints=(
            (1039, 4),
            (2079, 8),
            (3119, 12),
            (3899, 15),
            (5199, 20),
            (6499, 25),
            (7799, 30),
        ),
        selected_checkpoint=3899,
        global_batch=4096,
        train_root=OLD_TRAIN_ROOT,
    ),
    Recipe(
        key="splus_new",
        title=r"$\mathrm{S^+}$ new-recipe compute scaling by task (fixed 1M pool)",
        subtitle="SIGReg 0.05 | nominal GBS 1024 | 1-15 passes",
        curve_label=r"$\mathrm{S^+}$ new",
        checkpoints=(
            (1024, 1),
            (2049, 2),
            (4099, 4),
            (6149, 6),
            (8199, 8),
            (10249, 10),
            (12299, 12),
            (15374, 15),
        ),
        selected_checkpoint=8199,
        global_batch=1024,
        train_root=NEW_TRAIN_ROOT,
    ),
)

_NMI_CACHE: dict[str, tuple[list[float], float, float, float, float]] = {}


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


def old_eval_roots(checkpoint: int) -> tuple[Path, ...]:
    prefix = "S6b_rgb_robust_biosafe256_b4096_lr2e-4_wu2_e30"
    base = ROOT / "outputs/02_eval_runs"
    return (
        base / f"{prefix}__ckpt{checkpoint}_cls_chammi_cp",
        base / f"{prefix}__ckpt{checkpoint}_cls_chammi_hpa",
        base / f"{prefix}__S6_ck{checkpoint}_regret_fill3090_singlecard_20260712_1616",
    )


def eval_roots(recipe: Recipe, checkpoint: int) -> tuple[Path, ...]:
    if recipe.key == "s6b":
        return old_eval_roots(checkpoint)
    return (NEW_EVAL_ROOT,)


def validate_protocol(result: dict, path: Path, *, expected_channel: str = "auto") -> None:
    if result.get("channel_policy") != expected_channel:
        raise ValueError(f"Expected {expected_channel} channel policy in {path}")
    if int(result.get("channel_tta_samples", -1)) != 8:
        raise ValueError(f"Expected 8 channel TTA samples in {path}")
    if result.get("task") in {"classification", "regression"}:
        if result.get("resolution_protocol") != "best":
            raise ValueError(f"Expected best resolution protocol in {path}")


def load_checkpoint_results(recipe: Recipe, checkpoint: int) -> dict:
    bucket: dict[tuple[str, str], tuple[dict, Path]] = {}
    for root in eval_roots(recipe, checkpoint):
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("last_result.json")):
            result = json.loads(path.read_text())
            key = (str(result.get("task", "")), str(result.get("dataset", "")))
            if key not in EXPECTED_KEYS or checkpoint_id(result, path) != checkpoint:
                continue
            validate_protocol(result, path)
            if key in bucket:
                raise ValueError(f"Duplicate {recipe.key} ck{checkpoint} {key}: {path}")
            bucket[key] = (result, path)
    if set(bucket) != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - set(bucket))
        extra = sorted(set(bucket) - EXPECTED_KEYS)
        raise ValueError(
            f"{recipe.key} ck{checkpoint} coverage {len(bucket)}/9; "
            f"missing={missing}, extra={extra}"
        )
    return bucket


def load_base_results() -> dict:
    bucket: dict[tuple[str, str], tuple[dict, Path]] = {}
    for path in sorted(BASE_ROOT.rglob("last_result.json")):
        result = json.loads(path.read_text())
        key = (str(result.get("task", "")), str(result.get("dataset", "")))
        if key not in EXPECTED_KEYS:
            continue
        validate_protocol(result, path, expected_channel="first3")
        if key in bucket:
            raise ValueError(f"Duplicate official baseline {key}: {path}")
        bucket[key] = (result, path)
    if set(bucket) != EXPECTED_KEYS:
        raise ValueError(f"Official baseline coverage is {len(bucket)}/9")
    return bucket


def multiseed_nmi(result: dict) -> tuple[list[float], float, float, float, float]:
    feature_file = str(result["feature_file"])
    if feature_file in _NMI_CACHE:
        return _NMI_CACHE[feature_file]
    with np.load(feature_file) as cached:
        features = np.asarray(cached["features"], dtype=np.float32)
        labels = np.asarray(cached["labels"]).astype(int)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    values = []
    for seed in NMI_SEEDS:
        prediction = MiniBatchKMeans(
            n_clusters=int(len(np.unique(labels))),
            random_state=seed,
            batch_size=2048,
            n_init=1,
        ).fit_predict(features)
        values.append(float(normalized_mutual_info_score(labels, prediction)))
    summary = (
        values,
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
    )
    _NMI_CACHE[feature_file] = summary
    return summary


def load_nmi_cache() -> None:
    path = OUT / "lc25000_nmi_seeds.csv"
    if not path.is_file():
        return
    cache_mtime = path.stat().st_mtime_ns
    grouped: dict[str, dict[int, float]] = {}
    for row in csv.DictReader(path.open()):
        feature_file = row["feature_file"]
        feature_path = Path(feature_file)
        if not feature_path.is_file() or feature_path.stat().st_mtime_ns > cache_mtime:
            continue
        grouped.setdefault(feature_file, {})[int(row["seed"])] = float(row["lc25000_nmi"])
    for feature_file, by_seed in grouped.items():
        if set(by_seed) != set(NMI_SEEDS):
            continue
        values = [by_seed[seed] for seed in NMI_SEEDS]
        _NMI_CACHE[feature_file] = (
            values,
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.max(values)),
        )


def load_training_metadata(recipe: Recipe) -> dict[int, dict]:
    wanted = {checkpoint for checkpoint, _ in recipe.checkpoints}
    found = {}
    path = recipe.train_root / "raw_loss_metrics.jsonl"
    for line in path.read_text().splitlines():
        result = json.loads(line)
        checkpoint = int(result.get("optimizer_update", -1))
        if checkpoint not in wanted:
            continue
        if int(result["effective_global_batch_size"]) != recipe.global_batch:
            raise ValueError(f"Unexpected GBS for {recipe.key} ck{checkpoint}")
        found[checkpoint] = result
    if set(found) != wanted:
        raise ValueError(f"Missing training metadata for {recipe.key}: {sorted(wanted - set(found))}")
    return found


def load_s6b_five_family_scaling() -> tuple[dict, list[dict]]:
    with S6B_DATA_TABLE.open() as handle:
        table = {row["scale"]: row for row in csv.DictReader(handle)}
    with S6B_DATA_MANIFEST.open() as handle:
        manifest = {row["label"]: row for row in csv.DictReader(handle)}
    scale_to_label = {
        "0.1M": "random_10",
        "0.2M": "random_20",
        "0.5M": "random_50",
        "1.0M": "random_100",
    }

    def metrics(row: dict) -> dict:
        values = {
            "classification25_macro_f1": float(row["cls_macroF1"]),
            "regression2_spearman": float(row["reg_spearman"]),
            "retrieval4_map_at_5": float(row["ret_mAP@5"]),
            "segmentation8_mdice": float(row["seg_mDice"]),
            "ood_composite": float(row["ood_composite"]),
            "id4_overall": float(row["ID4_overall"]),
            "five_family_mean": float(row["5fam_mean"]),
        }
        expected_id4 = float(
            np.mean(
                [
                    values["classification25_macro_f1"],
                    values["regression2_spearman"],
                    values["retrieval4_map_at_5"],
                    values["segmentation8_mdice"],
                ]
            )
        )
        if not np.isclose(values["id4_overall"], expected_id4, atol=1e-12):
            raise ValueError(f"ID4 aggregate mismatch for {row['scale']}")
        return values

    baseline = {
        "recipe": "official_base_s6b_five_family",
        "scale": "base (0)",
        "unique_images": 0,
        "checkpoint": 0,
        "passes": 0.0,
        "image_visits": 0,
        "image_visits_m": 0.0,
        **metrics(table["base (0)"]),
    }
    rows = []
    for scale, label in scale_to_label.items():
        source = manifest[label]
        checkpoint = int(source["checkpoint"])
        log_path = ROOT / source["train_dir"] / "raw_loss_metrics.jsonl"
        training = None
        for line in log_path.read_text().splitlines():
            candidate = json.loads(line)
            if int(candidate.get("optimizer_update", -1)) == checkpoint:
                training = candidate
                break
        if training is None:
            raise ValueError(f"Missing S6b data-scaling training metadata for {scale}")
        if not np.isclose(float(training["epoch_float"]), 15.0):
            raise ValueError(f"Expected 15 passes for {scale}")
        rows.append(
            {
                "recipe": "s6b_five_family",
                "scale": scale,
                "unique_images": int(source["samples"]),
                "checkpoint": checkpoint,
                "passes": float(training["epoch_float"]),
                "image_visits": int(training["image_visits"]),
                "image_visits_m": float(training["image_visits"]) / 1e6,
                "patch_tokens_seen": int(training["patch_tokens_seen_estimate"]),
                "training_total_loss": float(training["total_loss"]),
                **metrics(table[scale]),
            }
        )
    return baseline, rows


def summarize(
    recipe: str,
    checkpoint: int,
    passes: float,
    training: dict | None,
    rows: dict,
) -> tuple[dict, list[dict], list[dict]]:
    classification = [
        float(rows[("classification", dataset)][0]["balanced_accuracy"])
        for dataset in CLASSIFICATION_DATASETS
    ]
    regression = float(rows[("regression", "bbbc005")][0]["r2"])
    retrieval_result = rows[("retrieval_clustering", "lc25000")][0]
    map_at_5 = float(retrieval_result["map_at_5"])
    nmi_seed0 = float(retrieval_result["nmi"])
    nmi_values, nmi, nmi_std, nmi_min, nmi_max = multiseed_nmi(retrieval_result)
    classification_mean = float(np.mean(classification))
    image_visits = int(training["image_visits"]) if training is not None else 0
    patch_tokens = int(training["patch_tokens_seen_estimate"]) if training is not None else 0
    aggregate = {
        "recipe": recipe,
        "checkpoint": checkpoint,
        "passes": passes,
        "image_visits": image_visits,
        "image_visits_m": image_visits / 1e6,
        "patch_tokens_seen": patch_tokens,
        "classification7_balanced_accuracy": classification_mean,
        "bbbc005_r2": regression,
        "lc25000_map_at_5": map_at_5,
        "lc25000_nmi_seed0": nmi_seed0,
        "lc25000_nmi": nmi,
        "lc25000_nmi_std": nmi_std,
        "lc25000_nmi_min": nmi_min,
        "lc25000_nmi_max": nmi_max,
        "family_balanced_mean": float(np.mean([classification_mean, regression, nmi])),
        "family_balanced_mean_std": nmi_std / 3.0,
        "proxy9_mean": float(np.mean([*classification, regression, nmi])),
        "proxy9_mean_std": nmi_std / 9.0,
        "training_total_loss": float(training["total_loss"]) if training is not None else "",
    }
    per_dataset = []
    for key, (result, path) in sorted(rows.items()):
        task, dataset = key
        if task == "classification":
            metric, value, std = "balanced_accuracy", float(result["balanced_accuracy"]), 0.0
        elif task == "regression":
            metric, value, std = "r2", float(result["r2"]), 0.0
        else:
            metric, value, std = "nmi_10seed_mean", nmi, nmi_std
        per_dataset.append(
            {
                "recipe": recipe,
                "checkpoint": checkpoint,
                "passes": passes,
                "image_visits": image_visits,
                "task": task,
                "dataset": dataset,
                "metric": metric,
                "value": value,
                "std": std,
                "result_path": str(path),
            }
        )
    nmi_rows = [
        {
            "recipe": recipe,
            "checkpoint": checkpoint,
            "passes": passes,
            "image_visits": image_visits,
            "seed": seed,
            "lc25000_nmi": value,
            "feature_file": str(retrieval_result["feature_file"]),
        }
        for seed, value in zip(NMI_SEEDS, nmi_values)
    ]
    return aggregate, per_dataset, nmi_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_all(figure: plt.Figure, base: Path) -> None:
    figure.savefig(base.with_suffix(".png"), dpi=220, facecolor="white")
    figure.savefig(base.with_suffix(".pdf"), facecolor="white")
    figure.savefig(base.with_suffix(".svg"), facecolor="white")


def value_limits(values: list[float], baseline: float, *, saturated: bool = False) -> tuple[float, float]:
    low = min([*values, baseline])
    high = max([*values, baseline])
    span = max(high - low, 1e-5)
    if saturated:
        return max(0.0, low - span * 0.45), min(1.00004, high + span * 1.8)
    return max(-1.0, low - span * 0.22 - 0.003), min(1.02, high + span * 0.38 + 0.004)


def style_axes(
    axis: plt.Axes,
    x: np.ndarray,
    labels: list[str],
    ymin: float,
    ymax: float,
) -> None:
    axis.set_xscale("log")
    axis.set_xlim(x[0] * 0.86, x[-1] * 1.17)
    axis.set_ylim(ymin, ymax)
    axis.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.set_xticks(x, labels)
    axis.minorticks_off()
    axis.tick_params(axis="both", which="both", labelsize=8.7, color=INK, length=0)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(INK)
    axis.plot(1, 0, ">", markersize=6.5, color=INK, transform=axis.get_yaxis_transform(), clip_on=False)
    axis.plot(0, 1, "^", markersize=6.5, color=INK, transform=axis.get_xaxis_transform(), clip_on=False)
    axis.set_xlabel("training image-visits (log)", fontsize=10.5)


def image_visit_labels(x: np.ndarray) -> list[str]:
    labels = [f"{value:.1f}M" for value in x]
    if len(labels) == 8:
        keep = {0, 1, 2, 4, 7}
        labels = [label if index in keep else "" for index, label in enumerate(labels)]
    return labels


def draw_curve(
    axis: plt.Axes,
    recipe: Recipe,
    rows: list[dict],
    baseline: dict,
    key: str,
    *,
    std_key: str | None = None,
    saturated: bool = False,
) -> None:
    x = np.array([float(row["image_visits_m"]) for row in rows])
    y = np.array([float(row[key]) for row in rows])
    yerr = np.array([float(row[std_key]) for row in rows]) if std_key else np.zeros_like(y)
    sizes = np.interp(np.log10(x), np.log10(x[[0, -1]]), [115, 500])
    color = COLORS[key]
    axis.axhline(float(baseline[key]), color="#AAB0B8", linewidth=1.1, linestyle=(0, (4, 4)), zorder=1)
    axis.plot(x, y, color="#4A4A4A", linewidth=1.35, linestyle=(0, (5, 4)), zorder=2)
    if std_key:
        axis.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.1, capsize=3, alpha=0.7, zorder=3)
    axis.scatter(x, y, s=sizes, facecolor=color, edgecolor=INK, linewidth=1.1, zorder=4)
    selected_index = next(
        index for index, row in enumerate(rows) if int(row["checkpoint"]) == recipe.selected_checkpoint
    )
    axis.scatter(
        [x[selected_index]],
        [y[selected_index]],
        s=[sizes[selected_index] + 115],
        facecolor="none",
        edgecolor=SELECTED,
        linewidth=2.1,
        zorder=5,
    )
    axis.annotate(
        "official base (first3)",
        xy=(x[0] * 0.9, float(baseline[key])),
        fontsize=7.8,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    axis.annotate(
        recipe.curve_label,
        xy=(x[-1], y[-1]),
        xytext=(0, 17),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=10.5,
        color="white",
        bbox=dict(boxstyle="round,pad=0.34", facecolor=color, edgecolor="none"),
        zorder=6,
    )
    limit_values = [*(y - yerr).tolist(), *(y + yerr).tolist()]
    ymin, ymax = value_limits(limit_values, float(baseline[key]), saturated=saturated)
    labels = image_visit_labels(x)
    style_axes(axis, x, labels, ymin, ymax)
    if saturated:
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))


def plot_tasks(
    recipe: Recipe,
    rows: list[dict],
    baseline: dict,
    *,
    output_prefix: str | None = None,
) -> None:
    panels = (
        ("classification7_balanced_accuracy", "(a) Classification (7 CHAMMI tasks)", "balanced accuracy", None, False),
        ("bbbc005_r2", "(b) Regression (BBBC005)", r"$R^2$", None, False),
        ("lc25000_map_at_5", "(c) Retrieval (LC25000)", "mAP@5", None, True),
        ("lc25000_nmi", "(d) Clustering (LC25000; 10 seeds)", "NMI mean +/- std", "lc25000_nmi_std", False),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.1))
    for axis, (key, title, ylabel, std_key, saturated) in zip(axes.flat, panels):
        draw_curve(axis, recipe, rows, baseline, key, std_key=std_key, saturated=saturated)
        axis.set_title(title, fontsize=12.2, fontweight="bold", pad=8)
        axis.set_ylabel(ylabel, fontsize=10.8)
    figure.suptitle(recipe.title, fontsize=15.5, fontweight="bold", y=0.985)
    figure.text(0.5, 0.936, recipe.subtitle, ha="center", fontsize=10.0, color="#5C6770")
    figure.text(
        0.5,
        0.014,
        "Bubble area increases with compute. Coral ring marks the checkpoint used by the corresponding data-scaling story. Clustering bars: 10-seed KMeans std.",
        ha="center",
        fontsize=8.8,
        color="#5C6770",
    )
    figure.subplots_adjust(left=0.075, right=0.98, top=0.88, bottom=0.09, wspace=0.25, hspace=0.38)
    prefix = output_prefix or recipe.key
    save_all(figure, OUT / f"{prefix}_compute_scaling_tasks")
    plt.close(figure)


def plot_overall(recipe: Recipe, rows: list[dict], baseline: dict) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 6.2))
    x = np.array([float(row["image_visits_m"]) for row in rows])
    sizes = np.interp(np.log10(x), np.log10(x[[0, -1]]), [120, 515])
    specs = (
        ("family_balanced_mean", "3-family balanced", "family_balanced_mean_std", 15),
        ("proxy9_mean", "Proxy-9 dataset-weighted", "proxy9_mean_std", -17),
    )
    values_for_limits = []
    selected_index = next(
        index for index, row in enumerate(rows) if int(row["checkpoint"]) == recipe.selected_checkpoint
    )
    for key, label, std_key, label_offset in specs:
        y = np.array([float(row[key]) for row in rows])
        yerr = np.array([float(row[std_key]) for row in rows])
        color = COLORS[key]
        values_for_limits.extend((y - yerr).tolist())
        values_for_limits.extend((y + yerr).tolist())
        axis.axhline(float(baseline[key]), color=color, linewidth=1.0, linestyle=(0, (4, 4)), alpha=0.35, zorder=1)
        axis.plot(x, y, color=color, linewidth=1.7, linestyle=(0, (5, 4)), zorder=2)
        axis.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.05, capsize=3, alpha=0.5, zorder=3)
        axis.scatter(x, y, s=sizes, facecolor=color, edgecolor=color, linewidth=1.0, zorder=4)
        axis.scatter(
            [x[selected_index]],
            [y[selected_index]],
            s=[sizes[selected_index] + 115],
            facecolor="none",
            edgecolor=SELECTED,
            linewidth=2.1,
            zorder=5,
        )
        axis.annotate(
            label,
            xy=(x[-1], y[-1]),
            xytext=(-8, label_offset),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9.8,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.32", facecolor=color, edgecolor="none"),
            zorder=6,
        )
    base_values = [float(baseline[key]) for key, _, _, _ in specs]
    low = min([*values_for_limits, *base_values])
    high = max([*values_for_limits, *base_values])
    pad = (high - low) * 0.13 + 0.004
    labels = image_visit_labels(x)
    style_axes(axis, x, labels, low - pad, high + pad)
    axis.set_ylabel("mean downstream score", fontsize=11.5)
    axis.set_title(recipe.title.replace(" by task", " overall") + "\n" + recipe.subtitle, fontsize=14.2, fontweight="bold", pad=12)
    axis.text(
        0.02,
        0.02,
        "official S+ baselines",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=8.2,
        ha="left",
        va="bottom",
    )
    figure.text(
        0.5,
        0.018,
        "Fixed 1M microscopy pool. Coral rings mark the data-scaling checkpoint; error bars propagate 10-seed KMeans NMI std.",
        ha="center",
        fontsize=8.7,
        color="#5C6770",
    )
    figure.subplots_adjust(left=0.12, right=0.96, top=0.84, bottom=0.14)
    save_all(figure, OUT / f"{recipe.key}_compute_scaling_overall")
    plt.close(figure)


def plot_proxy9_overall(
    recipe: Recipe,
    rows: list[dict],
    baseline: dict,
    *,
    output_prefix: str,
) -> None:
    x = np.array([float(row["image_visits_m"]) for row in rows])
    y = np.array([float(row["proxy9_mean"]) for row in rows])
    yerr = np.array([float(row["proxy9_mean_std"]) for row in rows])
    sizes = np.interp(np.log10(x), np.log10(x[[0, -1]]), [120, 515])
    color = COLORS["proxy9_mean"]
    selected_index = next(
        index for index, row in enumerate(rows) if int(row["checkpoint"]) == recipe.selected_checkpoint
    )
    figure, axis = plt.subplots(figsize=(8.0, 6.2))
    axis.axhline(
        float(baseline["proxy9_mean"]),
        color="#AAB0B8",
        linewidth=1.1,
        linestyle=(0, (4, 4)),
        zorder=1,
    )
    axis.plot(x, y, color="#4A4A4A", linewidth=1.45, linestyle=(0, (5, 4)), zorder=2)
    axis.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="none",
        ecolor=color,
        elinewidth=1.1,
        capsize=3,
        alpha=0.58,
        zorder=3,
    )
    axis.scatter(x, y, s=sizes, facecolor=color, edgecolor=INK, linewidth=1.1, zorder=4)
    axis.scatter(
        [x[selected_index]],
        [y[selected_index]],
        s=[sizes[selected_index] + 115],
        facecolor="none",
        edgecolor=SELECTED,
        linewidth=2.1,
        zorder=5,
    )
    axis.annotate(
        "Proxy-9 overall",
        xy=(x[-1], y[-1]),
        xytext=(-8, 17),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.34", facecolor=color, edgecolor="none"),
        zorder=6,
    )
    axis.annotate(
        "official base (first3)",
        xy=(x[0] * 0.9, float(baseline["proxy9_mean"])),
        fontsize=8.2,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    values = [*(y - yerr).tolist(), *(y + yerr).tolist()]
    ymin, ymax = value_limits(values, float(baseline["proxy9_mean"]))
    style_axes(axis, x, image_visit_labels(x), ymin, ymax)
    axis.set_ylabel("Proxy-9 mean downstream score", fontsize=11.5)
    axis.set_title(
        r"$\mathrm{S6b}$ Proxy-9 compute scaling (fixed 1M pool)" + "\n"
        + "7 CHAMMI classification + BBBC005 + LC25000 NMI",
        fontsize=14.2,
        fontweight="bold",
        pad=12,
    )
    figure.text(
        0.5,
        0.018,
        "Only training compute changes. Coral ring marks ck3899 (15 passes); bars propagate 10-seed KMeans NMI std.",
        ha="center",
        fontsize=8.8,
        color="#5C6770",
    )
    figure.subplots_adjust(left=0.12, right=0.96, top=0.84, bottom=0.14)
    save_all(figure, OUT / f"{output_prefix}_compute_scaling_overall")
    plt.close(figure)


def plot_s6b_five_family_tasks(rows: list[dict], baseline: dict) -> None:
    recipe = RECIPES[0]
    panels = (
        ("classification25_macro_f1", "(a) Classification (25 sets)", "macro-F1"),
        ("regression2_spearman", "(b) Regression (2 sets)", r"Spearman $\rho$"),
        ("retrieval4_map_at_5", "(c) Retrieval / clustering (4)", "mAP@5"),
        ("segmentation8_mdice", "(d) Segmentation (8 sets)", "mDice"),
        ("ood_composite", "(e) OOD (X-ray + cryo)", "composite"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.4))
    flat = axes.ravel()
    for axis, (key, title, ylabel) in zip(flat, panels):
        draw_curve(axis, recipe, rows, baseline, key)
        axis.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
        axis.set_ylabel(ylabel, fontsize=11.5)
    flat[5].axis("off")
    figure.suptitle(
        r"$\mathrm{S6b}$ training-compute scaling by task (0.1M $\to$ 1M; fixed 15 passes)",
        fontsize=15.2,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.014,
        "Same five task families as the original S6b data-scaling figure. Unique data and image-visits grow together.",
        ha="center",
        fontsize=9.0,
        color="#5C6770",
    )
    figure.subplots_adjust(left=0.055, right=0.985, top=0.915, bottom=0.065, wspace=0.26, hspace=0.42)
    save_all(figure, OUT / "s6b_compute_scaling_tasks")
    plt.close(figure)


def plot_s6b_id4_overall(rows: list[dict], baseline: dict) -> None:
    x = np.array([float(row["image_visits_m"]) for row in rows])
    y = np.array([float(row["id4_overall"]) for row in rows])
    sizes = np.interp(np.log10(x), np.log10(x[[0, -1]]), [130, 560])
    color = COLORS["id4_overall"]
    figure, axis = plt.subplots(figsize=(7.6, 6.0))
    axis.axhline(
        float(baseline["id4_overall"]),
        color="#AAB0B8",
        linewidth=1.1,
        linestyle=(0, (4, 4)),
        zorder=1,
    )
    axis.plot(x, y, color="#4A4A4A", linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
    axis.scatter(x, y, s=sizes, facecolor=color, edgecolor=INK, linewidth=1.2, zorder=4)
    axis.annotate(
        "ID-4 overall",
        xy=(x[-1], y[-1]),
        xytext=(0, 18),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.34", facecolor=color, edgecolor="none"),
        zorder=6,
    )
    axis.annotate(
        "official base",
        xy=(x[0] * 0.9, float(baseline["id4_overall"])),
        fontsize=8.2,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    ymin, ymax = value_limits(y.tolist(), float(baseline["id4_overall"]))
    style_axes(axis, x, image_visit_labels(x), ymin, ymax)
    axis.set_ylabel("ID-4 mean score", fontsize=12)
    axis.set_title(
        r"$\mathrm{S6b}$ training-compute scaling - ID overall" + "\n"
        + "Classification + regression + retrieval/clustering + segmentation",
        fontsize=14,
        fontweight="bold",
        pad=11,
    )
    axis.text(
        0.985,
        0.025,
        "OOD excluded from overall",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.0,
        color="#C6803B",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.018,
        "Fixed 15 passes: training compute rises with the nested data pool. ID-4 is an equal mean over four in-domain families.",
        ha="center",
        fontsize=8.7,
        color="#5C6770",
    )
    figure.subplots_adjust(left=0.12, right=0.965, top=0.86, bottom=0.13)
    save_all(figure, OUT / "s6b_compute_scaling_overall")
    plt.close(figure)


def write_readme(
    s6b_rows: list[dict],
    s6b_baseline: dict,
    s6b_proxy_rows: list[dict],
    new_rows: list[dict],
    new_baseline: dict,
) -> None:
    lines = [
        "# Separate S6b and current S+ compute-scaling figures",
        "",
        "The two reports are deliberately presented separately and use the coverage available for each study.",
        "",
        "## S6b: five-family fixed-pass scaling",
        "",
        "This is the original nested-data S6b study re-plotted against exact image-visits.",
        "All four runs use 15 passes, so unique data and training compute increase together.",
        "The task figure exactly preserves Classification-25 macro-F1, Regression-2 Spearman,",
        "Retrieval/Clustering-4 mAP@5, Segmentation-8 mDice, and the X-ray + cryo OOD composite.",
        "The overall figure is the equal mean of the first four in-domain families and excludes OOD.",
        "",
        f"Official ID-4 baseline: {s6b_baseline['id4_overall']:.6f}.",
        "",
        "| pool | checkpoint | image-visits | Classification-25 | Regression-2 | Retrieval-4 | Segmentation-8 | OOD | ID-4 overall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in s6b_rows:
        lines.append(
            f"| {row['scale']} | {row['checkpoint']} | {row['image_visits']:,} | "
            f"{row['classification25_macro_f1']:.6f} | {row['regression2_spearman']:.6f} | "
            f"{row['retrieval4_map_at_5']:.6f} | {row['segmentation8_mdice']:.6f} | "
            f"{row['ood_composite']:.6f} | {row['id4_overall']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## S6b: fixed-1M Proxy-9 compute scaling",
            "",
            "This is the strict compute-scaling view: the 1M pool is fixed and only checkpoints change.",
            "It uses complete common Proxy-9 coverage at all seven S6b checkpoints.",
            "",
            "| checkpoint | passes | image-visits | Classification-7 BA | BBBC005 R2 | LC25000 NMI | Proxy-9 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in s6b_proxy_rows:
        lines.append(
            f"| {row['checkpoint']} | {row['passes']} | {row['image_visits']:,} | "
            f"{row['classification7_balanced_accuracy']:.6f} | {row['bbbc005_r2']:.6f} | "
            f"{row['lc25000_nmi']:.6f} +/- {row['lc25000_nmi_std']:.6f} | "
            f"{row['proxy9_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Current S+ recipe: fixed-1M checkpoint scaling",
            "",
            "This report fixes the 1M pool and varies checkpoints from 1 to 15 passes.",
            "Coverage is Proxy-9: seven CHAMMI balanced accuracies, BBBC005 R2, and LC25000 NMI.",
            "LC25000 NMI is recomputed over KMeans seeds 0-9. The official reference uses first3;",
            "the trained curve uses auto channel handling with eight TTA samples.",
            "",
            f"Official Proxy-9 baseline: {new_baseline['proxy9_mean']:.6f}.",
            "",
            "| checkpoint | passes | image-visits | Classification-7 BA | BBBC005 R2 | LC25000 NMI | Proxy-9 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in new_rows:
        lines.append(
            f"| {row['checkpoint']} | {row['passes']} | {row['image_visits']:,} | "
            f"{row['classification7_balanced_accuracy']:.6f} | {row['bbbc005_r2']:.6f} | "
            f"{row['lc25000_nmi']:.6f} +/- {row['lc25000_nmi_std']:.6f} | "
            f"{row['proxy9_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "Each recipe has independent `*_compute_scaling_tasks` and `*_compute_scaling_overall` figures",
            "in PNG, PDF, and SVG. The S6b and current-recipe aggregate CSVs are kept separate.",
            "The x-axes use exact image-visits from raw loss logs.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    load_nmi_cache()
    s6b_baseline, s6b_rows = load_s6b_five_family_scaling()
    base_rows = load_base_results()
    baseline, baseline_per_dataset, baseline_nmi = summarize(
        "official_base", 0, 0.0, None, base_rows
    )
    aggregate = []
    per_dataset = [*baseline_per_dataset]
    nmi_rows = [*baseline_nmi]
    s6b_proxy_recipe = RECIPES[0]
    s6b_proxy_training = load_training_metadata(s6b_proxy_recipe)
    s6b_proxy_rows = []
    for checkpoint, passes in s6b_proxy_recipe.checkpoints:
        results = load_checkpoint_results(s6b_proxy_recipe, checkpoint)
        row, source_rows, seed_rows = summarize(
            "s6b_proxy9",
            checkpoint,
            float(passes),
            s6b_proxy_training[checkpoint],
            results,
        )
        s6b_proxy_rows.append(row)
        per_dataset.extend(source_rows)
        nmi_rows.extend(seed_rows)
    new_recipe = RECIPES[1]
    training = load_training_metadata(new_recipe)
    for checkpoint, passes in new_recipe.checkpoints:
        results = load_checkpoint_results(new_recipe, checkpoint)
        row, source_rows, seed_rows = summarize(
            new_recipe.key,
            checkpoint,
            float(passes),
            training[checkpoint],
            results,
        )
        aggregate.append(row)
        per_dataset.extend(source_rows)
        nmi_rows.extend(seed_rows)

    write_csv(OUT / "baseline.csv", [baseline])
    write_csv(OUT / "s6b_five_family_baseline.csv", [s6b_baseline])
    write_csv(OUT / "s6b_compute_scaling_five_families.csv", s6b_rows)
    write_csv(OUT / "s6b_proxy9_compute_aggregate.csv", s6b_proxy_rows)
    write_csv(OUT / "aggregate.csv", aggregate)
    write_csv(OUT / "splus_new_compute_aggregate.csv", aggregate)
    write_csv(OUT / "per_dataset.csv", per_dataset)
    write_csv(OUT / "lc25000_nmi_seeds.csv", nmi_rows)

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
    plot_s6b_five_family_tasks(s6b_rows, s6b_baseline)
    plot_s6b_id4_overall(s6b_rows, s6b_baseline)
    plot_tasks(
        s6b_proxy_recipe,
        s6b_proxy_rows,
        baseline,
        output_prefix="s6b_proxy9",
    )
    plot_proxy9_overall(
        s6b_proxy_recipe,
        s6b_proxy_rows,
        baseline,
        output_prefix="s6b_proxy9",
    )
    plot_tasks(new_recipe, aggregate, baseline)
    plot_overall(new_recipe, aggregate, baseline)
    write_readme(s6b_rows, s6b_baseline, s6b_proxy_rows, aggregate, baseline)
    print(f"Wrote separate S6b and current S+ compute-scaling reports to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
