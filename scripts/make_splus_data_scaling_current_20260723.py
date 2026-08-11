#!/usr/bin/env python3
"""Rebuild the two S+ data-scaling figures from the selected 1M sweet spot.

The scaling curve fixes the 1M-selected recipe and checkpoint (ck8199) across
all four nested data pools. The separately evaluated 1M deployable interpolation
is shown as a star, not mixed into the raw matched-checkpoint curve.
"""

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
OUT = ROOT / "outputs/00_reports/splus_data_scaling_current_20260723"

SELECTED_CHECKPOINT = 8199
GLOBAL_BATCH = 1024
NMI_SEEDS = tuple(range(10))
CLASSIFICATION_DATASETS = {
    "chammi-allen-task1",
    "chammi-allen-task2",
    "chammi-cp-task1",
    "chammi-cp-task2",
    "chammi-cp-task3",
    "chammi-hpa-task1",
    "chammi-hpa-task2",
}
EXPECTED_KEYS = {
    *(("classification", dataset) for dataset in CLASSIFICATION_DATASETS),
    ("regression", "bbbc005"),
    ("retrieval_clustering", "lc25000"),
}


@dataclass(frozen=True)
class Pool:
    label: str
    samples: int
    root: Path


POOLS = [
    Pool(
        "0.1M",
        104_877,
        ROOT
        / "outputs/02_eval_runs/"
        "DscaleFinal_splus_sigreg005_random10_fixed15M_qi4gbs64acc4__compute_proxy_curve",
    ),
    Pool(
        "0.2M",
        209_754,
        ROOT
        / "outputs/02_eval_runs/"
        "DscaleFinal_splus_sigreg005_random20_fixed15M_qi4gbs64acc4__compute_proxy_curve_auto_b64_clean_20260722",
    ),
    Pool(
        "0.5M",
        524_385,
        ROOT
        / "outputs/02_eval_runs/"
        "DscaleFinal_splus_sigreg005_random50_fixed15M_local8gbs64acc2__compute_proxy_curve",
    ),
    Pool(
        "1.0M",
        1_048_771,
        ROOT
        / "outputs/02_eval_runs/"
        "S6sigreg005_rgb_robust_biosafe256_b1024__compute_proxy_curve_20260721",
    ),
]
BASE_ROOT = ROOT / "outputs/02_eval_runs/S7z_splus_official_objective_proxy_cpu1"
SWEET_SPOT_ROOT = (
    ROOT
    / "outputs/02_eval_runs/"
    "S6interp_official_sigreg005repl8199_20260719__full_dense_local_bf16_20260722"
)

COLORS = {
    "classification7_balanced_accuracy": "#35698F",
    "bbbc005_r2": "#C29ACB",
    "lc25000_map_at_5": "#72A8CF",
    "lc25000_nmi": "#D18A45",
    "family_balanced_mean": "#2F6E8F",
    "proxy9_mean": "#C6803B",
}
GRID = "#DADDE2"
INK = "#263238"
MUTED = "#8A9099"
SWEET = "#E4572E"
_NMI_CACHE: dict[str, tuple[float, float, float, float]] = {}


def checkpoint_id(result: dict, path: Path) -> int:
    checkpoint = str(result.get("checkpoint", ""))
    match = re.search(r"/ckpt/(\d+)(?:/checkpoint\.pth)?$", checkpoint)
    if match:
        return int(match.group(1))
    for part in reversed(path.parts):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot infer checkpoint from {path}")


def results_by_checkpoint(root: Path) -> dict[int, dict[tuple[str, str], tuple[dict, Path]]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    grouped: dict[int, dict[tuple[str, str], tuple[dict, Path]]] = {}
    for path in sorted(root.rglob("last_result.json")):
        result = json.loads(path.read_text())
        key = (str(result.get("task", "")), str(result.get("dataset", "")))
        if key not in EXPECTED_KEYS:
            continue
        checkpoint = checkpoint_id(result, path)
        bucket = grouped.setdefault(checkpoint, {})
        if key in bucket:
            raise ValueError(f"Duplicate {key} at checkpoint {checkpoint} under {root}")
        bucket[key] = (result, path)
    return grouped


def multiseed_nmi(result: dict) -> tuple[float, float, float, float]:
    """Return mean/std/min/max NMI over deterministic KMeans initializations."""
    feature_file = str(result["feature_file"])
    if feature_file in _NMI_CACHE:
        return _NMI_CACHE[feature_file]
    with np.load(feature_file) as cached:
        features = np.asarray(cached["features"], dtype=np.float32)
        labels = np.asarray(cached["labels"]).astype(int)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    n_clusters = int(len(np.unique(labels)))
    values = []
    for seed in NMI_SEEDS:
        prediction = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=2048,
            n_init=1,
        ).fit_predict(features)
        values.append(float(normalized_mutual_info_score(labels, prediction)))
    summary = (
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
    )
    _NMI_CACHE[feature_file] = summary
    return summary


def summarize(
    label: str,
    samples: int,
    checkpoint: int,
    rows: dict,
    *,
    source_kind: str = "raw_matched_checkpoint",
) -> dict[str, object]:
    classification = [
        float(rows[("classification", dataset)][0]["balanced_accuracy"])
        for dataset in sorted(CLASSIFICATION_DATASETS)
    ]
    regression = float(rows[("regression", "bbbc005")][0]["r2"])
    retrieval_result = rows[("retrieval_clustering", "lc25000")][0]
    map_at_5 = float(retrieval_result["map_at_5"])
    nmi_seed0 = float(retrieval_result["nmi"])
    nmi, nmi_std, nmi_min, nmi_max = multiseed_nmi(retrieval_result)
    classification_mean = float(np.mean(classification))
    return {
        "pool": label,
        "source_kind": source_kind,
        "samples": samples,
        "checkpoint": checkpoint,
        "image_visits": checkpoint * GLOBAL_BATCH,
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
    }


def load_base() -> dict[str, object]:
    grouped = results_by_checkpoint(BASE_ROOT)
    complete = [checkpoint for checkpoint, rows in grouped.items() if set(rows) == EXPECTED_KEYS]
    if len(complete) != 1:
        raise ValueError(f"Expected one complete official baseline checkpoint, found {complete}")
    return summarize(
        "official_0",
        0,
        complete[0],
        grouped[complete[0]],
        source_kind="official_reference",
    )


def load_sweet_spot() -> dict[str, object]:
    grouped = results_by_checkpoint(SWEET_SPOT_ROOT)
    complete = [checkpoint for checkpoint, rows in grouped.items() if set(rows) == EXPECTED_KEYS]
    if len(complete) != 1:
        raise ValueError(f"Expected one complete 1M sweet-spot result, found {complete}")
    result = summarize(
        "1.0M_deployable_interp",
        POOLS[-1].samples,
        complete[0],
        grouped[complete[0]],
        source_kind="1M_0.25official_0.75bio_interp",
    )
    result["checkpoint"] = "interp75_from_ck8199"
    result["image_visits"] = SELECTED_CHECKPOINT * GLOBAL_BATCH
    return result


def style_axes(axis: plt.Axes, ymin: float, ymax: float) -> None:
    samples = np.array([pool.samples for pool in POOLS], dtype=float)
    axis.set_xscale("log")
    axis.set_xlim(samples[0] * 0.72, samples[-1] * 1.62)
    axis.set_ylim(ymin, ymax)
    axis.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.set_xticks(samples, [pool.label.replace("1.0M", "1M") for pool in POOLS])
    axis.tick_params(axis="both", which="both", labelsize=9.5, color=INK, length=0)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(INK)
    axis.plot(1, 0, ">", markersize=6.5, color=INK, transform=axis.get_yaxis_transform(), clip_on=False)
    axis.plot(0, 1, "^", markersize=6.5, color=INK, transform=axis.get_xaxis_transform(), clip_on=False)
    axis.set_xlabel("unique microscopy images (log)", fontsize=10.5)


def value_limits(values: list[float], base: float, *, saturated: bool = False) -> tuple[float, float]:
    low = min([*values, base])
    high = max([*values, base])
    span = high - low
    if saturated:
        pad = max(span * 0.38, 0.000035)
        return max(0.0, low - pad), min(1.00004, high + pad * 1.7)
    pad = span * 0.25 + 0.004
    return max(-1.0, low - pad), min(1.02, high + pad * 1.45)


def plot_curve(
    axis: plt.Axes,
    rows: list[dict[str, object]],
    key: str,
    base_value: float,
    sweet_value: float,
    *,
    saturated: bool = False,
    std_key: str | None = None,
    sweet_std: float | None = None,
) -> None:
    x = np.array([float(row["samples"]) for row in rows])
    y = np.array([float(row[key]) for row in rows])
    sizes = np.interp(np.log10(x), np.log10(x[[0, -1]]), [130, 520])
    color = COLORS[key]
    axis.axhline(base_value, color="#AAB0B8", linewidth=1.1, linestyle=(0, (4, 4)), zorder=1)
    axis.plot(x, y, color="#4A4A4A", linewidth=1.25, linestyle=(0, (5, 4)), zorder=2)
    if std_key is not None:
        yerr = np.array([float(row[std_key]) for row in rows])
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor=color,
            elinewidth=1.15,
            capsize=3,
            alpha=0.72,
            zorder=3,
        )
    axis.scatter(x, y, s=sizes, facecolor=color, edgecolor="#2B2B2B", linewidth=1.0, zorder=4)
    for index, (xx, yy) in enumerate(zip(x, y)):
        offset = (0, 11) if index % 2 == 0 else (0, -14)
        value_format = ".5f" if saturated else ".4f"
        axis.annotate(
            format(yy, value_format),
            xy=(xx, yy),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7.8,
            color=INK,
            fontweight="bold",
            zorder=6,
        )
    star_x = x[-1] * 1.075
    axis.plot([x[-1], star_x], [sweet_value, sweet_value], color=SWEET, linewidth=0.9, zorder=5)
    axis.scatter(
        [star_x],
        [sweet_value],
        marker="*",
        s=235,
        facecolor=SWEET,
        edgecolor="white",
        linewidth=0.9,
        zorder=8,
    )
    if std_key is not None:
        if sweet_std is None:
            raise ValueError("sweet_std is required when std_key is set")
        axis.errorbar(
            [star_x],
            [sweet_value],
            yerr=[sweet_std],
            fmt="none",
            ecolor=SWEET,
            elinewidth=1.15,
            capsize=3,
            alpha=0.8,
            zorder=7,
        )
    axis.annotate(
        f"{sweet_value:.5f}" if saturated else f"{sweet_value:.4f}",
        xy=(star_x, sweet_value),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=7.7,
        color=SWEET,
        fontweight="bold",
        zorder=9,
    )
    axis.annotate(
        f"official base  {base_value:.4f}",
        xy=(x[0] * 0.76, base_value),
        ha="left",
        va="bottom",
        fontsize=7.8,
        color=MUTED,
    )
    limit_values = [*y.tolist(), sweet_value]
    if std_key is not None:
        yerr = np.array([float(row[std_key]) for row in rows])
        limit_values.extend((y - yerr).tolist())
        limit_values.extend((y + yerr).tolist())
        assert sweet_std is not None
        limit_values.extend([sweet_value - sweet_std, sweet_value + sweet_std])
    ymin, ymax = value_limits(limit_values, base_value, saturated=saturated)
    style_axes(axis, ymin, ymax)
    if saturated:
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_all(figure: plt.Figure, base: Path) -> None:
    figure.savefig(base.with_suffix(".png"), dpi=220, facecolor="white")
    figure.savefig(base.with_suffix(".pdf"), facecolor="white")
    figure.savefig(base.with_suffix(".svg"), facecolor="white")


def plot_tasks(
    rows: list[dict[str, object]],
    base: dict[str, object],
    sweet: dict[str, object],
) -> None:
    panels = [
        ("classification7_balanced_accuracy", "(a) Classification (7 CHAMMI tasks)", "balanced accuracy", False, None),
        ("bbbc005_r2", "(b) Regression (BBBC005)", r"$R^2$", False, None),
        ("lc25000_map_at_5", "(c) Retrieval (LC25000)", "mAP@5", True, None),
        ("lc25000_nmi", "(d) Clustering (LC25000; 10 seeds)", "NMI mean +/- std", False, "lc25000_nmi_std"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.1))
    for axis, (key, title, ylabel, saturated, std_key) in zip(axes.flat, panels):
        plot_curve(
            axis,
            rows,
            key,
            float(base[key]),
            float(sweet[key]),
            saturated=saturated,
            std_key=std_key,
            sweet_std=float(sweet[std_key]) if std_key is not None else None,
        )
        axis.set_title(title, fontsize=12.2, fontweight="bold", pad=8)
        axis.set_ylabel(ylabel, fontsize=10.8)
    visits_m = SELECTED_CHECKPOINT * GLOBAL_BATCH / 1e6
    figure.suptitle(
        rf"$\mathrm{{S^+}}$ data scaling from the 1M-selected sweet spot  |  ck{SELECTED_CHECKPOINT}  |  {visits_m:.2f}M visits",
        fontsize=15.5,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.014,
        "Circles: raw matched ck8199 across all pools. Red star: final 1M deployable model. Clustering bars: 10-seed KMeans std.",
        ha="center",
        fontsize=9.2,
        color="#5C6770",
    )
    figure.subplots_adjust(left=0.075, right=0.98, top=0.90, bottom=0.09, wspace=0.25, hspace=0.38)
    save_all(figure, OUT / "splus_data_scaling_tasks")
    plt.close(figure)


def plot_overall(
    rows: list[dict[str, object]],
    base: dict[str, object],
    sweet: dict[str, object],
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 6.2))
    x = np.array([float(row["samples"]) for row in rows])
    sizes = np.interp(np.log10(x), np.log10(x[[0, -1]]), [135, 540])
    specs = [
        ("family_balanced_mean", "3-family balanced", 0.010, "family_balanced_mean_std"),
        ("proxy9_mean", "Proxy-9 dataset-weighted", -0.008, "proxy9_mean_std"),
    ]
    values_for_limits = []
    for key, label, label_offset, std_key in specs:
        y = np.array([float(row[key]) for row in rows])
        yerr = np.array([float(row[std_key]) for row in rows])
        values_for_limits.extend((y - yerr).tolist())
        values_for_limits.extend((y + yerr).tolist())
        values_for_limits.append(float(sweet[key]))
        color = COLORS[key]
        axis.axhline(float(base[key]), color=color, linewidth=1.0, linestyle=(0, (4, 4)), alpha=0.35, zorder=1)
        axis.plot(x, y, color=color, linewidth=1.7, linestyle=(0, (5, 4)), zorder=2)
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor=color,
            elinewidth=1.05,
            capsize=3,
            alpha=0.48,
            zorder=3,
        )
        axis.scatter(x, y, s=sizes, facecolor=color, edgecolor=color, linewidth=1.0, zorder=4)
        for index, (xx, yy) in enumerate(zip(x, y)):
            offset = (0, 11) if index % 2 == 0 else (0, -14)
            axis.annotate(
                f"{yy:.4f}",
                xy=(xx, yy),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=7.8,
                color=INK,
                fontweight="bold",
                zorder=6,
            )
        star_x = x[-1] * 1.025
        axis.scatter(
            [star_x],
            [float(sweet[key])],
            marker="*",
            s=260,
            facecolor=SWEET,
            edgecolor="white",
            linewidth=0.9,
            zorder=8,
        )
        axis.errorbar(
            [star_x],
            [float(sweet[key])],
            yerr=[float(sweet[std_key])],
            fmt="none",
            ecolor=SWEET,
            elinewidth=1.05,
            capsize=3,
            alpha=0.65,
            zorder=7,
        )
        axis.annotate(
            label,
            xy=(x[-1], y[-1]),
            xytext=(x[-1] * 1.07, y[-1] + label_offset),
            ha="left",
            va="center",
            fontsize=10.3,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.32", facecolor=color, edgecolor="none"),
            arrowprops=dict(arrowstyle="-", color=color, linewidth=1.0),
            zorder=7,
        )

    base_values = [float(base[key]) for key, _, _, _ in specs]
    low = min([*values_for_limits, *base_values])
    high = max([*values_for_limits, *base_values])
    pad = (high - low) * 0.10 + 0.006
    style_axes(axis, low - pad, high + pad)
    axis.set_xlim(x[0] * 0.72, x[-1] * 1.85)
    axis.set_ylabel("mean downstream score", fontsize=11.5)
    visits_m = SELECTED_CHECKPOINT * GLOBAL_BATCH / 1e6
    axis.set_title(
        rf"$\mathrm{{S^+}}$ data scaling from the 1M-selected sweet spot" + "\n"
        + f"matched ck{SELECTED_CHECKPOINT}  |  {visits_m:.2f}M image-visits",
        fontsize=14.5,
        fontweight="bold",
        pad=12,
    )
    axis.text(
        x[0] * 0.76,
        float(base["proxy9_mean"]) - 0.003,
        "official S+ baselines",
        color=MUTED,
        fontsize=8.2,
        ha="left",
        va="top",
    )
    axis.text(
        0.985,
        0.985,
        r"$\bigstar$  1M deployable interpolation",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color=SWEET,
        fontsize=9.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.018,
        "Circles = raw matched checkpoints; red stars = final 1M interpolated model; bars propagate 10-seed KMeans NMI std.",
        ha="center",
        fontsize=8.8,
        color="#5C6770",
    )
    figure.subplots_adjust(left=0.12, right=0.89, top=0.86, bottom=0.13)
    save_all(figure, OUT / "splus_data_scaling_overall")
    plt.close(figure)


def write_readme(
    rows: list[dict[str, object]],
    base: dict[str, object],
    sweet: dict[str, object],
) -> None:
    visits_m = SELECTED_CHECKPOINT * GLOBAL_BATCH / 1e6
    lines = [
        "# Current S+ data scaling figures",
        "",
        "Status: Proxy-9 diagnostic only. For presentation and full-suite conclusions, use",
        "`outputs/00_reports/splus_data_scaling_full_current_20260723`, which has strict",
        "C25/Reg-2/Ret-4/Seg-8/OOD coverage under one bf16 evaluation protocol.",
        "",
        f"The raw scaling curve fixes the 1M-selected sweet checkpoint: `{SELECTED_CHECKPOINT}` ({visits_m:.3f}M image-visits).",
        "",
        "This report uses the current locked S+ fixed-compute lane (robust decoder, BioSafe, SIGReg 0.05, GBS 1024).",
        "It does not reuse the older 15-fixed-pass/no-SIGReg values in `outputs/splus_data_scaling_*.png`.",
        "",
        "Current coverage is Proxy-9 only: seven CHAMMI classification datasets, BBBC005 regression, and LC25000 retrieval/clustering.",
        "New fixed-compute segmentation and OOD scaling results do not exist yet, so they are not copied from the old figures.",
        "The final 1M deployable interpolation is shown separately as a red star; it is not substituted for the raw 1M scaling point.",
        "",
        "LC25000 clustering is recomputed over KMeans seeds 0-9. The figure reports mean +/- standard deviation;",
        "the original evaluator used one seed with one initialization and is too unstable for model ranking.",
        "",
        "| pool | Classification-7 BA | BBBC005 R2 | LC25000 mAP@5 | LC25000 NMI mean +/- std | family-balanced | Proxy-9 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pool']} | {row['classification7_balanced_accuracy']:.6f} | "
            f"{row['bbbc005_r2']:.6f} | {row['lc25000_map_at_5']:.6f} | "
            f"{row['lc25000_nmi']:.6f} +/- {row['lc25000_nmi_std']:.6f} | "
            f"{row['family_balanced_mean']:.6f} | {row['proxy9_mean']:.6f} |"
        )
    lines.extend(
        [
            f"| 1M deployable interp | {sweet['classification7_balanced_accuracy']:.6f} | "
            f"{sweet['bbbc005_r2']:.6f} | {sweet['lc25000_map_at_5']:.6f} | "
            f"{sweet['lc25000_nmi']:.6f} +/- {sweet['lc25000_nmi_std']:.6f} | "
            f"{sweet['family_balanced_mean']:.6f} | {sweet['proxy9_mean']:.6f} |",
            f"| official base | {base['classification7_balanced_accuracy']:.6f} | "
            f"{base['bbbc005_r2']:.6f} | {base['lc25000_map_at_5']:.6f} | "
            f"{base['lc25000_nmi']:.6f} +/- {base['lc25000_nmi_std']:.6f} | "
            f"{base['family_balanced_mean']:.6f} | {base['proxy9_mean']:.6f} |",
            "",
            "Do not replace ck8199 with a later common checkpoint: ck8199 is the checkpoint selected during the 1M sweet-spot search.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    grouped = {pool.label: results_by_checkpoint(pool.root) for pool in POOLS}
    for label, by_checkpoint in grouped.items():
        coverage = set(by_checkpoint.get(SELECTED_CHECKPOINT, {}))
        if coverage != EXPECTED_KEYS:
            raise ValueError(
                f"{label} has {len(coverage)}/9 results at selected checkpoint {SELECTED_CHECKPOINT}"
            )
    rows = [
        summarize(
            pool.label,
            pool.samples,
            SELECTED_CHECKPOINT,
            grouped[pool.label][SELECTED_CHECKPOINT],
        )
        for pool in POOLS
    ]
    base = load_base()
    sweet = load_sweet_spot()
    OUT.mkdir(parents=True, exist_ok=True)

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

    plot_tasks(rows, base, sweet)
    plot_overall(rows, base, sweet)
    write_csv(OUT / "splus_data_scaling_current.csv", [base, *rows, sweet])
    write_readme(rows, base, sweet)
    print(f"wrote current S+ data-scaling figures at selected checkpoint {SELECTED_CHECKPOINT} to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
