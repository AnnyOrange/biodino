#!/usr/bin/env python3
"""
Aggregate linear-probe results under outputs/outputs_microscopy_continual_* and plot:
  1) Grouped bar chart: best test mIoU per dataset (S+ / B / L).
  2) Scaling-style line: x = ViT S+ / B / L, y = avg over datasets of (best test mIoU per dataset).
  2b) Grouped bars: mean @ step 1024 (five datasets) vs mean(per-dataset best over steps).
  2c) Heatmap: rows S+/B/L, columns datasets; cell = best test mIoU; colorbar = mIoU (0 → grid max).
  3) Per-dataset curves: test mIoU vs step for S+ / B / L (step 0 = linear_probe baseline,
     then all continual ckpt steps up to the last available).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LINEAR_PROBE = REPO_ROOT / "outputs" / "linear_probe"

DATASETS: List[str] = ["bbbc038", "conic", "monuseg", "pannuke", "tissuenet"]

# (dataset, arch s|b|l) -> baseline folder names under outputs/linear_probe (first hit wins)
BASELINE_CANDIDATES: Dict[Tuple[str, str], List[str]] = {
    ("bbbc038", "s"): ["bbbc038_vitS"],
    ("bbbc038", "b"): ["bbbc038_vitB", "bbbc038_vitb_new"],
    ("bbbc038", "l"): ["bbbc038_vitl", "bbbc038_vitL", "bbbc038_vitl_new"],
    ("conic", "s"): ["conic_vits"],
    ("conic", "b"): ["conic_vitb"],
    ("conic", "l"): ["conic_vitl"],
    ("monuseg", "s"): ["monuseg_vits", "monuseg_vitS"],
    ("monuseg", "b"): ["monuseg_vitB", "monuseg_vitb_new"],
    ("monuseg", "l"): ["monuseg_vitl", "monuseg_vitl_new"],
    ("pannuke", "s"): ["pannuke_vits", "pannuke_vitS"],
    ("pannuke", "b"): ["pannuke_vitb", "pannuke_vitB", "pannuke_vitb_new"],
    ("pannuke", "l"): ["pannuke_vitl", "pannuke_vitl_new"],
    ("tissuenet", "s"): ["tissuenet_vits", "tissuenet_vitS"],
    ("tissuenet", "b"): ["tissuenet_vitb", "tissuenet_vitB", "tissuenet_vitb_new"],
    ("tissuenet", "l"): ["tissuenet_vitl", "tissuenet_vitl_new"],
}

# keys -> display name, relative path under REPO_ROOT
MODELS: Dict[str, Tuple[str, Path]] = {
    "s": ("S+", REPO_ROOT / "outputs/outputs_microscopy_continual_vits16"),
    "b": ("B", REPO_ROOT / "outputs/outputs_microscopy_continual_vitb16"),
    "l": ("L", REPO_ROOT / "outputs/outputs_microscopy_continual_vitl16"),
}

COLORS = {"S+": "#9eb8c8", "B": "#2a3f6f", "L": "#e07020"}

# Heatmap column order (match common benchmark table; CoNIC = conic)
HEATMAP_DATASETS = ["bbbc038", "monuseg", "conic", "pannuke", "tissuenet"]
HEATMAP_DISPLAY = ["BBBC038", "MoNuSeg", "CoNIC", "PanNuke", "TissueNet"]


def _read_test_miou_json(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
        return float(data["test"]["mIoU"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def baseline_test_miou(dataset: str, arch_key: str) -> Optional[float]:
    """Pre-continual baseline test mIoU (step 0) from outputs/linear_probe."""
    for name in BASELINE_CANDIDATES.get((dataset, arch_key), []):
        m = _read_test_miou_json(LINEAR_PROBE / name / "results.json")
        if m is not None:
            return m
    return None


def load_step_metrics(outputs_root: Path) -> Dict[int, Dict[str, float]]:
    """step -> {dataset: test mIoU} for all available results.json under linear_probe."""
    out: Dict[int, Dict[str, float]] = {}
    if not outputs_root.is_dir():
        return out
    for ds in DATASETS:
        probe = outputs_root / ds / "linear_probe"
        if not probe.is_dir():
            continue
        for step_dir in sorted(probe.iterdir(), key=lambda p: p.name):
            if not step_dir.is_dir():
                continue
            try:
                step = int(step_dir.name)
            except ValueError:
                continue
            rj = step_dir / "results.json"
            if not rj.is_file():
                continue
            try:
                with rj.open() as f:
                    data = json.load(f)
                miou = float(data["test"]["mIoU"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            out.setdefault(step, {})[ds] = miou
    return out


def mean_test_miou_shared_at_step(
    step_metrics: Dict[int, Dict[str, float]], step: int
) -> float:
    """Mean test mIoU over DATASETS at ``step`` when all five have results; else nan."""
    row = step_metrics.get(step)
    if not row or not all(ds in row for ds in DATASETS):
        return float("nan")
    return float(np.mean([row[ds] for ds in DATASETS]))


def best_per_dataset(step_metrics: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    best: Dict[str, float] = {ds: float("nan") for ds in DATASETS}
    for _step, per_ds in step_metrics.items():
        for ds, v in per_ds.items():
            if ds not in best:
                continue
            if np.isnan(best[ds]) or v > best[ds]:
                best[ds] = v
    return best


def plot_best_miou_heatmap(
    best_by_model: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    """Rows = model scale (S+, B, L); columns = datasets; color = raw best test mIoU (global 0 → vmax)."""
    row_keys = ["s", "b", "l"]
    row_labels = ["S+", "B", "L"]
    nrows, ncols = 3, len(HEATMAP_DATASETS)
    M = np.full((nrows, ncols), np.nan, dtype=float)
    for i, k in enumerate(row_keys):
        for j, ds in enumerate(HEATMAP_DATASETS):
            v = best_by_model[k].get(ds, float("nan"))
            if np.isfinite(v):
                M[i, j] = v

    vmax = float(np.nanmax(M))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    else:
        vmax = min(1.0, vmax * 1.02)

    M_plot = np.ma.masked_invalid(M)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(
        M_plot,
        aspect="equal",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=vmax,
        origin="upper",
    )
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels(HEATMAP_DISPLAY, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("Model size", fontsize=11)
    ax.set_title("Best test mIoU", fontsize=12, pad=12)

    for i in range(nrows):
        for j in range(ncols):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="#666666")
                continue
            tcol = "white" if (v / vmax) >= 0.52 else "#1a1a1a"
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                fontsize=11,
                color=tcol,
                fontweight="500",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Best test mIoU", fontsize=10)
    cbar.ax.minorticks_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_1024_vs_best_grouped(
    all_step_metrics: Dict[str, Dict[int, Dict[str, float]]],
    out_path: Path,
    *,
    first_step: int = 1024,
) -> None:
    """Grouped bars: light = mean test mIoU at ``first_step`` (all 5 present); dark = mean of per-dataset bests."""
    keys = ["s", "b", "l"]
    tick_labels = ["S+", "B", "L"]
    light_blue = "#8ec5e8"
    dark_blue = "#1e4a78"
    x = np.arange(len(keys), dtype=float)
    width = 0.36

    m_first: List[float] = []
    m_best: List[float] = []
    for k in keys:
        sm = all_step_metrics[k]
        m_first.append(mean_test_miou_shared_at_step(sm, first_step))
        # "best avg" = mean over datasets of (max test mIoU over ckpt steps), same as scaling line / heatmap row mean
        m_best.append(_avg_best_over_datasets(best_per_dataset(sm)))

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, k in enumerate(keys):
        xi = x[i]
        mf, mb = m_first[i], m_best[i]
        if np.isfinite(mf):
            b0 = ax.bar(
                xi - width / 2,
                mf,
                width,
                color=light_blue,
                edgecolor="white",
                linewidth=0.8,
                label=f"{first_step} step avg" if i == 0 else None,
            )
            h = b0[0].get_height()
            ax.text(b0[0].get_x() + b0[0].get_width() / 2, h + 0.004, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
        if np.isfinite(mb):
            b1 = ax.bar(
                xi + width / 2,
                mb,
                width,
                color=dark_blue,
                edgecolor="white",
                linewidth=0.8,
                label="best avg" if i == 0 else None,
            )
            h = b1[0].get_height()
            ax.text(b1[0].get_x() + b1[0].get_width() / 2, h + 0.004, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

        if np.isfinite(mf) and np.isfinite(mb):
            d = mb - mf
            ymax = max(mf, mb)
            ax.text(
                xi,
                ymax + 0.018,
                f"Δ {d:+.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#222222",
            )
        elif not np.isfinite(mf) and np.isfinite(mb):
            print(
                f"[WARN] {tick_labels[i]} ({MODELS[k][1].name}): no complete row at step {first_step}; skip light bar.",
                file=sys.stderr,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel("Mean test mIoU on shared datasets", fontsize=11)
    ax.set_title(
        f"From first checkpoint to best: {first_step} mean vs mean(per-dataset best)",
        fontsize=12,
        pad=10,
    )
    ax.grid(axis="y", linestyle="-", color="#d8d8d8", linewidth=0.85, alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=10)

    all_vals = [v for v in m_first + m_best if np.isfinite(v)]
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = max(0.02, (hi - lo) * 0.35)
        ax.set_ylim(max(0.0, lo - pad * 0.15), hi + pad)

    fig.text(
        0.5,
        0.02,
        f"Light: mean test mIoU at step {first_step} (five datasets all present). "
        "Dark: mean of each dataset’s best test mIoU over ckpt steps. "
        "Data: outputs_microscopy_continual_vit{{s,b,l}}16.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    fig.subplots_adjust(bottom=0.14, top=0.9)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_best_bar(
    best_by_model: Dict[str, Dict[str, float]],
    out_path: Path,
    title: str = "Best comparison (ViT S+ / B / L)",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(DATASETS))
    width = 0.24
    offsets = [-width, 0.0, width]
    for off, (key, label) in zip(offsets, [("s", "S+"), ("b", "B"), ("l", "L")]):
        vals = [best_by_model[key].get(ds, float("nan")) for ds in DATASETS]
        bars = ax.bar(
            x + off,
            vals,
            width,
            label=label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.6,
        )
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    h + 0.012,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("Best test mIoU")
    ax.set_title(title)
    ax.set_ylim(0.0, min(1.0, max(0.1, max(
        (best_by_model[k][ds] for k in best_by_model for ds in DATASETS
         if np.isfinite(best_by_model[k].get(ds, float("nan")))),
        default=0.8,
    ) + 0.12)))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _avg_best_over_datasets(per_dataset_best: Dict[str, float]) -> float:
    vals = [per_dataset_best.get(ds, float("nan")) for ds in DATASETS]
    if not any(np.isfinite(v) for v in vals):
        return float("nan")
    return float(np.nanmean(vals))


# ~ViT-S+/B/L @patch16 参数量（示意，与常见 scaling 图同量级）
SCALING_XTICKS = ["S+\n(~29M)", "B\n(~86M)", "L\n(~304M)"]
LINE_CONNECT_SCALING = "#152238"


def plot_scaling_avg_best_miou(best_by_model: Dict[str, Dict[str, float]], out_path: Path) -> None:
    """x: ViT S+ / B / L; y: mean over datasets of (best test mIoU per dataset across ckpt steps)."""
    keys = ["s", "b", "l"]
    labels = ["S+", "B", "L"]
    x = np.arange(3, dtype=float)
    ys = np.array([_avg_best_over_datasets(best_by_model[k]) for k in keys], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    finite = [(i, float(ys[i])) for i in range(3) if np.isfinite(ys[i])]
    if not finite:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=180, facecolor="white")
        plt.close(fig)
        return

    fi, fy = zip(*finite)
    fi = np.asarray(fi, dtype=float)
    fy = np.asarray(fy, dtype=float)
    ax.plot(fi, fy, color=LINE_CONNECT_SCALING, linewidth=2.9, solid_capstyle="round", zorder=2)

    span = max(float(np.ptp(fy)), 0.02)
    for i in range(3):
        if not np.isfinite(ys[i]):
            continue
        lab = labels[i]
        ax.scatter(
            [x[i]],
            [ys[i]],
            s=140,
            facecolors=COLORS[lab],
            edgecolors="white",
            linewidths=1.7,
            zorder=5,
        )
        ax.text(
            x[i],
            ys[i] + span * 0.045,
            f"{ys[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222222",
            zorder=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SCALING_XTICKS, fontsize=10)
    ax.set_ylabel("Avg test best mIoU", fontsize=11)
    ax.set_xlabel("Model scale", fontsize=11)
    ax.set_title("Scaling: average of per-dataset best test mIoU", fontsize=12, pad=10)
    ax.grid(axis="y", linestyle="-", color="#d4d4d4", linewidth=0.9, alpha=0.95)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#1a1a1a")
        spine.set_linewidth(0.9)

    caption = (
        "Per dataset: max test mIoU over all continual linear_probe steps; "
        "y is the mean of those five values (S+ / B / L each from its own output tree)."
    )
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=8, color="#444444")

    lo, hi = float(np.nanmin(ys)), float(np.nanmax(ys))
    pad = max(0.015, (hi - lo) * 0.2)
    ax.set_ylim(lo - pad, hi + pad * 1.5)
    ax.set_xlim(-0.35, 2.35)

    fig.subplots_adjust(bottom=0.2, top=0.88)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_per_dataset_curves(
    all_step_metrics: Dict[str, Dict[int, Dict[str, float]]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    axes = axes.flatten()
    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        for key, label in [("s", "S+"), ("b", "B"), ("l", "L")]:
            sm = all_step_metrics[key]
            curve: List[Tuple[int, float]] = []
            b0 = baseline_test_miou(ds, key)
            if b0 is not None:
                curve.append((0, b0))
            curve.extend((s, sm[s][ds]) for s in sm if ds in sm[s] and s > 0)
            curve.sort(key=lambda t: t[0])
            if not curve:
                continue
            xs, ys = zip(*curve)
            ax.plot(xs, ys, marker=".", ms=4, lw=1.4, label=label, color=COLORS[label])
        ax.set_title(ds)
        ax.set_xlabel("step (0 = baseline)")
        ax.set_ylabel("test mIoU")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=False, fontsize=8)
    axes[-1].axis("off")
    fig.suptitle("Test mIoU vs step (0 = linear_probe baseline, then continual ckpts)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot microscopy continual linear-probe mIoU summaries.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "plot" / "figures",
        help="Directory to write PNG figures.",
    )
    p.add_argument(
        "--first-step",
        type=int,
        default=1024,
        help="Step for the 'first checkpoint' mean in the 1024-vs-best bar chart (default 1024).",
    )
    args = p.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_step_metrics: Dict[str, Dict[int, Dict[str, float]]] = {}
    best_by_model: Dict[str, Dict[str, float]] = {}

    for key, (_label, root) in MODELS.items():
        sm = load_step_metrics(root)
        all_step_metrics[key] = sm
        best_by_model[key] = best_per_dataset(sm)
        if not sm:
            print(f"[WARN] no results under {root}", file=sys.stderr)

    plot_best_bar(best_by_model, out_dir / "best_grouped_miou.png")
    plot_best_miou_heatmap(best_by_model, out_dir / "best_test_miou_heatmap.png")

    bar1024_path = out_dir / "mean_miou_1024_vs_best_shared.png"
    plot_1024_vs_best_grouped(all_step_metrics, bar1024_path, first_step=args.first_step)
    shutil.copy2(bar1024_path, out_dir / "first_ckpt_to_best_miou.png")

    scaling_path = out_dir / "scaling_avg_best_miou.png"
    plot_scaling_avg_best_miou(best_by_model, scaling_path)
    shutil.copy2(scaling_path, out_dir / "mean_test_miou_vs_step.png")

    plot_per_dataset_curves(all_step_metrics, out_dir / "per_dataset_miou_vs_step.png")

    print(f"Wrote figures under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
