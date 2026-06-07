#!/usr/bin/env python3
"""Plot DINOv3 OOD (X-ray + cryo-EM) evaluation results from the remote sweeps.

Reads the per-sweep ``results/summary.csv`` files and produces a set of figures:
  1. temporal_curves.png   - metric vs checkpoint iter, one line per run (all-ckpt sweep)
  2. run_comparison.png    - best-composite config per run, grouped bars across metrics
  3. protocol_effect.png   - marginal effect of protocol factors (nlb, pool, slice, invert)
  4. final_top3_fullcryo.png - the 3 final full-cryo (80k particle) configs side by side
  5. run_metric_heatmap.png  - best-per-run x metric heatmap (normalized)

Usage:
    conda run -n dinov3 python -m dinov3.eval.eval_ood.plot_results \
        --base-dir /mnt/huawei_deepcad/dinov3/benchmark_runs \
        --out benchmark_runs/eval_ood_analysis
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_fig(fig, path, dpi=130):
    """Save via an in-memory buffer + fsync.

    Direct ``fig.savefig`` onto an NFS share sometimes
    leaves a 0-byte file because the flush is not committed before exit; writing
    the encoded bytes explicitly with fsync avoids that.
    """
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, format="png")
    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    plt.close(fig)

RUNS = ["base", "vitl_oep1025", "channelvit_s6_fixed", "hplus_rgb3"]
RUN_LABEL = {
    "base": "DINOv3-base",
    "vitl_oep1025": "ViT-L OEP1025",
    "channelvit_s6_fixed": "ChannelViT s6",
    "hplus_rgb3": "ViT-H+ RGB3",
}
RUN_COLOR = {
    "base": "#1f77b4",
    "vitl_oep1025": "#d62728",
    "channelvit_s6_fixed": "#2ca02c",
    "hplus_rgb3": "#9467bd",
}
# launcher composite = mean of these (saturated OOD AUROC excluded)
COMPOSITE = [
    "xray_pair_recall_at_1",
    "xray_dose_r2",
    "cryo_class_accuracy",
    "cryo_quality_auroc",
    "cryo_retrieval_map_at_10",
]
# metrics shown in the temporal / comparison panels
PANEL_METRICS = [
    ("composite", "Composite (selection score)"),
    ("xray_dose_r2", "X-ray dose R²"),
    ("xray_resolution_r2", "X-ray resolution R²"),
    ("xray_pair_recall_at_1", "X-ray rigid/nonrigid R@1"),
    ("cryo_quality_auroc", "Cryo quality AUROC"),
    ("cryo_quality_score_spearman", "Cryo quality Spearman"),
    ("cryo_class_accuracy", "Cryo class acc (fine)"),
    ("cryo_retrieval_map_at_10", "Cryo retrieval mAP@10"),
]


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def parse_model(model: str):
    """`vitl_oep1025_nlb4_cls_three_slices_raw-8199` -> (run, protocol, ckpt)."""
    base, _, ckpt = model.rpartition("-")
    run = next((r for r in RUNS if base == r or base.startswith(r + "_")), base)
    protocol = base[len(run) + 1 :] if base.startswith(run + "_") else ""
    try:
        ck = int(ckpt)
    except ValueError:
        ck = -1
    return run, protocol, ck


def load(summary: Path):
    rows = []
    with summary.open(newline="") as f:
        for r in csv.DictReader(f):
            run, protocol, ckpt = parse_model(r.get("model", ""))
            rec = {k: fnum(v) for k, v in r.items()}
            vals = [rec.get(k, float("nan")) for k in COMPOSITE]
            vals = [v for v in vals if v == v]
            rec["composite"] = sum(vals) / len(vals) if vals else float("nan")
            rec["_run"], rec["_protocol"], rec["_ckpt"], rec["_model"] = run, protocol, ckpt, r.get("model", "")
            rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
def fig_temporal(rows, out: Path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (key, title) in zip(axes.flat, PANEL_METRICS):
        for run in RUNS:
            pts = sorted([(r["_ckpt"], r.get(key, float("nan"))) for r in rows if r["_run"] == run])
            pts = [(c, v) for c, v in pts if c >= 0 and v == v]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", ms=4, lw=1.6, color=RUN_COLOR[run], label=RUN_LABEL[run])
            bi = int(np.argmax(ys))
            ax.scatter([xs[bi]], [ys[bi]], s=90, facecolors="none", edgecolors=RUN_COLOR[run], lw=2, zorder=5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("checkpoint iter")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=10, loc="best")
    fig.suptitle(
        "DINOv3 OOD temporal sweep — metric vs checkpoint (protocol nlb4_cls_25d_raw; "
        "X-ray 124 vols, cryo 4k tuning). ○ = per-run best.",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, out)


def best_per_run(rows):
    best = {}
    for r in rows:
        run = r["_run"]
        if run not in best or r["composite"] > best[run]["composite"]:
            best[run] = r
    return best


def fig_run_comparison(rows, out: Path):
    best = best_per_run(rows)
    metrics = [m for m in PANEL_METRICS]
    x = np.arange(len(metrics))
    w = 0.2
    fig, ax = plt.subplots(figsize=(16, 7))
    for i, run in enumerate(RUNS):
        if run not in best:
            continue
        b = best[run]
        ys = [b.get(k, float("nan")) for k, _ in metrics]
        bars = ax.bar(x + (i - 1.5) * w, ys, w, color=RUN_COLOR[run], label=f"{RUN_LABEL[run]} @ {b['_ckpt']}")
        for bar, v in zip(bars, ys):
            if v == v:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in metrics], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("score")
    ax.set_title("Best-composite config per DINOv3 run (core_suggested sweep; cryo 4k tuning)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out)


def fig_protocol_effect(rows, out: Path):
    """Marginal mean composite for each protocol factor level."""
    factors = {
        "n_last_blocks": (lambda p: "nlb4" if "nlb4" in p else ("nlb1" if "nlb1" in p else "?")),
        "pooling": (lambda p: "avg" if "_avg" in p else ("cls" if "_cls" in p else "?")),
        "x-ray input": (lambda p: "three_slices" if "three_slices" in p or "_25d" in p else "single_slice"),
        "cryo invert": (lambda p: "inv" if p.endswith("_inv") or "_inv" in p else "raw"),
    }
    fig, axes = plt.subplots(1, len(factors), figsize=(18, 5))
    for ax, (fname, fn) in zip(axes, factors.items()):
        groups = {}
        for r in rows:
            lv = fn(r["_protocol"])
            if lv == "?":
                continue
            groups.setdefault(lv, []).append(r["composite"])
        levels = sorted(groups)
        means = [float(np.nanmean(groups[l])) for l in levels]
        errs = [float(np.nanstd(groups[l])) for l in levels]
        ax.bar(levels, means, yerr=errs, capsize=5, color="#4c72b0", alpha=0.85)
        for i, m in enumerate(means):
            ax.text(i, m + 0.002, f"{m:.3f}", ha="center", fontsize=9)
        ax.set_title(fname, fontsize=12)
        ax.set_ylabel("mean composite")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Protocol factor effects on composite (full-protocol top-ckpt sweep, 64 configs)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, out)


def fig_final_top3(rows, out: Path):
    rows = sorted(rows, key=lambda r: r["composite"], reverse=True)
    metrics = [m for m in PANEL_METRICS]
    x = np.arange(len(metrics))
    w = 0.25
    fig, ax = plt.subplots(figsize=(16, 7))
    colors = ["#d62728", "#ff7f0e", "#8c564b"]
    for i, r in enumerate(rows):
        ys = [r.get(k, float("nan")) for k, _ in metrics]
        lbl = r["_protocol"] + f"  (comp {r['composite']:.3f})"
        bars = ax.bar(x + (i - 1) * w, ys, w, color=colors[i % 3], label=lbl)
        for bar, v in zip(bars, ys):
            if v == v:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in metrics], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("score")
    ax.set_title("Final full-cryo (80k particles) — ViT-L OEP1025 @ 8199, top-3 protocols", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out)


def fig_heatmap(rows, out: Path):
    best = best_per_run(rows)
    metric_keys = [k for k, _ in PANEL_METRICS]
    runs = [r for r in RUNS if r in best]
    M = np.array([[best[run].get(k, float("nan")) for k in metric_keys] for run in runs], dtype=float)
    # column-normalize to [0,1] for visual comparability
    Mn = M.copy()
    for j in range(M.shape[1]):
        col = M[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        Mn[:, j] = (col - lo) / (hi - lo) if hi > lo else 0.5
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(Mn, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels([t for _, t in PANEL_METRICS], rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([f"{RUN_LABEL[r]} @{best[r]['_ckpt']}" for r in runs])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Best-per-run metrics (cell=raw value, color=column-normalized rank)", fontsize=12)
    fig.colorbar(im, ax=ax, label="column-normalized")
    fig.tight_layout()
    save_fig(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="benchmark_runs")
    ap.add_argument("--core", default="eval_ood_remote_core_suggested_20260602_200423")
    ap.add_argument("--selection", default="eval_ood_remote_selection_all_20260602_204156")
    ap.add_argument("--fullprotocol", default="eval_ood_remote_full_topckpt_20260602_211139")
    ap.add_argument("--fullcryo", default="eval_ood_remote_fullcryo_top3_20260602_212923")
    ap.add_argument("--out", default="benchmark_runs/eval_ood_analysis")
    args = ap.parse_args()

    base = Path(args.base_dir)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    sel = load(base / args.selection / "results" / "summary.csv")
    core = load(base / args.core / "results" / "summary.csv")
    fullp = load(base / args.fullprotocol / "results" / "summary.csv")
    fcryo = load(base / args.fullcryo / "results" / "summary.csv")

    fig_temporal(sel, out / "figures" / "1_temporal_curves.png")
    fig_run_comparison(core, out / "figures" / "2_run_comparison.png")
    fig_protocol_effect(fullp, out / "figures" / "3_protocol_effect.png")
    fig_final_top3(fcryo, out / "figures" / "4_final_top3_fullcryo.png")
    fig_heatmap(core, out / "figures" / "5_run_metric_heatmap.png")

    print(f"[plot] wrote 5 figures to {out / 'figures'}")
    # also dump best-per-run table for the summary
    best = best_per_run(core)
    print("[plot] best-composite per run (core_suggested):")
    for run in RUNS:
        if run in best:
            b = best[run]
            print(f"  {RUN_LABEL[run]:16} ckpt={b['_ckpt']:>6} protocol={b['_protocol']:28} composite={b['composite']:.4f}")


if __name__ == "__main__":
    main()
