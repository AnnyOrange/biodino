#!/usr/bin/env python3
"""Aggregate the 4 DINOv3 runs at 80k full-cryo (frozen protocol nlb4_cls_three_slices_raw)
into one final table + comparison figures.

  final_4run_fullcryo.csv
  10_final_4run_comparison.png
  11_final_4run_heatmap.png
"""
from __future__ import annotations

import csv
import io
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/mnt/huawei_deepcad/dinov3")
# (run_label, color, summary.csv, optional model-name filter for multi-row summaries, ckpt)
SOURCES = [
    ("DINOv3-base", "#1f77b4",
     "benchmark_runs/eval_ood_fullcryo_allruns_20260605/base/summary.csv", None, "12299"),
    ("ViT-L OEP1025", "#d62728",
     "benchmark_runs/eval_ood_remote_fullcryo_top3_20260602_212923/results/summary.csv",
     "vitl_oep1025_nlb4_cls_three_slices_raw-8199", "8199"),
    ("ChannelViT s6", "#2ca02c",
     "benchmark_runs/eval_ood_fullcryo_allruns_20260605/channelvit_s6_fixed/summary.csv", None, "1024"),
    ("ViT-H+ RGB3", "#9467bd",
     "benchmark_runs/eval_ood_fullcryo_allruns_20260605/hplus_rgb3/summary.csv", None, "1024"),
]
COMPOSITE = ["xray_pair_recall_at_1", "xray_dose_r2", "cryo_class_accuracy",
             "cryo_quality_auroc", "cryo_retrieval_map_at_10"]
PANEL = [
    ("composite", "Composite"),
    ("xray_dose_r2", "X-ray dose R²"),
    ("xray_resolution_r2", "X-ray resolution R²"),
    ("xray_pair_recall_at_1", "X-ray pair R@1"),
    ("xray_sample_accuracy", "X-ray sample acc"),
    ("cryo_quality_auroc", "Cryo quality AUROC"),
    ("cryo_quality_score_spearman", "Cryo quality Spearman"),
    ("cryo_class_accuracy", "Cryo class acc"),
    ("cryo_retrieval_map_at_10", "Cryo retrieval mAP@10"),
]


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def save_fig(fig, path, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, format="png", bbox_inches="tight")
    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    plt.close(fig)
    print(f"  wrote {path} ({len(data)/1e3:.0f} KB)")


def load_row(csv_path, model_filter):
    rows = list(csv.DictReader(open(BASE / csv_path)))
    if model_filter:
        rows = [r for r in rows if r.get("model") == model_filter]
    r = rows[0]
    rec = {k: fnum(v) for k, v in r.items()}
    vals = [rec.get(k, float("nan")) for k in COMPOSITE]
    vals = [v for v in vals if v == v]
    rec["composite"] = sum(vals) / len(vals) if vals else float("nan")
    return rec


def main():
    out = BASE / "benchmark_runs/eval_ood_analysis"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    data = []
    for label, color, path, mfilter, ckpt in SOURCES:
        rec = load_row(path, mfilter)
        rec["_label"], rec["_color"], rec["_ckpt"] = label, color, ckpt
        data.append(rec)

    # combined csv
    keys = [k for k, _ in PANEL] + ["cryo_n_particles", "xray_n_volumes"]
    with open(out / "final_4run_fullcryo.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "ckpt"] + keys)
        for d in data:
            w.writerow([d["_label"], d["_ckpt"]] + [f"{d.get(k, float('nan')):.4f}" for k in keys])
    print(f"  wrote {out/'final_4run_fullcryo.csv'}")

    # grouped bars
    x = np.arange(len(PANEL)); w = 0.2
    fig, ax = plt.subplots(figsize=(17, 7))
    for i, d in enumerate(data):
        ys = [d.get(k, float("nan")) for k, _ in PANEL]
        bars = ax.bar(x + (i - 1.5) * w, ys, w, color=d["_color"], label=f"{d['_label']} @{d['_ckpt']}")
        for b, v in zip(bars, ys):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([t for _, t in PANEL], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("score")
    ax.set_title("FINAL — 4 DINOv3 runs at 80k full-cryo, frozen protocol nlb4_cls_three_slices_raw", fontsize=13)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out / "figures" / "10_final_4run_comparison.png")

    # heatmap (column-normalized)
    mk = [k for k, _ in PANEL]
    M = np.array([[d.get(k, float("nan")) for k in mk] for d in data], dtype=float)
    Mn = M.copy()
    for j in range(M.shape[1]):
        lo, hi = np.nanmin(M[:, j]), np.nanmax(M[:, j])
        Mn[:, j] = (M[:, j] - lo) / (hi - lo) if hi > lo else 0.5
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(Mn, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(mk))); ax.set_xticklabels([t for _, t in PANEL], rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(data))); ax.set_yticklabels([f"{d['_label']} @{d['_ckpt']}" for d in data])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("FINAL 80k-cryo: best-per-run × metric (cell=value, color=column-normalized)", fontsize=12)
    fig.colorbar(im, ax=ax, label="column-normalized")
    fig.tight_layout()
    save_fig(fig, out / "figures" / "11_final_4run_heatmap.png")

    # print table
    print("\nFINAL 4-run @ 80k full-cryo (composite-ranked):")
    for d in sorted(data, key=lambda r: r["composite"], reverse=True):
        print(f"  {d['_label']:16}@{d['_ckpt']:>6}  comp={d['composite']:.4f}  "
              f"dose={d['xray_dose_r2']:.3f} pairR1={d['xray_pair_recall_at_1']:.3f} "
              f"qAUROC={d['cryo_quality_auroc']:.3f} class={d['cryo_class_accuracy']:.3f} map10={d['cryo_retrieval_map_at_10']:.4f}")


if __name__ == "__main__":
    main()
