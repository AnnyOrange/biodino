#!/usr/bin/env python3
"""Grouped-bar comparison of best-protocol segmentation across models.
Usage: python _plot_seg_comparison.py <out.png> LABEL1=dir1 LABEL2=dir2 ...
Each dir is a bio_eval output dir containing bio_segmentation_best/.../results.json
"""
import json, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1]
specs = [a.split("=", 1) for a in sys.argv[2:]]

DATASETS = ["bbbc038", "tissuenet", "monuseg", "pannuke", "conic"]
METRICS = ["mDice", "mIoU", "AJI", "bPQ"]


def load(dirpath):
    res = {}
    for f in glob.glob(f"{dirpath}/bio_segmentation_best/*/*/*/results.json"):
        ds = os.path.basename(os.path.dirname(os.path.dirname(f)))
        d = json.load(open(f))
        t = d.get("test") or d.get("val") or {}
        res[ds] = t
    return res

models = [(label, load(d)) for label, d in specs]

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.ravel()
x = np.arange(len(DATASETS))
w = 0.8 / max(1, len(models))
for ax, metric in zip(axes, METRICS):
    for i, (label, res) in enumerate(models):
        vals = [res.get(ds, {}).get(metric, float("nan")) for ds in DATASETS]
        bars = ax.bar(x + i * w, vals, w, label=label)
        for b, v in zip(bars, vals):
            if v == v:  # not nan
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_title(metric)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(DATASETS, rotation=20)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
fig.suptitle("best-protocol segmentation: " + " vs ".join(l for l, _ in models), fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(out, dpi=110)
print("saved", out)
for label, res in models:
    print(f"\n[{label}]")
    for ds in DATASETS:
        t = res.get(ds, {})
        print(f"  {ds:<10}", {m: round(t[m], 4) for m in METRICS if m in t} or "(missing)")
