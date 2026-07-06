#!/usr/bin/env python3
"""Cross-model best-protocol seg summary (final ckpt of each model). Rows=datasets, cols=models."""
import json, glob, os, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "outputs/bioseg_best_7models_20260622"
MODELS = ["sp", "b", "l", "hplus", "robust4", "gram5", "dualroute"]
DATASETS = ["bbbc038", "cellpose", "livecell", "tissuenet", "monuseg", "pannuke", "conic"]
METRICS = ["mDice", "mIoU", "AJI", "bPQ"]


def load_final(model):
    """dataset -> test-metrics dict, using the highest iter available per dataset."""
    best = {}
    for f in glob.glob(f"{ROOT}/{model}/bio_segmentation_best/*/*/*/results.json"):
        parts = f.split("/")
        ds, it = parts[-3], parts[-2]
        try:
            it = int(it)
        except ValueError:
            continue
        if ds not in DATASETS:
            continue
        if ds not in best or it > best[ds][0]:
            d = json.load(open(f))
            best[ds] = (it, d.get("test") or d.get("val") or {})
    return {ds: v[1] for ds, v in best.items()}, {ds: v[0] for ds, v in best.items()}


data = {}
iters = {}
for m in MODELS:
    data[m], iters[m] = load_final(m)

print(f"# Cross-model best-protocol segmentation (final ckpt) — {ROOT}\n")
print("final iter per model:", {m: (max(iters[m].values()) if iters[m] else "-") for m in MODELS})
for metric in METRICS:
    print(f"\n### {metric}")
    print("dataset".ljust(11) + "".join(m.rjust(10) for m in MODELS))
    for ds in DATASETS:
        row = ds.ljust(11)
        for m in MODELS:
            v = data[m].get(ds, {}).get(metric)
            row += (f"{v:.4f}" if isinstance(v, (int, float)) else "  -  ").rjust(10)
        print(row)
# coverage
print("\n### coverage (datasets done per model, at any ckpt)")
for m in MODELS:
    print(f"  {m:<10} {len(data[m])}/7  iters_seen=", sorted(set(iters[m].values())) if iters[m] else [])
