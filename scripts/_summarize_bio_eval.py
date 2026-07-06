#!/usr/bin/env python3
"""Summarize a bio_eval output dir (classification last_result.json + segmentation results.json)."""
import json, glob, os, sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
ckpt = sys.argv[2] if len(sys.argv) > 2 else "*"


def num(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "  -  "


print("==== CLASSIFICATION ====")
hdr = "dataset".ljust(22) + "acc".rjust(8) + "bal_acc".rjust(9) + "macroF1".rjust(9) + "   extra"
print(hdr)
for f in sorted(glob.glob(f"{root}/bio_classification/*/{ckpt}/last_result.json")):
    d = json.load(open(f))
    ds = d.get("dataset", os.path.basename(os.path.dirname(os.path.dirname(f))))
    row = ds.ljust(22) + num(d.get("accuracy")).rjust(8) + num(d.get("balanced_accuracy")).rjust(9) + num(d.get("macro_f1")).rjust(9)
    extra = {k: round(v, 4) for k, v in d.items()
             if k in ("auroc", "macro_auroc", "micro_auroc", "map", "mAP", "auc") and isinstance(v, (int, float))}
    print(row + "   " + (str(extra) if extra else ""))

print()
print("==== SEGMENTATION (test split; val in parens if no test) ====")
print("dataset".ljust(12) + "mDice".rjust(8) + "mIoU".rjust(8) + "AJI".rjust(8) + "bPQ".rjust(8))
for f in sorted(glob.glob(f"{root}/bio_segmentation*/*/*/{ckpt}/results.json")):
    d = json.load(open(f))
    ds = os.path.basename(os.path.dirname(os.path.dirname(f)))
    t = d.get("test") or d.get("val") or {}
    print(ds.ljust(12) + num(t.get("mDice")).rjust(8) + num(t.get("mIoU")).rjust(8)
          + num(t.get("AJI")).rjust(8) + num(t.get("bPQ")).rjust(8))
