#!/usr/bin/env python3
"""Summarize the full multi-task eval of dual-route ep15_all into per-task tables."""
import json, glob, os, sys

R = sys.argv[1] if len(sys.argv) > 1 else "outputs/bio_eval_dualroute_ep15all"
IT = "15374"


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def fmt(x, n=4):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "  -  "


def first_json(d):
    for p in sorted(glob.glob(f"{d}/*.json")):
        if "feature" not in p:
            return jload(p)
    return {}


print("\n############### CLASSIFICATION (acc / bal_acc / macroF1) ###############")
print("dataset".ljust(22) + "acc".rjust(8) + "bal_acc".rjust(9) + "macroF1".rjust(9) + "   note")
for f in sorted(glob.glob(f"{R}/bio_classification/*/{IT}/last_result.json")):
    d = jload(f); ds = d.get("dataset", f.split("/")[-3])
    extra = ""
    for k in ("auroc", "macro_auroc", "micro_auroc", "map", "mAP"):
        if isinstance(d.get(k), (int, float)):
            extra += f" {k}={d[k]:.4f}"
    print(ds.ljust(22) + fmt(d.get("accuracy")).rjust(8) + fmt(d.get("balanced_accuracy")).rjust(9)
          + fmt(d.get("macro_f1")).rjust(9) + "  " + (extra or ("multilabel" if ds == "chestmnist" else "")))

print("\n############### REGRESSION (MAE / R2 / Spearman) ###############")
print("dataset".ljust(14) + "MAE".rjust(10) + "R2".rjust(9) + "Spearman".rjust(10))
for d_ in sorted(glob.glob(f"{R}/bio_regression/*/{IT}")):
    d = first_json(d_); ds = d.get("dataset", d_.split("/")[-2])
    print(ds.ljust(14) + fmt(d.get("mae")).rjust(10) + fmt(d.get("r2")).rjust(9) + fmt(d.get("spearman")).rjust(10))

print("\n############### RETRIEVAL / CLUSTERING ###############")
print("dataset".ljust(16) + "recall@1".rjust(10) + "mAP@5".rjust(9) + "MRR".rjust(8) + "ARI".rjust(8) + "NMI".rjust(8))
for d_ in sorted(glob.glob(f"{R}/bio_retrieval/*/{IT}")):
    d = first_json(d_); ds = d.get("dataset", d_.split("/")[-2])
    print(ds.ljust(16) + fmt(d.get("recall_at_1")).rjust(10) + fmt(d.get("map_at_5")).rjust(9)
          + fmt(d.get("mrr")).rjust(8) + fmt(d.get("ari") or d.get("cluster_ari")).rjust(8)
          + fmt(d.get("nmi") or d.get("cluster_nmi")).rjust(8))

print("\n############### DETECTION (patch center-probe) ###############")
for f in sorted(glob.glob(f"{R}/bio_detection/*/{IT}/results_bio_detection.json")):
    d = jload(f); ds = d.get("dataset", f.split("/")[-3])
    print(f"{ds:<12} test F1={fmt(d.get('test_patch_f1'),2)}  acc={fmt(d.get('test_patch_accuracy'),2)}  "
          f"prec={fmt(d.get('test_patch_precision'),2)}  recall={fmt(d.get('test_patch_recall'),2)}")

print("\n############### OOD (xray + cryo) ###############")
for f in sorted(glob.glob(f"{R}/ood/*/{IT}/last_result.json")):
    d = jload(f)
    for mod in ("xray", "cryo"):
        keys = {k: v for k, v in d.items() if k.startswith(mod + "_")}
        if keys:
            print(f"[{mod}] ood_auroc={fmt(d.get(mod+'_ood_auroc'))}  ood_AP={fmt(d.get(mod+'_ood_average_precision'))}  "
                  f"class_acc={fmt(d.get(mod+'_class_accuracy'))}  proj_acc={fmt(d.get(mod+'_project_accuracy'))}  "
                  f"cluster_NMI={fmt(d.get(mod+'_cluster_nmi'))}  cluster_ARI={fmt(d.get(mod+'_cluster_ari'))}")

print("\n############### SEGMENTATION (best protocol, test) ###############")
print("dataset".ljust(12) + "mDice".rjust(8) + "mIoU".rjust(8) + "AJI".rjust(8) + "bPQ".rjust(8))
for f in sorted(glob.glob(f"{R}/bio_segmentation_best/*/*/{IT}/results.json")):
    d = jload(f); ds = f.split("/")[-3]; t = d.get("test") or d.get("val") or {}
    print(ds.ljust(12) + fmt(t.get("mDice")).rjust(8) + fmt(t.get("mIoU")).rjust(8)
          + fmt(t.get("AJI")).rjust(8) + fmt(t.get("bPQ")).rjust(8))
