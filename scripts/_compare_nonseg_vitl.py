#!/usr/bin/env python3
"""Compare the 4 ViT-L BioDINOv3 models on NON-segmentation frozen-feature tasks
(classification / regression / retrieval+clustering / detection). Same sklearn-probe
protocol for all. dualroute(#1) vs l(plain baseline) vs robust4(#4) vs gram5(#5)."""
import json, glob, os

# model label -> (eval_root, iter)
MODELS = {
    "sp":        ("outputs/bio_eval_sp_15374", 15374),
    "b":         ("outputs/bio_eval_b_16399", 16399),
    "l":         ("outputs/bio_eval_l_15374", 15374),
    "hplus":     ("outputs/bio_eval_hplus_14349", 14349),
    "robust4":   ("outputs/bio_continue_vitl16_robust_eval_all_sklearn_20260609", 15374),
    "gram5":     ("outputs/bio_continue_vitl16_robust_hires_gram_1024_eval_all_sklearn_20260609", 3074),
    "dualroute": ("outputs/bio_eval_dualroute_ep15all", 15374),
}
ORDER = ["sp", "b", "l", "hplus", "robust4", "gram5", "dualroute"]

CLS = ["bbbc048-cellcycle","bloodmnist","breastmnist","chestmnist","cyclops-protein-loc",
       "dermamnist","midog25-atypical","octmnist","organamnist","organcmnist","organsmnist",
       "pathmnist","pneumoniamnist","retinamnist","tissuemnist"]
REG = ["bbbc005","bbbc013"]
RET = ["crc-val-he-7k","lc25000","nct-crc-he-1k"]
DET = ["livecell"]


def load(root, task, ds, it, fname="last_result.json"):
    hits = glob.glob(f"{root}/**/{task}/{ds}/{it}/{fname}", recursive=True) or \
           glob.glob(f"{root}/{task}/{ds}/{it}/{fname}")
    if not hits:
        return None
    try:
        return json.load(open(hits[0]))
    except Exception:
        return None


def cls_metric(d):
    if d is None: return None
    if d.get("task") == "multilabel_classification":  # chestmnist
        return d.get("macro_auc")            # AUROC for multilabel
    return d.get("balanced_accuracy")


def table(title, datasets, task, getter, fmt="{:.4f}", fname="last_result.json"):
    print(f"\n### {title}")
    print("dataset".ljust(20) + "".join(m.rjust(11) for m in ORDER))
    means = {m: [] for m in ORDER}
    for ds in datasets:
        line = ds.ljust(20)
        for m in ORDER:
            root, it = MODELS[m]
            d = load(root, task, ds, it, fname)
            v = getter(d)
            if isinstance(v, (int, float)):
                line += fmt.format(v).rjust(11); means[m].append(v)
            else:
                line += "   -   ".rjust(11)
        print(line)
    # mean row
    ml = "MEAN".ljust(20)
    for m in ORDER:
        ml += (fmt.format(sum(means[m])/len(means[m])) if means[m] else "   -   ").rjust(11)
    print(ml)


print("# ViT-L 非分割对比 (frozen sklearn probe, 同协议)")
print("iters:", {m: MODELS[m][1] for m in ORDER}, " (gram5 仅训到 3074, 欠训)")

table("Classification — balanced_acc (chestmnist=macro_AUROC)", CLS, "bio_classification", cls_metric)
table("Regression — R²", REG, "bio_regression", lambda d: d.get("r2") if d else None)
table("Retrieval — recall@1", RET, "bio_retrieval", lambda d: d.get("recall_at_1") if d else None)
table("Clustering — NMI", RET, "bio_retrieval", lambda d: d.get("nmi") if d else None)
table("Detection — patch F1", DET, "bio_detection",
      lambda d: d.get("test_patch_f1") if d else None, fname="results_bio_detection.json")
