#!/usr/bin/env python3
"""Consolidate ALL 7-model BioDINOv3 eval results (segmentation + non-seg frozen probes + OOD)
into a single markdown report under outputs/."""
import json, glob, os

OUT = "outputs/BIODINOV3_7MODEL_FULL_EVAL_20260624.md"
ORDER = ["sp", "b", "l", "hplus", "robust4", "gram5", "dualroute"]
NICE = {"sp": "sp (ViT-S+)", "b": "b (ViT-B)", "l": "l (ViT-L)", "hplus": "hplus (ViT-H+)",
        "robust4": "robust4 (#4)", "gram5": "gram5 (#5)*", "dualroute": "dualroute (#1)"}

# ---------------- segmentation ----------------
SEG_ROOT = "outputs/bioseg_best_7models_20260622"
SEG_DS = ["bbbc038", "cellpose", "livecell", "tissuenet", "monuseg", "pannuke", "conic"]
SEG_METRICS = ["mDice", "mIoU", "AJI", "bPQ"]


def seg_load(model):
    best = {}
    for f in glob.glob(f"{SEG_ROOT}/{model}/bio_segmentation_best/*/*/*/results.json"):
        p = f.split("/"); ds, it = p[-3], p[-2]
        if ds not in SEG_DS:
            continue
        try:
            it = int(it)
        except ValueError:
            continue
        if ds not in best or it > best[ds][0]:
            d = json.load(open(f)); best[ds] = (it, d.get("test") or d.get("val") or {})
    return {ds: v[1] for ds, v in best.items()}


seg = {m: seg_load(m) for m in ORDER}

# ---------------- non-seg ----------------
NS = {"sp": ("outputs/bio_eval_sp_15374", 15374), "b": ("outputs/bio_eval_b_16399", 16399),
      "l": ("outputs/bio_eval_l_15374", 15374), "hplus": ("outputs/bio_eval_hplus_14349", 14349),
      "robust4": ("outputs/bio_continue_vitl16_robust_eval_all_sklearn_20260609", 15374),
      "gram5": ("outputs/bio_continue_vitl16_robust_hires_gram_1024_eval_all_sklearn_20260609", 3074),
      "dualroute": ("outputs/bio_eval_dualroute_ep15all", 15374)}
CLS = ["bbbc048-cellcycle", "bloodmnist", "breastmnist", "chestmnist", "cyclops-protein-loc",
       "dermamnist", "midog25-atypical", "octmnist", "organamnist", "organcmnist", "organsmnist",
       "pathmnist", "pneumoniamnist", "retinamnist", "tissuemnist"]


def ns_load(m, task, ds, fname="last_result.json"):
    root, it = NS[m]
    h = glob.glob(f"{root}/**/{task}/{ds}/{it}/{fname}", recursive=True) or glob.glob(f"{root}/{task}/{ds}/{it}/{fname}")
    try:
        return json.load(open(h[0])) if h else None
    except Exception:
        return None


def cls_metric(d):
    if not d:
        return None
    return d.get("macro_auc") if d.get("task") == "multilabel_classification" else d.get("balanced_accuracy")


def ns_mean(m, task, dss, getter, fname="last_result.json"):
    vs = [getter(ns_load(m, task, ds, fname)) for ds in dss]
    vs = [v for v in vs if isinstance(v, (int, float))]
    return sum(vs) / len(vs) if vs else None

# ---------------- OOD ----------------
OOD = {"sp": ("outputs/bio_eval_sp_ood/ood", "sp", 15374), "b": ("outputs/bio_eval_b_ood/ood", "b", 16399),
       "l": ("outputs/bio_eval_l_ood/ood", "l", 15374), "hplus": ("outputs/bio_eval_hplus_ood/ood", "hplus", 14349),
       "robust4": ("outputs/bio_eval_robust4_ood/ood", "robust4", 15374),
       "gram5": ("outputs/bio_eval_gram5_ood/ood", "gram5", 3074),
       "dualroute": ("outputs/bio_eval_dualroute_ep15all/ood", "dualroute_ep15all", 15374)}


def ood_load(m):
    r, mn, it = OOD[m]; h = glob.glob(f"{r}/{mn}/{it}/last_result.json")
    try:
        return json.load(open(h[0])) if h else {}
    except Exception:
        return {}


ood = {m: ood_load(m) for m in ORDER}

# ---------------- emit markdown ----------------
def fmt(v, p=4):
    return f"{v:.{p}f}" if isinstance(v, (int, float)) else "—"


def mdrow(cells):
    return "| " + " | ".join(cells) + " |"


L = []
L.append("# BioDINOv3 — 7-model full evaluation (segmentation + non-seg + OOD)\n")
L.append("**Date:** 2026-06-24  ·  **Machines:** seg on local 8×RTX-5090; non-seg+OOD on deepcad 8×A100 (172.16.0.230)\n")
L.append("All models are continual-pretrained DINOv3 backbones, evaluated with FROZEN features. "
         "`gram5` only reached iter 3074 (undertrained); all others are final iters.\n")
L.append("## Models\n")
L.append(mdrow(["label", "size / variant", "run dir", "iter"]))
L.append(mdrow(["---"] * 4))
MROWS = [("sp", "ViT-S+", "bio_continue_vits16_ep15_1025", 15374),
         ("b", "ViT-B", "bio_continue_1025_a100_grad_acc_2_base", 16399),
         ("l", "ViT-L (plain baseline)", "bio_continue_vitL16_OEP1025_ep15_b1024_1025", 15374),
         ("hplus", "ViT-H+", "bio_continue_rgb3_vith16plus (DCP→consolidated)", 14349),
         ("robust4", "ViT-L #4 16bit-robust-norm", "bio_continue_vitl16_robust", 15374),
         ("gram5", "ViT-L #5 robust+Gram+hires*", "bio_continue_vitl16_robust_hires_gram_1024", 3074),
         ("dualroute", "ViT-L #1 dual-route stem", "bio_continue_vitl16_dualroute_ep15_all", 15374)]
for lbl, sz, rd, it in MROWS:
    L.append(mdrow([lbl, sz, f"`{rd}`", str(it)]))

# Section 1: segmentation
L.append("\n## 1. Segmentation — best per-dataset linear probe (frozen, 50-epoch)\n")
L.append("Per-dataset 'best' protocol (resolution/layers/loss tuned per dataset); same for all models.\n")
for metric in SEG_METRICS:
    L.append(f"\n### {metric}\n")
    L.append(mdrow(["dataset"] + ORDER))
    L.append(mdrow(["---"] * (len(ORDER) + 1)))
    for ds in SEG_DS:
        L.append(mdrow([ds] + [fmt(seg[m].get(ds, {}).get(metric)) for m in ORDER]))

# Section 2: non-seg
L.append("\n## 2. Non-segmentation — frozen sklearn probes\n")
NS_TASKS = [
    ("Classification — balanced-acc (chestmnist=macro-AUROC), 15-dataset mean", CLS, "bio_classification", cls_metric, "last_result.json"),
    ("Regression — R² (bbbc005)", ["bbbc005"], "bio_regression", lambda d: d.get("r2") if d else None, "last_result.json"),
    ("Retrieval — recall@1 (3-dataset mean)", ["crc-val-he-7k", "lc25000", "nct-crc-he-1k"], "bio_retrieval", lambda d: d.get("recall_at_1") if d else None, "last_result.json"),
    ("Clustering — NMI (3-dataset mean)", ["crc-val-he-7k", "lc25000", "nct-crc-he-1k"], "bio_retrieval", lambda d: d.get("nmi") if d else None, "last_result.json"),
    ("Detection — livecell patch-F1", ["livecell"], "bio_detection", lambda d: d.get("test_patch_f1") if d else None, "results_bio_detection.json"),
]
L.append(mdrow(["task"] + ORDER))
L.append(mdrow(["---"] * (len(ORDER) + 1)))
for title, dss, task, getter, fname in NS_TASKS:
    L.append(mdrow([title] + [fmt(ns_mean(m, task, dss, getter, fname)) for m in ORDER]))

# Section 3: OOD
L.append("\n## 3. Out-of-distribution (X-ray tomography + cryo-EM), frozen features\n")
OOD_M = [("X-ray OOD-detect AUROC", "xray_ood_auroc"), ("cryo OOD-detect AUROC", "cryo_ood_auroc"),
         ("cryo QUALITY AUROC (discriminative)", "cryo_quality_auroc"),
         ("cryo clustering NMI", "cryo_cluster_nmi")]
L.append(mdrow(["metric"] + ORDER))
L.append(mdrow(["---"] * (len(ORDER) + 1)))
for title, k in OOD_M:
    L.append(mdrow([title] + [fmt(ood[m].get(k)) for m in ORDER]))
L.append("\n> OOD-detection AUROC saturates at 1.0 for all models — X-ray vs cryo-EM are entirely "
         "distinct imaging modalities, so ID/OOD separation is trivial and non-discriminative. The "
         "discriminative signals are cryo *quality* AUROC (good vs bad particles) and clustering NMI.\n")

# Section 4: conclusions
L.append("## 4. Conclusions\n")
L.append("**dualroute (#1 dual-route stem) vs the L baseline — across ALL tasks:** on every RGB task "
         "dualroute matches L (classification/regression/retrieval/OOD ≈ tied, clustering slightly better, "
         "detection within 0.2 pt); the *only* place it trails L is dense segmentation, where the extra "
         "pool-path parameters slightly dilute per-pixel features. Net: the dual-route stem does **not** "
         "hurt RGB performance. Its multichannel advantage is **untestable here** — every benchmark above is "
         "3-channel RGB, so dualroute routes through the standard RGB path. A true multichannel eval "
         "(CHAMMI / JUMP / IMC, >3 channels) is needed to demonstrate its value.\n")
L.append("**Cross-size observations:** segmentation is size-dominated (hplus wins all dense metrics); "
         "classification is nearly size-independent (sp ≈ hplus on the 15-dataset mean) and the "
         "normalization variants (robust4/gram5) lead slightly; clustering favors the largest model (hplus).\n")

# Section 5: provenance
L.append("## 5. Provenance\n")
L.append("- Seg: `scripts/_summarize_7models.py` over `outputs/bioseg_best_7models_20260622/` (651 jobs, all done).\n")
L.append("- Non-seg: `dinov3.eval.bio_frozen_eval` (sklearn probes, last1 layer, img 224, bf16) via `scripts/run_bio_benchmark_all.sh`; comparison `scripts/_compare_nonseg_vitl.py`.\n")
L.append("- OOD: `dinov3.eval.eval_ood.dinov3_runner` (tasks xray+cryo).\n")
L.append("- hplus is DCP-only at 14349; the frozen-eval harness can't load DCP, so a consolidated single-file `.pth` was used.\n")

os.makedirs("outputs", exist_ok=True)
open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT} ({len(L)} lines)")
