#!/usr/bin/env python3
"""Run all four frozen-feature fixed-budget tests for one encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, r2_score, recall_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "outputs/02_eval_runs/external4_hplus_fm_fixedbudget_3090qi_20260910"
HPLUS_CKPT = ROOT / "outputs/01_training_runs/HS6_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e15_seed0_4xH100_20260818/ckpt/15374/checkpoint.pth"
HPLUS_CONFIG = ROOT / "outputs/01_training_runs/HS6_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e15_seed0_4xH100_20260818/config.yaml"
DATASETS = ("ctc", "hest", "rxrx3", "midogpp")


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def build(model: str, device: str, batch: int, checkpoint: Path | None = None,
          train_config: Path | None = None):
    if checkpoint is not None:
        if train_config is None:
            raise ValueError("--train-config is required with --checkpoint")
        from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder
        return Dinov3CkptEncoder(
            checkpoint=checkpoint, train_config=train_config, device=device,
            n_last_blocks=1, use_avgpool=True, autocast_dtype=torch.bfloat16,
            image_size=224, resize_size=256, channel_policy="auto",
        )
    if model == "hs6_hplus":
        from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder
        return Dinov3CkptEncoder(
            checkpoint=HPLUS_CKPT, train_config=HPLUS_CONFIG, device=device,
            n_last_blocks=1, use_avgpool=True, autocast_dtype=torch.bfloat16,
            image_size=224, resize_size=256, channel_policy="auto",
        )
    sys.path[:0] = [
        "/mnt/huawei_deepcad/benchmark_model",
        "/mnt/huawei_deepcad/benchmark_model/_vendor/external_gapfill_py311",
        "/mnt/huawei_deepcad/benchmark_model/_vendor",
        "/home/bbnc/venvs/external_fm/lib/python3.11/site-packages",
    ]
    if model == "bioclip":
        # open_clip needs ftfy; reuse the already installed pure-Python copy
        # without putting the second environment ahead of this env's binaries.
        sys.path.append("/home/bbnc/anaconda3/envs/dinov3/lib/python3.11/site-packages")
    from benchmark_eval.encoders import build_encoder
    return build_encoder(model, device=device, batch_size=batch)


def extract(encoder, images_path: Path, feature_path: Path, batch: int) -> np.ndarray:
    if feature_path.exists():
        feats = np.load(feature_path)
        if len(feats) == len(np.load(images_path, mmap_mode="r")):
            print(f"[resume] {feature_path}", flush=True)
            return feats
    images = np.load(images_path, mmap_mode="r")
    chunks = []
    for start in range(0, len(images), batch):
        pil = [Image.fromarray(np.asarray(x)) for x in images[start:start + batch]]
        chunks.append(np.asarray(encoder.encode_pil(pil), dtype=np.float32))
        if start == 0 or (start // batch + 1) % 20 == 0:
            print(f"[features] {feature_path.parent.name}: {min(start+batch, len(images))}/{len(images)}", flush=True)
    feats = np.concatenate(chunks)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temp = feature_path.with_suffix(".tmp.npy")
    np.save(temp, feats.astype(np.float16))
    os.replace(temp, feature_path)
    return feats


def scaled(train: np.ndarray, test: np.ndarray):
    scaler = StandardScaler()
    return scaler.fit_transform(train), scaler.transform(test)


def ctc_result(x, y, meta):
    rows = meta["rows"]
    train = np.asarray([r["split"] == "train" for r in rows])
    test = ~train
    a, b = scaled(x[train], x[test])
    pred = Ridge(alpha=10.0, solver="lsqr").fit(a, y[train]).predict(b)
    truth = y[test]
    per_domain = {}
    domains = np.asarray([r["domain"] for r in rows])[test]
    for domain in sorted(set(domains)):
        mask = domains == domain
        per_domain[domain] = {"n": int(mask.sum()), "mae": finite(mean_absolute_error(truth[mask], pred[mask]))}
    return {
        "protocol_id": meta["protocol_id"], "n_train": int(train.sum()), "n_test": int(test.sum()),
        "r2": finite(r2_score(truth, pred)), "mae": finite(mean_absolute_error(truth, pred)),
        "spearman": finite(spearmanr(truth, pred).statistic), "per_domain": per_domain,
        "status": "OBSERVATIONAL", "proxy": True,
    }


def safe_corr(a, b, fn):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return finite(fn(a, b).statistic)


def hest_result(x, y, meta):
    rows = meta["rows"]
    task_arr = np.asarray([r["task"] for r in rows])
    split_arr = np.asarray([r["split"] for r in rows])
    tasks = {}
    for task in sorted(set(task_arr)):
        train = (task_arr == task) & (split_arr == "train")
        test = (task_arr == task) & (split_arr == "test")
        a, b = scaled(x[train], x[test])
        pred = Ridge(alpha=10.0, solver="lsqr").fit(a, y[train]).predict(b)
        truth = y[test]
        pears = [safe_corr(truth[:, j], pred[:, j], pearsonr) for j in range(truth.shape[1])]
        spears = [safe_corr(truth[:, j], pred[:, j], spearmanr) for j in range(truth.shape[1])]
        pears = [v for v in pears if v is not None]; spears = [v for v in spears if v is not None]
        tasks[task] = {
            "n_train": int(train.sum()), "n_test": int(test.sum()),
            "gene_wise_pearson": finite(np.mean(pears)) if pears else None,
            "gene_wise_spearman": finite(np.mean(spears)) if spears else None,
            "r2": finite(r2_score(truth, pred, multioutput="variance_weighted")),
            "mae": finite(mean_absolute_error(truth, pred)),
        }
    def macro(key):
        vals = [v[key] for v in tasks.values() if v[key] is not None]
        return finite(np.mean(vals)) if vals else None
    return {
        "protocol_id": meta["protocol_id"], "n": len(x), "tasks": tasks,
        "gene_wise_pearson": macro("gene_wise_pearson"),
        "gene_wise_spearman": macro("gene_wise_spearman"),
        "r2": macro("r2"), "mae": macro("mae"), "status": "OBSERVATIONAL", "proxy": False,
    }


def rxrx3_result(x, y, meta):
    rows = meta["rows"]
    gallery = np.asarray([r["split"] == "gallery" for r in rows])
    query = ~gallery
    g, q = x[gallery].astype(np.float32), x[query].astype(np.float32)
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    scores = q @ g.T
    order = np.argsort(-scores, axis=1)
    gy, qy = y[gallery], y[query]
    ranks = np.asarray([np.flatnonzero(gy[idx] == qy[i])[0] + 1 for i, idx in enumerate(order)])
    labels = np.concatenate((gy, qy))
    n_clusters = len(set(labels.tolist()))
    cluster_input = np.concatenate((g, q))
    if meta.get("protocol_id") == "crispr-query-guide-plate-disjoint-all-eligible-genes-v1":
        pred_clusters = MiniBatchKMeans(
            n_clusters=n_clusters, n_init=5, random_state=0,
            batch_size=min(1024, len(cluster_input)), max_iter=200,
        ).fit_predict(cluster_input)
        clustering_method = "MiniBatchKMeans(n_init=5,max_iter=200,seed=0)"
    else:
        pred_clusters = KMeans(n_clusters=n_clusters, n_init=5, random_state=0).fit_predict(cluster_input)
        clustering_method = "KMeans(n_init=5,seed=0)"
    from sklearn.metrics import normalized_mutual_info_score
    reciprocal_ranks = 1.0 / ranks
    is_formal = meta.get("protocol_id") == "crispr-query-guide-plate-disjoint-all-eligible-genes-v1"
    return {
        "protocol_id": meta["protocol_id"], "n_gallery": int(gallery.sum()), "n_query": int(query.sum()),
        "recall_at_1": finite(np.mean(ranks <= 1)), "recall_at_5": finite(np.mean(ranks <= 5)),
        "recall_at_10": finite(np.mean(ranks <= 10)), "mrr_at_10": finite(np.mean(np.where(ranks <= 10, 1/ranks, 0))),
        "mrr": finite(np.mean(reciprocal_ranks)),
        "map": finite(np.mean(reciprocal_ranks)),
        "nmi": finite(normalized_mutual_info_score(labels, pred_clusters)),
        "clustering_method": clustering_method,
        "status": "FORMAL" if is_formal else "OBSERVATIONAL", "proxy": False,
    }


def midog_result(x, y, meta):
    rows = meta["rows"]
    train = np.asarray([r["split"] == "train" for r in rows])
    test = ~train
    a, b = scaled(x[train], x[test])
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=0).fit(a, y[train])
    score = clf.predict_proba(b)[:, 1]
    pred = score >= 0.5
    truth = y[test]
    tumors = np.asarray([r["tumor"] for r in rows])[test]
    per_domain = {}
    for tumor in sorted(set(tumors)):
        mask = tumors == tumor
        per_domain[tumor] = {"n": int(mask.sum()), "f1": finite(f1_score(truth[mask], pred[mask], zero_division=0))}
    return {
        "protocol_id": meta["protocol_id"], "n_train": int(train.sum()), "n_test": int(test.sum()),
        "f1": finite(f1_score(truth, pred)), "ap": finite(average_precision_score(truth, score)),
        "recall": finite(recall_score(truth, pred)), "per_domain": per_domain,
        "status": "OBSERVATIONAL", "proxy": True,
    }


EVALUATORS = {"ctc": ctc_result, "hest": hest_result, "rxrx3": rxrx3_result, "midogpp": midog_result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--logical-batch-size", type=int,
        help="Protocol batch size when --batch-size is only an inference microbatch (features are sample-independent).",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument(
        "--datasets", default=",".join(DATASETS),
        help=f"Comma-separated subset of: {','.join(DATASETS)}",
    )
    parser.add_argument(
        "--cache-root", type=Path,
        help="Cache directory to reuse; defaults to CAMPAIGN/cache",
    )
    args = parser.parse_args()
    datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        parser.error(f"unknown datasets: {','.join(unknown)}")
    if not datasets:
        parser.error("--datasets must select at least one dataset")
    if (args.checkpoint is None) != (args.train_config is None):
        parser.error("--checkpoint and --train-config must be provided together")
    checkpoint = args.checkpoint
    train_config = args.train_config
    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        train_config = train_config.resolve()
        if not checkpoint.is_file() or not train_config.is_file():
            parser.error("checkpoint/config path does not exist")
    elif args.model == "hs6_hplus":
        checkpoint, train_config = HPLUS_CKPT, HPLUS_CONFIG
    cache_root = (args.cache_root or (args.campaign / "cache")).resolve()
    model_dir = args.campaign / "models" / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    result_path = model_dir / "results.json"
    started = time.time()
    manifest_path = args.campaign / "campaign_manifest.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None
    result = {
        "model": args.model, "status": "RUNNING", "host": platform.node(), "pid": os.getpid(),
        "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_size": args.logical_batch_size or args.batch_size,
        "inference_microbatch": args.batch_size,
        "started_unix": started, "tests": {},
        "checkpoint": str(checkpoint) if checkpoint else None,
        "train_config": str(train_config) if train_config else None,
        "teacher_branch": "teacher" if checkpoint else None,
        "readout": "final_cls_plus_final_patch_mean_l2" if checkpoint else "registry_default_global_l2",
        "resolution": 224, "seed": 0, "campaign_manifest_sha256_at_start": manifest_sha,
        "datasets": list(datasets), "cache_root": str(cache_root),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    try:
        encoder = build(args.model, args.device, args.batch_size, checkpoint, train_config)
        for dataset in datasets:
            folder = cache_root / dataset
            meta = json.loads((folder / "metadata.json").read_text())
            features = extract(encoder, folder / "images.npy", model_dir / f"{dataset}_features.npy", args.batch_size)
            labels = np.load(folder / "labels.npy")
            result["tests"][dataset] = EVALUATORS[dataset](features.astype(np.float32), labels, meta)
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
            print(f"[done] {args.model} {dataset}: {result['tests'][dataset]}", flush=True)
        result["status"] = "VALID_COMPLETE"
    except Exception as exc:
        result["status"] = "FAILED"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["finished_unix"] = time.time()
        result["elapsed_seconds"] = result["finished_unix"] - started
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
