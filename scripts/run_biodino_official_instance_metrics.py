#!/usr/bin/env python3
"""Eval-only official instance metrics for BioDINOv3 TEST heads.

This runner does not train, tune on TEST, or modify model weights. It uses the
existing No-trick and Fixed-trick heads, caches raw HoVerNet outputs, and writes
official-protocol metric artifacts for Cellpose, TissueNet/Mesmer, and
LIVECell.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

import scripts.instseg_optimization_workflow as w
from dinov3.eval.bio_segmentation.datasets.livecell import get_livecell_paths
from dinov3.eval.bio_segmentation.datasets.tissuenet import get_tissuenet_paths
from dinov3.eval.bio_segmentation.instance_seg.eval_full import _load_hover_state
from dinov3.eval.bio_segmentation.instance_seg.model import build_dino_hovernet
from dinov3.eval.bio_segmentation.instance_seg.postproc import postprocess
from dinov3.eval.bio_segmentation.instance_seg.tiling import sliding_window_predict
from dinov3.eval.bio_segmentation.instance_seg.train import DATASET_NUM_TYPES, _build_instance_dataset
from dinov3.eval.bio_segmentation.metrics.instance import _pairwise_iou, compute_aji, compute_pq


CHECKPOINT = ROOT / "outputs/01_training_runs/bio_continue_rgb3_vith16plus/ckpt/14349"
TRAIN_CONFIG = ROOT / "outputs/01_training_runs/bio_continue_rgb3_vith16plus/config.yaml"
HEAD_ROOT = ROOT / "outputs/instance_seg_tuning/bio_continue_rgb3_vith16plus_seven_dataset"
OUTPUT_ROOT = ROOT / "outputs/instance_seg_tuning/bio_continue_rgb3_vith16plus_official_metric_eval"
EXISTING_LIVECELL_FIXED_CACHE = (
    ROOT / "outputs/instance_seg_tuning/bio_continue_rgb3_vith16plus_livecell_cached_perf/cache/livecell"
)
DATA_ROOTS = {
    "cellpose": Path("/mnt/huawei_deepcad/benchmark/segmentation/cellpose/extracted"),
    "tissuenet": Path("/mnt/huawei_deepcad/benchmark/segmentation/tissuenet/extracted"),
    "livecell": Path("/mnt/huawei_deepcad/benchmark/segmentation/LIVECell"),
}
THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]
NO_TRICK_POSTPROC = (0.50, 0.40, 10, 21)
FIXED_POSTPROC = {
    "cellpose": (0.50, 0.45, 10, 21),
    "tissuenet": (0.57, 0.40, 10, 21),
    "livecell": (0.28, 0.42, 10, 21),
}


@dataclass(frozen=True)
class HeadSpec:
    dataset: str
    mode: str
    head: Path
    postproc: Tuple[float, float, int, int]
    head_kind: str


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_path(path: Path) -> str:
    h = __import__("hashlib").sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            h.update(str(child.relative_to(path)).encode())
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
        return h.hexdigest()
    return "missing"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_status(output_root: Path, message: Dict[str, Any]) -> None:
    path = output_root / "official_eval_status.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": now(), **message}, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(message, ensure_ascii=False, sort_keys=True), flush=True)


def head_specs() -> List[HeadSpec]:
    return [
        HeadSpec("cellpose", "no_trick", HEAD_ROOT / "cellpose/no_trick/best_head.pth", NO_TRICK_POSTPROC, "no_trick_head"),
        HeadSpec("cellpose", "fixed_trick", HEAD_ROOT / "cellpose/cellpose_validated/best_head.pth", FIXED_POSTPROC["cellpose"], "existing_fixed_head"),
        HeadSpec("tissuenet", "no_trick", HEAD_ROOT / "tissuenet/no_trick/best_head.pth", NO_TRICK_POSTPROC, "no_trick_head"),
        HeadSpec("tissuenet", "fixed_trick", HEAD_ROOT / "tissuenet/no_trick/best_head.pth", FIXED_POSTPROC["tissuenet"], "no_trick_head_with_validation_selected_postproc"),
        HeadSpec("livecell", "no_trick", HEAD_ROOT / "livecell/no_trick/best_head.pth", NO_TRICK_POSTPROC, "no_trick_head"),
        HeadSpec("livecell", "fixed_trick", HEAD_ROOT / "livecell/livecell_validated/best_head.pth", FIXED_POSTPROC["livecell"], "existing_fixed_head"),
    ]


def make_run_config(spec: HeadSpec) -> w.RunConfig:
    return w.RunConfig(
        dataset=spec.dataset,
        split="test",
        checkpoint=CHECKPOINT,
        train_config=TRAIN_CONFIG,
        head_path=spec.head,
        layers=[7, 15, 23, 31],
        feature_size=32,
        embed_proj=384,
        crop_size=256,
        stride=192,
        blend_mode="uniform",
        fg_thresh=spec.postproc[0],
        energy_thresh=spec.postproc[1],
        min_size=spec.postproc[2],
        sobel_ksize=spec.postproc[3],
        tta=False,
        tta_mode="flip4",
        max_images=None,
        output_tag=spec.mode,
        save_overlays=False,
        top_k=8,
        post_workers=1,
    )


def build_dataset(dataset: str, normalize: bool = True):
    return _build_instance_dataset(
        SimpleNamespace(dataset=dataset, data_root=str(DATA_ROOTS[dataset])),
        "test",
        do_normalize=normalize,
    )


def sample_file(cache_dir: Path, index: int) -> Path:
    return cache_dir / "samples" / f"sample_{index:06d}.npz"


def valid_cache(cache_dir: Path, spec: HeadSpec) -> bool:
    manifest = cache_dir / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text())
    except Exception:
        return False
    return (
        payload.get("dataset") == spec.dataset
        and payload.get("split") == "test"
        and payload.get("head_sha256") == sha256_path(spec.head)
        and payload.get("sample_count", 0) > 0
        and sample_file(cache_dir, 0).is_file()
    )


def write_cache(
    cache_dir: Path,
    spec: HeadSpec,
    collected: Tuple[List[Tuple[Dict[str, np.ndarray], np.ndarray, Optional[np.ndarray]]], str, int, int, float, float],
) -> Dict[str, Any]:
    cached, loaded_kind, tensor_count, patch_size, infer_seconds, peak_mem = collected
    samples = cache_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (out, gt_inst, gt_sem) in enumerate(tqdm(cached, desc=f"{spec.dataset}:{spec.mode}:write-cache", leave=False)):
        payload: Dict[str, Any] = {
            "np_logits": np.asarray(out["np"], dtype=np.float32),
            "hv": np.asarray(out["hv"], dtype=np.float32),
            "gt_inst": np.asarray(gt_inst, dtype=np.int32),
            "sample_id": np.asarray(str(index)),
        }
        if out.get("tp") is not None:
            payload["tp_logits"] = np.asarray(out["tp"], dtype=np.float32)
        if gt_sem is not None:
            payload["gt_sem"] = np.asarray(gt_sem, dtype=np.int32)
        path = sample_file(cache_dir, index)
        np.savez_compressed(path, **payload)
        records.append({"index": index, "sample_id": str(index), "path": str(path.relative_to(cache_dir)), "sha256": sha256_path(path)})
    manifest = {
        "schema": 2,
        "dataset": spec.dataset,
        "split": "test",
        "mode": spec.mode,
        "head": str(spec.head),
        "head_kind": spec.head_kind,
        "head_sha256": sha256_path(spec.head),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_path(CHECKPOINT),
        "train_config": str(TRAIN_CONFIG),
        "loaded_kind": loaded_kind,
        "checkpoint_tensor_count": tensor_count,
        "patch_size": patch_size,
        "sample_count": len(records),
        "samples": records,
        "layers": [7, 15, 23, 31],
        "feature_size": 32,
        "embed_proj": 384,
        "crop_size": 256,
        "stride": 192,
        "blend_mode": "uniform",
        "tta": False,
        "raw_inference_seconds": infer_seconds,
        "peak_cuda_gib": peak_mem,
        "postproc_for_metrics": {
            "fg_thresh": spec.postproc[0],
            "energy_thresh": spec.postproc[1],
            "min_size": spec.postproc[2],
            "sobel_ksize": spec.postproc[3],
            "source": "no_trick_default" if spec.mode == "no_trick" else "validation_selected_fixed_trick",
        },
        "created_at": now(),
        "host": socket.gethostname(),
    }
    write_json(cache_dir / "manifest.json", manifest)
    return manifest


def collect_cache(cache_dir: Path, spec: HeadSpec, output_root: Path) -> Path:
    if valid_cache(cache_dir, spec):
        append_status(output_root, {"stage": "cache-ready", "dataset": spec.dataset, "mode": spec.mode, "cache": str(cache_dir), "source": "existing"})
        return cache_dir
    if not spec.head.is_file():
        raise FileNotFoundError(f"missing head for {spec.dataset}/{spec.mode}: {spec.head}")
    append_status(output_root, {"stage": "cache-start", "dataset": spec.dataset, "mode": spec.mode, "head": str(spec.head)})
    collected = w._collect_continuous_outputs(make_run_config(spec))
    manifest = write_cache(cache_dir, spec, collected)
    append_status(
        output_root,
        {
            "stage": "cache-done",
            "dataset": spec.dataset,
            "mode": spec.mode,
            "samples": manifest["sample_count"],
            "raw_inference_seconds": manifest["raw_inference_seconds"],
            "peak_cuda_gib": manifest["peak_cuda_gib"],
        },
    )
    return cache_dir


def maybe_existing_livecell_fixed_cache(spec: HeadSpec, output_root: Path) -> Optional[Path]:
    if spec.dataset != "livecell" or spec.mode != "fixed_trick":
        return None
    manifest = EXISTING_LIVECELL_FIXED_CACHE / "manifest.json"
    if not manifest.is_file():
        return None
    payload = json.loads(manifest.read_text())
    if payload.get("dataset") != "livecell" or payload.get("sample_count") != 1564:
        return None
    if payload.get("head_sha256") != sha256_path(spec.head):
        return None
    pointer = output_root / "raw_cache/livecell/fixed_trick/manifest_pointer.json"
    write_json(
        pointer,
        {
            "schema": 2,
            "dataset": "livecell",
            "mode": "fixed_trick",
            "cache": str(EXISTING_LIVECELL_FIXED_CACHE),
            "cache_manifest": str(manifest),
            "cache_manifest_sha256": sha256_path(manifest),
            "head": str(spec.head),
            "head_sha256": sha256_path(spec.head),
            "note": "Fixed-trick uses the existing raw cache as requested; sample files are not duplicated.",
            "created_at": now(),
        },
    )
    append_status(output_root, {"stage": "cache-ready", "dataset": "livecell", "mode": "fixed_trick", "cache": str(EXISTING_LIVECELL_FIXED_CACHE), "source": "existing-livecell-fixed"})
    return EXISTING_LIVECELL_FIXED_CACHE


def load_cache_manifest(cache_dir: Path) -> Dict[str, Any]:
    return json.loads((cache_dir / "manifest.json").read_text())


def load_sample(cache_dir: Path, record: Dict[str, Any]):
    with np.load(cache_dir / record["path"], allow_pickle=False) as z:
        out = {"np": z["np_logits"], "hv": z["hv"], "tp": z["tp_logits"] if "tp_logits" in z else None}
        gt_inst = z["gt_inst"].astype(np.int32, copy=False)
        gt_sem = z["gt_sem"].astype(np.int32, copy=False) if "gt_sem" in z else None
    return out, gt_inst, gt_sem


def softmax_foreground(np_logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(np_logits, dtype=np.float32)
    z = logits - logits.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e[1] / (e.sum(axis=0) + 1e-8)


def instance_ids(inst: np.ndarray) -> np.ndarray:
    return np.unique(inst[inst > 0])


def match_counts(pred: np.ndarray, gt: np.ndarray, threshold: float) -> Tuple[int, int, int]:
    pred_ids = instance_ids(pred)
    gt_ids = instance_ids(gt)
    n_pred, n_gt = len(pred_ids), len(gt_ids)
    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0
    iou = _pairwise_iou(pred, gt, pred_ids, gt_ids)
    matched_gt = set()
    matched_pred = set()
    for flat_idx in np.argsort(-iou, axis=None):
        gi, pi = divmod(int(flat_idx), n_pred)
        if float(iou[gi, pi]) < threshold:
            break
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
    tp = len(matched_gt)
    return tp, n_pred - tp, n_gt - tp


def prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)


def cellpose_vector_for_cache(cache_dir: Path, spec: HeadSpec, output_root: Path) -> Dict[str, Any]:
    manifest = load_cache_manifest(cache_dir)
    per_image: List[Dict[str, Any]] = []
    vectors = {f"{t:.2f}": [] for t in THRESHOLDS}
    totals = {f"{t:.2f}": {"tp": 0, "fp": 0, "fn": 0} for t in THRESHOLDS}
    fg, energy, min_size, sobel = spec.postproc
    aji_values, bpq_values = [], []
    for rec in tqdm(manifest["samples"], desc=f"cellpose:{spec.mode}:official-ap", leave=False):
        out, gt, _ = load_sample(cache_dir, rec)
        pred, _ = postprocess(out["np"], out["hv"], out["tp"], fg_thresh=fg, energy_thresh=energy, min_size=min_size, sobel_ksize=sobel)
        row: Dict[str, Any] = {"index": int(rec["index"]), "gt_objects": int(len(instance_ids(gt))), "pred_objects": int(len(instance_ids(pred)))}
        aji = compute_aji(pred, gt)
        bpq = compute_pq(pred, gt)["pq"]
        row["AJI"] = float(aji)
        row["bPQ"] = float(bpq)
        aji_values.append(aji)
        bpq_values.append(bpq)
        for t in THRESHOLDS:
            key = f"{t:.2f}"
            tp, fp, fn = match_counts(pred, gt, t)
            ap = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
            row[f"AP{int(t * 100):02d}"] = float(ap)
            row[f"TP{int(t * 100):02d}"] = tp
            row[f"FP{int(t * 100):02d}"] = fp
            row[f"FN{int(t * 100):02d}"] = fn
            vectors[key].append(float(ap))
            totals[key]["tp"] += tp
            totals[key]["fp"] += fp
            totals[key]["fn"] += fn
        row["AP_mean_50_95"] = float(np.mean([row[f"AP{int(t * 100):02d}"] for t in THRESHOLDS]))
        per_image.append(row)
    threshold_vector = {key: float(np.mean(values)) for key, values in vectors.items()}
    payload = {
        "name": "Cellpose official average_precision / object AP",
        "definition": "TP/(TP+FP+FN) after one-to-one instance matching at each IoU threshold; mean is averaged over thresholds 0.50:0.05:0.95. This is Cellpose-style object AP, not COCO AP.",
        "mode": spec.mode,
        "postproc": {"fg_thresh": fg, "energy_thresh": energy, "min_size": min_size, "sobel_ksize": sobel},
        "thresholds": THRESHOLDS,
        "threshold_vector": threshold_vector,
        "threshold_counts_global": totals,
        "AP@[0.50:0.95]": float(np.mean(list(threshold_vector.values()))),
        "AP50": threshold_vector["0.50"],
        "AP75": threshold_vector["0.75"],
        "AP90": threshold_vector["0.90"],
        "AP95": threshold_vector["0.95"],
        "AJI": float(np.mean(aji_values)),
        "bPQ": float(np.mean(bpq_values)),
        "per_image_path": str(output_root / f"cellpose_{spec.mode}_per_image_ap.csv"),
    }
    write_csv(output_root / f"cellpose_{spec.mode}_per_image_ap.csv", per_image)
    write_json(output_root / f"cellpose_{spec.mode}_official_ap_vector.json", payload)
    return payload


def load_tissuenet_npz() -> Tuple[Path, Any]:
    path = Path(get_tissuenet_paths(str(DATA_ROOTS["tissuenet"]), split="test"))
    return path, np.load(path, mmap_mode="r")


def tissuenet_channel_gt(data: Any, index: int, channel: int) -> np.ndarray:
    return data["y"][index, :, :, channel].astype(np.int32)


def tissuenet_diagnostics(output_root: Path, sample_count: int = 16) -> Dict[str, Any]:
    npz_path, data = load_tissuenet_npz()
    n = int(data["y"].shape[0])
    rng = np.random.default_rng(0)
    indices = sorted(int(i) for i in rng.choice(n, size=min(sample_count, n), replace=False))
    rows = []
    for i in indices:
        rows.append(
            {
                "index": i,
                "channel0_label": "nuclear",
                "channel0_instances": int(len(np.unique(data["y"][i, :, :, 0][data["y"][i, :, :, 0] > 0]))),
                "channel1_label": "whole-cell",
                "channel1_instances": int(len(np.unique(data["y"][i, :, :, 1][data["y"][i, :, :, 1] > 0]))),
            }
        )
    save_tissuenet_visual(output_root / "tissuenet_target_channel_diagnostic.png", data, indices[:8])
    return {
        "npz": str(npz_path),
        "metadata_evidence": {
            "loader_doc": "dinov3/eval/bio_segmentation/datasets/tissuenet.py documents y[...,0]=nuclear instance map and y[...,1]=whole-cell instance map.",
            "loader_default": "TissueNetDataset(target='nuclear') maps target index 0.",
            "training_call_path": "instance_seg.train._build_instance_dataset -> feature_extractor._build_dataset -> TissueNetDataset(...) without overriding target, so current head was trained/evaluated on nuclear y[...,0].",
            "screening_summary": "scripts/consolidate_vitl16_screening_results.py labels tissuenet as nuclear target.",
        },
        "chosen_target": "nuclear",
        "chosen_target_index": 0,
        "diagnostic_samples": rows,
        "visualization": str(output_root / "tissuenet_target_channel_diagnostic.png"),
    }


def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / max(float(hi - lo), 1e-6), 0, 1)


def boundary(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return grad > 0


def save_tissuenet_visual(path: Path, data: Any, indices: Sequence[int]) -> None:
    if not indices:
        return
    tiles = []
    for idx in indices:
        x = data["X"][idx]
        base = np.stack([norm01(x[:, :, 0]), norm01(x[:, :, 1]), norm01(x[:, :, 0])], axis=-1)
        rgb = (base * 255).astype(np.uint8)
        rgb[boundary(data["y"][idx, :, :, 0])] = np.array([255, 32, 32], dtype=np.uint8)
        rgb[boundary(data["y"][idx, :, :, 1])] = np.array([32, 220, 32], dtype=np.uint8)
        im = Image.fromarray(rgb).resize((192, 192))
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, 191, 18], fill=(0, 0, 0))
        draw.text((4, 4), f"idx {idx} red=nuc green=cell", fill=(255, 255, 255))
        tiles.append(im)
    wcols = 4
    rows = math.ceil(len(tiles) / wcols)
    canvas = Image.new("RGB", (wcols * 192, rows * 192), (0, 0, 0))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % wcols) * 192, (i // wcols) * 192))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def tissuenet_f1_for_cache(cache_dir: Path, spec: HeadSpec, output_root: Path) -> Dict[str, Any]:
    manifest = load_cache_manifest(cache_dir)
    npz_path, data = load_tissuenet_npz()
    fg, energy, min_size, sobel = spec.postproc
    per_channel_rows: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
    aji_values, bpq_values = [], []
    global_counts = {0: {"tp": 0, "fp": 0, "fn": 0}, 1: {"tp": 0, "fp": 0, "fn": 0}}
    for rec in tqdm(manifest["samples"], desc=f"tissuenet:{spec.mode}:official-f1", leave=False):
        index = int(rec["index"])
        out, gt0_from_cache, _ = load_sample(cache_dir, rec)
        pred, _ = postprocess(out["np"], out["hv"], out["tp"], fg_thresh=fg, energy_thresh=energy, min_size=min_size, sobel_ksize=sobel)
        aji_values.append(compute_aji(pred, gt0_from_cache))
        bpq_values.append(compute_pq(pred, gt0_from_cache)["pq"])
        for channel in (0, 1):
            gt = tissuenet_channel_gt(data, index, channel)
            tp, fp, fn = match_counts(pred, gt, 0.5)
            precision, recall, f1 = prf_from_counts(tp, fp, fn)
            row = {
                "index": index,
                "channel": channel,
                "target": "nuclear" if channel == 0 else "whole-cell",
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "gt_objects": int(len(instance_ids(gt))),
                "pred_objects": int(len(instance_ids(pred))),
            }
            per_channel_rows[channel].append(row)
            global_counts[channel]["tp"] += tp
            global_counts[channel]["fp"] += fp
            global_counts[channel]["fn"] += fn
    diagnostics = tissuenet_diagnostics(output_root)
    channel_summaries: Dict[str, Any] = {}
    for channel, rows in per_channel_rows.items():
        label = "nuclear" if channel == 0 else "whole-cell"
        arr_p = np.asarray([r["precision"] for r in rows], dtype=np.float64)
        arr_r = np.asarray([r["recall"] for r in rows], dtype=np.float64)
        arr_f = np.asarray([r["f1"] for r in rows], dtype=np.float64)
        tp, fp, fn = (global_counts[channel][k] for k in ("tp", "fp", "fn"))
        gp, gr, gf = prf_from_counts(tp, fp, fn)
        channel_summaries[label] = {
            "channel": channel,
            "per_image_precision_mean": float(arr_p.mean()),
            "per_image_precision_std": float(arr_p.std(ddof=1)) if len(arr_p) > 1 else 0.0,
            "per_image_recall_mean": float(arr_r.mean()),
            "per_image_recall_std": float(arr_r.std(ddof=1)) if len(arr_r) > 1 else 0.0,
            "per_image_f1_mean": float(arr_f.mean()),
            "per_image_f1_std": float(arr_f.std(ddof=1)) if len(arr_f) > 1 else 0.0,
            "global_tp": int(tp),
            "global_fp": int(fp),
            "global_fn": int(fn),
            "global_precision": gp,
            "global_recall": gr,
            "global_f1": gf,
        }
        write_csv(output_root / f"tissuenet_{spec.mode}_{label}_per_image_f1.csv", rows)
    chosen = "nuclear"
    payload = {
        "name": "TissueNet/Mesmer official Precision / Recall / F1",
        "definition": "One-to-one instance matching at IoU >= 0.5. Per-image P/R/F1 mean and std are reported; global F1 is computed from summed TP/FP/FN.",
        "mode": spec.mode,
        "postproc": {"fg_thresh": fg, "energy_thresh": energy, "min_size": min_size, "sobel_ksize": sobel},
        "diagnostics": diagnostics,
        "target": chosen,
        "target_index": 0,
        "target_selection": "Current training path uses TissueNetDataset default target='nuclear' -> y[...,0]. Channel 1 whole-cell is retained as diagnostic only.",
        "paper_main_table_aggregation": "per-image F1 mean is used as the primary TissueNet/Mesmer row; global F1 is reported as a secondary diagnostic.",
        "reported": channel_summaries[chosen],
        "diagnostic_by_target": channel_summaries,
        "AJI": float(np.mean(aji_values)),
        "bPQ": float(np.mean(bpq_values)),
        "per_image_path": str(output_root / f"tissuenet_{spec.mode}_{chosen}_per_image_f1.csv"),
    }
    write_json(output_root / f"tissuenet_{spec.mode}_official_f1.json", payload)
    return payload


def require_pycocotools(site_dir: Optional[Path]) -> None:
    if site_dir is not None:
        sys.path.insert(0, str(site_dir))
    try:
        import pycocotools  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "pycocotools is required for LIVECell official COCO mask AP. "
            "Install it into an isolated --pycocotools-site and set PYTHONPATH."
        ) from exc


def load_livecell_coco() -> Tuple[Path, Path, Dict[str, Any]]:
    coco_json, img_root = get_livecell_paths(str(DATA_ROOTS["livecell"]), split="test")
    path = Path(coco_json)
    with path.open() as handle:
        data = json.load(handle)
    return path, Path(img_root), data


def coco_eval_metrics(coco_json: Path, pred_json: Path, output_prefix: Path, site_dir: Optional[Path]) -> Dict[str, Any]:
    if site_dir is not None:
        sys.path.insert(0, str(site_dir))
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(coco_json))
    coco_dt = coco_gt.loadRes(str(pred_json)) if pred_json.stat().st_size > 2 else coco_gt.loadRes([])
    ev = COCOeval(coco_gt, coco_dt, iouType="segm")
    ev.params.iouThrs = np.round(np.arange(0.50, 0.96, 0.05), 2)
    ev.params.maxDets = [1, 10, 2000]
    ev.params.areaRng = [[0, 1e10], [0, 324], [324, 961], [961, 1e10]]
    ev.params.areaRngLbl = ["all", "small", "medium", "large"]
    ev.params.imgIds = sorted(coco_gt.getImgIds())
    ev.params.catIds = sorted(coco_gt.getCatIds())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    raw_text = buf.getvalue()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    (output_prefix.with_suffix(".txt")).write_text(raw_text, encoding="utf-8")

    precision = ev.eval["precision"]  # [T, R, K, A, M]
    recall = ev.eval["recall"]        # [T, K, A, M]
    iou_thrs = [float(x) for x in ev.params.iouThrs]
    area_labels = list(ev.params.areaRngLbl)
    maxdet_index = len(ev.params.maxDets) - 1

    def mean_precision(t_index: Optional[int], area_label: str) -> float:
        a = area_labels.index(area_label)
        vals = precision[:, :, :, a, maxdet_index] if t_index is None else precision[t_index : t_index + 1, :, :, a, maxdet_index]
        vals = vals[vals > -1]
        return float(vals.mean()) if vals.size else float("nan")

    def mean_recall(t_index: Optional[int], area_label: str = "all") -> float:
        a = area_labels.index(area_label)
        vals = recall[:, :, a, maxdet_index] if t_index is None else recall[t_index : t_index + 1, :, a, maxdet_index]
        vals = vals[vals > -1]
        return float(vals.mean()) if vals.size else float("nan")

    recall_vector = {f"{thr:.2f}": mean_recall(i, "all") for i, thr in enumerate(iou_thrs)}
    fnr_vector = {k: 1.0 - v if math.isfinite(v) else float("nan") for k, v in recall_vector.items()}
    metrics = {
        "mask_AP@[0.50:0.95]": mean_precision(None, "all"),
        "AP50": mean_precision(0, "all"),
        "AP75": mean_precision(5, "all"),
        "AP-small": mean_precision(None, "small"),
        "AP-medium": mean_precision(None, "medium"),
        "AP-large": mean_precision(None, "large"),
        "AR@2000": mean_recall(None, "all"),
        "recall_by_iou": recall_vector,
        "fnr_by_iou": fnr_vector,
        "AFNR": float(np.mean([v for v in fnr_vector.values() if math.isfinite(v)])),
        "cocoeval_stats": [float(x) for x in ev.stats],
        "raw_output_txt": str(output_prefix.with_suffix(".txt")),
    }
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        precision=precision,
        recall=recall,
        scores=ev.eval["scores"],
        iouThrs=np.asarray(ev.params.iouThrs),
        recThrs=np.asarray(ev.params.recThrs),
    )
    metrics["raw_eval_npz"] = str(output_prefix.with_suffix(".npz"))
    return metrics


def livecell_predictions_for_cache(cache_dir: Path, spec: HeadSpec, output_root: Path, site_dir: Optional[Path]) -> Dict[str, Any]:
    require_pycocotools(site_dir)
    from pycocotools import mask as mask_utils

    manifest = load_cache_manifest(cache_dir)
    coco_json, _img_root, coco_data = load_livecell_coco()
    images = sorted(coco_data["images"], key=lambda x: x["id"])
    category_id = int(coco_data.get("categories", [{"id": 1}])[0]["id"])
    if len(images) != manifest["sample_count"]:
        raise RuntimeError(f"LIVECell image/cache count mismatch: coco={len(images)} cache={manifest['sample_count']}")
    fg, energy, min_size, sobel = spec.postproc
    preds: List[Dict[str, Any]] = []
    aji_values, bpq_values = [], []
    per_image = []
    for rec in tqdm(manifest["samples"], desc=f"livecell:{spec.mode}:coco-preds", leave=False):
        idx = int(rec["index"])
        image_info = images[idx]
        out, gt, _ = load_sample(cache_dir, rec)
        pred, _ = postprocess(out["np"], out["hv"], out["tp"], fg_thresh=fg, energy_thresh=energy, min_size=min_size, sobel_ksize=sobel)
        fg_prob = softmax_foreground(out["np"])
        aji = compute_aji(pred, gt)
        bpq = compute_pq(pred, gt)["pq"]
        aji_values.append(aji)
        bpq_values.append(bpq)
        pred_count = 0
        for pid in instance_ids(pred):
            m = np.asfortranarray((pred == pid).astype(np.uint8))
            if int(m.sum()) == 0:
                continue
            rle = mask_utils.encode(m)
            rle["counts"] = rle["counts"].decode("ascii")
            score = float(np.mean(fg_prob[pred == pid]))
            bbox = [float(x) for x in mask_utils.toBbox(rle)]
            area = float(mask_utils.area(rle))
            preds.append(
                {
                    "image_id": int(image_info["id"]),
                    "category_id": category_id,
                    "segmentation": rle,
                    "score": score,
                    "bbox": bbox,
                    "area": area,
                }
            )
            pred_count += 1
        per_image.append({"index": idx, "image_id": int(image_info["id"]), "gt_objects": int(len(instance_ids(gt))), "pred_objects": pred_count, "AJI": float(aji), "bPQ": float(bpq)})
    pred_json = output_root / f"livecell_{spec.mode}_coco_predictions.json"
    write_json(pred_json, preds)
    write_csv(output_root / f"livecell_{spec.mode}_per_image_internal.csv", per_image)
    metrics = coco_eval_metrics(coco_json, pred_json, output_root / f"livecell_{spec.mode}_cocoeval_raw", site_dir)
    payload = {
        "name": "LIVECell official COCO mask AP + AFNR",
        "mode": spec.mode,
        "definition": "COCO mask evaluation with iouType='segm', IoU thresholds 0.50:0.05:0.95, maxDets=2000, and LIVECell area bins all/0-324/324-961/961+.",
        "confidence_score": "score = mean NP foreground softmax probability inside each predicted instance region.",
        "postproc": {"fg_thresh": fg, "energy_thresh": energy, "min_size": min_size, "sobel_ksize": sobel},
        "annotation_json": str(coco_json),
        "image_count": len(images),
        "prediction_json": str(pred_json),
        "prediction_count": len(preds),
        "official_scores": metrics,
        "AJI": float(np.mean(aji_values)),
        "bPQ": float(np.mean(bpq_values)),
    }
    write_json(output_root / f"livecell_{spec.mode}_official_coco_metrics.json", payload)
    return payload


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(output_root: Path, cellpose: Dict[str, Dict[str, Any]], tissuenet: Dict[str, Dict[str, Any]], livecell: Dict[str, Dict[str, Any]]) -> None:
    rows = []
    rows.append(
        {
            "Dataset": "Cellpose",
            "Target": "cell instances",
            "Official metric": "Cellpose average_precision mean AP@[0.50:0.95]",
            "No-trick": cellpose["no_trick"]["AP@[0.50:0.95]"],
            "Fixed-trick": cellpose["fixed_trick"]["AP@[0.50:0.95]"],
            "提升": cellpose["fixed_trick"]["AP@[0.50:0.95]"] - cellpose["no_trick"]["AP@[0.50:0.95]"],
            "AJI": cellpose["fixed_trick"]["AJI"],
            "bPQ": cellpose["fixed_trick"]["bPQ"],
            "协议说明": "Cellpose-style object AP; one-to-one matching; not COCO AP. Full threshold vector saved.",
        }
    )
    rows.append(
        {
            "Dataset": "TissueNet",
            "Target": "nuclear y[...,0]",
            "Official metric": "TissueNet/Mesmer per-image F1 mean @ IoU>=0.5",
            "No-trick": tissuenet["no_trick"]["reported"]["per_image_f1_mean"],
            "Fixed-trick": tissuenet["fixed_trick"]["reported"]["per_image_f1_mean"],
            "提升": tissuenet["fixed_trick"]["reported"]["per_image_f1_mean"] - tissuenet["no_trick"]["reported"]["per_image_f1_mean"],
            "AJI": tissuenet["fixed_trick"]["AJI"],
            "bPQ": tissuenet["fixed_trick"]["bPQ"],
            "协议说明": "IoU>=0.5 one-to-one matching; per-image mean/std and global TP/FP/FN F1 saved; whole-cell channel diagnostic saved.",
        }
    )
    rows.append(
        {
            "Dataset": "LIVECell",
            "Target": "single-category cell masks",
            "Official metric": "COCO mask AP@[0.50:0.95]",
            "No-trick": livecell["no_trick"]["official_scores"]["mask_AP@[0.50:0.95]"],
            "Fixed-trick": livecell["fixed_trick"]["official_scores"]["mask_AP@[0.50:0.95]"],
            "提升": livecell["fixed_trick"]["official_scores"]["mask_AP@[0.50:0.95]"] - livecell["no_trick"]["official_scores"]["mask_AP@[0.50:0.95]"],
            "AJI": livecell["fixed_trick"]["AJI"],
            "bPQ": livecell["fixed_trick"]["bPQ"],
            "协议说明": "pycocotools COCOeval segm; maxDets=2000; LIVECell area bins; confidence is mean NP foreground softmax inside instance; AFNR saved separately.",
        }
    )
    write_csv(output_root / "official_metrics_summary.csv", rows)
    lines = [
        "# BioDINOv3 Official Metric Eval",
        "",
        f"- Created: {now()}",
        f"- Host: {socket.gethostname()}",
        f"- Checkpoint: `{CHECKPOINT}`",
        f"- Output root: `{output_root}`",
        "- Eval-only: yes",
        "- TEST-tuned parameters: no",
        "",
        "| Dataset | Target | Official metric | No-trick | Fixed-trick | 提升 | AJI | bPQ | 协议说明 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Dataset']} | {row['Target']} | {row['Official metric']} | "
            f"{float(row['No-trick']):.9f} | {float(row['Fixed-trick']):.9f} | {float(row['提升']):+.9f} | "
            f"{float(row['AJI']):.9f} | {float(row['bPQ']):.9f} | {row['协议说明']} |"
        )
    lines += [
        "",
        "## Additional Official Outputs",
        "",
        f"- Cellpose vectors: `{output_root / 'cellpose_no_trick_official_ap_vector.json'}`, `{output_root / 'cellpose_fixed_trick_official_ap_vector.json'}`",
        f"- TissueNet per-image F1 CSVs: `{output_root / 'tissuenet_no_trick_nuclear_per_image_f1.csv'}`, `{output_root / 'tissuenet_fixed_trick_nuclear_per_image_f1.csv'}`",
        f"- LIVECell COCO prediction JSONs: `{output_root / 'livecell_no_trick_coco_predictions.json'}`, `{output_root / 'livecell_fixed_trick_coco_predictions.json'}`",
        f"- LIVECell COCOeval raw outputs: `{output_root / 'livecell_no_trick_cocoeval_raw.txt'}`, `{output_root / 'livecell_fixed_trick_cocoeval_raw.txt'}`",
    ]
    (output_root / "official_metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_json(
        output_root / "cellpose_official_metrics.json",
        {"dataset": "Cellpose", "created_at": now(), "checkpoint": str(CHECKPOINT), "official_metric": cellpose},
    )
    write_json(
        output_root / "tissuenet_official_metrics.json",
        {"dataset": "TissueNet", "created_at": now(), "checkpoint": str(CHECKPOINT), "official_metric": tissuenet},
    )
    write_json(
        output_root / "livecell_official_metrics.json",
        {"dataset": "LIVECell", "created_at": now(), "checkpoint": str(CHECKPOINT), "official_metric": livecell},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=["cellpose", "tissuenet", "livecell"], choices=["cellpose", "tissuenet", "livecell"])
    parser.add_argument("--pycocotools-site", type=Path, default=None)
    parser.add_argument("--reuse-livecell-fixed-cache", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    append_status(output_root, {"stage": "start", "pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"), "datasets": args.datasets})
    specs = [s for s in head_specs() if s.dataset in set(args.datasets)]
    caches: Dict[Tuple[str, str], Path] = {}
    for spec in specs:
        existing = maybe_existing_livecell_fixed_cache(spec, output_root) if args.reuse_livecell_fixed_cache else None
        caches[(spec.dataset, spec.mode)] = existing or collect_cache(output_root / "raw_cache" / spec.dataset / spec.mode, spec, output_root)

    cellpose: Dict[str, Dict[str, Any]] = {}
    tissuenet: Dict[str, Dict[str, Any]] = {}
    livecell: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        cache_dir = caches[(spec.dataset, spec.mode)]
        if spec.dataset == "cellpose":
            cellpose[spec.mode] = cellpose_vector_for_cache(cache_dir, spec, output_root)
        elif spec.dataset == "tissuenet":
            tissuenet[spec.mode] = tissuenet_f1_for_cache(cache_dir, spec, output_root)
        elif spec.dataset == "livecell":
            livecell[spec.mode] = livecell_predictions_for_cache(cache_dir, spec, output_root, args.pycocotools_site)

    if all(k in cellpose for k in ("no_trick", "fixed_trick")) and all(k in tissuenet for k in ("no_trick", "fixed_trick")) and all(k in livecell for k in ("no_trick", "fixed_trick")):
        summarize(output_root, cellpose, tissuenet, livecell)
    append_status(output_root, {"stage": "done", "pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
