#!/usr/bin/env python
"""Audited instance-seg evaluation and eval-only sweeps for the DINOHoVerNet runs.

This script intentionally wraps the existing model, dataset, tiling, postprocess,
and metric code instead of redefining the training/evaluation stack.  It adds the
bookkeeping needed for optimization work: full-val CSV rows, per-image error
analysis, overlays, experiment logs, and method-gain rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from dinov3.eval.bio_segmentation.instance_seg.eval_full import _load_hover_state
from dinov3.eval.bio_segmentation.instance_seg.model import build_dino_hovernet
from dinov3.eval.bio_segmentation.instance_seg.postproc import postprocess
from dinov3.eval.bio_segmentation.instance_seg.tiling import sliding_window_predict
from dinov3.eval.bio_segmentation.instance_seg.train import DATASET_NUM_TYPES, _build_instance_dataset
from dinov3.eval.bio_segmentation.metrics import accumulate_instance_metrics
from dinov3.eval.bio_segmentation.metrics.instance import (
    _pairwise_iou,
    compute_aji,
    compute_ap,
    compute_multi_class_pq,
    compute_object_ap,
    compute_pq,
    compute_seg,
)


OUT_ROOT = ROOT / "outputs" / "instance_seg_tuning"
RUN_ROOT = OUT_ROOT / "5tb_idweak10_last4_7ds"
PRETRAIN_ROOT = ROOT / "outputs" / "01_training_runs" / "5tb_idweak10_vitl16_robust_b1024_8gpu" / "ckpt"
DEFAULT_CHECKPOINT = PRETRAIN_ROOT / "15374" / "checkpoint.pth"
DEFAULT_TRAIN_CONFIG = ROOT / "dinov3" / "configs" / "train" / "microscopy_continual_vitl16_robust_5tb_idweak10.yaml"
SEG_ROOT = Path("/mnt/huawei_deepcad/benchmark/segmentation")

DATASET_ORDER = ["pannuke", "conic", "monuseg", "livecell", "bbbc038", "tissuenet", "cellpose"]
DATA_ROOTS = {
    "pannuke": SEG_ROOT / "pannuke" / "extracted",
    "conic": SEG_ROOT / "conic" / "extracted",
    "monuseg": SEG_ROOT / "monuseg" / "extracted",
    "livecell": SEG_ROOT / "LIVECell",
    "bbbc038": SEG_ROOT / "bbbc038" / "extracted",
    "tissuenet": SEG_ROOT / "tissuenet" / "extracted",
    "cellpose": SEG_ROOT / "cellpose" / "extracted",
}
PRIMARY_METRIC = {
    "pannuke": "mPQ",
    "conic": "mPQ",
    "monuseg": "AJI",
    "livecell": "SEG",
    "bbbc038": "CellposeStyleAP",
    "tissuenet": "CellposeStyleAP",
    "cellpose": "CellposeStyleAP",
}
METRIC_COLUMNS = [
    "AJI",
    "Dice",
    "AP",
    "AP50",
    "AP75",
    "COCOProxyAP",
    "COCOProxyAP50",
    "COCOProxyAP75",
    "CellposeStyleAP",
    "CellposeStyleAP50",
    "CellposeStyleAP75",
    "SEG",
    "bPQ",
    "bSQ",
    "bDQ",
    "mPQ",
    "mSQ",
    "mDQ",
]


@dataclass
class RunConfig:
    dataset: str
    split: str
    checkpoint: Path
    train_config: Path
    head_path: Path
    layers: List[int]
    feature_size: int
    embed_proj: int
    crop_size: int
    stride: int
    blend_mode: str
    fg_thresh: float
    energy_thresh: float
    sobel_ksize: int
    min_size: int
    tta: bool
    tta_mode: str
    max_images: Optional[int]
    output_tag: str
    save_overlays: bool
    top_k: int
    post_workers: int


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _append_csv(path: Path, row: Dict[str, object], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            missing = [name for name in fieldnames if name not in old_fields]
            fieldnames = old_fields + missing
            if missing:
                rows = list(reader)
                with path.open("w", newline="") as out:
                    writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                write_header = False
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _head_path(dataset: str) -> Path:
    return RUN_ROOT / dataset / "15374" / "last4" / "best_head.pth"


def _build_model(cfg: RunConfig, device: torch.device):
    num_types = DATASET_NUM_TYPES.get(cfg.dataset, 0)
    model = build_dino_hovernet(
        checkpoint=str(cfg.checkpoint),
        train_config=str(cfg.train_config),
        layers=cfg.layers,
        num_types=num_types,
        freeze_backbone=True,
        trainable_backbone_blocks=None,
        feature_size=cfg.feature_size,
        embed_proj=cfg.embed_proj,
        device=device,
    )
    loaded_kind, tensor_count = _load_hover_state(model, str(cfg.head_path), device, "auto")
    return model, loaded_kind, tensor_count


def _build_ds(dataset: str, split: str, normalize: bool):
    return _build_instance_dataset(
        SimpleNamespace(dataset=dataset, data_root=str(DATA_ROOTS[dataset])),
        split,
        do_normalize=normalize,
    )


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _instance_ids(inst: np.ndarray) -> np.ndarray:
    return np.unique(inst[inst > 0])


def _per_image_metrics(
    pred_inst: np.ndarray,
    gt_inst: np.ndarray,
    pred_sem: Optional[np.ndarray],
    gt_sem: Optional[np.ndarray],
    num_types: int,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["AJI"] = compute_aji(pred_inst, gt_inst)
    pred_fg = pred_inst > 0
    gt_fg = gt_inst > 0
    out["Dice"] = float(2.0 * np.logical_and(pred_fg, gt_fg).sum() / (pred_fg.sum() + gt_fg.sum() + 1e-8))
    ap = compute_ap(pred_inst, gt_inst)
    out.update({"COCOProxyAP": ap["AP"], "COCOProxyAP50": ap["AP50"], "COCOProxyAP75": ap["AP75"]})
    out.update(compute_object_ap(pred_inst, gt_inst))
    out["SEG"] = compute_seg(pred_inst, gt_inst)
    if num_types > 0 and pred_sem is not None and gt_sem is not None:
        pq = compute_multi_class_pq(pred_inst, pred_sem, gt_inst, gt_sem, num_classes=num_types)
        out.update(pq)
        sq_values = [v for k, v in pq.items() if k.startswith("sq_class_")]
        dq_values = [v for k, v in pq.items() if k.startswith("rq_class_")]
        out["mSQ"] = float(np.nanmean(sq_values)) if sq_values else float("nan")
        out["mDQ"] = float(np.nanmean(dq_values)) if dq_values else float("nan")
        out["bDQ"] = pq.get("bRQ", float("nan"))
    else:
        pq = compute_pq(pred_inst, gt_inst)
        out["bPQ"] = pq["pq"]
        out["bSQ"] = pq["sq"]
        out["bDQ"] = pq["rq"]
        out["mPQ"] = float("nan")
        out["mSQ"] = float("nan")
        out["mDQ"] = float("nan")
    return out


def _error_counts(pred_inst: np.ndarray, gt_inst: np.ndarray) -> Dict[str, int]:
    pred_ids = _instance_ids(pred_inst)
    gt_ids = _instance_ids(gt_inst)
    if len(pred_ids) == 0 and len(gt_ids) == 0:
        return {"merge": 0, "split": 0, "miss": 0, "false_positive": 0, "boundary": 0}
    if len(pred_ids) == 0:
        return {"merge": 0, "split": 0, "miss": int(len(gt_ids)), "false_positive": 0, "boundary": 0}
    if len(gt_ids) == 0:
        return {"merge": 0, "split": 0, "miss": 0, "false_positive": int(len(pred_ids)), "boundary": 0}

    iou = _pairwise_iou(pred_inst, gt_inst, pred_ids, gt_ids)
    matched_gt = set()
    matched_pred = set()
    boundary = 0
    for flat_idx in np.argsort(-iou, axis=None):
        gi, pi = divmod(int(flat_idx), len(pred_ids))
        if iou[gi, pi] < 0.5:
            break
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        if iou[gi, pi] < 0.75:
            boundary += 1

    overlap = iou > 0.10
    split = int(np.sum(overlap.sum(axis=1) > 1))
    merge = int(np.sum(overlap.sum(axis=0) > 1))
    return {
        "merge": merge,
        "split": split,
        "miss": int(len(gt_ids) - len(matched_gt)),
        "false_positive": int(len(pred_ids) - len(matched_pred)),
        "boundary": int(boundary),
    }


def _size_recall(pred_inst: np.ndarray, gt_inst: np.ndarray) -> Dict[str, Tuple[int, int]]:
    pred_ids = _instance_ids(pred_inst)
    gt_ids = _instance_ids(gt_inst)
    bins = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    if len(gt_ids) == 0:
        return {k: (v[0], v[1]) for k, v in bins.items()}
    if len(pred_ids) == 0:
        for gid in gt_ids:
            area = int((gt_inst == gid).sum())
            label = "small" if area < 100 else "medium" if area < 500 else "large"
            bins[label][1] += 1
        return {k: (v[0], v[1]) for k, v in bins.items()}
    iou = _pairwise_iou(pred_inst, gt_inst, pred_ids, gt_ids)
    for gi, gid in enumerate(gt_ids):
        area = int((gt_inst == gid).sum())
        label = "small" if area < 100 else "medium" if area < 500 else "large"
        bins[label][1] += 1
        if float(iou[gi].max(initial=0.0)) >= 0.5:
            bins[label][0] += 1
    return {k: (v[0], v[1]) for k, v in bins.items()}


def _to_rgb_uint8(img_t: torch.Tensor) -> np.ndarray:
    img = img_t.detach().cpu().float().permute(1, 2, 0).numpy()
    img = np.nan_to_num(img)
    if img.min() < 0 or img.max() > 1:
        lo, hi = np.percentile(img, [1, 99])
        img = (img - lo) / max(hi - lo, 1e-6)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def _save_overlay(path: Path, image: np.ndarray, pred_inst: np.ndarray, gt_inst: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = np.repeat(canvas[..., None], 3, axis=2)
    gt_b = find_boundaries(gt_inst, mode="outer")
    pr_b = find_boundaries(pred_inst, mode="outer")
    canvas[gt_b] = np.array([0, 220, 0], dtype=np.uint8)
    canvas[pr_b] = np.array([240, 0, 0], dtype=np.uint8)
    im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, min(im.width, 560), 18], fill=(0, 0, 0))
    draw.text((4, 3), title[:92], fill=(255, 255, 255))
    im.save(path)


def _aggregate_extra(per_image: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in METRIC_COLUMNS:
        values = [_float_or_nan(row.get(key)) for row in per_image]
        finite = [v for v in values if math.isfinite(v)]
        if finite:
            out[key] = float(np.mean(finite))
    return out


def _error_analysis_block(
    dataset: str,
    tag: str,
    primary: str,
    per_image: List[Dict[str, float]],
    output_dir: Path,
    error_totals: Dict[str, int],
    size_totals: Dict[str, Tuple[int, int]],
) -> List[str]:
    scores = np.asarray([_float_or_nan(row.get(primary)) for row in per_image], dtype=np.float64)
    density = np.asarray([_float_or_nan(row.get("gt_objects")) for row in per_image], dtype=np.float64)
    density_lines = []
    if len(scores) >= 3 and np.isfinite(scores).any():
        q1, q2 = np.nanquantile(density, [1 / 3, 2 / 3])
        for name, mask in [
            ("low_density", density <= q1),
            ("mid_density", (density > q1) & (density <= q2)),
            ("high_density", density > q2),
        ]:
            if mask.any():
                density_lines.append(f"- {name}: n={int(mask.sum())}, {primary}={float(np.nanmean(scores[mask])):.4f}")
    size_lines = []
    for name, (hit, total) in size_totals.items():
        score = hit / total if total else float("nan")
        size_lines.append(f"- {name}: recall@0.5={score:.4f} ({hit}/{total})")
    worst = sorted(per_image, key=lambda r: _float_or_nan(r.get(primary)))[: min(10, len(per_image))]
    worst_lines = [
        f"- sample={int(row['index'])}, {primary}={_float_or_nan(row.get(primary)):.4f}, "
        f"gt={int(row['gt_objects'])}, pred={int(row['pred_objects'])}"
        for row in worst
    ]
    block = [
        f"\n## {dataset} / {tag}",
        "",
        f"- Primary metric: {primary}",
        f"- Images evaluated: {len(per_image)}",
        f"- Error totals: merge={error_totals['merge']}, split={error_totals['split']}, "
        f"miss={error_totals['miss']}, false_positive={error_totals['false_positive']}, "
        f"boundary={error_totals['boundary']}",
        f"- Overlay directory: `{output_dir}`",
        "",
        "Density strata:",
        *density_lines,
        "",
        "Object size strata:",
        *size_lines,
        "",
        "Worst samples:",
        *worst_lines,
        "",
    ]
    return block


def _write_error_analysis(
    dataset: str,
    tag: str,
    primary: str,
    per_image: List[Dict[str, float]],
    output_dir: Path,
    error_totals: Dict[str, int],
    size_totals: Dict[str, Tuple[int, int]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = OUT_ROOT / "error_analysis.md"
    block = _error_analysis_block(dataset, tag, primary, per_image, output_dir, error_totals, size_totals)
    if not md_path.exists():
        md_path.write_text("# Full-Validation Error Analysis\n", encoding="utf-8")
    with md_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(block))


def run_eval(cfg: RunConfig, mode: str) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_ts = _now()
    wall_start = time.time()
    num_types = DATASET_NUM_TYPES.get(cfg.dataset, 0)
    primary = PRIMARY_METRIC[cfg.dataset]
    model, loaded_kind, tensor_count = _build_model(cfg, device)
    patch_size = int(model.backbone.patch_size)
    ds = _build_ds(cfg.dataset, cfg.split, normalize=True)
    raw_ds = _build_ds(cfg.dataset, cfg.split, normalize=False) if cfg.save_overlays else None
    n = len(ds) if cfg.max_images is None else min(len(ds), int(cfg.max_images))

    per_image: List[Dict[str, float]] = []
    preds_i: List[np.ndarray] = []
    gts_i: List[np.ndarray] = []
    preds_s: List[np.ndarray] = []
    gts_s: List[np.ndarray] = []
    error_totals = {"merge": 0, "split": 0, "miss": 0, "false_positive": 0, "boundary": 0}
    size_totals = {"small": (0, 0), "medium": (0, 0), "large": (0, 0)}
    overlay_candidates: List[Tuple[float, int, np.ndarray, np.ndarray, np.ndarray]] = []

    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(n), desc=f"{cfg.dataset}:{cfg.output_tag}", leave=False):
            img, sem, inst = ds[i][:3]
            out = sliding_window_predict(
                model,
                img.to(device),
                crop_size=cfg.crop_size,
                stride=cfg.stride,
                patch_size=patch_size,
                num_types=num_types,
                tta=cfg.tta,
                tta_mode=cfg.tta_mode,
                blend_mode=cfg.blend_mode,
            )
            pred_inst, pred_sem = postprocess(
                out["np"],
                out["hv"],
                out["tp"],
                fg_thresh=cfg.fg_thresh,
                energy_thresh=cfg.energy_thresh,
                sobel_ksize=cfg.sobel_ksize,
                min_size=cfg.min_size,
            )
            gt_inst = inst.numpy().astype(np.int32)
            preds_i.append(pred_inst)
            gts_i.append(gt_inst)
            gt_sem = None
            if num_types > 0:
                gt_sem = sem.numpy().astype(np.int32)
                gt_sem[gt_sem == 255] = 0
                preds_s.append(pred_sem)
                gts_s.append(gt_sem)
            row = _per_image_metrics(pred_inst, gt_inst, pred_sem if num_types > 0 else None, gt_sem, num_types)
            row["index"] = i
            row["gt_objects"] = len(_instance_ids(gt_inst))
            row["pred_objects"] = len(_instance_ids(pred_inst))
            errs = _error_counts(pred_inst, gt_inst)
            for k, v in errs.items():
                row[f"err_{k}"] = v
                error_totals[k] += v
            size = _size_recall(pred_inst, gt_inst)
            for key, (hit, total) in size.items():
                old_hit, old_total = size_totals[key]
                size_totals[key] = (old_hit + hit, old_total + total)
            per_image.append(row)
            score = _float_or_nan(row.get(primary))
            if cfg.save_overlays and raw_ds is not None:
                raw_img = _to_rgb_uint8(raw_ds[i][0])
                overlay_candidates.append((score, i, raw_img, pred_inst.copy(), gt_inst.copy()))

    if num_types > 0:
        metrics = accumulate_instance_metrics(preds_i, gts_i, preds_s, gts_s, num_classes=num_types)
    else:
        metrics = accumulate_instance_metrics(preds_i, gts_i)
    metrics.update(_aggregate_extra(per_image))
    wall_seconds = time.time() - wall_start
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
    end_ts = _now()

    result_dir = OUT_ROOT / mode / cfg.dataset / cfg.output_tag
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": cfg.dataset,
                "split": cfg.split,
                "n": n,
                "primary_metric": primary,
                "metrics": metrics,
                "per_image": per_image,
                "config": cfg.__dict__ | {
                    "checkpoint": str(cfg.checkpoint),
                    "train_config": str(cfg.train_config),
                    "head_path": str(cfg.head_path),
                },
                "meta": {
                    "start_time": start_ts,
                    "end_time": end_ts,
                    "wall_seconds": wall_seconds,
                    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
                    "peak_cuda_gib": peak_mem,
                    "checkpoint_kind": loaded_kind,
                    "checkpoint_tensor_count": tensor_count,
                    "patch_size": patch_size,
                    "host": socket.gethostname(),
                    "git_commit": _git_commit(),
                },
                "errors": error_totals,
                "size_strata": {k: {"matched": v[0], "total": v[1]} for k, v in size_totals.items()},
            },
            f,
            indent=2,
        )

    if cfg.save_overlays:
        overlay_dir = OUT_ROOT / "error_overlays" / cfg.dataset / cfg.output_tag
        for score, i, raw_img, pred_inst, gt_inst in sorted(overlay_candidates, key=lambda x: x[0])[: cfg.top_k]:
            name = f"{cfg.dataset}_{i:05d}_{primary}_{score:.4f}.png"
            _save_overlay(overlay_dir / name, raw_img, pred_inst, gt_inst, f"{cfg.dataset} idx={i} {primary}={score:.4f}")
        _write_error_analysis(cfg.dataset, cfg.output_tag, primary, per_image, overlay_dir, error_totals, size_totals)

    common_row: Dict[str, object] = {
        "timestamp": end_ts,
        "dataset": cfg.dataset,
        "split": cfg.split,
        "mode": mode,
        "tag": cfg.output_tag,
        "n_images": n,
        "checkpoint": str(cfg.checkpoint),
        "head_path": str(cfg.head_path),
        "layers": " ".join(str(x) for x in cfg.layers),
        "feature_size": cfg.feature_size,
        "embed_proj": cfg.embed_proj,
        "crop_size": cfg.crop_size,
        "stride": cfg.stride,
        "blend_mode": cfg.blend_mode,
        "fg_thresh": cfg.fg_thresh,
        "energy_thresh": cfg.energy_thresh,
        "sobel_ksize": cfg.sobel_ksize,
        "min_size": cfg.min_size,
        "tta": int(cfg.tta),
        "tta_mode": cfg.tta_mode,
        "primary_metric": primary,
        "primary_value": metrics.get(primary, float("nan")),
        "wall_seconds": wall_seconds,
        "peak_cuda_gib": peak_mem,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "pid": os.getpid(),
        "exit_code": 0,
    }
    for key in METRIC_COLUMNS:
        common_row[key] = metrics.get(key, float("nan"))
    full_fields = list(common_row.keys())
    if mode == "full_val_baseline":
        _append_csv(OUT_ROOT / "full_val_baseline.csv", common_row, full_fields)
    else:
        _append_csv(OUT_ROOT / "eval_only_results.csv", common_row, full_fields)

    exp_fields = [
        "timestamp",
        "dataset",
        "stage",
        "experiment",
        "changed_variable",
        "config",
        "checkpoint",
        "split",
        "n_images",
        "primary_metric",
        "primary_value",
        "wall_seconds",
        "peak_cuda_gib",
        "gpu",
        "pid",
        "exit_code",
        "log_path",
    ]
    _append_csv(
        OUT_ROOT / "metric_improvement_experiment_log.csv",
        {
            "timestamp": end_ts,
            "dataset": cfg.dataset,
            "stage": "stage1" if mode == "full_val_baseline" else "stage2",
            "experiment": cfg.output_tag,
            "changed_variable": "protocol_full_validation" if mode == "full_val_baseline" else "eval_config",
            "config": json.dumps({
                "crop_size": cfg.crop_size,
                "stride": cfg.stride,
                "blend_mode": cfg.blend_mode,
                "fg_thresh": cfg.fg_thresh,
                "energy_thresh": cfg.energy_thresh,
                "sobel_ksize": cfg.sobel_ksize,
                "min_size": cfg.min_size,
                "tta": cfg.tta,
                "tta_mode": cfg.tta_mode,
                "max_images": cfg.max_images,
            }, sort_keys=True),
            "checkpoint": str(cfg.head_path),
            "split": cfg.split,
            "n_images": n,
            "primary_metric": primary,
            "primary_value": metrics.get(primary, float("nan")),
            "wall_seconds": wall_seconds,
            "peak_cuda_gib": peak_mem,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "pid": os.getpid(),
            "exit_code": 0,
            "log_path": os.environ.get("INSTSEG_LOG_PATH", ""),
        },
        exp_fields,
    )
    return {"metrics": metrics, "row": common_row}


def _eval_row_from_cached_outputs(
    cfg: RunConfig,
    mode: str,
    primary: str,
    num_types: int,
    n: int,
    cached: Sequence[Tuple[Dict[str, np.ndarray], np.ndarray, Optional[np.ndarray]]],
    loaded_kind: str,
    tensor_count: int,
    patch_size: int,
    infer_wall_seconds: float,
    peak_mem: float,
) -> Dict[str, object]:
    start_ts = _now()
    wall_start = time.time()
    preds_i: List[np.ndarray] = []
    gts_i: List[np.ndarray] = []
    preds_s: List[np.ndarray] = []
    gts_s: List[np.ndarray] = []
    per_image: List[Dict[str, float]] = []
    error_totals = {"merge": 0, "split": 0, "miss": 0, "false_positive": 0, "boundary": 0}
    size_totals = {"small": (0, 0), "medium": (0, 0), "large": (0, 0)}

    def process_one(item: Tuple[int, Tuple[Dict[str, np.ndarray], np.ndarray, Optional[np.ndarray]]]):
        i, (out, gt_inst, gt_sem) = item
        pred_inst, pred_sem = postprocess(
            out["np"],
            out["hv"],
            out["tp"],
            fg_thresh=cfg.fg_thresh,
            energy_thresh=cfg.energy_thresh,
            sobel_ksize=cfg.sobel_ksize,
            min_size=cfg.min_size,
        )
        row = _per_image_metrics(pred_inst, gt_inst, pred_sem if num_types > 0 else None, gt_sem, num_types)
        row["index"] = i
        row["gt_objects"] = len(_instance_ids(gt_inst))
        row["pred_objects"] = len(_instance_ids(pred_inst))
        errs = _error_counts(pred_inst, gt_inst)
        size = _size_recall(pred_inst, gt_inst)
        return i, row, errs, size

    items = list(enumerate(cached))
    if cfg.post_workers > 1:
        with ThreadPoolExecutor(max_workers=cfg.post_workers) as executor:
            processed = list(tqdm(
                executor.map(process_one, items),
                total=len(items),
                desc=f"{cfg.dataset}:{cfg.output_tag}:post",
                leave=False,
            ))
    else:
        processed = [
            process_one(item)
            for item in tqdm(items, desc=f"{cfg.dataset}:{cfg.output_tag}:post", leave=False)
        ]
    processed.sort(key=lambda x: x[0])

    for _, row, errs, size in processed:
        for k, v in errs.items():
            row[f"err_{k}"] = v
            error_totals[k] += v
        for key, (hit, total) in size.items():
            old_hit, old_total = size_totals[key]
            size_totals[key] = (old_hit + hit, old_total + total)
        per_image.append(row)

    # process_one already computes the same per-image metrics as
    # accumulate_instance_metrics; aggregate once to avoid recomputing pairwise
    # IoU matrices for every full-validation threshold candidate.
    metrics = _aggregate_extra(per_image)
    wall_seconds = infer_wall_seconds + (time.time() - wall_start)
    end_ts = _now()

    result_dir = OUT_ROOT / mode / cfg.dataset / cfg.output_tag
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": cfg.dataset,
                "split": cfg.split,
                "n": n,
                "primary_metric": primary,
                "metrics": metrics,
                "per_image": per_image,
                "config": cfg.__dict__ | {
                    "checkpoint": str(cfg.checkpoint),
                    "train_config": str(cfg.train_config),
                    "head_path": str(cfg.head_path),
                    "fast_threshold_reuse_outputs": True,
                    "post_workers": cfg.post_workers,
                },
                "meta": {
                    "start_time": start_ts,
                    "end_time": end_ts,
                    "wall_seconds": wall_seconds,
                    "infer_wall_seconds": infer_wall_seconds,
                    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
                    "peak_cuda_gib": peak_mem,
                    "checkpoint_kind": loaded_kind,
                    "checkpoint_tensor_count": tensor_count,
                    "patch_size": patch_size,
                    "host": socket.gethostname(),
                    "git_commit": _git_commit(),
                },
                "errors": error_totals,
                "size_strata": {k: {"matched": v[0], "total": v[1]} for k, v in size_totals.items()},
            },
            f,
            indent=2,
        )

    common_row: Dict[str, object] = {
        "timestamp": end_ts,
        "dataset": cfg.dataset,
        "split": cfg.split,
        "mode": mode,
        "tag": cfg.output_tag,
        "n_images": n,
        "checkpoint": str(cfg.checkpoint),
        "head_path": str(cfg.head_path),
        "layers": " ".join(str(x) for x in cfg.layers),
        "feature_size": cfg.feature_size,
        "embed_proj": cfg.embed_proj,
        "crop_size": cfg.crop_size,
        "stride": cfg.stride,
        "blend_mode": cfg.blend_mode,
        "fg_thresh": cfg.fg_thresh,
        "energy_thresh": cfg.energy_thresh,
        "sobel_ksize": cfg.sobel_ksize,
        "min_size": cfg.min_size,
        "tta": int(cfg.tta),
        "tta_mode": cfg.tta_mode,
        "primary_metric": primary,
        "primary_value": metrics.get(primary, float("nan")),
        "wall_seconds": wall_seconds,
        "peak_cuda_gib": peak_mem,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "pid": os.getpid(),
        "exit_code": 0,
    }
    for key in METRIC_COLUMNS:
        common_row[key] = metrics.get(key, float("nan"))
    _append_csv(OUT_ROOT / "eval_only_results.csv", common_row, list(common_row.keys()))

    exp_fields = [
        "timestamp",
        "dataset",
        "stage",
        "experiment",
        "changed_variable",
        "config",
        "checkpoint",
        "split",
        "n_images",
        "primary_metric",
        "primary_value",
        "wall_seconds",
        "peak_cuda_gib",
        "gpu",
        "pid",
        "exit_code",
        "log_path",
    ]
    _append_csv(
        OUT_ROOT / "metric_improvement_experiment_log.csv",
        {
            "timestamp": end_ts,
            "dataset": cfg.dataset,
            "stage": "stage2",
            "experiment": cfg.output_tag,
            "changed_variable": "eval_config",
            "config": json.dumps({
                "crop_size": cfg.crop_size,
                "stride": cfg.stride,
                "blend_mode": cfg.blend_mode,
                "fg_thresh": cfg.fg_thresh,
                "energy_thresh": cfg.energy_thresh,
                "sobel_ksize": cfg.sobel_ksize,
                "min_size": cfg.min_size,
                "tta": cfg.tta,
                "tta_mode": cfg.tta_mode,
                "max_images": cfg.max_images,
                "fast_threshold_reuse_outputs": True,
                "post_workers": cfg.post_workers,
            }, sort_keys=True),
            "checkpoint": str(cfg.head_path),
            "split": cfg.split,
            "n_images": n,
            "primary_metric": primary,
            "primary_value": metrics.get(primary, float("nan")),
            "wall_seconds": wall_seconds,
            "peak_cuda_gib": peak_mem,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "pid": os.getpid(),
            "exit_code": 0,
            "log_path": os.environ.get("INSTSEG_LOG_PATH", ""),
        },
        exp_fields,
    )
    return {"metrics": metrics, "row": common_row}


def _collect_continuous_outputs(cfg: RunConfig) -> Tuple[
    List[Tuple[Dict[str, np.ndarray], np.ndarray, Optional[np.ndarray]]],
    str,
    int,
    int,
    float,
    float,
]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.time()
    num_types = DATASET_NUM_TYPES.get(cfg.dataset, 0)
    model, loaded_kind, tensor_count = _build_model(cfg, device)
    patch_size = int(model.backbone.patch_size)
    ds = _build_ds(cfg.dataset, cfg.split, normalize=True)
    n = len(ds) if cfg.max_images is None else min(len(ds), int(cfg.max_images))
    cached: List[Tuple[Dict[str, np.ndarray], np.ndarray, Optional[np.ndarray]]] = []
    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(n), desc=f"{cfg.dataset}:continuous", leave=False):
            img, sem, inst = ds[i][:3]
            out = sliding_window_predict(
                model,
                img.to(device),
                crop_size=cfg.crop_size,
                stride=cfg.stride,
                patch_size=patch_size,
                num_types=num_types,
                tta=cfg.tta,
                tta_mode=cfg.tta_mode,
                blend_mode=cfg.blend_mode,
            )
            gt_inst = inst.numpy().astype(np.int32)
            gt_sem = None
            if num_types > 0:
                gt_sem = sem.numpy().astype(np.int32)
                gt_sem[gt_sem == 255] = 0
            cached.append((out, gt_inst, gt_sem))
    infer_wall_seconds = time.time() - wall_start
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
    return cached, loaded_kind, tensor_count, patch_size, infer_wall_seconds, peak_mem


def _run_threshold_candidates_reuse_outputs(
    args,
    dataset: str,
    candidates: Sequence[Tuple[str, Dict[str, object]]],
) -> Tuple[float, str, Dict[str, object]]:
    base_cfg = _make_config(args, dataset, "_continuous_cache", save_overlays=False)
    cached, loaded_kind, tensor_count, patch_size, infer_wall_seconds, peak_mem = _collect_continuous_outputs(base_cfg)
    num_types = DATASET_NUM_TYPES.get(dataset, 0)
    primary = PRIMARY_METRIC[dataset]
    n = len(cached)
    best_value = -float("inf")
    best_tag = ""
    best_config: Dict[str, object] = {}
    for tag, overrides in candidates:
        cfg = _make_config(args, dataset, tag, save_overlays=False, **overrides)
        result = _eval_row_from_cached_outputs(
            cfg,
            "eval_only",
            primary,
            num_types,
            n,
            cached,
            loaded_kind,
            tensor_count,
            patch_size,
            infer_wall_seconds,
            peak_mem,
        )
        value = _float_or_nan(result["metrics"].get(primary))
        if value > best_value:
            best_value = value
            best_tag = tag
            best_config = dict(overrides)
    return best_value, best_tag, best_config


def _make_config(args, dataset: str, tag: str, **overrides) -> RunConfig:
    values = {
        "dataset": dataset,
        "split": args.split,
        "checkpoint": Path(args.checkpoint),
        "train_config": Path(args.train_config),
        "head_path": Path(args.head_path) if args.head_path else _head_path(dataset),
        "layers": [int(x) for x in args.layers],
        "feature_size": args.feature_size,
        "embed_proj": args.embed_proj,
        "crop_size": args.crop_size,
        "stride": args.stride,
        "blend_mode": args.blend_mode,
        "fg_thresh": args.fg_thresh,
        "energy_thresh": args.energy_thresh,
        "sobel_ksize": args.sobel_ksize,
        "min_size": args.min_size,
        "tta": args.tta,
        "tta_mode": args.tta_mode,
        "max_images": args.max_images,
        "output_tag": tag,
        "save_overlays": args.save_overlays,
        "top_k": args.top_k,
        "post_workers": args.post_workers,
    }
    values.update(overrides)
    return RunConfig(**values)


def run_baseline(args) -> None:
    datasets = args.datasets or DATASET_ORDER
    for dataset in datasets:
        cfg = _make_config(args, dataset, "default_full_val")
        run_eval(cfg, "full_val_baseline")
    summarize_baseline()


def run_eval_one(args) -> None:
    datasets = args.datasets or DATASET_ORDER
    for dataset in datasets:
        cfg = _make_config(args, dataset, args.tag)
        run_eval(cfg, "eval_only")
    summarize_eval_only()


def _load_baseline_value(dataset: str) -> Optional[float]:
    path = OUT_ROOT / "full_val_baseline.csv"
    if not path.exists():
        return None
    primary = PRIMARY_METRIC[dataset]
    best = None
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("dataset") != dataset or row.get("tag") != "default_full_val":
                continue
            value = _float_or_nan(row.get(primary))
            if math.isfinite(value):
                best = value
    return best


def _append_gain(
    dataset: str,
    method: str,
    baseline: Optional[float],
    optimized: float,
    config: Dict[str, object],
    cost: str,
    adopted: bool,
    gain_type: str = "single_item",
) -> None:
    primary = PRIMARY_METRIC[dataset]
    row = {
        "timestamp": _now(),
        "dataset": dataset,
        "optimization_method": method,
        "metric": primary,
        "baseline_metric": "" if baseline is None else baseline,
        "optimized_metric": optimized,
        "absolute_gain": "" if baseline is None else optimized - baseline,
        "config": json.dumps(config, sort_keys=True),
        "train_infer_cost": cost,
        "adopted": int(adopted),
        "gain_type": gain_type,
    }
    _append_csv(
        OUT_ROOT / "method_gain_ledger.csv",
        row,
        [
            "timestamp",
            "dataset",
            "optimization_method",
            "metric",
            "baseline_metric",
            "optimized_metric",
            "absolute_gain",
            "config",
            "train_infer_cost",
            "adopted",
            "gain_type",
        ],
    )


def run_threshold_sweep(args) -> None:
    datasets = args.datasets or DATASET_ORDER
    for dataset in datasets:
        baseline = _load_baseline_value(dataset)
        candidates: List[Tuple[str, Dict[str, object]]] = []
        seen_candidates: set[Tuple[float, float]] = set()
        def add_candidate(tag: str, fg: float, energy: float) -> None:
            key = (round(float(fg), 4), round(float(energy), 4))
            if key in seen_candidates:
                return
            seen_candidates.add(key)
            candidates.append((tag, {"fg_thresh": float(fg), "energy_thresh": float(energy)}))

        for fg in np.round(np.arange(0.30, 0.7001, 0.05), 2):
            add_candidate(f"fg_{fg:.2f}_energy_0.40", float(fg), 0.40)
        for energy in np.round(np.arange(0.20, 0.7001, 0.05), 2):
            add_candidate(f"fg_0.50_energy_{energy:.2f}", 0.50, float(energy))
        best_value = -float("inf")
        best_tag = ""
        best_config: Dict[str, object] = {}
        if args.reuse_outputs:
            best_value, best_tag, best_config = _run_threshold_candidates_reuse_outputs(args, dataset, candidates)
        else:
            for tag, overrides in candidates:
                cfg = _make_config(args, dataset, tag, save_overlays=False, **overrides)
                result = run_eval(cfg, "eval_only")
                value = _float_or_nan(result["metrics"].get(PRIMARY_METRIC[dataset]))
                if value > best_value:
                    best_value = value
                    best_tag = tag
                    best_config = overrides
        if args.max_images is None:
            _append_gain(
                dataset,
                "threshold_coarse_single_factor",
                baseline,
                best_value,
                best_config | {"best_tag": best_tag},
                "eval-only; full validation per candidate",
                adopted=baseline is not None and best_value >= baseline,
            )
    summarize_eval_only()


def run_threshold_local(args) -> None:
    datasets = args.datasets or DATASET_ORDER
    for dataset in datasets:
        baseline = _load_baseline_value(dataset)
        fg_center = float(args.fg_center)
        energy_center = float(args.energy_center)
        fg_values = [round(fg_center + 0.02 * offset, 2) for offset in range(-2, 3)]
        energy_values = [round(energy_center + 0.02 * offset, 2) for offset in range(-2, 3)]
        best_value = -float("inf")
        best_tag = ""
        best_config: Dict[str, object] = {}
        candidates: List[Tuple[str, Dict[str, object]]] = []
        for fg in fg_values:
            for energy in energy_values:
                if fg < 0.0 or fg > 1.0 or energy < 0.0 or energy > 1.0:
                    continue
                tag = f"local_fg_{fg:.2f}_energy_{energy:.2f}"
                candidates.append((tag, {"fg_thresh": float(fg), "energy_thresh": float(energy)}))
        if args.reuse_outputs:
            best_value, best_tag, best_config = _run_threshold_candidates_reuse_outputs(args, dataset, candidates)
        else:
            for tag, overrides in candidates:
                cfg = _make_config(args, dataset, tag, save_overlays=False, **overrides)
                result = run_eval(cfg, "eval_only")
                value = _float_or_nan(result["metrics"].get(PRIMARY_METRIC[dataset]))
                if value > best_value:
                    best_value = value
                    best_tag = tag
                    best_config = dict(overrides)
        if args.max_images is None:
            _append_gain(
                dataset,
                "threshold_local_2d",
                baseline,
                best_value,
                best_config | {"best_tag": best_tag, "fg_center": fg_center, "energy_center": energy_center},
                "eval-only; 5x5 local full-validation grid",
                adopted=baseline is not None and best_value >= baseline,
            )
    summarize_eval_only()


def run_tiling_sweep(args) -> None:
    datasets = args.datasets or DATASET_ORDER
    for dataset in datasets:
        baseline = _load_baseline_value(dataset)
        candidates: List[Tuple[str, Dict[str, object]]] = []
        for crop in (256, 384, 512):
            strides = sorted({args.stride, max(1, int(round(crop * 0.50))), max(1, int(round(crop * 0.375)))})
            for stride in strides:
                if stride > crop:
                    continue
                for blend in ("uniform", "gaussian"):
                    candidates.append(
                        (
                            f"tile_crop{crop}_stride{stride}_{blend}",
                            {"crop_size": crop, "stride": stride, "blend_mode": blend},
                        )
                    )
        best_value = -float("inf")
        best_tag = ""
        best_config: Dict[str, object] = {}
        for tag, overrides in candidates:
            cfg = _make_config(args, dataset, tag, save_overlays=False, **overrides)
            result = run_eval(cfg, "eval_only")
            value = _float_or_nan(result["metrics"].get(PRIMARY_METRIC[dataset]))
            if value > best_value:
                best_value = value
                best_tag = tag
                best_config = overrides
        if args.max_images is None:
            _append_gain(
                dataset,
                "tiling_crop_stride_blending",
                baseline,
                best_value,
                best_config | {"best_tag": best_tag},
                "eval-only; crop/stride/blending full-validation sweep",
                adopted=baseline is not None and best_value >= baseline,
            )
    summarize_eval_only()


def run_tta_sweep(args) -> None:
    datasets = args.datasets or DATASET_ORDER
    for dataset in datasets:
        baseline = _load_baseline_value(dataset)
        candidates: List[Tuple[str, Dict[str, object]]] = [
            ("tta_off", {"tta": False, "tta_mode": "flip4"}),
            ("tta_flip4", {"tta": True, "tta_mode": "flip4"}),
        ]
        if args.include_dihedral8:
            candidates.append(("tta_dihedral8", {"tta": True, "tta_mode": "dihedral8"}))
        best_value = -float("inf")
        best_tag = ""
        best_config: Dict[str, object] = {}
        for tag, overrides in candidates:
            cfg = _make_config(args, dataset, tag, save_overlays=False, **overrides)
            result = run_eval(cfg, "eval_only")
            value = _float_or_nan(result["metrics"].get(PRIMARY_METRIC[dataset]))
            if value > best_value:
                best_value = value
                best_tag = tag
                best_config = overrides
        if args.max_images is None:
            _append_gain(
                dataset,
                "tta_mode",
                baseline,
                best_value,
                best_config | {"best_tag": best_tag},
                "eval-only; continuous logit/HV TTA merge",
                adopted=baseline is not None and best_value >= baseline,
            )
    summarize_eval_only()


def summarize_baseline() -> None:
    csv_path = OUT_ROOT / "full_val_baseline.csv"
    md_path = OUT_ROOT / "full_val_baseline_summary.md"
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open(newline="")))
    latest: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if row.get("tag") == "default_full_val":
            latest[row["dataset"]] = row
    lines = [
        "# Full Validation Baseline Summary",
        "",
        "Protocol note: these rows use complete validation splits and are protocol corrections, not optimization gains.",
        "",
        "| dataset | n | primary | value | AJI | bPQ | mPQ | ObjectAP | SEG |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASET_ORDER:
        row = latest.get(dataset)
        if not row:
            continue
        primary = row.get("primary_metric", PRIMARY_METRIC[dataset])
        lines.append(
            f"| {dataset} | {row.get('n_images','')} | {primary} | {_float_or_nan(row.get(primary)):.4f} | "
            f"{_float_or_nan(row.get('AJI')):.4f} | {_float_or_nan(row.get('bPQ')):.4f} | "
            f"{_float_or_nan(row.get('mPQ')):.4f} | {_float_or_nan(row.get('ObjectAP')):.4f} | "
            f"{_float_or_nan(row.get('SEG')):.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_eval_only() -> None:
    csv_path = OUT_ROOT / "eval_only_results.csv"
    md_path = OUT_ROOT / "eval_only_summary.md"
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open(newline="")))
    lines = [
        "# Eval-Only Optimization Summary",
        "",
        "| dataset | best tag | primary | value | crop | stride | fg | energy | tta |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASET_ORDER:
        primary = PRIMARY_METRIC[dataset]
        full_n = None
        baseline_path = OUT_ROOT / "full_val_baseline.csv"
        if baseline_path.exists():
            for base_row in csv.DictReader(baseline_path.open(newline="")):
                if base_row.get("dataset") == dataset and base_row.get("tag") == "default_full_val":
                    full_n = base_row.get("n_images")
        ds_rows = [
            r for r in rows
            if r.get("dataset") == dataset and (full_n is None or r.get("n_images") == full_n)
        ]
        if not ds_rows:
            continue
        best = max(ds_rows, key=lambda r: _float_or_nan(r.get(primary)))
        lines.append(
            f"| {dataset} | {best.get('tag','')} | {primary} | {_float_or_nan(best.get(primary)):.4f} | "
            f"{best.get('crop_size','')} | {best.get('stride','')} | {best.get('fg_thresh','')} | "
            f"{best.get('energy_thresh','')} | {best.get('tta','')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_gain() -> None:
    path = OUT_ROOT / "method_gain_ledger.csv"
    out = OUT_ROOT / "method_gain_summary.md"
    lines = ["# Method Gain Summary", ""]
    if not path.exists():
        lines.append("No optimization-gain rows have completed yet.")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    rows = list(csv.DictReader(path.open(newline="")))
    lines.extend([
        "| dataset | method | metric | baseline | optimized | gain | adopted |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row.get('dataset','')} | {row.get('optimization_method','')} | {row.get('metric','')} | "
            f"{_float_or_nan(row.get('baseline_metric')):.4f} | {_float_or_nan(row.get('optimized_metric')):.4f} | "
            f"{_float_or_nan(row.get('absolute_gain')):.4f} | {row.get('adopted','')} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rebuild_error_analysis() -> None:
    lines = [
        "# Full-Validation Error Analysis",
        "",
        "Regenerated from `full_val_baseline/*/default_full_val/metrics.json`; only complete validation runs are included.",
    ]
    for dataset in DATASET_ORDER:
        path = OUT_ROOT / "full_val_baseline" / dataset / "default_full_val" / "metrics.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        tag = str(data.get("config", {}).get("output_tag") or "default_full_val")
        primary = str(data.get("primary_metric") or PRIMARY_METRIC[dataset])
        per_image = data.get("per_image", [])
        errors = data.get("errors", {})
        size = data.get("size_strata", {})
        error_totals = {
            "merge": int(errors.get("merge", 0)),
            "split": int(errors.get("split", 0)),
            "miss": int(errors.get("miss", 0)),
            "false_positive": int(errors.get("false_positive", 0)),
            "boundary": int(errors.get("boundary", 0)),
        }
        size_totals = {
            key: (int(value.get("matched", 0)), int(value.get("total", 0)))
            for key, value in size.items()
        }
        for key in ("small", "medium", "large"):
            size_totals.setdefault(key, (0, 0))
        overlay_dir = OUT_ROOT / "error_overlays" / dataset / tag
        lines.extend(_error_analysis_block(dataset, tag, primary, per_image, overlay_dir, error_totals, size_totals))
    (OUT_ROOT / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def record_adaptation_gain() -> None:
    src = OUT_ROOT / "adaptation_results.csv"
    dst = OUT_ROOT / "method_gain_ledger.csv"
    if not src.exists():
        raise FileNotFoundError(src)

    preserved_rows: List[Dict[str, str]] = []
    preserved_fields: List[str] = []
    if dst.exists() and dst.stat().st_size > 0:
        with dst.open(newline="") as f:
            reader = csv.DictReader(f)
            preserved_fields = list(reader.fieldnames or [])
            preserved_rows = [
                row for row in reader
                if not row.get("optimization_method", "").startswith("adaptation_")
            ]
        with dst.open("w", newline="") as f:
            fields = preserved_fields or [
                "timestamp",
                "dataset",
                "optimization_method",
                "metric",
                "baseline_metric",
                "optimized_metric",
                "absolute_gain",
                "config",
                "train_infer_cost",
                "adopted",
                "gain_type",
            ]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(preserved_rows)

    rows = list(csv.DictReader(src.open(newline="")))
    best_by_dataset_metric: Dict[Tuple[str, str], float] = {}
    for row in rows:
        if row.get("max_eval_images") or row.get("max_train_batches"):
            continue
        dataset = row.get("dataset", "")
        primary = row.get("primary_metric") or PRIMARY_METRIC.get(dataset, "")
        value = _float_or_nan(row.get("primary_value"))
        key = (dataset, primary)
        if math.isfinite(value):
            best_by_dataset_metric[key] = max(best_by_dataset_metric.get(key, -float("inf")), value)

    for row in rows:
        if row.get("max_eval_images") or row.get("max_train_batches"):
            continue
        dataset = row.get("dataset", "")
        if dataset not in PRIMARY_METRIC:
            continue
        primary = row.get("primary_metric") or PRIMARY_METRIC[dataset]
        optimized = _float_or_nan(row.get("primary_value"))
        if not math.isfinite(optimized):
            continue
        baseline = _load_baseline_value(dataset)
        config = {
            "mode": row.get("mode", ""),
            "layers": row.get("layers", ""),
            "fusion_mode": row.get("fusion_mode", ""),
            "epochs": int(float(row.get("epochs") or 0)),
            "decoder_lr": _float_or_nan(row.get("decoder_lr")),
            "backbone_lr": row.get("backbone_lr", ""),
            "lora_rank": row.get("lora_rank", ""),
            "lora_alpha": row.get("lora_alpha", ""),
            "lora_dropout": row.get("lora_dropout", ""),
            "warmup_ratio": _float_or_nan(row.get("warmup_ratio")),
            "grad_clip_norm": _float_or_nan(row.get("grad_clip_norm")),
            "layer_wise_lr_decay": _float_or_nan(row.get("layer_wise_lr_decay") or 1.0),
            "amp_dtype": row.get("amp_dtype", ""),
            "seed": int(float(row.get("seed") or 0)),
            "np_loss_mode": row.get("np_loss_mode", "ce_dice") or "ce_dice",
            "focal_gamma": _float_or_nan(row.get("focal_gamma") or 2.0),
            "tversky_alpha": _float_or_nan(row.get("tversky_alpha") or 0.3),
            "tversky_beta": _float_or_nan(row.get("tversky_beta") or 0.7),
        }
        config_text = json.dumps(config, sort_keys=True)
        np_loss_mode = config["np_loss_mode"]
        loss_suffix = "" if np_loss_mode == "ce_dice" else f"_{np_loss_mode}"
        method = f"adaptation_{row.get('mode', '')}{loss_suffix}"
        cost = (
            f"{row.get('epochs', '')}ep full-val; wall_seconds={row.get('wall_seconds', '')}; "
            f"effective_batch={row.get('effective_batch_size', '')}; "
            f"trainable_params={row.get('trainable_params', '')}; "
            f"trainable_backbone_params={row.get('trainable_backbone_params', '')}; no test tuning"
        )
        _append_gain(
            dataset,
            method,
            baseline,
            optimized,
            config,
            cost,
            adopted=optimized >= best_by_dataset_metric.get((dataset, primary), float("inf")),
        )
    if dst.exists() and dst.stat().st_size > 0:
        ledger_rows = list(csv.DictReader(dst.open(newline="")))
        fieldnames = list(ledger_rows[0].keys()) if ledger_rows else []
        best_adaptation: Dict[Tuple[str, str], float] = {}
        for row in ledger_rows:
            if not row.get("optimization_method", "").startswith("adaptation_"):
                continue
            key = (row.get("dataset", ""), row.get("metric", ""))
            value = _float_or_nan(row.get("optimized_metric"))
            if math.isfinite(value):
                best_adaptation[key] = max(best_adaptation.get(key, -float("inf")), value)
        for row in ledger_rows:
            if not row.get("optimization_method", "").startswith("adaptation_"):
                continue
            key = (row.get("dataset", ""), row.get("metric", ""))
            row["adopted"] = int(_float_or_nan(row.get("optimized_metric")) >= best_adaptation.get(key, float("inf")))
        with dst.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ledger_rows)
    summarize_gain()


def record_eval_gain(args) -> None:
    src = OUT_ROOT / "eval_only_results.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    rows = list(csv.DictReader(src.open(newline="")))
    matches = [
        row for row in rows
        if row.get("dataset") == args.dataset
        and row.get("tag") == args.tag
        and row.get("split") == args.split
        and not row.get("max_images")
    ]
    if not matches:
        raise RuntimeError(f"No full-validation eval row found for dataset={args.dataset} split={args.split} tag={args.tag}")
    row = matches[-1]
    baseline = _load_baseline_value(args.dataset)
    optimized = _float_or_nan(row.get(PRIMARY_METRIC[args.dataset]))
    if not math.isfinite(optimized):
        raise RuntimeError(f"Primary metric missing for {args.dataset}/{args.tag}")
    config = {
        "tag": row.get("tag", ""),
        "crop_size": int(float(row.get("crop_size") or 0)),
        "stride": int(float(row.get("stride") or 0)),
        "blend_mode": row.get("blend_mode", ""),
        "fg_thresh": _float_or_nan(row.get("fg_thresh")),
        "energy_thresh": _float_or_nan(row.get("energy_thresh")),
        "sobel_ksize": int(float(row.get("sobel_ksize") or 0)),
        "min_size": int(float(row.get("min_size") or 0)),
        "tta": bool(int(float(row.get("tta") or 0))),
        "tta_mode": row.get("tta_mode", ""),
    }
    adopted = bool(args.adopted)
    if args.auto_adopt and baseline is not None:
        adopted = optimized >= baseline
    _append_gain(
        args.dataset,
        args.method,
        baseline,
        optimized,
        config,
        args.cost,
        adopted=adopted,
        gain_type=args.gain_type,
    )
    summarize_gain()


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("baseline", "eval-one", "threshold-sweep", "threshold-local", "tiling-sweep", "tta-sweep"):
        q = sub.add_parser(name)
        q.add_argument("--datasets", nargs="+", choices=DATASET_ORDER)
        q.add_argument("--split", default="val")
        q.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
        q.add_argument("--train-config", default=str(DEFAULT_TRAIN_CONFIG))
        q.add_argument("--head-path", default="")
        q.add_argument("--layers", nargs="+", type=int, default=[4, 11, 17, 23])
        q.add_argument("--feature-size", type=int, default=32)
        q.add_argument("--embed-proj", type=int, default=384)
        q.add_argument("--crop-size", type=int, default=256)
        q.add_argument("--stride", type=int, default=192)
        q.add_argument("--blend-mode", choices=["uniform", "gaussian"], default="uniform")
        q.add_argument("--fg-thresh", type=float, default=0.50)
        q.add_argument("--energy-thresh", type=float, default=0.40)
        q.add_argument("--sobel-ksize", type=int, default=21)
        q.add_argument("--min-size", type=int, default=10)
        q.add_argument("--tta", action="store_true")
        q.add_argument("--tta-mode", choices=["flip4", "dihedral8"], default="flip4")
        q.add_argument("--include-dihedral8", action="store_true")
        q.add_argument("--max-images", type=int, default=None)
        q.add_argument("--save-overlays", action="store_true")
        q.add_argument("--top-k", type=int, default=8)
        q.add_argument("--post-workers", type=int, default=1)
        if name in ("threshold-sweep", "threshold-local"):
            q.add_argument("--reuse-outputs", action="store_true")
        if name == "eval-one":
            q.add_argument("--tag", required=True)
        if name == "threshold-local":
            q.add_argument("--fg-center", type=float, required=True)
            q.add_argument("--energy-center", type=float, required=True)
    sub.add_parser("record-adaptation-gain")
    sub.add_parser("rebuild-error-analysis")
    q = sub.add_parser("record-eval-gain")
    q.add_argument("--dataset", required=True, choices=DATASET_ORDER)
    q.add_argument("--split", default="val")
    q.add_argument("--tag", required=True)
    q.add_argument("--method", required=True)
    q.add_argument("--cost", required=True)
    q.add_argument("--adopted", type=int, choices=[0, 1], default=0)
    q.add_argument("--auto-adopt", action="store_true")
    q.add_argument("--gain-type", choices=["single_item", "cumulative"], default="single_item")
    sub.add_parser("summarize-gain")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "baseline":
        run_baseline(args)
    elif args.cmd == "eval-one":
        run_eval_one(args)
    elif args.cmd == "threshold-sweep":
        run_threshold_sweep(args)
    elif args.cmd == "threshold-local":
        run_threshold_local(args)
    elif args.cmd == "tiling-sweep":
        run_tiling_sweep(args)
    elif args.cmd == "tta-sweep":
        run_tta_sweep(args)
    elif args.cmd == "summarize-gain":
        summarize_gain()
    elif args.cmd == "record-adaptation-gain":
        record_adaptation_gain()
    elif args.cmd == "record-eval-gain":
        record_eval_gain(args)
    elif args.cmd == "rebuild-error-analysis":
        rebuild_error_analysis()


if __name__ == "__main__":
    main()
