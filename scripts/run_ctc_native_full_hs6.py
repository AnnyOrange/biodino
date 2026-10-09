#!/usr/bin/env python3
"""Run the precommitted 12-checkpoint HS6 matrix on native 20-domain CTC."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch


ROOT = Path(__file__).resolve().parents[1]
IMAGECODECS_VENDOR = ROOT / "outputs/02_eval_runtime/imagecodecs_py311_wheel_2026.3.6"
for path in (IMAGECODECS_VENDOR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_ctc_native_2d_observation as base  # noqa: E402
from dinov3.eval.bio_segmentation.instance_seg.model import build_dino_hovernet  # noqa: E402
from dinov3.eval.bio_segmentation.instance_seg.postproc import postprocess  # noqa: E402
from dinov3.eval.bio_segmentation.instance_seg.tiling import sliding_window_predict  # noqa: E402
from dinov3.eval.bio_segmentation.metrics import compute_ap  # noqa: E402
from dinov3.eval.bio_tracking.ctc_linker import (  # noqa: E402
    TrackLinker,
    equivalent_diameters,
)
from dinov3.eval.bio_tracking.ctc_metrics_requested import score_requested_metrics  # noqa: E402
from dinov3.eval.bio_tracking.ctc_native import (  # noqa: E402
    CTC_DOMAINS,
    CTCNativeTrainDataset,
    PROTOCOL_ID,
    atomic_json,
    load_cache_manifest,
    normalized_image_tensor,
    prepare_cache,
    read_tiff,
    sha256,
)


CAMPAIGN = ROOT / "outputs/02_eval_runs/ctc_native_full_hs6_12ckpt_20260916"
CACHE = ROOT / "outputs/02_eval_inputs/formal_v3/ctc_native_full"
ARCHIVES = Path(
    "/mnt/huawei_deepcad/benchmark/external_benchmarks_20260901/"
    "CellTrackingChallenge/training_zips"
)
SPLIT = ROOT / "outputs/02_eval_inputs/formal_v3/ctc/split_manifest.jsonl"
PY_CTCMETRICS = ROOT / "outputs/02_eval_runtime/py-ctcmetrics"
PLAN = ROOT / "Evaluation Rules/plans/ctc_native_hs6_candidates_20260911.md"
PROTOCOL = ROOT / "Evaluation Rules/06_CTC_Native_Protocol.md"

SPLUS = ROOT / (
    "outputs/01_training_runs/"
    "HS6_Splus_robust_biosafe256_gb1024_lr2e4_wu3_tw30_nosig_e15_seed0_8x5090xr_20260821b"
)
B = ROOT / (
    "outputs/01_training_runs/"
    "HS6_B_robust_biosafe256_gb1024_lr1p5e4_wu3_tw30_nosig_e15_seed0_8x5090hxw_20260818"
)
L = ROOT / (
    "outputs/01_training_runs/"
    "HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_seed0_20260818"
)
HPLUS = ROOT / (
    "outputs/01_training_runs/"
    "HS6_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e15_seed0_4xH100_20260818"
)
L5 = ROOT / (
    "outputs/01_training_runs/"
    "HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907"
)

CROP_SIZE = 256
STRIDE = 192
TILE_BATCH_SIZE = 8
CTCMETRICS_THREADS = 16


def candidates() -> list[dict[str, Any]]:
    rows = []
    for family, root, steps, layer in (
        ("splus_1tb", SPLUS, (8199, 15374), 11),
        ("b_1tb", B, (6149, 9224, 14349), 11),
        ("l_1tb", L, (12299, 14349), 23),
        ("hplus_1tb", HPLUS, (15374,), 31),
        ("l_5tb", L5, (15615, 17079, 20007, 21959), 23),
    ):
        for step in steps:
            if family == "l_5tb":
                checkpoint = root / "eval" / f"training_{step}" / "teacher_checkpoint.pth"
            elif family == "hplus_1tb":
                checkpoint = root / "ckpt" / str(step) / "teacher_backbone.pth"
            elif family == "b_1tb":
                checkpoint = root / "eval" / f"training_{step}" / "teacher_checkpoint.pth"
            elif family == "l_1tb" and step == 14349:
                checkpoint = root / "eval" / f"training_{step}" / "teacher_checkpoint.pth"
            else:
                checkpoint = root / "ckpt" / str(step) / "checkpoint.pth"
            rows.append({
                "model": f"hs6_{family}_ck{step}",
                "family": family,
                "checkpoint_step": step,
                "checkpoint_path": str(checkpoint.resolve()),
                "config_path": str((root / "config.yaml").resolve()),
                "layers": [layer],
            })
    if len(rows) != 12:
        raise AssertionError("precommitted native CTC set must contain 12 candidates")
    return rows


def fold_record(data: dict[str, Any], fold: int) -> dict[str, Any]:
    rows = [row for row in data["folds"] if int(row["fold"]) == fold]
    if len(rows) != 1:
        raise RuntimeError(f"missing or duplicate fold {fold}")
    return rows[0]


def training_records(data: dict[str, Any], fold: int) -> list[dict[str, Any]]:
    records = []
    for domain in fold_record(data, fold)["train_domains"]:
        records.extend({"domain": domain, **item} for item in data["domains"][domain]["train_samples"])
    return sorted(
        records,
        key=lambda item: (item["domain"], int(item["frame"]), -1 if item.get("z") is None else int(item["z"])),
    )


def training_diameter(
    records: list[dict[str, Any]], array_cache: dict[str, np.ndarray]
) -> tuple[float, int]:
    values = []
    for record in records:
        path = record["mask"]
        if path not in array_cache:
            array_cache[path] = read_tiff(path)
        values.extend(equivalent_diameters(array_cache[path]).tolist())
    if not values:
        raise RuntimeError("cannot estimate linker diameter without training instances")
    return float(np.median(np.asarray(values, dtype=np.float64))), len(values)


# Reuse the already exercised epoch-50 training implementation with the native
# dataset/records and formal protocol identity.
base.PROTOCOL_ID = PROTOCOL_ID
base.CTC2DTrainDataset = CTCNativeTrainDataset
base.training_records = training_records


def _file_record(path: Path, include_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()
    record = {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if include_hash:
        record["sha256"] = sha256(path)
    return record


def prepare_campaign(campaign: Path, data_manifest_path: Path) -> Path:
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / ".manifest.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _prepare_campaign_locked(campaign, data_manifest_path)


def _prepare_campaign_locked(campaign: Path, data_manifest_path: Path) -> Path:
    manifest_path = campaign / "campaign_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text())
        if payload.get("status") != "LOCKED_BEFORE_RUN" or payload.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError(f"incompatible campaign manifest: {manifest_path}")
        return manifest_path
    evaluator_commit = base.ctcmetrics_commit()
    if evaluator_commit != base.PINNED_CTCMETRICS_COMMIT:
        raise RuntimeError(
            f"py-ctcmetrics commit drift: {evaluator_commit} != {base.PINNED_CTCMETRICS_COMMIT}"
        )
    base.verify_imagecodecs_vendor()
    data = load_cache_manifest(data_manifest_path)
    rows = []
    for candidate in candidates():
        checkpoint = Path(candidate["checkpoint_path"])
        config = Path(candidate["config_path"])
        if not config.is_file():
            raise FileNotFoundError(config)
        rows.append({
            **candidate,
            "available_at_lock": checkpoint.is_file(),
            "checkpoint": _file_record(checkpoint) if checkpoint.is_file() else {
                "path": str(checkpoint), "status": "PENDING_RESTORE"
            },
            "config": _file_record(config),
        })
    code_paths = [
        Path(__file__),
        ROOT / "dinov3/eval/bio_tracking/ctc_native.py",
        ROOT / "dinov3/eval/bio_tracking/ctc_linker.py",
        ROOT / "dinov3/eval/bio_tracking/ctc_metrics_requested.py",
        ROOT / "scripts/run_ctc_native_2d_observation.py",
        PROTOCOL,
        PLAN,
    ]
    payload = {
        "status": "LOCKED_BEFORE_RUN",
        "admission": "FORMAL_PENDING_COMPLETE",
        "protocol_id": PROTOCOL_ID,
        "created_unix": time.time(),
        "created_host": platform.node(),
        "candidate_selection": "12 precommitted rows in ctc_native_hs6_candidates_20260911.md",
        "models": rows,
        "data_manifest": _file_record(data_manifest_path),
        "source_split_manifest_sha256": data["source_split_manifest_sha256"],
        "folds": data["folds"],
        "code": [_file_record(path) for path in code_paths],
        "head": {
            "architecture": "DINOHoVerNet binary final block",
            "feature_size": 32, "embed_proj": 384, "backbone": "frozen",
            "epochs": base.EPOCHS, "batch_size": base.TRAIN_BATCH_SIZE,
            "optimizer": "AdamW", "learning_rate": base.LEARNING_RATE,
            "weight_decay": base.WEIGHT_DECAY, "seed": base.SEED,
            "selection": "epoch_50_no_validation",
            "volume_annotation_sampling": "one deterministic foreground z page per complete SEG volume per epoch",
        },
        "inference": {
            "native_geometry": True, "crop_size": CROP_SIZE, "stride": STRIDE,
            "tile_batch_size": TILE_BATCH_SIZE, "tta": False,
            "np_threshold": 0.5, "hv_energy_threshold": 0.4,
            "minimum_area_or_volume": 10,
            "3d": "slice-wise logits/watershed, 26-connected instance consolidation, volume-size filtering",
            "input_scaling": "exact per-plane p01/p99 then microscopy RGB normalization",
        },
        "scoring": {
            "metrics": ["DET", "SEG", "TRA", "instance_mDice", "detection_AP"],
            "aggregation": "unweighted macro over 20 domains",
            "py_ctcmetrics_commit": evaluator_commit,
        },
    }
    atomic_json(manifest_path, payload)
    return manifest_path


def _materialize_candidate(candidate: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    checkpoint = Path(candidate["checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"precommitted checkpoint is not restored on the shared filesystem: {checkpoint}"
        )
    actual = _file_record(checkpoint)
    locked = candidate["checkpoint"]
    if locked.get("sha256") and locked["sha256"] != actual["sha256"]:
        raise RuntimeError(f"checkpoint hash drift: {checkpoint}")
    record = {
        "status": "MATERIALIZED", "model": candidate["model"],
        "checkpoint": actual, "config": candidate["config"], "layers": candidate["layers"],
    }
    atomic_json(model_dir / "model_input_manifest.json", record)
    return actual


def _remove_small_volumes(mask: np.ndarray, minimum: int = 10) -> np.ndarray:
    counts = np.bincount(np.asarray(mask, dtype=np.int64).reshape(-1))
    keep = np.flatnonzero(counts >= minimum)
    keep = keep[keep > 0]
    lookup = np.zeros(len(counts), dtype=np.int32)
    lookup[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    return lookup[np.asarray(mask, dtype=np.int64)]


def _consolidate_slices_3d(slices: list[np.ndarray]) -> np.ndarray:
    """26-connected consolidation with a vectorized final relabel pass."""
    if not slices:
        raise ValueError("at least one 3-D slice is required")
    shape = np.asarray(slices[0]).shape
    if len(shape) != 2 or any(np.asarray(item).shape != shape for item in slices):
        raise ValueError("3-D slices must have one common planar shape")
    volume = np.zeros((len(slices), *shape), dtype=np.int32)
    next_label = 1
    for z, item in enumerate(slices):
        array = np.asarray(item, dtype=np.int64)
        ids = np.unique(array)
        ids = ids[ids > 0]
        if not len(ids):
            continue
        lookup = np.zeros(int(array.max()) + 1, dtype=np.int32)
        lookup[ids] = np.arange(next_label, next_label + len(ids), dtype=np.int32)
        volume[z] = lookup[array]
        next_label += len(ids)
    parent = np.arange(next_label, dtype=np.int32)

    def find(value: int) -> int:
        while int(parent[value]) != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    height, width = shape
    for z in range(1, len(slices)):
        previous, current = volume[z - 1], volume[z]
        for dy in (-1, 0, 1):
            py0, py1 = max(0, -dy), min(height, height - dy)
            cy0, cy1 = max(0, dy), min(height, height + dy)
            for dx in (-1, 0, 1):
                px0, px1 = max(0, -dx), min(width, width - dx)
                cx0, cx1 = max(0, dx), min(width, width + dx)
                left = previous[py0:py1, px0:px1]
                right = current[cy0:cy1, cx0:cx1]
                valid = (left > 0) & (right > 0)
                if valid.any():
                    for first, second in np.unique(
                        np.stack((left[valid], right[valid]), axis=1), axis=0
                    ):
                        union(int(first), int(second))
    roots = np.zeros(next_label, dtype=np.int32)
    for value in range(1, next_label):
        roots[value] = find(value)
    unique_roots = np.unique(roots[1:])
    root_labels = np.zeros(next_label, dtype=np.int32)
    root_labels[unique_roots] = np.arange(1, len(unique_roots) + 1, dtype=np.int32)
    labels = root_labels[roots]
    return labels[volume]


@torch.inference_mode()
def _infer_plane(model, plane: np.ndarray, device: torch.device, min_size: int) -> np.ndarray:
    image = normalized_image_tensor(plane).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = sliding_window_predict(
            model, image, crop_size=CROP_SIZE, stride=STRIDE,
            patch_size=int(model.backbone.patch_size), num_types=0, tta=False,
            blend_mode="uniform", tile_batch_size=TILE_BATCH_SIZE,
        )
    instances, _ = postprocess(
        output["np"], output["hv"], None,
        fg_thresh=0.5, energy_thresh=0.4, min_size=min_size,
    )
    return instances.astype(np.int32, copy=False)


@torch.inference_mode()
def _infer_frame(model, image_path: str, ndim: int, device: torch.device) -> np.ndarray:
    if ndim == 2:
        image = read_tiff(image_path)
        if image.ndim != 2:
            raise ValueError(f"expected 2-D image, got {image.shape}: {image_path}")
        return _infer_plane(model, image, device, 10)
    slices = []
    with tifffile.TiffFile(image_path) as tif:
        if len(tif.pages) <= 1:
            volume = np.asarray(tif.asarray())
            if volume.ndim != 3:
                raise ValueError(f"expected 3-D volume, got {volume.shape}: {image_path}")
            iterable = volume
        else:
            iterable = (page.asarray() for page in tif.pages)
        for plane in iterable:
            slices.append(_infer_plane(model, np.asarray(plane), device, 1))
    return _remove_small_volumes(_consolidate_slices_3d(slices), minimum=10)


def _extra_segmentation(
    prediction: np.ndarray, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        gt = read_tiff(record["mask"]).astype(np.int32, copy=False)
        z = record.get("z")
        pred = prediction[int(z)] if z is not None else prediction
        if pred.shape != gt.shape:
            raise RuntimeError(f"SEG shape mismatch: {pred.shape} != {gt.shape} for {record}")
        ap = compute_ap(pred, gt)
        pred_fg, gt_fg = pred > 0, gt > 0
        dice = float(
            2.0 * np.logical_and(pred_fg, gt_fg).sum()
            / (pred_fg.sum() + gt_fg.sum() + 1.0e-8)
        )
        rows.append({"frame": int(record["frame"]), "z": z, "Dice": dice, **ap})
    return rows


def _replace_directory(working: Path, destination: Path) -> None:
    if destination.exists():
        stale = destination.with_name(f"{destination.name}.invalid.{int(time.time())}.{os.getpid()}")
        os.replace(destination, stale)
    os.replace(working, destination)


def _valid_domain(path: Path, manifest_sha: str, head_sha: str, frames: int) -> dict[str, Any] | None:
    try:
        row = json.loads((path / "result.json").read_text())
        valid = (
            row.get("status") == "VALID_COMPLETE"
            and row.get("protocol_id") == PROTOCOL_ID
            and row.get("campaign_manifest_sha256") == manifest_sha
            and row.get("head_sha256") == head_sha
            and int(row.get("test_frames")) == frames
            and len(list((path / "02_RES").glob("mask*.tif"))) == frames
            and int(row["ctc_metrics"]["Valid"]) == 1
        )
        return row if valid else None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


@torch.inference_mode()
def evaluate_domain(
    model, data: dict[str, Any], domain: str, fold: int, fold_dir: Path,
    diameter: float, diameter_instances: int, manifest_sha: str, head_sha: str,
    device: torch.device,
) -> dict[str, Any]:
    spec = data["domains"][domain]
    frames = spec["test_frames"]
    domain_dir = fold_dir / "domains" / domain
    existing = _valid_domain(domain_dir, manifest_sha, head_sha, len(frames))
    if existing is not None:
        print(f"[fold {fold}] reuse domain {domain}", flush=True)
        return existing
    working = domain_dir.with_name(f".{domain}.{os.getpid()}.tmp")
    if working.exists():
        raise RuntimeError(f"stale active-looking temporary directory: {working}")
    result_dir = working / "02_RES"
    result_dir.mkdir(parents=True)
    linker = TrackLinker(diameter=diameter)
    seg_by_frame: dict[int, list[dict[str, Any]]] = {}
    for record in spec["test_segmentation"]:
        seg_by_frame.setdefault(int(record["frame"]), []).append(record)
    segmentation_rows = []
    started = time.perf_counter()
    model.eval()
    for index, frame in enumerate(frames, start=1):
        local = _infer_frame(model, frame["image"], int(spec["ndim"]), device)
        tracked = linker.step(local)
        if int(tracked.max(initial=0)) > np.iinfo(np.uint16).max:
            raise RuntimeError(f"{domain} exceeded CTC uint16 track-label limit")
        tifffile.imwrite(
            result_dir / f"mask{frame['token']}.tif",
            tracked.astype(np.uint16), compression="zlib",
        )
        segmentation_rows.extend(_extra_segmentation(local, seg_by_frame.get(int(frame["frame"]), [])))
        if index == 1 or index % 25 == 0 or index == len(frames):
            print(
                f"[fold {fold}] {domain} frame={index}/{len(frames)} "
                f"tracks={linker.next_track_id - 1}", flush=True,
            )
        del local, tracked
    table = linker.track_table()
    np.savetxt(result_dir / "res_track.txt", table, fmt="%d")
    metrics, diagnostics = score_requested_metrics(
        result_dir, Path(spec["test_gt_dir"]), PY_CTCMETRICS, threads=CTCMETRICS_THREADS
    )
    if int(metrics.get("Valid", 0)) != 1:
        raise RuntimeError(f"invalid official CTC output for {domain}: {metrics}")
    if len(segmentation_rows) != len(spec["test_segmentation"]):
        raise RuntimeError(f"did not score every SEG annotation for {domain}")
    extras = {
        "instance_mDice": float(np.mean([row["Dice"] for row in segmentation_rows])),
        "detection_AP": float(np.mean([row["AP"] for row in segmentation_rows])),
        "AP50": float(np.mean([row["AP50"] for row in segmentation_rows])),
        "AP75": float(np.mean([row["AP75"] for row in segmentation_rows])),
    }
    row = {
        "status": "VALID_COMPLETE", "admission": "FORMAL_PENDING_COMPLETE",
        "protocol_id": PROTOCOL_ID, "campaign_manifest_sha256": manifest_sha,
        "head_sha256": head_sha, "fold": fold, "domain": domain,
        "ndim": int(spec["ndim"]), "host": platform.node(),
        "test_frames": len(frames), "seg_annotations": len(segmentation_rows),
        "tracks": len(table), "training_only_median_diameter": diameter,
        "training_instances_for_diameter": diameter_instances,
        "ctc_metrics": metrics, "ctc_scoring": diagnostics,
        "extra_segmentation_metrics": extras,
        "segmentation_annotation_metrics": segmentation_rows,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(working / "result.json", row)
    _replace_directory(working, domain_dir)
    return row


def _macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    result = {
        key: float(np.mean([float(row["ctc_metrics"][key]) for row in rows]))
        for key in ("DET", "SEG", "TRA")
    }
    result.update({
        key: float(np.mean([float(row["extra_segmentation_metrics"][key]) for row in rows]))
        for key in ("instance_mDice", "detection_AP", "AP50", "AP75")
    })
    return result


def run_candidate(
    campaign: Path, manifest_path: Path, data_path: Path,
    candidate: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    manifest_sha = sha256(manifest_path)
    data = load_cache_manifest(data_path)
    model_dir = campaign / "models" / candidate["model"]
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _materialize_candidate(candidate, model_dir)
    candidate["checkpoint"] = checkpoint
    base.set_determinism()
    model = build_dino_hovernet(
        checkpoint=checkpoint["path"], train_config=candidate["config_path"],
        layers=candidate["layers"], num_types=0, freeze_backbone=True,
        feature_size=32, embed_proj=384, fusion_mode="bucket_concat",
        decoder_variant="current", device=device,
    )
    initial_decoder = {
        name: tensor.detach().cpu().clone() for name, tensor in model.decoder.state_dict().items()
    }
    array_cache: dict[str, np.ndarray] = {}
    domain_rows, fold_rows = [], []
    started = time.perf_counter()
    for fold in range(5):
        fold_dir = model_dir / "folds" / f"fold{fold}"
        head = base.train_or_load_head(
            model, initial_decoder, data, candidate, fold, fold_dir,
            manifest_sha, device, array_cache,
        )
        records = training_records(data, fold)
        diameter, n_diameter = training_diameter(records, array_cache)
        current = [
            evaluate_domain(
                model, data, domain, fold, fold_dir, diameter, n_diameter,
                manifest_sha, head["head_sha256"], device,
            )
            for domain in fold_record(data, fold)["test_domains"]
        ]
        fold_payload = {
            "status": "VALID_COMPLETE", "protocol_id": PROTOCOL_ID,
            "campaign_manifest_sha256": manifest_sha, "model": candidate["model"],
            "fold": fold, "head_sha256": head["head_sha256"],
            "train_domains": fold_record(data, fold)["train_domains"],
            "test_domains": fold_record(data, fold)["test_domains"],
            "train_samples": len(records), "domain_rows": current, "macro": _macro(current),
        }
        atomic_json(fold_dir / "result.json", fold_payload)
        fold_rows.append(fold_payload)
        domain_rows.extend(current)
    if len(domain_rows) != 20 or {row["domain"] for row in domain_rows} != set(CTC_DOMAINS):
        raise RuntimeError("candidate does not contain exactly one result for all 20 CTC domains")
    result = {
        "status": "VALID_COMPLETE", "admission": "FORMAL_PENDING_CAMPAIGN_VALIDATION",
        "protocol_id": PROTOCOL_ID, "campaign_manifest_sha256": manifest_sha,
        "model": candidate["model"], "family": candidate["family"],
        "checkpoint_step": candidate["checkpoint_step"],
        "checkpoint_sha256": checkpoint["sha256"], "host": platform.node(),
        "folds": fold_rows, "domain_rows": domain_rows, "macro": _macro(domain_rows),
        "seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
    }
    atomic_json(model_dir / "results.json", result)
    return result


def validate(campaign: Path) -> dict[str, Any]:
    with (campaign / ".validation.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _validate_locked(campaign)


def _validate_locked(campaign: Path) -> dict[str, Any]:
    manifest_path = campaign / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256(manifest_path)
    rows, errors = [], []
    for candidate in manifest["models"]:
        path = campaign / "models" / candidate["model"] / "results.json"
        try:
            result = json.loads(path.read_text())
            domains = result["domain_rows"]
            valid = (
                result.get("status") == "VALID_COMPLETE"
                and result.get("protocol_id") == PROTOCOL_ID
                and result.get("campaign_manifest_sha256") == manifest_sha
                and len(result.get("folds", [])) == 5
                and len(domains) == 20
                and {row["domain"] for row in domains} == set(CTC_DOMAINS)
                and all(int(row["ctc_metrics"]["Valid"]) == 1 for row in domains)
            )
            if not valid:
                raise ValueError("completeness/provenance mismatch")
            rows.append({"model": candidate["model"], "checkpoint": candidate["checkpoint_step"], **result["macro"]})
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{candidate['model']}: {error}")
    if rows:
        path = campaign / "checkpoint_metrics.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    report = {
        "status": "VALID_COMPLETE" if len(rows) == 12 and not errors else "INVALID_INCOMPLETE",
        "admission": "FORMAL" if len(rows) == 12 and not errors else "FORMAL_PENDING_COMPLETE",
        "protocol_id": PROTOCOL_ID, "campaign_manifest_sha256": manifest_sha,
        "expected_models": 12, "valid_models": len(rows),
        "expected_folds_per_model": 5, "expected_domains_per_model": 20,
        "errors": errors,
    }
    atomic_json(campaign / "validation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--model-index", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cache_manifest = prepare_cache(args.cache.resolve(), ARCHIVES, SPLIT)
    manifest_path = prepare_campaign(args.campaign.resolve(), cache_manifest)
    if args.prepare_only:
        return
    if args.validate_only:
        raise SystemExit(0 if validate(args.campaign.resolve())["status"] == "VALID_COMPLETE" else 1)
    if args.model_index is None or not 0 <= args.model_index < 12:
        parser.error("--model-index must be in [0, 11]")
    if not torch.cuda.is_available():
        raise RuntimeError("native CTC evaluation requires CUDA")
    manifest = json.loads(manifest_path.read_text())
    candidate = manifest["models"][args.model_index]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    try:
        run_candidate(args.campaign.resolve(), manifest_path, cache_manifest, candidate, device)
    except Exception as error:
        atomic_json(args.campaign.resolve() / "models" / candidate["model"] / "failure.json", {
            "status": "FAILED", "model": candidate["model"], "protocol_id": PROTOCOL_ID,
            "campaign_manifest_sha256": sha256(manifest_path), "host": platform.node(),
            "error_type": type(error).__name__, "error": str(error), "failed_unix": time.time(),
        })
        raise
    validate(args.campaign.resolve())


if __name__ == "__main__":
    main()
