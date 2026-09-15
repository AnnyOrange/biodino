#!/usr/bin/env python3
"""Run the precommitted native 2-D CTC observation on four HS6-L checkpoints."""

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
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
IMAGECODECS_VENDOR = ROOT / "outputs/02_eval_runtime/imagecodecs_py311_wheel_2026.3.6"
if str(IMAGECODECS_VENDOR) not in sys.path:
    sys.path.insert(0, str(IMAGECODECS_VENDOR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dinov3.eval.bio_segmentation.instance_seg.losses import HoVerNetLoss  # noqa: E402
from dinov3.eval.bio_segmentation.instance_seg.model import build_dino_hovernet  # noqa: E402
from dinov3.eval.bio_segmentation.instance_seg.postproc import postprocess  # noqa: E402
from dinov3.eval.bio_segmentation.instance_seg.tiling import sliding_window_predict  # noqa: E402
from dinov3.eval.bio_segmentation.metrics import compute_ap  # noqa: E402
from dinov3.eval.bio_tracking.ctc_2d import (  # noqa: E402
    CTC2DTrainDataset,
    PROTOCOL_ID,
    atomic_json,
    load_cache_manifest,
    normalized_image_tensor,
    prepare_cache,
    read_2d_tiff,
    sha256,
    verify_cache_content,
)
from dinov3.eval.bio_tracking.ctc_linker import TrackLinker, equivalent_diameters  # noqa: E402
from dinov3.eval.bio_tracking.ctc_metrics_requested import score_requested_metrics  # noqa: E402


DEFAULT_CAMPAIGN = ROOT / "outputs/02_eval_runs/ctc_native_2d_l5_candidates_observation_v2_20260911"
DEFAULT_CACHE = ROOT / "outputs/02_eval_inputs/observational_v1/ctc_native_2d"
ARCHIVES = Path(
    "/mnt/huawei_deepcad/benchmark/external_benchmarks_20260901/"
    "CellTrackingChallenge/training_zips"
)
FORMAL_SPLIT = ROOT / "outputs/02_eval_inputs/formal_v3/ctc/split_manifest.jsonl"
PY_CTCMETRICS = ROOT / "outputs/02_eval_runtime/py-ctcmetrics"
PINNED_CTCMETRICS_COMMIT = "59481c48a62d4376fe34bed3e3606b4ec4d60972"
IMAGECODECS_WHEEL = ROOT / (
    "outputs/02_eval_runtime/imagecodecs_wheel/"
    "imagecodecs-2026.3.6-cp311-abi3-manylinux_2_28_x86_64.whl"
)
IMAGECODECS_WHEEL_SHA256 = "e30a14aa2e1c6c90e00375292726486c1d90bf003b1414d608ea4d1f62fd8a79"
PLAN = ROOT / (
    "Evaluation Rules/plans/ctc_native_2d_l5_candidates_observation_v2_20260911.md"
)
TRAIN_RUN = ROOT / (
    "outputs/01_training_runs/"
    "HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907"
)
CANDIDATE_STEPS = (15615, 17079, 20007, 21959)
LAYERS = [23]
CROP_SIZE = 256
STRIDE = 192
TRAIN_BATCH_SIZE = 8
TILE_BATCH_SIZE = 8
EPOCHS = 50
SEED = 0
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
CTCMETRICS_THREADS = 16


def checkpoint_path(step: int) -> Path:
    return TRAIN_RUN / "eval" / f"training_{step}" / "teacher_checkpoint.pth"


def file_record(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    record: dict[str, Any] = {
        "path": str(resolved),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        record["sha256"] = sha256(resolved)
    return record


def git_head() -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return process.stdout.strip() or None


def ctcmetrics_commit() -> str:
    process = subprocess.run(
        [
            "git", "-c", f"safe.directory={PY_CTCMETRICS}",
            "-C", str(PY_CTCMETRICS), "rev-parse", "HEAD",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(f"cannot inspect py-ctcmetrics: {process.stderr.strip()}")
    return process.stdout.strip()


def verify_imagecodecs_vendor() -> None:
    if sha256(IMAGECODECS_WHEEL) != IMAGECODECS_WHEEL_SHA256:
        raise RuntimeError("imagecodecs wheel failed its pinned SHA256 check")
    with zipfile.ZipFile(IMAGECODECS_WHEEL) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            extracted = IMAGECODECS_VENDOR / member.filename
            if not extracted.is_file() or extracted.stat().st_size != member.file_size:
                raise RuntimeError(f"missing or truncated imagecodecs wheel member: {extracted}")
            extracted_digest = hashlib.sha256(extracted.read_bytes()).digest()
            archive_digest = hashlib.sha256(archive.read(member)).digest()
            if extracted_digest != archive_digest:
                raise RuntimeError(f"imagecodecs wheel extraction drift: {extracted}")


def prepare_campaign_manifest(campaign: Path, cache_manifest_path: Path) -> Path:
    manifest_path = campaign / "campaign_manifest.json"
    campaign.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        if payload.get("status") != "LOCKED_BEFORE_RUN" or payload.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError(f"refusing incompatible campaign manifest: {manifest_path}")
        return manifest_path

    commit = ctcmetrics_commit()
    if commit != PINNED_CTCMETRICS_COMMIT:
        raise RuntimeError(
            f"py-ctcmetrics commit drift: {commit} != {PINNED_CTCMETRICS_COMMIT}"
        )
    verify_imagecodecs_vendor()
    config = TRAIN_RUN / "config.yaml"
    code_paths = [
        Path(__file__),
        ROOT / "dinov3/eval/bio_tracking/ctc_2d.py",
        ROOT / "dinov3/eval/bio_tracking/ctc_linker.py",
        ROOT / "dinov3/eval/bio_tracking/ctc_metrics_requested.py",
        ROOT / "dinov3/eval/bio_segmentation/constants.py",
        ROOT / "dinov3/eval/bio_segmentation/model_utils.py",
        ROOT / "dinov3/eval/bio_segmentation/instance_seg/model.py",
        ROOT / "dinov3/eval/bio_segmentation/instance_seg/decoder.py",
        ROOT / "dinov3/eval/bio_segmentation/instance_seg/losses.py",
        ROOT / "dinov3/eval/bio_segmentation/instance_seg/targets.py",
        ROOT / "dinov3/eval/bio_segmentation/instance_seg/tiling.py",
        ROOT / "dinov3/eval/bio_segmentation/instance_seg/postproc.py",
        ROOT / "dinov3/eval/bio_segmentation/metrics/instance.py",
        PLAN,
    ]
    required = [config, cache_manifest_path, IMAGECODECS_WHEEL, *code_paths]
    required.extend(checkpoint_path(step) for step in CANDIDATE_STEPS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing CTC campaign inputs:\n" + "\n".join(missing))

    data_manifest = load_cache_manifest(cache_manifest_path)
    candidates = []
    for index, step in enumerate(CANDIDATE_STEPS, start=1):
        path = checkpoint_path(step)
        print(f"[manifest] hashing checkpoint {index}/{len(CANDIDATE_STEPS)}: {step}", flush=True)
        candidates.append({
            "model": f"hs6_l_5tb_ck{step}",
            "checkpoint_step": step,
            "checkpoint": file_record(path),
        })
    fold_records = []
    for fold in data_manifest["folds"]:
        train_samples = sum(
            len(data_manifest["domains"][domain]["train_samples"])
            for domain in fold["train_domains"]
        )
        test_frames = sum(
            len(data_manifest["domains"][domain]["test_frames"])
            for domain in fold["test_domains"]
        )
        fold_records.append({**fold, "train_samples": train_samples, "test_frames": test_frames})

    payload = {
        "campaign": campaign.name,
        "status": "LOCKED_BEFORE_RUN",
        "admission": "OBSERVATIONAL_NATIVE_2D",
        "created_unix": time.time(),
        "created_host": platform.node(),
        "git_head": git_head(),
        "protocol_id": PROTOCOL_ID,
        "teacher_branch": "teacher",
        "candidate_selection": "precommitted L5 subset from ctc_native_hs6_candidates_20260911.md",
        "models": candidates,
        "config": file_record(config),
        "data_manifest": file_record(cache_manifest_path),
        "source_split_manifest_sha256": data_manifest["source_split_manifest_sha256"],
        "folds": fold_records,
        "code": [file_record(path) for path in code_paths],
        "py_ctcmetrics": {
            "path": str(PY_CTCMETRICS.resolve()),
            "commit": commit,
        },
        "imagecodecs": {
            "version": "2026.3.6",
            "vendor_path": str(IMAGECODECS_VENDOR.resolve()),
            "wheel": file_record(IMAGECODECS_WHEEL),
            "expected_wheel_sha256": IMAGECODECS_WHEEL_SHA256,
        },
        "head": {
            "architecture": "DINOHoVerNet binary current bucket_concat",
            "layers": LAYERS,
            "feature_size": 32,
            "embed_proj": 384,
            "backbone": "frozen",
            "epochs": EPOCHS,
            "batch_size": TRAIN_BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": None,
            "amp": "bf16",
            "seed": SEED,
            "drop_last": False,
            "selection": "epoch_50_no_validation",
            "sample_unit": "one foreground-aware 256 crop per available seq01 SEG frame per epoch",
            "crop_rng": "SeedSequence([seed, zero_based_epoch, sorted_sample_index])",
            "augmentation": "uniform D4 (rot90 k in 0..3, horizontal flip in 0..1)",
            "input_scaling": "exact full-source-image p01/p99 then clip to [0,1]",
            "channel_mapping": "repeat grayscale to RGB",
            "normalization_mean": [0.511375, 0.598449, 0.683452],
            "normalization_std": [0.340017, 0.306132, 0.284308],
        },
        "inference": {
            "native_geometry": True,
            "crop_size": CROP_SIZE,
            "stride": STRIDE,
            "overlap": CROP_SIZE - STRIDE,
            "tile_batch_size": TILE_BATCH_SIZE,
            "blend": "uniform_logit_mean",
            "tta": False,
            "np_threshold": 0.5,
            "hv_energy_threshold": 0.4,
            "minimum_area": 10,
            "input_scaling": "exact per-image p01/p99 then microscopy RGB normalization",
        },
        "linker": {
            "cost": "0.7*(1-IoU)+0.3*normalized_centroid_distance",
            "distance_normalizer": "fold-training-only median equivalent instance diameter",
            "forbid": "IoU=0 and normalized centroid distance>1",
            "division_overlap_fraction": 0.1,
        },
        "scoring": {
            "native": ["Valid", "DET", "SEG", "TRA"],
            "extra": ["mean_foreground_dice", "AP", "AP50", "AP75"],
            "aggregation": "unweighted macro over 10 domains",
            "implementation": "pinned_primitives_requested_only_indexed_parent_lookup_v1",
            "mask_matching_threads": CTCMETRICS_THREADS,
            "merged_track_products": "not computed because requested metrics do not consume them",
        },
    }
    atomic_json(manifest_path, payload)
    print(f"[manifest] locked {manifest_path} sha256={sha256(manifest_path)}", flush=True)
    return manifest_path


def _verify_file_record(record: dict[str, Any], *, hash_content: bool) -> None:
    path = Path(record["path"])
    stat = path.stat()
    if stat.st_size != int(record["bytes"]) or stat.st_mtime_ns != int(record["mtime_ns"]):
        raise RuntimeError(f"campaign input metadata drift: {path}")
    if hash_content and sha256(path) != record["sha256"]:
        raise RuntimeError(f"campaign input hash drift: {path}")


def verify_worker_inputs(manifest: dict[str, Any], candidate: dict[str, Any]) -> None:
    if ctcmetrics_commit() != manifest["py_ctcmetrics"]["commit"]:
        raise RuntimeError("py-ctcmetrics commit changed after campaign lock")
    verify_imagecodecs_vendor()
    _verify_file_record(manifest["config"], hash_content=True)
    _verify_file_record(manifest["data_manifest"], hash_content=True)
    for record in manifest["code"]:
        _verify_file_record(record, hash_content=True)
    _verify_file_record(manifest["imagecodecs"]["wheel"], hash_content=True)
    if manifest["imagecodecs"]["wheel"]["sha256"] != IMAGECODECS_WHEEL_SHA256:
        raise RuntimeError("imagecodecs wheel hash differs from the audited v2 wheel")
    _verify_file_record(candidate["checkpoint"], hash_content=True)
    verify_cache_content(load_cache_manifest(Path(manifest["data_manifest"]["path"])))


def set_determinism(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fold_record(data_manifest: dict[str, Any], fold: int) -> dict[str, Any]:
    records = [item for item in data_manifest["folds"] if int(item["fold"]) == fold]
    if len(records) != 1:
        raise RuntimeError(f"missing or duplicate fold {fold}")
    return records[0]


def training_records(data_manifest: dict[str, Any], fold: int) -> list[dict[str, Any]]:
    spec = fold_record(data_manifest, fold)
    records = []
    for domain in spec["train_domains"]:
        records.extend({"domain": domain, **item} for item in data_manifest["domains"][domain]["train_samples"])
    return sorted(records, key=lambda item: (item["domain"], int(item["frame"])))


def training_diameter(
    records: list[dict[str, Any]], array_cache: dict[str, np.ndarray]
) -> tuple[float, int]:
    values = []
    for record in records:
        path = record["mask"]
        if path not in array_cache:
            array_cache[path] = read_2d_tiff(path)
        values.extend(equivalent_diameters(array_cache[path]).tolist())
    if not values:
        raise RuntimeError("cannot estimate linker diameter without training instances")
    diameter = float(np.median(np.asarray(values, dtype=np.float64)))
    if not math.isfinite(diameter) or diameter <= 0:
        raise RuntimeError(f"invalid training-only linker diameter: {diameter}")
    return diameter, len(values)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _valid_head(
    head_path: Path,
    metadata_path: Path,
    manifest_sha: str,
    candidate: dict[str, Any],
    fold: int,
) -> dict[str, Any] | None:
    try:
        metadata = json.loads(metadata_path.read_text())
        valid = (
            metadata.get("status") == "VALID_EPOCH_50"
            and metadata.get("protocol_id") == PROTOCOL_ID
            and metadata.get("campaign_manifest_sha256") == manifest_sha
            and metadata.get("model") == candidate["model"]
            and int(metadata.get("fold")) == fold
            and metadata.get("head_sha256") == sha256(head_path)
            and int(metadata.get("epochs")) == EPOCHS
        )
        return metadata if valid else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def train_or_load_head(
    model,
    initial_decoder_state: dict[str, torch.Tensor],
    data_manifest: dict[str, Any],
    candidate: dict[str, Any],
    fold: int,
    fold_dir: Path,
    manifest_sha: str,
    device: torch.device,
    array_cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    head_path = fold_dir / "head_epoch50.pth"
    metadata_path = fold_dir / "head_metadata.json"
    existing = _valid_head(head_path, metadata_path, manifest_sha, candidate, fold)
    if existing is not None:
        payload = torch.load(head_path, map_location="cpu", weights_only=True)
        model.decoder.load_state_dict(payload["decoder"])
        print(f"[fold {fold}] reuse epoch-50 head", flush=True)
        return existing

    set_determinism()
    model.decoder.load_state_dict(initial_decoder_state)
    records = training_records(data_manifest, fold)
    dataset = CTC2DTrainDataset(records, crop_size=CROP_SIZE, seed=SEED, array_cache=array_cache)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )
    criterion = HoVerNetLoss(num_types=0).to(device)
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    log_path = fold_dir / "train_epoch50.jsonl"
    fold_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    final_loss = float("nan")
    with log_path.open("w") as log:
        for epoch in range(1, EPOCHS + 1):
            dataset.set_epoch(epoch - 1)
            model.train()
            loss_sum = 0.0
            batches = 0
            epoch_started = time.perf_counter()
            for image, np_target, hv_target, tp_target in loader:
                image = image.to(device, non_blocking=True)
                target = {
                    "np": np_target.to(device, non_blocking=True),
                    "hv": hv_target.to(device, non_blocking=True),
                    "tp": tp_target.to(device, non_blocking=True),
                }
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = model(image)
                    loss, _ = criterion(prediction, target)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at fold={fold} epoch={epoch}")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach())
                batches += 1
            final_loss = loss_sum / max(1, batches)
            row = {
                "epoch": epoch,
                "loss": final_loss,
                "batches": batches,
                "seconds": time.perf_counter() - epoch_started,
            }
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()
            if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
                print(
                    f"[fold {fold}] epoch={epoch:02d}/{EPOCHS} loss={final_loss:.5f} "
                    f"seconds={row['seconds']:.1f}",
                    flush=True,
                )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    state = {name: tensor.detach().cpu() for name, tensor in model.decoder.state_dict().items()}
    _atomic_torch_save({"decoder": state}, head_path)
    metadata = {
        "status": "VALID_EPOCH_50",
        "protocol_id": PROTOCOL_ID,
        "campaign_manifest_sha256": manifest_sha,
        "model": candidate["model"],
        "checkpoint_sha256": candidate["checkpoint"]["sha256"],
        "fold": fold,
        "epochs": EPOCHS,
        "samples": len(records),
        "batches_per_epoch": len(loader),
        "final_loss": final_loss,
        "training_seconds": time.perf_counter() - started,
        "head_sha256": sha256(head_path),
        "head_bytes": head_path.stat().st_size,
        "host": platform.node(),
    }
    atomic_json(metadata_path, metadata)
    return metadata


def _json_scalars(payload: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in payload.items():
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def _score_ctc(
    result_dir: Path, gt_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics, diagnostics = score_requested_metrics(
        result_dir,
        gt_dir,
        PY_CTCMETRICS,
        threads=CTCMETRICS_THREADS,
    )
    return _json_scalars(metrics), _json_scalars(diagnostics)


def _valid_domain_result(
    domain_dir: Path,
    manifest_sha: str,
    head_sha: str,
    domain: str,
    expected_frames: int,
) -> dict[str, Any] | None:
    try:
        payload = json.loads((domain_dir / "result.json").read_text())
        masks = list((domain_dir / "02_RES").glob("mask*.tif"))
        metrics = payload["ctc_metrics"]
        extras = payload["extra_segmentation_metrics"]
        valid = (
            payload.get("status") == "VALID_COMPLETE"
            and payload.get("protocol_id") == PROTOCOL_ID
            and payload.get("campaign_manifest_sha256") == manifest_sha
            and payload.get("head_sha256") == head_sha
            and payload.get("domain") == domain
            and int(payload.get("test_frames")) == expected_frames
            and len(masks) == expected_frames
            and (domain_dir / "02_RES/res_track.txt").is_file()
            and int(metrics.get("Valid")) == 1
            and all(math.isfinite(float(metrics[key])) for key in ("DET", "SEG", "TRA"))
            and all(
                math.isfinite(float(extras[key]))
                for key in ("mean_foreground_dice", "AP", "AP50", "AP75")
            )
        )
        return payload if valid else None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _replace_directory(working: Path, destination: Path) -> None:
    if destination.exists():
        stale = destination.with_name(f"{destination.name}.invalid.{int(time.time())}.{os.getpid()}")
        os.replace(destination, stale)
    os.replace(working, destination)


@torch.inference_mode()
def evaluate_domain(
    model,
    data_manifest: dict[str, Any],
    domain: str,
    fold: int,
    fold_dir: Path,
    diameter: float,
    diameter_instances: int,
    manifest_sha: str,
    head_sha: str,
    device: torch.device,
) -> dict[str, Any]:
    domain_data = data_manifest["domains"][domain]
    frames = domain_data["test_frames"]
    domain_dir = fold_dir / "domains" / domain
    existing = _valid_domain_result(domain_dir, manifest_sha, head_sha, domain, len(frames))
    if existing is not None:
        print(f"[fold {fold}] reuse domain {domain}", flush=True)
        return existing

    working = domain_dir.with_name(f".{domain}.{os.getpid()}.tmp")
    if working.exists():
        shutil.rmtree(working)
    result_dir = working / "02_RES"
    result_dir.mkdir(parents=True)
    linker = TrackLinker(diameter=diameter)
    segmentation_rows = []
    started = time.perf_counter()
    model.eval()
    for index, frame in enumerate(frames, start=1):
        image = normalized_image_tensor(read_2d_tiff(frame["image"])).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = sliding_window_predict(
                model,
                image,
                crop_size=CROP_SIZE,
                stride=STRIDE,
                patch_size=int(model.backbone.patch_size),
                num_types=0,
                tta=False,
                blend_mode="uniform",
                tile_batch_size=TILE_BATCH_SIZE,
            )
        local_instances, _ = postprocess(
            output["np"],
            output["hv"],
            None,
            fg_thresh=0.5,
            energy_thresh=0.4,
            min_size=10,
        )
        tracked = linker.step(local_instances)
        maximum_track = int(tracked.max(initial=0))
        if maximum_track > np.iinfo(np.uint16).max:
            raise RuntimeError(f"{domain} exceeded the CTC uint16 track-label limit")
        output_path = result_dir / f"mask{frame['token']}.tif"
        tifffile.imwrite(output_path, tracked.astype(np.uint16), compression="zlib")

        segmentation_path = domain_data["test_segmentation"].get(str(frame["frame"]))
        if segmentation_path is not None:
            ground_truth = read_2d_tiff(segmentation_path).astype(np.int32, copy=False)
            if ground_truth.shape != local_instances.shape:
                raise RuntimeError(f"SEG shape mismatch for {domain} frame {frame['frame']}")
            ap = compute_ap(local_instances, ground_truth)
            pred_fg, gt_fg = local_instances > 0, ground_truth > 0
            dice = float(
                2.0 * np.logical_and(pred_fg, gt_fg).sum()
                / (pred_fg.sum() + gt_fg.sum() + 1.0e-8)
            )
            segmentation_rows.append({"frame": int(frame["frame"]), "Dice": dice, **ap})
        if index == 1 or index % 50 == 0 or index == len(frames):
            print(
                f"[fold {fold}] {domain} frame={index}/{len(frames)} "
                f"tracks={linker.next_track_id - 1}",
                flush=True,
            )
        del image

    table = linker.track_table()
    np.savetxt(result_dir / "res_track.txt", table, fmt="%d")
    metrics, scoring_diagnostics = _score_ctc(
        result_dir, Path(domain_data["test_gt_dir"])
    )
    if int(metrics.get("Valid", 0)) != 1:
        raise RuntimeError(f"invalid CTC output for {domain}: {metrics}")
    if len(segmentation_rows) != len(domain_data["test_segmentation"]):
        raise RuntimeError(f"did not score every SEG-annotated frame for {domain}")
    extras = {
        "mean_foreground_dice": float(np.mean([row["Dice"] for row in segmentation_rows])),
        "AP": float(np.mean([row["AP"] for row in segmentation_rows])),
        "AP50": float(np.mean([row["AP50"] for row in segmentation_rows])),
        "AP75": float(np.mean([row["AP75"] for row in segmentation_rows])),
    }
    payload = {
        "status": "VALID_COMPLETE",
        "admission": "OBSERVATIONAL_NATIVE_2D",
        "protocol_id": PROTOCOL_ID,
        "campaign_manifest_sha256": manifest_sha,
        "head_sha256": head_sha,
        "fold": fold,
        "domain": domain,
        "host": platform.node(),
        "test_frames": len(frames),
        "seg_annotated_frames": len(segmentation_rows),
        "tracks": len(table),
        "maximum_track_id": int(table[:, 0].max()) if len(table) else 0,
        "training_only_median_diameter": diameter,
        "training_instances_for_diameter": diameter_instances,
        "ctc_metrics": metrics,
        "ctc_scoring": scoring_diagnostics,
        "extra_segmentation_metrics": extras,
        "segmentation_frame_metrics": segmentation_rows,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(working / "result.json", payload)
    _replace_directory(working, domain_dir)
    return payload


def _macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    result = {}
    for key in ("DET", "SEG", "TRA"):
        result[key] = float(np.mean([float(row["ctc_metrics"][key]) for row in rows]))
    for key in ("mean_foreground_dice", "AP", "AP50", "AP75"):
        result[key] = float(np.mean([float(row["extra_segmentation_metrics"][key]) for row in rows]))
    return result


def run_candidate(
    campaign: Path,
    manifest_path: Path,
    cache_manifest_path: Path,
    candidate: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256(manifest_path)
    verify_worker_inputs(manifest, candidate)
    data_manifest = load_cache_manifest(cache_manifest_path)
    model_dir = campaign / "models" / candidate["model"]
    model_dir.mkdir(parents=True, exist_ok=True)

    set_determinism()
    model = build_dino_hovernet(
        checkpoint=candidate["checkpoint"]["path"],
        train_config=manifest["config"]["path"],
        layers=LAYERS,
        num_types=0,
        freeze_backbone=True,
        feature_size=32,
        embed_proj=384,
        fusion_mode="bucket_concat",
        decoder_variant="current",
        device=device,
    )
    initial_decoder_state = {
        name: tensor.detach().cpu().clone() for name, tensor in model.decoder.state_dict().items()
    }
    array_cache: dict[str, np.ndarray] = {}
    domain_rows = []
    fold_rows = []
    started = time.perf_counter()
    for fold in range(5):
        fold_dir = model_dir / "folds" / f"fold{fold}"
        head = train_or_load_head(
            model,
            initial_decoder_state,
            data_manifest,
            candidate,
            fold,
            fold_dir,
            manifest_sha,
            device,
            array_cache,
        )
        records = training_records(data_manifest, fold)
        diameter, diameter_instances = training_diameter(records, array_cache)
        current_rows = []
        for domain in fold_record(data_manifest, fold)["test_domains"]:
            current_rows.append(evaluate_domain(
                model,
                data_manifest,
                domain,
                fold,
                fold_dir,
                diameter,
                diameter_instances,
                manifest_sha,
                head["head_sha256"],
                device,
            ))
        fold_payload = {
            "status": "VALID_COMPLETE",
            "protocol_id": PROTOCOL_ID,
            "campaign_manifest_sha256": manifest_sha,
            "model": candidate["model"],
            "fold": fold,
            "head_sha256": head["head_sha256"],
            "train_domains": fold_record(data_manifest, fold)["train_domains"],
            "test_domains": fold_record(data_manifest, fold)["test_domains"],
            "train_samples": len(records),
            "training_only_median_diameter": diameter,
            "training_instances_for_diameter": diameter_instances,
            "domain_rows": current_rows,
            "macro": _macro(current_rows),
        }
        atomic_json(fold_dir / "result.json", fold_payload)
        fold_rows.append(fold_payload)
        domain_rows.extend(current_rows)

    if len(domain_rows) != 10 or {row["domain"] for row in domain_rows} != set(data_manifest["domains"]):
        raise RuntimeError("candidate does not contain exactly one result for each 2-D CTC domain")
    result = {
        "status": "VALID_COMPLETE",
        "admission": "OBSERVATIONAL_NATIVE_2D",
        "protocol_id": PROTOCOL_ID,
        "campaign_manifest_sha256": manifest_sha,
        "model": candidate["model"],
        "checkpoint_step": candidate["checkpoint_step"],
        "checkpoint_sha256": candidate["checkpoint"]["sha256"],
        "host": platform.node(),
        "folds": fold_rows,
        "domain_rows": domain_rows,
        "macro": _macro(domain_rows),
        "seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
    }
    atomic_json(model_dir / "results.json", result)
    return result


def _valid_candidate_result(
    path: Path, manifest_sha: str, candidate: dict[str, Any]
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
        rows = payload["domain_rows"]
        valid = (
            payload.get("status") == "VALID_COMPLETE"
            and payload.get("admission") == "OBSERVATIONAL_NATIVE_2D"
            and payload.get("protocol_id") == PROTOCOL_ID
            and payload.get("campaign_manifest_sha256") == manifest_sha
            and payload.get("model") == candidate["model"]
            and payload.get("checkpoint_sha256") == candidate["checkpoint"]["sha256"]
            and len(payload.get("folds", [])) == 5
            and len(rows) == 10
            and len({row["domain"] for row in rows}) == 10
            and all(row.get("status") == "VALID_COMPLETE" for row in rows)
            and all(row.get("campaign_manifest_sha256") == manifest_sha for row in rows)
            and all(int(row["ctc_metrics"]["Valid"]) == 1 for row in rows)
            and all(
                math.isfinite(float(row["ctc_metrics"][key]))
                for row in rows for key in ("DET", "SEG", "TRA")
            )
            and all(
                math.isfinite(float(row["extra_segmentation_metrics"][key]))
                for row in rows
                for key in ("mean_foreground_dice", "AP", "AP50", "AP75")
            )
        )
        return payload if valid else None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def validate_campaign(campaign: Path) -> dict[str, Any]:
    lock_path = campaign / ".validation.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest_path = campaign / "campaign_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest_sha = sha256(manifest_path)
        rows = []
        errors = []
        for candidate in manifest["models"]:
            path = campaign / "models" / candidate["model"] / "results.json"
            result = _valid_candidate_result(path, manifest_sha, candidate)
            if result is None:
                errors.append(f"invalid or missing result: {candidate['model']}")
                continue
            rows.append({
                "checkpoint": candidate["checkpoint_step"],
                **result["macro"],
            })
        if rows:
            csv_path = campaign / "checkpoint_metrics.csv"
            temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
            with temporary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda item: item["checkpoint"]))
            os.replace(temporary, csv_path)
        report = {
            "status": "VALID_COMPLETE" if not errors and len(rows) == len(CANDIDATE_STEPS) else "INVALID_INCOMPLETE",
            "admission": "OBSERVATIONAL_NATIVE_2D",
            "protocol_id": PROTOCOL_ID,
            "campaign_manifest_sha256": manifest_sha,
            "expected_models": len(CANDIDATE_STEPS),
            "valid_models": len(rows),
            "expected_folds_per_model": 5,
            "expected_domains_per_model": 10,
            "errors": errors,
        }
        atomic_json(campaign / "validation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--archives", type=Path, default=ARCHIVES)
    parser.add_argument("--split-manifest", type=Path, default=FORMAL_SPLIT)
    parser.add_argument("--model-index", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    args.cache = args.cache.resolve()

    cache_manifest = prepare_cache(args.cache, args.archives, args.split_manifest)
    manifest_path = prepare_campaign_manifest(args.campaign, cache_manifest)
    if args.prepare_only:
        return
    if args.validate_only:
        raise SystemExit(0 if validate_campaign(args.campaign)["status"] == "VALID_COMPLETE" else 1)
    if args.model_index is None or not 0 <= args.model_index < len(CANDIDATE_STEPS):
        parser.error(f"--model-index must be in [0, {len(CANDIDATE_STEPS) - 1}]")
    if not torch.cuda.is_available():
        raise RuntimeError("CTC native observation requires CUDA")
    manifest = json.loads(manifest_path.read_text())
    candidate = manifest["models"][args.model_index]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    try:
        run_candidate(args.campaign, manifest_path, cache_manifest, candidate, device)
    except Exception as error:
        failure = {
            "status": "FAILED",
            "model": candidate["model"],
            "protocol_id": PROTOCOL_ID,
            "campaign_manifest_sha256": sha256(manifest_path),
            "host": platform.node(),
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_unix": time.time(),
        }
        atomic_json(args.campaign / "models" / candidate["model"] / "failure.json", failure)
        raise
    validate_campaign(args.campaign)


if __name__ == "__main__":
    main()
