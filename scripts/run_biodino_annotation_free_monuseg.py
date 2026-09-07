#!/usr/bin/env python3
"""Strict annotation-free MoNuSeg instance segmentation with BioDINO tokens.

The prediction phase reads only TEST images.  Ground-truth XML files are parsed
only after prediction masks, method configs, and hashes have been written to the
output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label as cc_label
from skimage.segmentation import watershed
from tqdm import tqdm

from dinov3.eval.bio_segmentation.constants import MICRO_RGB_MEAN, MICRO_RGB_STD
from dinov3.eval.bio_segmentation.datasets.monuseg import _parse_xml_to_instance_map
from dinov3.eval.bio_segmentation.metrics.instance import (
    compute_aji,
    compute_object_ap,
    compute_pq,
    compute_seg,
)
from dinov3.eval.bio_segmentation.model_utils import load_dinov3_backbone
from dinov3.utils.bio_io import _normalize_to_float32, read_bio_image_as_numpy


DEFAULT_OUT = Path("/mnt/huawei_deepcad/dinov3/outputs/instance_seg_tuning/biodino_annotation_free_monuseg")
DEFAULT_DATA = Path("/mnt/huawei_deepcad/benchmark/segmentation/monuseg/extracted/MoNuSegTestData")
DEFAULT_CKPT = Path("/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/bio_continue_rgb3_vith16plus/ckpt/14349")
DEFAULT_CFG = Path("/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/bio_continue_rgb3_vith16plus/config.yaml")
LAYERS = [7, 15, 23, 31]
THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]


@dataclass(frozen=True)
class MethodSpec:
    key: str
    name: str
    layers: str
    affinity_source: str
    use_multispectral: bool
    use_feature_boundary: bool


METHODS = [
    MethodSpec(
        key="a_layer31_only",
        name="BioDINO-ML-Spectral-Watershed A: layer31-only",
        layers="31",
        affinity_source="layer31",
        use_multispectral=False,
        use_feature_boundary=False,
    ),
    MethodSpec(
        key="b_four_layer_fusion",
        name="BioDINO-ML-Spectral-Watershed B: four-layer affinity fusion",
        layers="7,15,23,31",
        affinity_source="four_layer_equal_affinity_fusion",
        use_multispectral=False,
        use_feature_boundary=False,
    ),
    MethodSpec(
        key="c_four_layer_multispectral_boundary_ws",
        name="BioDINO-ML-Spectral-Watershed C: fusion+multi-spectral+feature-boundary watershed",
        layers="7,15,23,31",
        affinity_source="four_layer_equal_affinity_fusion",
        use_multispectral=True,
        use_feature_boundary=True,
    ),
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_log(out: Path, event: Dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {"time": now(), **event}
    with (out / "run_status.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def list_test_images(data_dir: Path) -> List[Path]:
    images = sorted(
        p for p in data_dir.glob("*.tif")
        if p.is_file() and not p.name.startswith(".")
    )
    if not images:
        raise FileNotFoundError(f"No MoNuSeg TEST .tif images found under {data_dir}")
    return images


def wait_for_gpu(device: str, min_free_mb: int, poll_seconds: int, out: Path) -> None:
    if not device.startswith("cuda") or min_free_mb <= 0:
        return
    gpu_idx = device.split(":", 1)[1] if ":" in device else "0"
    while True:
        try:
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={gpu_idx}",
                    "--query-gpu=memory.free,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            free_mb, used_mb, util = [int(x.strip()) for x in raw.split(",")[:3]]
        except Exception as exc:
            append_log(out, {"stage": "gpu-wait-query-failed", "device": device, "error": repr(exc)})
            time.sleep(poll_seconds)
            continue
        if free_mb >= min_free_mb:
            append_log(out, {"stage": "gpu-ready", "device": device, "free_mb": free_mb, "used_mb": used_mb, "utilization_gpu": util})
            return
        append_log(out, {"stage": "gpu-wait", "device": device, "free_mb": free_mb, "used_mb": used_mb, "required_free_mb": min_free_mb, "utilization_gpu": util})
        time.sleep(poll_seconds)


def load_rgb01(path: Path) -> np.ndarray:
    img = read_bio_image_as_numpy(str(path), target_channels=3, normalize=False)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    return _normalize_to_float32(img)


def prep_tensor(rgb01: np.ndarray, work_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[np.ndarray, torch.Tensor]:
    resized = cv2.resize(rgb01, (work_size, work_size), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(resized).permute(2, 0, 1).float()
    mean = torch.tensor(MICRO_RGB_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(MICRO_RGB_STD, dtype=torch.float32).view(3, 1, 1)
    x = ((x - mean) / std).unsqueeze(0).to(device=device, dtype=dtype)
    return resized, x


def normalize_feature_map(feat: torch.Tensor) -> np.ndarray:
    # feat: [1, C, h, w]. Apply an extra per-layer LayerNorm over channels and L2 normalize tokens.
    f = feat[0].permute(1, 2, 0).contiguous().float()
    f = F.layer_norm(f, (f.shape[-1],))
    flat = F.normalize(f.reshape(-1, f.shape[-1]), p=2, dim=1)
    return flat.cpu().numpy().astype(np.float32)


def extract_layer_features(model: torch.nn.Module, x: torch.Tensor, layers: Sequence[int]) -> Tuple[Dict[int, np.ndarray], Tuple[int, int], Dict[str, Any]]:
    with torch.inference_mode():
        outs = model.get_intermediate_layers(x, n=list(layers), reshape=True, return_class_token=False, norm=True)
    feats: Dict[int, np.ndarray] = {}
    shapes: Dict[str, Any] = {}
    h_patch = w_patch = -1
    for layer, out in zip(layers, outs):
        h_patch, w_patch = int(out.shape[-2]), int(out.shape[-1])
        feats[int(layer)] = normalize_feature_map(out)
        shapes[str(layer)] = {
            "model_output_shape_after_cls_register_removal": list(out.shape),
            "token_matrix_shape": list(feats[int(layer)].shape),
        }
    return feats, (h_patch, w_patch), shapes


def fused_affinity(layer_feats: Sequence[np.ndarray], topk: int) -> np.ndarray:
    mats = []
    for feat in layer_feats:
        sim = (feat @ feat.T + 1.0) * 0.5
        np.fill_diagonal(sim, 0.0)
        if 0 < topk < sim.shape[0]:
            kth = np.partition(sim, -topk, axis=1)[:, -topk]
            keep = sim >= kth[:, None]
            sim = np.where(keep, sim, 0.0)
        sim = np.maximum(sim, sim.T)
        mats.append(sim.astype(np.float32, copy=False))
    W = np.mean(mats, axis=0).astype(np.float32)
    np.fill_diagonal(W, 0.0)
    return W


def spectral_decomposition(W: np.ndarray, max_nontrivial: int) -> Dict[str, Any]:
    degree = W.sum(axis=1).astype(np.float64)
    inv_sqrt = np.zeros_like(degree)
    valid = degree > 1e-8
    inv_sqrt[valid] = 1.0 / np.sqrt(degree[valid])
    S = (W.astype(np.float64) * inv_sqrt[:, None]) * inv_sqrt[None, :]
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    available = max(1, min(max_nontrivial, vecs.shape[1] - 1))
    nontriv_vals = vals[1 : available + 2]
    if len(nontriv_vals) >= 2:
        gaps = nontriv_vals[:-1] - nontriv_vals[1:]
        k = int(np.argmax(gaps) + 1)
    else:
        gaps = np.asarray([], dtype=np.float64)
        k = 1
    k = max(1, min(k, available))
    return {
        "eigenvalues": vals[: available + 2].astype(float).tolist(),
        "eigengaps": gaps.astype(float).tolist(),
        "selected_nontrivial_count": k,
        "vectors": vecs[:, 1 : available + 1].astype(np.float32),
    }


def normalize01(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def otsu_or_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or float(values.max() - values.min()) < 1e-6:
        return float(np.median(values)) if values.size else 0.5
    try:
        return float(threshold_otsu(values))
    except Exception:
        return float(np.median(values))


def choose_foreground(binary: np.ndarray, rgb_work: np.ndarray) -> np.ndarray:
    gray = 0.299 * rgb_work[..., 0] + 0.587 * rgb_work[..., 1] + 0.114 * rgb_work[..., 2]
    darkness = 1.0 - gray
    candidates = [binary.astype(bool), ~binary.astype(bool)]
    scored = []
    for idx, cand in enumerate(candidates):
        frac = float(cand.mean())
        if cand.any():
            dark = float(darkness[cand].mean())
        else:
            dark = -1.0
        valid = 0.005 <= frac <= 0.75
        border = np.concatenate([cand[0], cand[-1], cand[:, 0], cand[:, -1]])
        border_frac = float(border.mean())
        score = (2.0 if valid else 0.0) + dark - 0.25 * border_frac - 0.15 * abs(frac - 0.22)
        scored.append((score, idx, frac))
    best = max(scored, key=lambda x: x[0])[1]
    return candidates[best]


def cleanup_binary(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = ndi.binary_fill_holes(mask.astype(bool))
    lab, n = ndi.label(mask)
    if n == 0:
        return mask.astype(bool)
    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, n + 1):
        comp = lab == i
        if int(comp.sum()) >= min_area:
            out |= comp
    return out


def foreground_from_vectors(vectors: np.ndarray, h_patch: int, w_patch: int, rgb_work: np.ndarray, multi: bool, work_size: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if multi:
        oriented_maps = []
        used = vectors.shape[1]
        for j in range(used):
            raw = vectors[:, j].reshape(h_patch, w_patch)
            thr = otsu_or_median(raw)
            pos = cv2.resize((raw > thr).astype(np.uint8), (work_size, work_size), interpolation=cv2.INTER_NEAREST).astype(bool)
            chosen = choose_foreground(pos, rgb_work)
            sign = 1.0 if np.array_equal(chosen, pos) else -1.0
            oriented_maps.append(normalize01(sign * cv2.resize(raw, (work_size, work_size), interpolation=cv2.INTER_CUBIC)))
        score = np.mean(oriented_maps, axis=0).astype(np.float32)
        thr = otsu_or_median(score)
        fg = choose_foreground(score > thr, rgb_work)
        meta = {"foreground_rule": "eigengap_selected_oriented_spectral_maps_plus_otsu", "spectral_maps_used": int(used), "otsu_threshold": float(thr)}
    else:
        raw = vectors[:, 0].reshape(h_patch, w_patch)
        thr = otsu_or_median(raw)
        coarse = cv2.resize((raw > thr).astype(np.uint8), (work_size, work_size), interpolation=cv2.INTER_NEAREST).astype(bool)
        fg = choose_foreground(coarse, rgb_work)
        sign = 1.0 if np.array_equal(fg, coarse) else -1.0
        score = normalize01(sign * cv2.resize(raw, (work_size, work_size), interpolation=cv2.INTER_CUBIC))
        meta = {"foreground_rule": "fiedler_vector_otsu_with_input_darkness_orientation", "spectral_maps_used": 1, "otsu_threshold": float(thr)}
    fg = cleanup_binary(fg, min_area=max(4, int(round(work_size * work_size * 0.00002))))
    return fg, score, meta


def feature_boundary(feats: Dict[int, np.ndarray], h_patch: int, w_patch: int, work_size: int) -> np.ndarray:
    maps = []
    for layer in (7, 15):
        f = feats[layer].reshape(h_patch, w_patch, -1)
        gy = np.zeros((h_patch, w_patch), dtype=np.float32)
        gx = np.zeros((h_patch, w_patch), dtype=np.float32)
        gy[1:, :] = np.linalg.norm(f[1:, :, :] - f[:-1, :, :], axis=2)
        gx[:, 1:] = np.linalg.norm(f[:, 1:, :] - f[:, :-1, :], axis=2)
        maps.append(normalize01(np.maximum(gx, gy)))
    b = np.mean(maps, axis=0).astype(np.float32)
    b = cv2.resize(b, (work_size, work_size), interpolation=cv2.INTER_CUBIC)
    b = cv2.GaussianBlur(normalize01(b), (0, 0), sigmaX=1.0)
    return normalize01(b)


def estimate_seed_distance(fg: np.ndarray) -> int:
    lab, n = ndi.label(fg)
    if n == 0:
        return 4
    areas = np.asarray(ndi.sum(np.ones_like(fg, dtype=np.float32), lab, index=np.arange(1, n + 1)))
    areas = areas[areas >= 4]
    if areas.size == 0:
        return 4
    radius = np.sqrt(float(np.median(areas)) / np.pi)
    return int(np.clip(round(radius * 0.65), 3, 10))


def watershed_instances(fg: np.ndarray, boundary: np.ndarray, use_boundary: bool, min_area: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    fg = cleanup_binary(fg, min_area=min_area)
    if not fg.any():
        return np.zeros_like(fg, dtype=np.int32), np.zeros_like(fg, dtype=np.int32), np.zeros_like(fg, dtype=np.float32), {"seed_count": 0, "fg_fraction": 0.0}
    dist = ndi.distance_transform_edt(fg).astype(np.float32)
    if use_boundary:
        surface = dist * (1.0 - 0.55 * np.clip(boundary, 0.0, 1.0))
        elevation = -surface + 0.35 * np.clip(boundary, 0.0, 1.0)
    else:
        surface = dist
        elevation = -dist
    min_distance = estimate_seed_distance(fg)
    vals = surface[fg]
    threshold_abs = max(1.0, otsu_or_median(vals) * 0.35)
    coords = peak_local_max(
        surface,
        labels=fg.astype(np.uint8),
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        exclude_border=False,
    )
    markers = np.zeros_like(fg, dtype=np.int32)
    for sid, (r, c) in enumerate(coords, start=1):
        markers[int(r), int(c)] = sid
    comp_lab, n_comp = ndi.label(fg)
    next_id = int(markers.max()) + 1
    for comp_id in range(1, n_comp + 1):
        comp = comp_lab == comp_id
        if not np.any(markers[comp]):
            idx = int(np.argmax(surface * comp))
            r, c = np.unravel_index(idx, surface.shape)
            markers[r, c] = next_id
            next_id += 1
    markers, _ = ndi.label(markers > 0)
    inst = watershed(elevation, markers=markers, mask=fg).astype(np.int32)
    out = np.zeros_like(inst, dtype=np.int32)
    oid = 1
    for pid in np.unique(inst):
        if pid == 0:
            continue
        comp = inst == pid
        if int(comp.sum()) >= min_area:
            out[comp] = oid
            oid += 1
    meta = {
        "seed_count": int(markers.max()),
        "fg_fraction": float(fg.mean()),
        "min_seed_distance": int(min_distance),
        "seed_threshold_abs": float(threshold_abs),
    }
    return out, markers.astype(np.int32), dist, meta


def resize_instance_nearest(inst: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return cv2.resize(inst.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)


def color_instances(inst: np.ndarray) -> np.ndarray:
    h, w = inst.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(inst)
    for pid in ids:
        if pid == 0:
            continue
        seed = int(pid) * 2654435761 % (2**32)
        rng = np.random.default_rng(seed)
        out[inst == pid] = rng.integers(40, 255, size=3, dtype=np.uint8)
    return out


def save_visual(path: Path, rgb: np.ndarray, fg: np.ndarray, boundary: np.ndarray, seeds: np.ndarray, inst: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    fg8 = np.stack([fg.astype(np.uint8) * 255] * 3, axis=-1)
    b8 = np.stack([np.clip(boundary * 255.0, 0, 255).astype(np.uint8)] * 3, axis=-1)
    seed_vis = rgb8.copy()
    ys, xs = np.nonzero(seeds > 0)
    for y, x in zip(ys, xs):
        cv2.circle(seed_vis, (int(x), int(y)), 2, (255, 0, 0), -1)
    inst_vis = color_instances(inst)
    panels = [rgb8, fg8, b8, seed_vis, inst_vis]
    labels = ["input", "foreground", "boundary", "seeds", "instances"]
    tile_h, tile_w = rgb8.shape[:2]
    canvas = Image.new("RGB", (tile_w * len(panels), tile_h + 28), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel), (i * tile_w, 28))
        draw.text((i * tile_w + 4, 8), labels[i], fill=(0, 0, 0))
    draw.text((4, 0), title[:120], fill=(0, 0, 0))
    canvas.save(path)


def run_methods_for_image(
    model: torch.nn.Module,
    image_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    rgb_native = load_rgb01(image_path)
    rgb_work, x = prep_tensor(rgb_native, args.work_size, device, dtype)
    feats, patch_hw, shape_meta = extract_layer_features(model, x, LAYERS)
    h_patch, w_patch = patch_hw
    aff31 = fused_affinity([feats[31]], args.affinity_topk)
    aff4 = fused_affinity([feats[layer] for layer in LAYERS], args.affinity_topk)
    spec31 = spectral_decomposition(aff31, max_nontrivial=args.max_nontrivial_eigs)
    spec4 = spectral_decomposition(aff4, max_nontrivial=args.max_nontrivial_eigs)
    boundary = feature_boundary(feats, h_patch, w_patch, args.work_size)

    outputs: Dict[str, Dict[str, Any]] = {}
    min_area = max(4, int(round(args.work_size * args.work_size * args.min_area_fraction)))
    for method in METHODS:
        spectral = spec31 if method.affinity_source == "layer31" else spec4
        k = int(spectral["selected_nontrivial_count"]) if method.use_multispectral else 1
        vectors = spectral["vectors"][:, :k]
        fg, fg_score, fg_meta = foreground_from_vectors(
            vectors,
            h_patch,
            w_patch,
            rgb_work,
            multi=method.use_multispectral,
            work_size=args.work_size,
        )
        bmap = boundary if method.use_feature_boundary else np.zeros_like(boundary)
        inst_work, seeds, dist, ws_meta = watershed_instances(
            fg,
            bmap,
            use_boundary=method.use_feature_boundary,
            min_area=min_area,
        )
        inst_native = resize_instance_nearest(inst_work, rgb_native.shape[:2])
        outputs[method.key] = {
            "instance_work": inst_work,
            "instance_native": inst_native,
            "foreground": fg,
            "foreground_score": fg_score,
            "boundary": bmap,
            "seeds": seeds,
            "meta": {
                "method": method.name,
                "layers": method.layers,
                "image": image_path.name,
                "native_shape": list(rgb_native.shape[:2]),
                "work_shape": [args.work_size, args.work_size],
                "patch_shape": [h_patch, w_patch],
                "instance_count": int(inst_native.max()),
                "foreground_fraction": float((inst_work > 0).mean()),
                "spectral": {k2: v for k2, v in spectral.items() if k2 != "vectors"},
                **fg_meta,
                **ws_meta,
            },
        }
    image_meta = {"feature_shapes": shape_meta, "patch_shape": [h_patch, w_patch]}
    return outputs, image_meta


def smoke_check(outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    report = {}
    failures = []
    for key, payload in outputs.items():
        inst = payload["instance_work"]
        frac = float((inst > 0).mean())
        n = int(inst.max())
        ok = n > 0 and 0.001 < frac < 0.95
        report[key] = {"instance_count": n, "foreground_fraction": frac, "ok": ok}
        if not ok:
            failures.append(key)
    return {"ok": not failures, "failures": failures, "methods": report}


def save_predictions(
    out: Path,
    image_path: Path,
    outputs: Dict[str, Dict[str, Any]],
    save_vis: bool,
    visual_index: int,
) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    stem = image_path.stem
    for key, payload in outputs.items():
        method_dir = out / "predictions" / key
        method_dir.mkdir(parents=True, exist_ok=True)
        mask_path = method_dir / f"{stem}.npz"
        np.savez_compressed(
            mask_path,
            instance_mask=payload["instance_native"].astype(np.int32),
            instance_mask_work=payload["instance_work"].astype(np.int32),
            foreground=payload["foreground"].astype(np.uint8),
            boundary=payload["boundary"].astype(np.float32),
            seeds=payload["seeds"].astype(np.int32),
        )
        meta_path = method_dir / f"{stem}.json"
        write_json(meta_path, payload["meta"])
        if save_vis:
            vis_path = out / "visualizations" / key / f"{visual_index:02d}_{stem}.png"
            rgb_work = cv2.resize(load_rgb01(image_path), payload["foreground"].shape[::-1], interpolation=cv2.INTER_LINEAR)
            save_visual(
                vis_path,
                rgb_work,
                payload["foreground"],
                payload["boundary"],
                payload["seeds"],
                payload["instance_work"],
                f"{key} | {image_path.name}",
            )
        records[key] = {
            "image": image_path.name,
            "mask_path": str(mask_path),
            "mask_sha256": sha256_path(mask_path),
            "meta_path": str(meta_path),
            "meta_sha256": sha256_path(meta_path),
            "instance_count": int(payload["instance_native"].max()),
        }
    return records


def parse_gt_for_image(image_path: Path) -> np.ndarray:
    xml_path = image_path.with_suffix(".xml")
    if not xml_path.is_file():
        raise FileNotFoundError(f"Missing TEST GT XML for final evaluation: {xml_path}")
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image size for GT parse: {image_path}")
    return _parse_xml_to_instance_map(str(xml_path), img.shape[0], img.shape[1])


def object_ap_full(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, Dict[str, float]]:
    values = {}
    pred_ids = np.unique(pred[pred > 0])
    gt_ids = np.unique(gt[gt > 0])
    if len(pred_ids) == 0 and len(gt_ids) == 0:
        values = {f"AP{int(t * 100)}": 1.0 for t in THRESHOLDS}
    elif len(pred_ids) == 0 or len(gt_ids) == 0:
        values = {f"AP{int(t * 100)}": 0.0 for t in THRESHOLDS}
    else:
        from dinov3.eval.bio_segmentation.metrics.instance import _pairwise_iou

        iou = _pairwise_iou(pred, gt, pred_ids, gt_ids)
        for t in THRESHOLDS:
            matched_gt = set()
            matched_pred = set()
            for flat_idx in np.argsort(-iou, axis=None):
                gi, pi = divmod(int(flat_idx), len(pred_ids))
                if iou[gi, pi] < t:
                    break
                if gi in matched_gt or pi in matched_pred:
                    continue
                matched_gt.add(gi)
                matched_pred.add(pi)
            tp = len(matched_gt)
            fp = len(pred_ids) - tp
            fn = len(gt_ids) - tp
            values[f"AP{int(t * 100)}"] = float(tp / (tp + fp + fn + 1e-8))
    return float(np.mean(list(values.values()))), values


def evaluate_locked_predictions(out: Path, images: Sequence[Path], protocol_hash: str) -> Dict[str, Any]:
    per_method_rows: Dict[str, List[Dict[str, Any]]] = {m.key: [] for m in METHODS}
    for image_path in tqdm(images, desc="final-eval-with-gt"):
        gt = parse_gt_for_image(image_path)
        for method in METHODS:
            pred_path = out / "predictions" / method.key / f"{image_path.stem}.npz"
            with np.load(pred_path) as data:
                pred = data["instance_mask"].astype(np.int32)
            pq = compute_pq(pred, gt)
            ap_mean, ap_vec = object_ap_full(pred, gt)
            row = {
                "method_key": method.key,
                "method": method.name,
                "image": image_path.name,
                "gt_instance_count": int(gt.max()),
                "pred_instance_count": int(pred.max()),
                "AJI": float(compute_aji(pred, gt)),
                "bPQ": float(pq["pq"]),
                "bSQ": float(pq["sq"]),
                "bDQ": float(pq["rq"]),
                "TP": int(pq["n_tp"]),
                "FP": int(pq["n_fp"]),
                "FN": int(pq["n_fn"]),
                "SEG": float(compute_seg(pred, gt)),
                "CellposeStyleAP": ap_mean,
                **ap_vec,
            }
            per_method_rows[method.key].append(row)

    summary = {}
    for method in METHODS:
        rows = per_method_rows[method.key]
        metrics = {}
        for name in ("AJI", "bPQ", "bSQ", "bDQ", "SEG", "CellposeStyleAP"):
            vals = np.asarray([r[name] for r in rows], dtype=np.float64)
            metrics[name] = float(np.nanmean(vals))
            metrics[f"{name}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        for name in ("pred_instance_count", "gt_instance_count", "TP", "FP", "FN"):
            vals = np.asarray([r[name] for r in rows], dtype=np.float64)
            metrics[name] = float(np.sum(vals)) if name in {"TP", "FP", "FN"} else float(np.mean(vals))
        metrics["failure_mode"] = (
            "miss/leak-detection dominated" if metrics["FN"] >= metrics["FP"] * 1.25
            else "over-segmentation/false-positive dominated" if metrics["FP"] >= metrics["FN"] * 1.25
            else "mixed merge/split errors"
        )
        summary[method.key] = {
            "method": method.name,
            "layers": method.layers,
            "protocol_hash": protocol_hash,
            "metrics": metrics,
            "per_image_json": str(out / "metrics" / f"{method.key}_per_image.json"),
        }
        write_json(out / "metrics" / f"{method.key}_per_image.json", rows)

    fields = ["method_key", "method", "image", "gt_instance_count", "pred_instance_count", "AJI", "bPQ", "bSQ", "bDQ", "TP", "FP", "FN", "SEG", "CellposeStyleAP"] + [f"AP{int(t * 100)}" for t in THRESHOLDS]
    with (out / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for method in METHODS:
            writer.writerows(per_method_rows[method.key])
    write_json(out / "metrics_summary.json", summary)
    return summary


def write_summary_files(out: Path, summary: Dict[str, Any], runtimes: Dict[str, float], peak_gib: float) -> None:
    fields = ["Method", "Layers", "AJI", "bPQ", "SEG", "CellposeStyleAP", "Instances", "RuntimeSeconds", "PeakMemoryGiB", "FailureMode"]
    with (out / "annotation_free_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, item in summary.items():
            m = item["metrics"]
            writer.writerow({
                "Method": item["method"],
                "Layers": item["layers"],
                "AJI": f"{m['AJI']:.9f}",
                "bPQ": f"{m['bPQ']:.9f}",
                "SEG": f"{m['SEG']:.9f}",
                "CellposeStyleAP": f"{m['CellposeStyleAP']:.9f}",
                "Instances": f"{m['pred_instance_count']:.3f}",
                "RuntimeSeconds": f"{runtimes.get(key, 0.0):.3f}",
                "PeakMemoryGiB": f"{peak_gib:.3f}",
                "FailureMode": m["failure_mode"],
            })

    lines = [
        "# BioDINO Annotation-Free MoNuSeg",
        "",
        "Protocol: BioDINO-ML-Spectral-Watershed; frozen ViT-H+/16 BioDINO checkpoint; no decoder, no SAM/Cellpose/CellSAM, no pseudo-label training, no fine-tuning.",
        "",
        "| Method | Layers | AJI | bPQ | SEG | CellposeStyleAP | Instances/image | Runtime(s) | Peak GPU GiB | Failure mode |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, item in summary.items():
        m = item["metrics"]
        lines.append(
            f"| {item['method']} | {item['layers']} | {m['AJI']:.6f} | {m['bPQ']:.6f} | {m['SEG']:.6f} | "
            f"{m['CellposeStyleAP']:.6f} | {m['pred_instance_count']:.1f} | {runtimes.get(key, 0.0):.1f} | {peak_gib:.3f} | {m['failure_mode']} |"
        )
    a = summary["a_layer31_only"]["metrics"]
    b = summary["b_four_layer_fusion"]["metrics"]
    c = summary["c_four_layer_multispectral_boundary_ws"]["metrics"]
    lines.extend([
        "",
        "Supervised decoder references are not annotation-free and are listed only for context: No-trick TEST AJI 0.594572, bPQ 0.522405; Fixed-trick TEST AJI 0.617761, bPQ 0.546376.",
        "",
        f"Four-layer fusion vs layer31-only: AJI delta {b['AJI'] - a['AJI']:+.6f}, bPQ delta {b['bPQ'] - a['bPQ']:+.6f}.",
        f"Full multi-spectral boundary watershed vs layer31-only: AJI delta {c['AJI'] - a['AJI']:+.6f}, bPQ delta {c['bPQ'] - a['bPQ']:+.6f}.",
        "",
        "Prediction-stage GT access: none. TEST XML parsing happens after `prediction_lock.json` is written.",
        "Gradient updates: none; all inference is under `torch.inference_mode()` with frozen BioDINO parameters.",
    ])
    (out / "annotation_free_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--train-config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--wait-gpu-free-mb", type=int, default=12000)
    p.add_argument("--gpu-poll-seconds", type=int, default=60)
    p.add_argument("--work-size", type=int, default=512)
    p.add_argument("--affinity-topk", type=int, default=64)
    p.add_argument("--max-nontrivial-eigs", type=int, default=6)
    p.add_argument("--min-area-fraction", type=float, default=0.00002)
    p.add_argument("--max-images", type=int, default=0, help="Debug cap only; 0 means full TEST.")
    args = p.parse_args()

    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    images = list_test_images(args.data_dir)
    if args.max_images:
        images = images[: args.max_images]

    protocol = {
        "created_at": now(),
        "host": socket.gethostname(),
        "method_family": "BioDINO-ML-Spectral-Watershed",
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "layers": LAYERS,
        "split": "MoNuSeg official TEST",
        "data_dir": str(args.data_dir),
        "image_count": len(images),
        "prediction_reads": "TEST .tif images only",
        "gt_read_policy": "TEST XML files are not parsed until prediction_lock.json has been written.",
        "training": "none",
        "decoder": "none",
        "sam_cellpose_cellsam": "not used",
        "parameter_selection": "fixed parameters plus image-only Otsu/eigengap/connected-component statistics; no train/val/test annotation tuning",
        "input_work_size": [args.work_size, args.work_size],
        "patch_size": 16,
        "affinity": {"metric": "cosine", "per_layer": "extra LayerNorm plus L2 normalize", "fusion": "equal mean", "topk_per_row": args.affinity_topk},
        "spectral": {"max_nontrivial_eigs": args.max_nontrivial_eigs, "count_rule": "largest eigengap among nontrivial normalized-affinity eigenvalues"},
        "watershed": {"seed_rule": "image-derived foreground distance local maxima plus one fallback seed per component", "min_area_fraction": args.min_area_fraction},
        "methods": [m.__dict__ for m in METHODS],
    }
    protocol_hash = sha256_json(protocol)
    protocol["protocol_sha256"] = protocol_hash
    write_json(out / "frozen_protocol.json", protocol)
    for method in METHODS:
        cfg = {**protocol, "selected_method": method.__dict__}
        cfg_path = out / "method_configs" / f"{method.key}.json"
        write_json(cfg_path, cfg)

    append_log(out, {"stage": "start", "pid": os.getpid(), "device": args.device, "image_count": len(images), "protocol_sha256": protocol_hash})
    wait_for_gpu(args.device, args.wait_gpu_free_mb, args.gpu_poll_seconds, out)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    append_log(out, {"stage": "load-backbone-start", "device": str(device), "dtype": str(dtype), "checkpoint": str(args.checkpoint)})
    model = load_dinov3_backbone(str(args.checkpoint), str(args.train_config), device=torch.device("cpu"), freeze=True)
    model = model.to(dtype=dtype).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    append_log(out, {"stage": "load-backbone-done", "embed_dim": int(getattr(model, "embed_dim", -1)), "patch_size": int(getattr(model, "patch_size", -1))})

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Feature shape check and single-image image-only smoke test.
    t0 = time.time()
    first_outputs, first_meta = run_methods_for_image(model, images[0], args, device, dtype)
    write_json(out / "feature_shape_check.json", {"image": images[0].name, **first_meta})
    smoke = smoke_check(first_outputs)
    write_json(out / "image_only_smoke_test.json", smoke)
    save_predictions(out / "smoke", images[0], first_outputs, save_vis=True, visual_index=0)
    append_log(out, {"stage": "smoke-done", "image": images[0].name, "elapsed_seconds": time.time() - t0, **smoke})
    if not smoke["ok"]:
        append_log(out, {"stage": "stopped-after-smoke", "reason": "empty/full/all-invalid image-only prediction"})
        return 2

    records: Dict[str, List[Dict[str, Any]]] = {m.key: [] for m in METHODS}
    runtimes: Dict[str, float] = {m.key: 0.0 for m in METHODS}
    append_log(out, {"stage": "full-prediction-start", "image_count": len(images)})
    for idx, image_path in enumerate(tqdm(images, desc="predict-without-gt")):
        start = time.time()
        outputs, meta = run_methods_for_image(model, image_path, args, device, dtype)
        elapsed = time.time() - start
        for key in runtimes:
            runtimes[key] += elapsed / len(METHODS)
        saved = save_predictions(out, image_path, outputs, save_vis=idx < 6, visual_index=idx)
        for key, rec in saved.items():
            rec["feature_shape"] = meta["patch_shape"]
            records[key].append(rec)
        append_log(out, {"stage": "predicted-image", "index": idx, "image": image_path.name, "elapsed_seconds": elapsed, "instances": {k: v["instance_count"] for k, v in saved.items()}})

    for method in METHODS:
        manifest = {
            "method": method.name,
            "method_key": method.key,
            "protocol_sha256": protocol_hash,
            "config_path": str(out / "method_configs" / f"{method.key}.json"),
            "config_sha256": sha256_path(out / "method_configs" / f"{method.key}.json"),
            "predictions": records[method.key],
            "prediction_count": len(records[method.key]),
        }
        write_json(out / "manifests" / f"{method.key}_prediction_manifest.json", manifest)

    lock = {
        "locked_at": now(),
        "protocol_sha256": protocol_hash,
        "prediction_phase_gt_read": False,
        "gt_read_allowed_after_this_file": True,
        "method_manifests": [str(out / "manifests" / f"{m.key}_prediction_manifest.json") for m in METHODS],
        "method_manifest_sha256": {m.key: sha256_path(out / "manifests" / f"{m.key}_prediction_manifest.json") for m in METHODS},
    }
    write_json(out / "prediction_lock.json", lock)
    append_log(out, {"stage": "prediction-lock-written", "prediction_phase_gt_read": False, "lock": str(out / "prediction_lock.json")})

    summary = evaluate_locked_predictions(out, images, protocol_hash)
    peak_gib = float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
    write_json(out / "runtime_and_memory.json", {"runtimes_seconds": runtimes, "peak_cuda_memory_gib": peak_gib, "device": str(device)})
    write_summary_files(out, summary, runtimes, peak_gib)
    append_log(out, {"stage": "complete", "summary_csv": str(out / "annotation_free_summary.csv"), "summary_md": str(out / "annotation_free_summary.md"), "peak_cuda_memory_gib": peak_gib})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
