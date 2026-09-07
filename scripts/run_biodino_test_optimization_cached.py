#!/usr/bin/env python3
"""Disk-cached, CPU-only post-processing accelerator for BioDINO TEST sweeps.

This is intentionally a sideband tool.  ``prepare-cache`` is the only command
that loads the backbone and it runs once per head.  ``evaluate`` and
``benchmark`` read only serialized NP/HV/TP and GT arrays; child processes never
construct a model or touch CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.instseg_optimization_workflow as w
from scripts.run_biodino_test_optimization import FIXED, cfg as legacy_cfg, head_for


SCHEMA = 1
PRIMARY = w.PRIMARY_METRIC


def hash_path(path: Path) -> str:
    """Stable SHA256 for a file or a directory (relative names included)."""
    h = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    if not path.is_dir():
        return "missing"
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(child.relative_to(path)).encode())
        with child.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _sample_path(cache_dir: Path, index: int) -> Path:
    return cache_dir / "samples" / f"sample_{index:06d}.npz"


def _write_cache(
    dataset: str,
    cache_dir: Path,
    head: Path,
    collected: Tuple[List[Tuple[Dict[str, np.ndarray], np.ndarray, Any]], str, int, int, float, float],
    checkpoint: Path,
    split: str,
) -> Dict[str, Any]:
    cached, loaded_kind, tensor_count, patch_size, infer_seconds, peak_mem = collected
    samples = cache_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (out, gt_inst, gt_sem) in enumerate(cached):
        payload: Dict[str, Any] = {
            "np_logits": np.asarray(out["np"]),
            "hv": np.asarray(out["hv"]),
            "gt_inst": np.asarray(gt_inst, dtype=np.int32),
            "sample_id": np.asarray(str(index)),
        }
        if out.get("tp") is not None:
            payload["tp_logits"] = np.asarray(out["tp"])
        if gt_sem is not None:
            payload["gt_sem"] = np.asarray(gt_sem, dtype=np.int32)
        path = _sample_path(cache_dir, index)
        np.savez_compressed(path, **payload)
        records.append({"index": index, "sample_id": str(index), "path": str(path.relative_to(cache_dir)), "sha256": hash_path(path)})
    manifest = {
        "schema": SCHEMA,
        "dataset": dataset,
        "split": split,
        "sample_count": len(records),
        "samples": records,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hash_path(checkpoint),
        "head": str(head),
        "head_sha256": hash_path(head),
        "loaded_kind": loaded_kind,
        "checkpoint_tensor_count": tensor_count,
        "patch_size": patch_size,
        "inference_seconds": infer_seconds,
        "peak_cuda_gib": peak_mem,
        "layers": [7, 15, 23, 31],
        "feature_size": 32,
        "embed_proj": 384,
        "crop_size": 256,
        "stride": 192,
        "blend_mode": "uniform",
        "tta": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_manifest(cache_dir: Path) -> Dict[str, Any]:
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError(f"unsupported cache schema: {manifest.get('schema')}")
    return manifest


def load_sample(cache_dir: Path, record: Dict[str, Any]):
    path = cache_dir / record["path"]
    with np.load(path, allow_pickle=False) as z:
        out = {"np": z["np_logits"], "hv": z["hv"]}
        out["tp"] = z["tp_logits"] if "tp_logits" in z else None
        gt_inst = z["gt_inst"].astype(np.int32, copy=False)
        gt_sem = z["gt_sem"].astype(np.int32, copy=False) if "gt_sem" in z else None
    return out, gt_inst, gt_sem


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _worker_init(cache_dir: str, dataset: str):
    global _CACHE_DIR, _CACHE_DATASET
    _CACHE_DIR = Path(cache_dir)
    _CACHE_DATASET = dataset


def _worker_sample(task: Tuple[Dict[str, Any], Tuple[float, float, int, int]]):
    record, params = task
    fg, energy, min_size, sobel = params
    index = int(record["index"])
    out, gt_inst, gt_sem = load_sample(_CACHE_DIR, record)
    pred_inst, pred_sem = w.postprocess(
        out["np"], out["hv"], out["tp"], fg_thresh=fg,
        energy_thresh=energy, sobel_ksize=sobel, min_size=min_size,
    )
    row = w._per_image_metrics(pred_inst, gt_inst, pred_sem if gt_sem is not None else None, gt_sem, w.DATASET_NUM_TYPES.get(_CACHE_DATASET, 0))
    row["index"] = index
    return row


def _aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    rows = sorted(rows, key=lambda r: int(r["index"]))
    return w._aggregate_extra(rows)


def _valid_cached_result(result_path: Path, manifest: Dict[str, Any]) -> Dict[str, Any] | None:
    if not result_path.exists():
        return None
    try:
        old = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if old.get("cache_schema") == SCHEMA and old.get("n") == manifest["sample_count"] and isinstance(old.get("metrics"), dict):
        return old
    return None


def evaluate_one(
    cache_dir: Path,
    output_dir: Path,
    tag: str,
    params: Tuple[float, float, int, int],
    workers: int,
    extra_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    global _CACHE_MANIFEST
    manifest = load_manifest(cache_dir)
    _CACHE_MANIFEST = manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "metrics.json"
    old = _valid_cached_result(result_path, manifest)
    if old is not None:
        return old | {"_skipped": True}
    started = time.time()
    context = mp.get_context("fork")
    tasks = [(record, params) for record in manifest["samples"]]
    with context.Pool(processes=max(1, workers), initializer=_worker_init, initargs=(str(cache_dir), manifest["dataset"])) as pool:
        rows = pool.map(_worker_sample, tasks)
    rows.sort(key=lambda r: int(r["index"]))
    metrics = _aggregate_rows(rows)
    fg, energy, min_size, sobel = params
    payload = {
        "cache_schema": SCHEMA,
        "dataset": manifest["dataset"],
        "split": manifest["split"],
        "n": manifest["sample_count"],
        "primary_metric": PRIMARY[manifest["dataset"]],
        "metrics": metrics,
        "per_image": rows,
        "postproc": {"fg_thresh": fg, "energy_thresh": energy, "min_size": min_size, "sobel_ksize": sobel, "blend_mode": "uniform", "crop_size": 256, "stride": 192, "tta": False},
        "checkpoint": manifest["checkpoint"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "head": manifest["head"],
        "head_sha256": manifest["head_sha256"],
        "workers": workers,
        "post_seconds": time.time() - started,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if extra_payload:
        payload.update(extra_payload)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def grid(dataset: str) -> Iterable[Tuple[str, Tuple[float, float, int, int]]]:
    fg0, e0 = FIXED[dataset]
    for fg in np.round(np.arange(max(0.05, fg0 - 0.10), min(0.95, fg0 + 0.1001), 0.025), 3):
        for energy in np.round(np.arange(max(0.05, e0 - 0.10), min(0.95, e0 + 0.1001), 0.025), 3):
            for min_size in (5, 10, 20):
                for sobel in (15, 21, 31):
                    yield f"fg_{fg:.3f}_energy_{energy:.3f}_min_{min_size}_sobel_{sobel}", (float(fg), float(energy), min_size, sobel)


def indexed_grid(dataset: str) -> List[Tuple[int, str, Tuple[float, float, int, int]]]:
    return [(i, tag, params) for i, (tag, params) in enumerate(grid(dataset))]


def _selected_candidate(
    index: int,
    start: int | None,
    end: int | None,
    shard_id: int | None,
    shard_count: int | None,
) -> bool:
    if start is not None and index < start:
        return False
    if end is not None and index >= end:
        return False
    if shard_id is not None and shard_count is not None and index % shard_count != shard_id:
        return False
    return True


def command_prepare(args):
    cache_dir = Path(args.cache_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    head = Path(args.head).resolve()
    if (cache_dir / "manifest.json").exists() and not args.force:
        raise FileExistsError(f"cache exists; use a new directory or --force: {cache_dir}")
    c = legacy_cfg(args.dataset, head, "cache", *FIXED[args.dataset], 10, 21)
    collected = w._collect_continuous_outputs(c)
    manifest = _write_cache(args.dataset, cache_dir, head, collected, checkpoint, "test")
    print(json.dumps({"cache_dir": str(cache_dir), "sample_count": manifest["sample_count"], "cache_bytes": sum((cache_dir / r["path"]).stat().st_size for r in manifest["samples"])}, indent=2))


def command_evaluate(args):
    cache_dir = Path(args.cache_dir).resolve()
    manifest = load_manifest(cache_dir)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, params in grid(manifest["dataset"]):
        payload = evaluate_one(cache_dir, output / tag, tag, params, args.workers)
        rows.append({"tag": tag, "skipped": bool(payload.get("_skipped")), "primary": payload["metrics"].get(payload["primary_metric"]), "post_seconds": payload.get("post_seconds")})
    best = max(rows, key=lambda x: x["primary"] if math.isfinite(float(x["primary"])) else -float("inf"))
    (output / "manifest.json").write_text(json.dumps({"dataset": manifest["dataset"], "cache": str(cache_dir), "cache_manifest_sha256": hash_path(cache_dir / "manifest.json"), "candidate_count": len(rows), "skipped_count": sum(x["skipped"] for x in rows), "best": best, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"dataset": manifest["dataset"], "candidates": len(rows), "skipped": sum(x["skipped"] for x in rows), "best": best}, indent=2))


def _shard_args(args) -> Tuple[int | None, int | None, int | None, int | None]:
    start = args.candidate_start
    end = args.candidate_end
    shard_id = args.shard_id
    shard_count = args.shard_count
    if (shard_id is None) != (shard_count is None):
        raise ValueError("--shard-id and --shard-count must be provided together")
    if shard_id is not None and not (0 <= shard_id < shard_count):
        raise ValueError(f"invalid shard {shard_id}/{shard_count}")
    if start is not None and end is not None and end < start:
        raise ValueError("--candidate-end must be >= --candidate-start")
    return start, end, shard_id, shard_count


def command_evaluate_shard(args):
    cache_dir = Path(args.cache_dir).resolve()
    manifest = load_manifest(cache_dir)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    start, end, shard_id, shard_count = _shard_args(args)
    candidates = [
        (index, tag, params)
        for index, tag, params in indexed_grid(manifest["dataset"])
        if _selected_candidate(index, start, end, shard_id, shard_count)
    ]
    rows = []
    extra = {
        "candidate_mode": "test_tuned_candidates",
        "candidate_note": "test-set-tuned upper bound candidate; do not treat as unbiased generalization metric",
        "candidate_shard": {"start": start, "end": end, "shard_id": shard_id, "shard_count": shard_count},
    }
    for index, tag, params in candidates:
        payload = evaluate_one(cache_dir, output / tag, tag, params, args.workers, extra | {"candidate_index": index})
        rows.append({
            "candidate_index": index,
            "tag": tag,
            "params": {"fg_thresh": params[0], "energy_thresh": params[1], "min_size": params[2], "sobel_ksize": params[3]},
            "skipped": bool(payload.get("_skipped")),
            "primary": payload["metrics"].get(payload["primary_metric"]),
            "post_seconds": payload.get("post_seconds"),
        })
    finite_rows = [r for r in rows if math.isfinite(float(r["primary"]))]
    best = max(finite_rows, key=lambda x: float(x["primary"])) if finite_rows else None
    shard_manifest = {
        "dataset": manifest["dataset"],
        "cache": str(cache_dir),
        "cache_manifest_sha256": hash_path(cache_dir / "manifest.json"),
        "mode": "test_tuned_candidates",
        "candidate_count_total": len(indexed_grid(manifest["dataset"])),
        "candidate_count_selected": len(rows),
        "selected_indices": [r["candidate_index"] for r in rows],
        "skipped_count": sum(x["skipped"] for x in rows),
        "best": best,
        "test_tuned_label": "test-set-tuned upper bound",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "workers": args.workers,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output / args.manifest_name).write_text(json.dumps(shard_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"dataset": manifest["dataset"], "selected": len(rows), "skipped": sum(x["skipped"] for x in rows), "best": best}, indent=2))


def command_evaluate_fixed(args):
    cache_dir = Path(args.cache_dir).resolve()
    manifest = load_manifest(cache_dir)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    params = (args.fg_thresh, args.energy_thresh, args.min_size, args.sobel_ksize)
    extra = {
        "candidate_mode": args.mode,
        "candidate_note": args.note,
        "candidate_index": args.candidate_index,
    }
    payload = evaluate_one(cache_dir, output, args.tag, params, args.workers, extra)
    fixed_manifest = {
        "dataset": manifest["dataset"],
        "cache": str(cache_dir),
        "cache_manifest_sha256": hash_path(cache_dir / "manifest.json"),
        "mode": args.mode,
        "tag": args.tag,
        "candidate_index": args.candidate_index,
        "skipped": bool(payload.get("_skipped")),
        "primary_metric": payload["primary_metric"],
        "primary": payload["metrics"].get(payload["primary_metric"]),
        "postproc": {"fg_thresh": params[0], "energy_thresh": params[1], "min_size": params[2], "sobel_ksize": params[3]},
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "workers": args.workers,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output / args.manifest_name).write_text(json.dumps(fixed_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"dataset": manifest["dataset"], "mode": args.mode, "skipped": bool(payload.get("_skipped")), "primary": fixed_manifest["primary"]}, indent=2))


def _prepare_if_needed(dataset: str, cache_dir: Path, head: Path, checkpoint: Path, force: bool) -> Dict[str, Any]:
    if (cache_dir / "manifest.json").exists() and not force:
        return load_manifest(cache_dir)
    if (cache_dir / "manifest.json").exists() and force:
        raise FileExistsError(f"refusing to overwrite cache in worker mode: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "prepare_cache_start", "dataset": dataset, "cache_dir": str(cache_dir), "head": str(head), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}), flush=True)
    c = legacy_cfg(dataset, head, "cache", *FIXED[dataset], 10, 21)
    collected = w._collect_continuous_outputs(c)
    manifest = _write_cache(dataset, cache_dir, head, collected, checkpoint, "test")
    print(json.dumps({"event": "prepare_cache_done", "dataset": dataset, "cache_dir": str(cache_dir), "sample_count": manifest["sample_count"], "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}), flush=True)
    return manifest


def command_run_worker(args):
    dataset = args.dataset
    cache_dir = Path(args.cache_dir).resolve()
    eval_root = Path(args.eval_root).resolve()
    eval_root.mkdir(parents=True, exist_ok=True)
    head = Path(args.head).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    manifest = _prepare_if_needed(dataset, cache_dir, head, checkpoint, args.force_prepare)
    rows = []
    if args.no_trick:
        payload = evaluate_one(
            cache_dir,
            eval_root / "no_trick_test",
            "default",
            (0.50, 0.40, 10, 21),
            args.workers,
            {"candidate_mode": "no_trick_test", "candidate_note": "TEST evaluation; no TEST-set tuning", "candidate_index": -2},
        )
        rows.append({"mode": "no_trick_test", "skipped": bool(payload.get("_skipped")), "primary": payload["metrics"].get(payload["primary_metric"])})
        print(json.dumps({"event": "evaluate_fixed_done", "mode": "no_trick_test", "skipped": bool(payload.get("_skipped")), "primary": rows[-1]["primary"]}), flush=True)
    if args.fixed_trick:
        fg, energy = FIXED[dataset]
        payload = evaluate_one(
            cache_dir,
            eval_root / "fixed_trick_test",
            "val_fixed",
            (fg, energy, 10, 21),
            args.workers,
            {"candidate_mode": "fixed_trick_test", "candidate_note": "validation-selected post-processing applied to TEST", "candidate_index": -1},
        )
        rows.append({"mode": "fixed_trick_test", "skipped": bool(payload.get("_skipped")), "primary": payload["metrics"].get(payload["primary_metric"])})
        print(json.dumps({"event": "evaluate_fixed_done", "mode": "fixed_trick_test", "skipped": bool(payload.get("_skipped")), "primary": rows[-1]["primary"]}), flush=True)
    selected_indices: List[int] = []
    if args.test_tuned_shard:
        start, end, shard_id, shard_count = _shard_args(args)
        candidates = [
            (index, tag, params)
            for index, tag, params in indexed_grid(dataset)
            if _selected_candidate(index, start, end, shard_id, shard_count)
        ]
        selected_indices = [index for index, _, _ in candidates]
        print(json.dumps({"event": "evaluate_shard_start", "dataset": dataset, "selected": len(candidates), "first": selected_indices[:3], "last": selected_indices[-3:], "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}), flush=True)
        shard_rows = []
        extra = {
            "candidate_mode": "test_tuned_candidates",
            "candidate_note": "test-set-tuned upper bound candidate; do not treat as unbiased generalization metric",
            "candidate_shard": {"start": start, "end": end, "shard_id": shard_id, "shard_count": shard_count},
        }
        shard_output = eval_root / "test_tuned_candidates"
        for index, tag, params in candidates:
            payload = evaluate_one(cache_dir, shard_output / tag, tag, params, args.workers, extra | {"candidate_index": index})
            shard_rows.append({
                "candidate_index": index,
                "tag": tag,
                "params": {"fg_thresh": params[0], "energy_thresh": params[1], "min_size": params[2], "sobel_ksize": params[3]},
                "skipped": bool(payload.get("_skipped")),
                "primary": payload["metrics"].get(payload["primary_metric"]),
                "post_seconds": payload.get("post_seconds"),
            })
        finite_rows = [r for r in shard_rows if math.isfinite(float(r["primary"]))]
        best = max(finite_rows, key=lambda x: float(x["primary"])) if finite_rows else None
        (shard_output / args.shard_manifest_name).write_text(json.dumps({
            "dataset": dataset,
            "cache": str(cache_dir),
            "cache_manifest_sha256": hash_path(cache_dir / "manifest.json"),
            "mode": "test_tuned_candidates",
            "candidate_count_total": len(indexed_grid(dataset)),
            "candidate_count_selected": len(shard_rows),
            "selected_indices": selected_indices,
            "skipped_count": sum(x["skipped"] for x in shard_rows),
            "best": best,
            "test_tuned_label": "test-set-tuned upper bound",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "workers": args.workers,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, indent=2, sort_keys=True) + "\n")
        rows.append({"mode": "test_tuned_candidates", "selected": len(shard_rows), "skipped": sum(x["skipped"] for x in shard_rows), "best": best})
        print(json.dumps({"event": "evaluate_shard_done", "dataset": dataset, "selected": len(shard_rows), "skipped": sum(x["skipped"] for x in shard_rows), "best": best}), flush=True)
    worker_manifest = {
        "dataset": dataset,
        "cache": str(cache_dir),
        "cache_manifest_sha256": hash_path(cache_dir / "manifest.json"),
        "eval_root": str(eval_root),
        "modes": rows,
        "selected_indices": selected_indices,
        "test_tuned_label": "test-set-tuned upper bound" if args.test_tuned_shard else None,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "workers": args.workers,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (eval_root / args.manifest_name).write_text(json.dumps(worker_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "worker_done", "dataset": dataset, "eval_root": str(eval_root), "modes": [r["mode"] for r in rows]}), flush=True)


def _fixture_cache(cache_dir: Path, dataset: str = "monuseg", count: int = 5) -> None:
    rng = np.random.default_rng(12345)
    records = []
    samples = cache_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        h = w0 = 64
        np_logits = rng.normal(size=(2, h, w0)).astype(np.float32)
        hv = rng.normal(size=(2, h, w0)).astype(np.float32)
        gt = np.zeros((h, w0), dtype=np.int32)
        gt[8:24, 8:24] = 1
        gt[34:52, 34:54] = 2
        p = _sample_path(cache_dir, i)
        np.savez_compressed(p, np_logits=np_logits, hv=hv, gt_inst=gt, sample_id=np.asarray(str(i)))
        records.append({"index": i, "sample_id": str(i), "path": str(p.relative_to(cache_dir)), "sha256": hash_path(p)})
    manifest = {"schema": SCHEMA, "dataset": dataset, "split": "test", "sample_count": count, "samples": records, "checkpoint": "fixture", "checkpoint_sha256": "fixture", "head": "fixture", "head_sha256": "fixture"}
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def command_validate(args):
    import tempfile
    cache_dir = Path(args.cache_dir).resolve()
    if cache_dir.exists():
        raise FileExistsError(cache_dir)
    _fixture_cache(cache_dir)
    manifest = load_manifest(cache_dir)
    global _CACHE_MANIFEST
    old_root = w.OUT_ROOT
    with tempfile.TemporaryDirectory(prefix="biodino-old-eval-") as tmp:
        w.OUT_ROOT = Path(tmp)
        for j, (_, params) in enumerate(list(grid("monuseg"))[:5]):
            cached = [load_sample(cache_dir, r) for r in manifest["samples"]]
            old = w._eval_row_from_cached_outputs(legacy_cfg("monuseg", Path("fixture"), f"old_{j}", *params), "eval_only", "AJI", 0, len(cached), cached, "fixture", 0, 16, 0.0, 0.0)
            new = evaluate_one(cache_dir, cache_dir / "validated" / f"candidate_{j}", f"candidate_{j}", params, 1)
            keys = sorted(set(old["metrics"]) | set(new["metrics"]))
            diffs = {k: abs(_float(old["metrics"].get(k)) - _float(new["metrics"].get(k))) for k in keys}
            max_diff = max(diffs.values()) if diffs else 0.0
            print(json.dumps({"candidate": j, "params": params, "max_abs_diff": max_diff, "AJI": new["metrics"].get("AJI"), "bPQ": new["metrics"].get("bPQ"), "pass": max_diff < 1e-10}, sort_keys=True))
        w.OUT_ROOT = old_root


def command_benchmark(args):
    cache_dir = Path(args.cache_dir).resolve()
    manifest = load_manifest(cache_dir)
    params = next(grid(manifest["dataset"]))[1]
    for workers in args.workers:
        target = cache_dir / "benchmark" / f"workers_{workers}"
        started = time.time()
        payload = evaluate_one(cache_dir, target, f"benchmark_{workers}", params, workers)
        rss_mib = int(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 1024)
        print(json.dumps({"workers": workers, "seconds": time.time() - started, "post_seconds": payload.get("post_seconds"), "rss_mib": rss_mib, "sample_count": manifest["sample_count"]}, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-cache"); p.add_argument("--dataset", required=True); p.add_argument("--head", required=True); p.add_argument("--checkpoint", default="/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/bio_continue_rgb3_vith16plus/ckpt/14349"); p.add_argument("--cache-dir", required=True); p.add_argument("--force", action="store_true"); p.set_defaults(func=command_prepare)
    p = sub.add_parser("evaluate"); p.add_argument("--cache-dir", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--workers", type=int, default=8); p.set_defaults(func=command_evaluate)
    p = sub.add_parser("evaluate-shard"); p.add_argument("--cache-dir", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--workers", type=int, default=4); p.add_argument("--candidate-start", type=int); p.add_argument("--candidate-end", type=int); p.add_argument("--shard-id", type=int); p.add_argument("--shard-count", type=int); p.add_argument("--manifest-name", default="manifest.json"); p.set_defaults(func=command_evaluate_shard)
    p = sub.add_parser("evaluate-fixed"); p.add_argument("--cache-dir", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--workers", type=int, default=4); p.add_argument("--mode", required=True); p.add_argument("--tag", required=True); p.add_argument("--fg-thresh", type=float, required=True); p.add_argument("--energy-thresh", type=float, required=True); p.add_argument("--min-size", type=int, default=10); p.add_argument("--sobel-ksize", type=int, default=21); p.add_argument("--candidate-index", type=int); p.add_argument("--note", default=""); p.add_argument("--manifest-name", default="manifest.json"); p.set_defaults(func=command_evaluate_fixed)
    p = sub.add_parser("run-worker"); p.add_argument("--dataset", required=True); p.add_argument("--head", required=True); p.add_argument("--checkpoint", default="/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/bio_continue_rgb3_vith16plus/ckpt/14349"); p.add_argument("--cache-dir", required=True); p.add_argument("--eval-root", required=True); p.add_argument("--workers", type=int, default=4); p.add_argument("--no-trick", action="store_true"); p.add_argument("--fixed-trick", action="store_true"); p.add_argument("--test-tuned-shard", action="store_true"); p.add_argument("--candidate-start", type=int); p.add_argument("--candidate-end", type=int); p.add_argument("--shard-id", type=int); p.add_argument("--shard-count", type=int); p.add_argument("--force-prepare", action="store_true"); p.add_argument("--manifest-name", default="worker_manifest.json"); p.add_argument("--shard-manifest-name", default="shard_manifest.json"); p.set_defaults(func=command_run_worker)
    p = sub.add_parser("validate-five"); p.add_argument("--cache-dir", required=True); p.set_defaults(func=command_validate)
    p = sub.add_parser("benchmark"); p.add_argument("--cache-dir", required=True); p.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16]); p.set_defaults(func=command_benchmark)
    args = ap.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
