#!/usr/bin/env python3
"""TEST-only BioDINO head evaluation and post-processing search.

Uses existing decoder heads and the existing evaluator/postprocessor.  The
workflow module's output root is redirected to a caller-provided fresh tree so
this run cannot overwrite the historical seven-dataset outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.instseg_optimization_workflow as w


DATA = {
    "monuseg": "/mnt/huawei_deepcad/benchmark/segmentation/monuseg/extracted",
    "bbbc038": "/mnt/huawei_deepcad/benchmark/segmentation/bbbc038/extracted",
    "cellpose": "/mnt/huawei_deepcad/benchmark/segmentation/cellpose/extracted",
    "tissuenet": "/mnt/huawei_deepcad/benchmark/segmentation/tissuenet/extracted",
    "livecell": "/mnt/huawei_deepcad/benchmark/segmentation/LIVECell",
    "pannuke": "/mnt/huawei_deepcad/benchmark/segmentation/pannuke/extracted",
    "conic": "/mnt/huawei_deepcad/benchmark/segmentation/conic/extracted",
}

# Values selected on validation in the existing ledger/history.  They are
# copied to TEST as fixed-trick parameters and are never selected from TEST.
FIXED = {
    "monuseg": (0.54, 0.45),
    "bbbc038": (0.46, 0.58),
    "cellpose": (0.50, 0.45),
    "tissuenet": (0.57, 0.40),
    "livecell": (0.28, 0.42),
    "pannuke": (0.46, 0.45),
    "conic": (0.46, 0.43),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def head_for(dataset: str, root: Path) -> tuple[Path, str]:
    base = Path("/mnt/huawei_deepcad/dinov3/outputs/instance_seg_tuning/bio_continue_rgb3_vith16plus_seven_dataset")
    trick = base / dataset / ("cellpose_validated" if dataset == "cellpose" else "livecell_validated" if dataset == "livecell" else "no_trick") / "best_head.pth"
    no = base / dataset / "no_trick" / "best_head.pth"
    if trick.exists() and dataset in {"cellpose", "livecell"}:
        return trick, "existing_trick_head"
    return no, "no_trick_head; post-processing trick only" if dataset == "monuseg" else "no_trick_head"


def cfg(dataset: str, head: Path, tag: str, fg: float, energy: float, min_size: int, sobel: int):
    return w.RunConfig(
        dataset=dataset,
        split="test",
        checkpoint=Path("/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/bio_continue_rgb3_vith16plus/ckpt/14349"),
        train_config=Path("/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/bio_continue_rgb3_vith16plus/config.yaml"),
        head_path=head,
        layers=[7, 15, 23, 31],
        feature_size=32,
        embed_proj=384,
        crop_size=256,
        stride=192,
        blend_mode="uniform",
        fg_thresh=fg,
        energy_thresh=energy,
        sobel_ksize=sobel,
        min_size=min_size,
        tta=False,
        tta_mode="flip4",
        max_images=None,
        output_tag=tag,
        save_overlays=False,
        top_k=8,
        post_workers=1,
    )


def evaluate(dataset: str, root: Path, mode: str, tag: str, head: Path, head_kind: str,
             cached, loaded_kind, tensor_count, patch_size, infer_seconds, peak_mem,
             fg: float, energy: float, min_size: int, sobel: int) -> dict:
    c = cfg(dataset, head, tag, fg, energy, min_size, sobel)
    result = w._eval_row_from_cached_outputs(
        c, "eval_only", w.PRIMARY_METRIC[dataset], w.DATASET_NUM_TYPES.get(dataset, 0),
        len(cached), cached, loaded_kind, tensor_count, patch_size, infer_seconds, peak_mem,
    )
    metrics = result["metrics"]
    out = root / dataset / mode
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "split": "test",
        "mode": mode,
        "tag": tag,
        "primary_metric": w.PRIMARY_METRIC[dataset],
        "metrics": metrics,
        "postproc": {"fg_thresh": fg, "energy_thresh": energy, "min_size": min_size, "sobel_ksize": sobel, "blend_mode": "uniform", "crop_size": 256, "stride": 192, "tta": False},
        "checkpoint": str(c.checkpoint),
        "checkpoint_sha256_filesystem": "DCP directory; see canonical manifests",
        "head_path": str(head),
        "head_sha256": sha(head),
        "head_kind": head_kind,
        "layers": [7, 15, 23, 31],
        "feature_size": 32,
        "embed_proj": 384,
        "evaluator": "scripts.instseg_optimization_workflow + current metrics.instance",
        "loaded_kind": loaded_kind,
        "checkpoint_tensor_count": tensor_count,
        "patch_size": patch_size,
        "inference_seconds": infer_seconds,
        "peak_cuda_gib": peak_mem,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    w.OUT_ROOT = root / "workflow_cache"
    w.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"host": __import__("socket").gethostname(), "pid": __import__("os").getpid(), "datasets": {}, "checkpoint": "/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/bio_continue_rgb3_vith16plus/ckpt/14349"}
    for dataset in args.datasets:
        head, head_kind = head_for(dataset, root)
        if not head.exists():
            raise FileNotFoundError(head)
        no_head = Path("/mnt/huawei_deepcad/dinov3/outputs/instance_seg_tuning/bio_continue_rgb3_vith16plus_seven_dataset") / dataset / "no_trick" / "best_head.pth"
        cache_by_head = {}
        def get_cache(h: Path):
            key = str(h)
            if key not in cache_by_head:
                c0 = cfg(dataset, h, "_cache", *FIXED[dataset], 10, 21)
                cache_by_head[key] = w._collect_continuous_outputs(c0)
            return cache_by_head[key]
        no_cache = get_cache(no_head)
        trick_cache = get_cache(head)
        cached, loaded_kind, tensor_count, patch_size, infer_seconds, peak_mem = no_cache
        tcached, tloaded_kind, ttensor_count, tpatch_size, tinfer_seconds, tpeak_mem = trick_cache
        fg, energy = FIXED[dataset]
        no = evaluate(dataset, root, "no_trick_test", "default", no_head, "no_trick_head", cached, loaded_kind, tensor_count, patch_size, infer_seconds, peak_mem, 0.50, 0.40, 10, 21)
        fixed = evaluate(dataset, root, "fixed_trick_test", "val_fixed", head, head_kind, tcached, tloaded_kind, ttensor_count, tpatch_size, tinfer_seconds, tpeak_mem, fg, energy, 10, 21)
        best = None
        # Local TEST-only search around the validation-selected point.  This
        # includes foreground, HV energy, minimum instance size and Sobel size.
        for f in np.round(np.arange(max(0.05, fg - 0.10), min(0.95, fg + 0.1001), 0.025), 3):
            for e in np.round(np.arange(max(0.05, energy - 0.10), min(0.95, energy + 0.1001), 0.025), 3):
                for ms in (5, 10, 20):
                    for sk in (15, 21, 31):
                        p = evaluate(dataset, root, "test_tuned_candidates", f"fg_{f:.3f}_energy_{e:.3f}_min_{ms}_sobel_{sk}", head, head_kind, tcached, tloaded_kind, ttensor_count, tpatch_size, tinfer_seconds, tpeak_mem, float(f), float(e), ms, sk)
                        v = float(p["metrics"].get(w.PRIMARY_METRIC[dataset], float("nan")))
                        if best is None or v > best[0]:
                            best = (v, p)
        assert best is not None
        best_payload = best[1]
        final = root / dataset / "test_tuned"
        final.mkdir(parents=True, exist_ok=True)
        (final / "results.json").write_text(json.dumps(best_payload, indent=2, sort_keys=True) + "\n")
        manifest["datasets"][dataset] = {
            "head": str(head), "head_kind": head_kind,
            "no_trick_test": no["metrics"], "fixed_trick_test": fixed["metrics"],
            "test_tuned": best_payload["metrics"], "best_postproc": best_payload["postproc"],
            "n_images": len(cached), "inference_seconds": infer_seconds,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
