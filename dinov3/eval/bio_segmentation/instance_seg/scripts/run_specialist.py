"""
Specialist-model adapter (Cellpose / Cellpose-SAM) — scored with the SAME metrics.

This is the fairness keystone of Line 2: we run a specialist to produce instance
maps and feed them into the EXACT same ``accumulate_instance_metrics`` that scores
DINOHoVerNet. No model is allowed to be graded by its own private metric.

Cellpose is intentionally NOT vendored. Run this in an environment that has it
installed (e.g. a cellpose env); it is lazy-imported so the rest of the
instance_seg package never depends on it.

Usage:
    python -m dinov3.eval.bio_segmentation.instance_seg.scripts.run_specialist \\
        --dataset monuseg --data-root /data/monuseg/extracted \\
        --model nuclei --output-dir ./outputs/specialist/cellpose/monuseg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from dinov3.eval.bio_segmentation.feature_extractor import _build_dataset
from dinov3.eval.bio_segmentation.metrics import accumulate_instance_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bio_seg.instance_seg.specialist")


def _build_cellpose(model_name: str, gpu: bool):
    """Construct a Cellpose model, tolerating API differences across versions."""
    try:
        from cellpose import models  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "cellpose is not installed in this environment. Install it (pip install "
            "cellpose) or run this script from a cellpose-enabled env."
        ) from e

    name = model_name.lower()
    # Cellpose-SAM / Cellpose>=4 ships a single generalist model.
    if name in {"cpsam", "sam", "cellpose-sam"}:
        if hasattr(models, "CellposeModel"):
            return ("model", models.CellposeModel(gpu=gpu))
        return ("model", models.Cellpose(gpu=gpu))
    # Classic builtin model types ('nuclei', 'cyto', 'cyto2', 'cyto3').
    if hasattr(models, "Cellpose"):
        try:
            return ("cellpose", models.Cellpose(gpu=gpu, model_type=name))
        except TypeError:
            pass
    return ("model", models.CellposeModel(gpu=gpu, model_type=name))


def _eval_one(kind, model, img_uint8: np.ndarray, diameter, channels: List[int], omni: bool = False) -> np.ndarray:
    """Run one image through Cellpose/Omnipose and return an instance map."""
    kwargs = {}
    if diameter is not None:
        kwargs["diameter"] = diameter
    if omni:  # Omnipose mask reconstruction
        kwargs["omni"] = True
    # Channels only matter for the classic models; cpsam handles RGB itself.
    try:
        out = model.eval(img_uint8, channels=channels, **kwargs)
    except TypeError:
        out = model.eval(img_uint8, **kwargs)
    masks = out[0]
    return np.asarray(masks).astype(np.int32)


def run(args):
    # Un-normalized [0,1] RGB so we can hand Cellpose a clean uint8 image.
    ds = _build_dataset(args.dataset, args.data_root, args.split, None, do_normalize=False)
    kind, model = _build_cellpose(args.model, gpu=args.gpu)
    channels = [int(c) for c in args.channels]

    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    n = len(ds) if args.max_images is None else min(args.max_images, len(ds))
    for i in tqdm(range(n), desc=f"{args.model}:{args.dataset}"):
        sample = ds[i]
        img, _sem, inst = sample[0], sample[1], sample[2]
        img_uint8 = (img.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        pred = _eval_one(kind, model, img_uint8, args.diameter, channels, omni=args.omni)
        preds.append(pred)
        gts.append(inst.numpy().astype(np.int32))

    metrics = accumulate_instance_metrics(preds, gts)  # binary: AJI/AP/bPQ
    results = {
        args.split: metrics,
        "_meta": {
            "specialist": args.model,
            "dataset": args.dataset,
            "num_images": n,
            "channels": channels,
            "diameter": args.diameter,
        },
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("[%s/%s] %s", args.model, args.dataset, {k: round(v, 4) for k, v in metrics.items()})
    logger.info("Results saved → %s", out_json)
    return results


def main():
    p = argparse.ArgumentParser(description="Cellpose/cpsam adapter scored by the shared metrics")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--model", default="nuclei",
                   help="nuclei | cyto | cyto2 | cyto3 | cpsam")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--diameter", type=float, default=None,
                   help="Cell diameter; None lets Cellpose estimate.")
    p.add_argument("--channels", nargs=2, default=[0, 0],
                   help="Cellpose channels [cyto, nucleus]; default grayscale [0,0].")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--omni", action="store_true", help="Omnipose mask reconstruction (omni=True).")
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
