"""
Eval-only: load a trained DINOHoVerNet checkpoint and evaluate a full split.

``best_head.pth`` may be either:
  * a frozen-backbone decoder-only state dict (keys like ``fuse.0.weight``), or
  * a fine-tuned full-model state dict (keys like ``backbone.*``/``decoder.*``).

The output JSON records which checkpoint kind was loaded so final tables do not
mix frozen decoder-only and fine-tuned full-model results by accident.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import OrderedDict
from typing import Dict, Tuple

import torch

from ..feature_extractor import _build_dataset
from .model import build_dino_hovernet
from .train import DATASET_NUM_TYPES, evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bio_seg.instance_seg.eval_full")


def _strip_prefix_if_present(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not state or not all(k.startswith(prefix) for k in state):
        return state
    return OrderedDict((k[len(prefix) :], v) for k, v in state.items())


def _normalise_state_dict(obj) -> Dict[str, torch.Tensor]:
    """Return a plain state dict from common torch checkpoint wrappers."""
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a state-dict-like checkpoint, got {type(obj).__name__}")
    for key in ("state_dict", "model", "module"):
        value = obj.get(key)
        if isinstance(value, dict):
            obj = value
            break
    state = _strip_prefix_if_present(obj, "module.")
    state = _strip_prefix_if_present(state, "model.")
    return state


def _infer_checkpoint_kind(state: Dict[str, torch.Tensor], requested: str) -> str:
    if requested != "auto":
        return requested
    keys = set(state)
    if any(k.startswith("backbone.") for k in keys) or any(k.startswith("decoder.") for k in keys):
        return "full"
    return "decoder"


def _load_hover_state(model, path: str, device: torch.device, kind: str) -> Tuple[str, int]:
    state = _normalise_state_dict(torch.load(path, map_location=device))
    kind = _infer_checkpoint_kind(state, kind)
    if kind == "decoder":
        model.decoder.load_state_dict(state, strict=True)
    elif kind == "full":
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError(f"Unknown checkpoint kind: {kind}")
    logger.info("Loaded %s checkpoint (%d tensors) from %s", kind, len(state), path)
    return kind, len(state)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-config", required=True)
    p.add_argument("--head-path", required=True,
                   help="best_head.pth: decoder-only state or full DINOHoVerNet state")
    p.add_argument("--checkpoint-kind", choices=["auto", "decoder", "full"], default="auto",
                   help="How to load --head-path. Default auto-detects from state-dict keys.")
    p.add_argument("--output", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[7, 15, 23, 31])
    p.add_argument("--feature-size", type=int, default=64)
    p.add_argument("--embed-proj", type=int, default=512)
    p.add_argument("--split", default="test")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--max-eval-images", type=int, default=None)
    p.add_argument("--tta", action="store_true", help="Enable 4-way flip TTA at eval.")
    p.add_argument("--fg-thresh", type=float, default=0.5)
    p.add_argument("--energy-thresh", type=float, default=0.4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_types = DATASET_NUM_TYPES.get(args.dataset, 0)
    # Build frozen for eval. A full fine-tuned checkpoint still loads all
    # backbone tensors and runs under inference_mode, so gradients are irrelevant.
    model = build_dino_hovernet(
        checkpoint=args.checkpoint, train_config=args.train_config, layers=args.layers,
        num_types=num_types, freeze_backbone=True, feature_size=args.feature_size,
        embed_proj=args.embed_proj, device=device,
    )
    loaded_kind, n_tensors = _load_hover_state(model, args.head_path, device, args.checkpoint_kind)
    base = _build_dataset(args.dataset, args.data_root, args.split, None)
    metrics = evaluate(
        model, base, device, num_types, crop_size=args.crop_size, stride=args.stride,
        patch_size=int(model.backbone.patch_size), max_images=args.max_eval_images,
        tta=args.tta, fg_thresh=args.fg_thresh, energy_thresh=args.energy_thresh,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out = {
        "split": args.split,
        "n": args.max_eval_images,
        "metrics": metrics,
        "_meta": {
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "train_config": args.train_config,
            "head_path": args.head_path,
            "checkpoint_kind": loaded_kind,
            "num_tensors": n_tensors,
            "layers": model.layers,
            "num_types": num_types,
            "feature_size": args.feature_size,
            "embed_proj": args.embed_proj,
            "crop_size": args.crop_size,
            "stride": args.stride,
            "tta": bool(args.tta),
            "fg_thresh": args.fg_thresh,
            "energy_thresh": args.energy_thresh,
        },
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print({k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
