#!/usr/bin/env python3
"""Measure CMGI's actual ViT-L parameter displacement against a baseline."""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import torch


def _canonical_backbone_name(name: str) -> str:
    """Map compatible residual-MC stem keys to released DINOv3 names."""
    return name.replace("patch_embed.rgb.", "patch_embed.")


def _teacher_backbone(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    prefix = "teacher.backbone."
    tensors = {
        _canonical_backbone_name(name[len(prefix) :]): tensor.float().clone()
        for name, tensor in state.items()
        if name.startswith(prefix) and torch.is_floating_point(tensor)
    }
    # Released DINOv3 checkpoints contain the backbone directly, while
    # continued-training artifacts prefix it with ``teacher.backbone.``.
    # Supporting both lets calibration measure drift from the actual released
    # initialization instead of requiring an artificial zero-update checkpoint.
    if not tensors and isinstance(state, dict):
        tensors = {
            _canonical_backbone_name(name): tensor.float().clone()
            for name, tensor in state.items()
            if torch.is_tensor(tensor) and torch.is_floating_point(tensor)
        }
    if not tensors:
        raise RuntimeError(f"No floating backbone tensors in {path}")
    del checkpoint
    gc.collect()
    return tensors


def _group(name: str) -> str:
    fields = name.split(".")
    if len(fields) >= 2 and fields[0] == "blocks" and fields[1].isdigit():
        return f"block_{int(fields[1]):02d}"
    if name.startswith("patch_embed"):
        return "patch_embed"
    return "other"


def _stats(reference: dict[str, torch.Tensor], baseline: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]):
    totals = defaultdict(lambda: {"base_sq": 0.0, "effect_sq": 0.0, "anchor_sq": 0.0, "dot": 0.0})
    for name, ref in reference.items():
        if name not in baseline or name not in candidate:
            raise KeyError(f"Missing {name} in a calibration checkpoint")
        base = baseline[name] - ref
        effect = candidate[name] - baseline[name]
        group = totals[_group(name)]
        overall = totals["__overall__"]
        for target in (group, overall):
            target["base_sq"] += float(base.square().sum())
            target["effect_sq"] += float(effect.square().sum())
            target["anchor_sq"] += float(ref.square().sum())
            target["dot"] += float((base * effect).sum())

    def finalize(value):
        base_norm = value["base_sq"] ** 0.5
        effect_norm = value["effect_sq"] ** 0.5
        anchor_norm = value["anchor_sq"] ** 0.5
        return {
            "baseline_relative_drift": base_norm / max(anchor_norm, 1e-12),
            "candidate_specific_relative_effect": effect_norm / max(anchor_norm, 1e-12),
            # Kept for compatibility with the original CMGI calibration JSON.
            "cmgi_specific_relative_effect": effect_norm / max(anchor_norm, 1e-12),
            "effect_over_baseline_drift": effect_norm / max(base_norm, 1e-12),
            "effect_vs_baseline_cosine": value["dot"] / max(base_norm * effect_norm, 1e-12),
        }

    return {name: finalize(value) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True, help="zero-LR / no-update compatible ViT-L checkpoint")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference = _teacher_backbone(Path(args.anchor))
    baseline = _teacher_backbone(Path(args.baseline))
    report = {"anchor": args.anchor, "baseline": args.baseline, "candidates": {}}
    for entry in args.candidate:
        name, raw_path = entry.split("=", 1)
        candidate = _teacher_backbone(Path(raw_path))
        stats = _stats(reference, baseline, candidate)
        report["candidates"][name] = {
            "overall": stats.pop("__overall__"),
            "by_group": stats,
        }
        del candidate
        gc.collect()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
