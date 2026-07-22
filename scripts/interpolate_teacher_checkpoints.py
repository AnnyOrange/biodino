#!/usr/bin/env python3
"""Interpolate an official raw backbone with an EMA teacher from a training checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path

import torch


TEACHER_PREFIX = "teacher.backbone."


def load_teacher(path: Path) -> OrderedDict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Expected a consolidated training checkpoint with a model state: {path}")
    teacher = OrderedDict(
        (key[len(TEACHER_PREFIX) :], value)
        for key, value in checkpoint["model"].items()
        if key.startswith(TEACHER_PREFIX)
    )
    if not teacher:
        raise ValueError(f"No {TEACHER_PREFIX} tensors found in {path}")
    return teacher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--bio-checkpoint", type=Path, required=True)
    parser.add_argument("--bio-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0.9])
    args = parser.parse_args()

    official = torch.load(
        args.official_checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(official, dict) or not official:
        raise ValueError(f"Expected a raw backbone state dict: {args.official_checkpoint}")
    bio = load_teacher(args.bio_checkpoint)
    if set(official) != set(bio):
        raise ValueError(
            f"Backbone key mismatch: official_only={sorted(set(official) - set(bio))[:8]} "
            f"bio_only={sorted(set(bio) - set(official))[:8]}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.bio_config, args.output_root / "config.yaml")
    manifest = {
        "official_checkpoint": str(args.official_checkpoint),
        "bio_checkpoint": str(args.bio_checkpoint),
        "bio_config": str(args.bio_config),
        "alphas": args.alphas,
        "definition": "theta=(1-alpha)*official + alpha*bio_ema_teacher",
    }
    (args.output_root / "interpolation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for alpha in args.alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        checkpoint_id = int(round(alpha * 100))
        checkpoint_dir = args.output_root / "ckpt" / str(checkpoint_id)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        interpolated = OrderedDict()
        for key, official_value in official.items():
            bio_value = bio[key]
            if official_value.shape != bio_value.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: official={tuple(official_value.shape)} "
                    f"bio={tuple(bio_value.shape)}"
                )
            if official_value.is_floating_point() or official_value.is_complex():
                target_dtype = bio_value.dtype
                value = torch.lerp(
                    official_value.to(dtype=torch.float32),
                    bio_value.to(dtype=torch.float32),
                    alpha,
                ).to(dtype=target_dtype)
            else:
                value = bio_value.clone()
            interpolated[key] = value
        output = checkpoint_dir / "checkpoint.pth"
        torch.save(interpolated, output)
        print(f"alpha={alpha:.4f} checkpoint={checkpoint_id} output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
