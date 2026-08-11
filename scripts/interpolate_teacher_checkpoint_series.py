#!/usr/bin/env python3
"""Interpolate an official backbone with a series of EMA teacher checkpoints."""

from __future__ import annotations

import argparse
import gc
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


def interpolate(
    official: dict[str, torch.Tensor],
    bio: dict[str, torch.Tensor],
    alpha: float,
) -> OrderedDict[str, torch.Tensor]:
    if set(official) != set(bio):
        raise ValueError(
            f"Backbone key mismatch: official_only={sorted(set(official) - set(bio))[:8]} "
            f"bio_only={sorted(set(bio) - set(official))[:8]}"
        )
    result = OrderedDict()
    for key, official_value in official.items():
        bio_value = bio[key]
        if official_value.shape != bio_value.shape:
            raise ValueError(
                f"Shape mismatch for {key}: official={tuple(official_value.shape)} "
                f"bio={tuple(bio_value.shape)}"
            )
        if official_value.is_floating_point() or official_value.is_complex():
            result[key] = torch.lerp(
                official_value.to(dtype=torch.float32),
                bio_value.to(dtype=torch.float32),
                alpha,
            ).to(dtype=bio_value.dtype)
        else:
            result[key] = bio_value.clone()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-iters", type=int, nargs="+", required=True)
    parser.add_argument("--bio-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error(f"--alpha must be in [0, 1], got {args.alpha}")

    official = torch.load(
        args.official_checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(official, dict) or not official:
        raise ValueError(f"Expected a raw backbone state dict: {args.official_checkpoint}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.bio_config, args.output_root / "config.yaml")
    manifest = {
        "official_checkpoint": str(args.official_checkpoint),
        "checkpoints_dir": str(args.checkpoints_dir),
        "checkpoint_iters": args.checkpoint_iters,
        "bio_config": str(args.bio_config),
        "alpha": args.alpha,
        "definition": "theta=(1-alpha)*official + alpha*bio_ema_teacher",
    }
    (args.output_root / "interpolation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    for iteration in args.checkpoint_iters:
        source = args.checkpoints_dir / str(iteration) / "checkpoint.pth"
        output = args.output_root / "ckpt" / str(iteration) / "checkpoint.pth"
        if output.is_file() and not args.overwrite:
            print(f"skip checkpoint={iteration} output={output}", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        bio = load_teacher(source)
        state = interpolate(official, bio, args.alpha)
        partial = output.with_suffix(output.suffix + ".part")
        torch.save(state, partial)
        partial.replace(output)
        print(f"alpha={args.alpha:.4f} checkpoint={iteration} output={output}", flush=True)
        del bio, state
        gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
