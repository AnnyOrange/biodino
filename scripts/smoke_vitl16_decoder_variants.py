#!/usr/bin/env python3
"""Shape, one-batch backward, and CUDA peak-memory smoke tests for Stage 2 decoders."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.eval.bio_segmentation.instance_seg.decoder import HoVerNetDecoder


def run(variant: str, device: torch.device) -> dict:
    torch.manual_seed(0)
    model = HoVerNetDecoder(
        tap_dims=[1024, 1024, 1024, 1024],
        num_types=0,
        feature_size=32,
        embed_proj=384,
        patch_size=16,
        decoder_variant=variant,
    ).to(device)
    taps = [torch.randn(1, 1024, 16, 16, device=device, requires_grad=True) for _ in range(4)]
    image = torch.randn(1, 3, 256, 256, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    out = model(image, taps)
    assert tuple(out["np"].shape) == (1, 2, 256, 256), out["np"].shape
    assert tuple(out["hv"].shape) == (1, 2, 256, 256), out["hv"].shape
    loss = out["np"].square().mean() + out["hv"].square().mean()
    loss.backward()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        "variant": variant,
        "device": str(device),
        "shape": list(out["np"].shape),
        "trainable_parameters": trainable,
        "peak_memory_gib": peak,
        "forward_backward_seconds": elapsed,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["current", "fpn", "unet", "multi_layer_fpn"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args.variant, torch.device(args.device))


if __name__ == "__main__":
    main()
