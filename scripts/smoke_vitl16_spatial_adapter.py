#!/usr/bin/env python3
"""Shape, one-batch backward, and CUDA peak-memory test for the CNN spatial adapter."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.eval.bio_segmentation.instance_seg.decoder import HoVerNetDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    model = HoVerNetDecoder(
        tap_dims=[1024, 1024, 1024, 1024],
        num_types=0,
        feature_size=32,
        embed_proj=384,
        patch_size=16,
        decoder_variant="current",
        spatial_adapter=True,
        spatial_adapter_width=32,
    ).to(device)
    image = torch.randn(1, 3, 256, 256, device=device)
    taps = [torch.randn(1, 1024, 16, 16, device=device, requires_grad=True) for _ in range(4)]
    local = model.spatial_adapter(image)
    expected = {"quarter": (1, 128, 64, 64), "eighth": (1, 256, 32, 32), "sixteenth": (1, 384, 16, 16)}
    shapes = {key: tuple(value.shape) for key, value in local.items()}
    assert shapes == expected, shapes
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    out = model(image, taps)
    assert tuple(out["np"].shape) == (1, 2, 256, 256), out["np"].shape
    assert tuple(out["hv"].shape) == (1, 2, 256, 256), out["hv"].shape
    (out["np"].square().mean() + out["hv"].square().mean()).backward()
    elapsed = time.perf_counter() - started
    adapter_params = sum(p.numel() for p in model.spatial_adapter.parameters() if p.requires_grad)
    adapter_grad = sum(float(p.grad.abs().sum()) for p in model.spatial_adapter.parameters() if p.grad is not None)
    assert adapter_grad > 0.0, "spatial adapter received no gradient"
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
    result = {
        "variant": "current_plus_cnn_spatial_adapter_additive",
        "device": str(device),
        "local_shapes": {key: list(value) for key, value in shapes.items()},
        "output_shape": list(out["np"].shape),
        "trainable_parameters_total": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "spatial_adapter_parameters": adapter_params,
        "spatial_adapter_grad_l1": adapter_grad,
        "peak_memory_gib": peak,
        "forward_backward_seconds": elapsed,
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
