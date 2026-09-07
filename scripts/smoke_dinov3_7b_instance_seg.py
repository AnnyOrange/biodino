#!/usr/bin/env python3
"""Shape and forward/backward smoke test for the Frozen DINOv3-7B HoVerNet path."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", nargs=4, type=int, default=[7, 19, 29, 39])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 7B smoke test")

    from dinov3.eval.bio_segmentation.instance_seg.model import build_dino_hovernet

    started = time.perf_counter()
    device = torch.device("cuda")
    model = build_dino_hovernet(
        checkpoint=args.checkpoint,
        train_config=args.train_config,
        layers=args.layers,
        num_types=0,
        freeze_backbone=True,
        feature_size=32,
        embed_proj=384,
        fusion_mode="bucket_concat",
        decoder_variant="current",
        device=device,
    )
    model.eval()
    x = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float32)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = model(x)
        loss = sum(value.float().square().mean() for value in pred.values() if value is not None)
    loss.backward()
    torch.cuda.synchronize()
    payload = {
        "status": "passed",
        "layers": list(model.layers),
        "depth": len(model.backbone.blocks),
        "embed_dim": int(model.backbone.embed_dim),
        "patch_size": int(model.backbone.patch_size),
        "input_shape": list(x.shape),
        "output_shapes": {key: (list(value.shape) if value is not None else None) for key, value in pred.items()},
        "loss": float(loss.detach().cpu()),
        "backward": True,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
