#!/usr/bin/env python3
"""BioDINO Frozen instance-seg shape/gradient smoke test."""
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
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layers", nargs=4, type=int, default=[7, 15, 23, 31])
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from dinov3.eval.bio_segmentation.instance_seg.model import build_dino_hovernet

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = build_dino_hovernet(
        checkpoint=args.checkpoint, train_config=args.train_config,
        layers=args.layers, num_types=0, freeze_backbone=True,
        feature_size=32, embed_proj=384, fusion_mode="bucket_concat",
        decoder_variant="current", device=device,
    )
    model.backbone.to(dtype=torch.bfloat16)
    model._bb_dtype = torch.bfloat16
    model.train()
    x = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float32)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = model(x)
        loss = sum(value.float().square().mean() for value in pred.values() if value is not None)
    loss.backward()
    torch.cuda.synchronize(device)
    backbone_grads = [param.grad for param in model.backbone.parameters()]
    decoder_grad = sum(float(param.grad.detach().abs().sum()) for param in model.decoder.parameters() if param.grad is not None)
    if any(grad is not None for grad in backbone_grads):
        raise RuntimeError("Frozen backbone received gradients")
    if not decoder_grad > 0.0:
        raise RuntimeError(f"Decoder gradient is zero: {decoder_grad}")
    payload = {
        "status": "passed", "layers": list(model.layers),
        "depth": len(model.backbone.blocks), "embed_dim": int(model.backbone.embed_dim),
        "patch_size": int(model.backbone.patch_size), "backbone_dtype": str(next(model.backbone.parameters()).dtype),
        "input_shape": list(x.shape),
        "output_shapes": {key: (list(value.shape) if value is not None else None) for key, value in pred.items()},
        "loss": float(loss.detach().cpu()), "backbone_grad": False,
        "decoder_grad_l1": decoder_grad,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
