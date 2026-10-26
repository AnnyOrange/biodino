#!/usr/bin/env python3
"""Evaluate 2026 encoder patch maps with the existing v4 detection proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dinov3.eval.bio_detection import center_probe
from run_new_fm_v4_linear_20261016 import MODELS, NewEncoder, ROOT


class NewSpatialBackbone(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.encoder = NewEncoder(name, "cuda:0", 8)
        self.model = self.encoder.model
        self.patch_size = 16
        self.native_patch_size = 14 if name == "openpath" else 16
        self.input_mean = (0.485, 0.456, 0.406)
        self.input_std = (0.229, 0.224, 0.225)
        self.embed_dim = {"distillpath": 384, "morphem": 1152, "genbio_pathfm": 4608, "openpath": 1536}[name]

    @torch.inference_mode()
    def get_intermediate_layers(self, images: torch.Tensor, n=1, reshape=True, **kwargs):
        if not reshape or n != 1:
            raise ValueError("The v4 detection proxy expects one final spatial feature map")
        if self.name in ("genbio_pathfm", "openpath") and len(images) > 2:
            maps = [self.get_intermediate_layers(images[start:start + 2], n=n, reshape=reshape)[0]
                    for start in range(0, len(images), 2)]
            return [torch.cat(maps, dim=0)]
        mean = images.new_tensor(self.input_mean).view(1, 3, 1, 1)
        std = images.new_tensor(self.input_std).view(1, 3, 1, 1)
        pixels = (images * std + mean).clamp(0, 1)
        if self.name == "morphem":
            patches = []
            for channel in range(3):
                gray = pixels[:, channel:channel + 1]
                gray = (gray - gray.mean((2, 3), keepdim=True)) / gray.std((2, 3), keepdim=True).clamp_min(1e-7)
                patches.append(self.model.forward_features(gray)["x_norm_patchtokens"])
            tokens = torch.cat(patches, dim=-1)
        else:
            published_mean = images.new_tensor(self.encoder.mean).view(1, 3, 1, 1)
            published_std = images.new_tensor(self.encoder.std).view(1, 3, 1, 1)
            pixels = (pixels - published_mean) / published_std
            if self.name == "distillpath":
                tokens = self.model.forward_features(pixels)[:, 1:]
            elif self.name == "genbio_pathfm":
                tokens = self.model.forward_with_patches(pixels)[1]
            else:
                tokens = self.model.forward_features(pixels)["x_norm_patchtokens"]
        grid = images.shape[-1] // self.native_patch_size
        if tokens.shape[1] != grid * grid:
            raise ValueError(f"Expected {grid * grid} patch tokens; got {tokens.shape[1]}")
        spatial = tokens.transpose(1, 2).reshape(len(images), -1, grid, grid)
        target_grid = images.shape[-1] // self.patch_size
        if grid != target_grid:
            spatial = torch.nn.functional.interpolate(
                spatial.float(), size=(target_grid, target_grid), mode="bilinear", align_corners=False,
            )
        return [spatial.float()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--dataset", required=True, choices=("bbbc038", "conic", "livecell"))
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016")
    args = parser.parse_args()
    output = args.output_root / args.model / "detection" / args.dataset / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "component_result.json"
    if result_path.exists():
        print(f"[skip] {result_path}", flush=True)
        return 0
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    center_probe.load_backbone = lambda **kwargs: NewSpatialBackbone(args.model)
    result = center_probe.run_bio_detection_eval(
        arch="vit_base", weights="", checkpoint=str(MODELS[args.model]), train_config=None,
        benchmark_root="/mnt/huawei_deepcad/benchmark", output_dir=str(output),
        dataset=args.dataset, batch_size=8, num_workers=2, image_size=224,
        epochs=5, lr=1e-3, autocast_dtype=torch.bfloat16, channel_policy="auto",
        max_samples_per_split=0, seed=args.seed,
        conic_split_protocol="official-baseline-fold0-nested-v1",
    )
    result.update(
        status="OBSERVATIONAL", aggregate_allowed=False,
        union_status="V4_DETECTION_PROXY_MULTISEED", model=args.model,
        protocol_id="bio-eval-union-v4", probe_seed=args.seed,
        feature_grid_policy="published-final-map-to-common-stride16-decoder-grid",
    )
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(result_path)
    print(f"[complete] {args.model}/{args.dataset}/seed{args.seed}: {result['test_patch_f1']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
