#!/usr/bin/env python3
"""Measure whether nested local/global crops contain usable spatial SSL signal.

This is a frozen-backbone, label-free diagnostic.  A local crop is sampled
inside a global crop and its patch footprints are mapped exactly onto the
global patch grid.  Local tokens are compared with:

* their area-weighted true parent region;
* a coherent but spatially shifted region from the same image; and
* the same coordinates in another image.

Inference is performed at the image level.  Patch tokens are averaged within
each image before bootstrapping, so the confidence intervals do not treat
correlated patches as independent observations.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor
from torchvision.transforms import v2

from dinov3.configs import get_default_config
from dinov3.data import make_dataset
from dinov3.data.augmentations import BioSafeIntensityJitter, ChannelAgnosticGaussianNoise, ChannelDropout
from dinov3.data.loaders import make_data_loader
from dinov3.eval.bio_segmentation.model_utils import load_dinov3_backbone


LOGGER = logging.getLogger("local_global_spatial_signal")


@dataclass(frozen=True)
class NestedCropSpec:
    global_size: int
    local_size: int
    global_area_min: float
    local_area_min: float
    local_area_max: float
    photometric_policy: str
    crop_seed: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


class NestedCropTransform:
    """Create a square local crop with coordinates relative to its global crop."""

    def __init__(self, spec: NestedCropSpec) -> None:
        self.spec = spec
        self.crop_generator = torch.Generator().manual_seed(spec.crop_seed)
        self.mean = torch.tensor(spec.mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(spec.std, dtype=torch.float32)[:, None, None]
        if spec.photometric_policy == "bio_safe":
            def common() -> list[torch.nn.Module]:
                return [
                    BioSafeIntensityJitter(
                        brightness=0.20,
                        contrast=0.25,
                        gamma=(0.85, 1.20),
                        offset=0.03,
                        p=0.8,
                    ),
                    ChannelAgnosticGaussianNoise(sigma_min=0.005, sigma_max=0.025, p=0.25),
                    ChannelDropout(drop_prob=0.15, p=0.10),
                ]

            self.global_photometric = v2.Compose(
                common()
                + [v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.15)]
            )
            self.local_photometric = v2.Compose(
                common()
                + [v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.7))], p=0.10)]
            )
        elif spec.photometric_policy == "none":
            self.global_photometric = self.local_photometric = torch.nn.Identity()
        else:
            raise ValueError(f"Unsupported photometric policy: {spec.photometric_policy}")

    def _uniform(self, low: float, high: float) -> float:
        return float(torch.empty(()).uniform_(low, high, generator=self.crop_generator))

    @staticmethod
    def _resize(image: Tensor, size: int) -> Tensor:
        return F.interpolate(
            image.unsqueeze(0),
            size=(size, size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    def __call__(self, image: Tensor) -> dict[str, Tensor | float]:
        if image.ndim != 3:
            raise ValueError(f"Expected CHW tensor, got shape={tuple(image.shape)}")
        image = image.float()
        channels, height, width = image.shape
        if channels != len(self.spec.mean):
            raise ValueError(f"Expected {len(self.spec.mean)} channels, got {channels}")

        max_side = min(height, width)
        global_area = self._uniform(self.spec.global_area_min, 1.0)
        global_side = max(2, min(max_side, int(round(math.sqrt(global_area) * max_side))))
        global_top = int(torch.randint(height - global_side + 1, (), generator=self.crop_generator).item())
        global_left = int(torch.randint(width - global_side + 1, (), generator=self.crop_generator).item())
        global_raw = image[
            :, global_top : global_top + global_side, global_left : global_left + global_side
        ]

        local_area = self._uniform(self.spec.local_area_min, self.spec.local_area_max)
        local_side = max(2, min(global_side, int(round(math.sqrt(local_area) * global_side))))
        local_top = int(torch.randint(global_side - local_side + 1, (), generator=self.crop_generator).item())
        local_left = int(torch.randint(global_side - local_side + 1, (), generator=self.crop_generator).item())
        local_raw = global_raw[
            :, local_top : local_top + local_side, local_left : local_left + local_side
        ]

        global_view = self._resize(global_raw, self.spec.global_size)
        local_view = self._resize(local_raw, self.spec.local_size)
        global_view = self.global_photometric(global_view)
        local_view = self.local_photometric(local_view)
        global_view = (global_view - self.mean) / self.std
        local_view = (local_view - self.mean) / self.std
        box = torch.tensor(
            [
                local_left / global_side,
                local_top / global_side,
                (local_left + local_side) / global_side,
                (local_top + local_side) / global_side,
            ],
            dtype=torch.float32,
        )
        return {
            "global": global_view,
            "local": local_view,
            "box_xyxy": box,
            "local_area": float((local_side / global_side) ** 2),
        }


def _collate_nested(samples: list[tuple[dict[str, Any], tuple]]) -> dict[str, Any]:
    views = [sample[0] for sample in samples]
    return {
        "global": torch.stack([view["global"] for view in views]),
        "local": torch.stack([view["local"] for view in views]),
        "box_xyxy": torch.stack([view["box_xyxy"] for view in views]),
        "local_area": torch.tensor([view["local_area"] for view in views]),
        "key": [str(view.get("__key__", "")) for view in views],
        "url": [str(view.get("__url__", "")) for view in views],
    }


def _overlap_weights(boxes: Tensor, local_grid: int, global_grid: int) -> Tensor:
    """Return normalized intersection areas with shape [B, L, G]."""
    dtype, device = boxes.dtype, boxes.device
    local_edges = torch.arange(local_grid + 1, device=device, dtype=dtype) / local_grid
    global_edges = torch.arange(global_grid + 1, device=device, dtype=dtype) / global_grid

    bx0, by0, bx1, by1 = boxes.unbind(dim=1)
    width, height = bx1 - bx0, by1 - by0
    local_x0 = bx0[:, None] + width[:, None] * local_edges[:-1]
    local_x1 = bx0[:, None] + width[:, None] * local_edges[1:]
    local_y0 = by0[:, None] + height[:, None] * local_edges[:-1]
    local_y1 = by0[:, None] + height[:, None] * local_edges[1:]

    overlap_x = (
        torch.minimum(local_x1[:, :, None], global_edges[None, None, 1:])
        - torch.maximum(local_x0[:, :, None], global_edges[None, None, :-1])
    ).clamp_min(0)
    overlap_y = (
        torch.minimum(local_y1[:, :, None], global_edges[None, None, 1:])
        - torch.maximum(local_y0[:, :, None], global_edges[None, None, :-1])
    ).clamp_min(0)
    weights = torch.einsum("byh,bxw->byxhw", overlap_y, overlap_x)
    weights = weights.reshape(boxes.shape[0], local_grid**2, global_grid**2)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).eps)


def _coherent_shift(tokens: Tensor, grid: int, generator: torch.Generator) -> Tensor:
    """Roll each feature map to a wrong location while preserving local structure."""
    shifted = []
    for token_map in tokens.reshape(tokens.shape[0], grid, grid, tokens.shape[-1]):
        while True:
            dy = int(torch.randint(grid, (), generator=generator).item())
            dx = int(torch.randint(grid, (), generator=generator).item())
            if dy != 0 or dx != 0:
                break
        shifted.append(torch.roll(token_map, shifts=(dy, dx), dims=(0, 1)).reshape(grid**2, -1))
    return torch.stack(shifted)


def _image_metrics(
    local_tokens: Tensor,
    global_tokens: Tensor,
    weights: Tensor,
    shift_generator: torch.Generator,
) -> dict[str, Tensor]:
    local_tokens = F.normalize(local_tokens.float(), dim=-1)
    global_tokens = F.normalize(global_tokens.float(), dim=-1)
    global_grid = math.isqrt(global_tokens.shape[1])
    if global_grid**2 != global_tokens.shape[1]:
        raise ValueError(f"Global token count is not square: {global_tokens.shape[1]}")

    true_target = F.normalize(weights @ global_tokens, dim=-1)
    shifted_global = _coherent_shift(global_tokens, global_grid, shift_generator)
    shifted_target = F.normalize(weights @ shifted_global, dim=-1)
    cross_target = F.normalize(weights @ global_tokens.roll(1, dims=0), dim=-1)

    true_cosine = (local_tokens * true_target).sum(dim=-1)
    shifted_cosine = (local_tokens * shifted_target).sum(dim=-1)
    cross_cosine = (local_tokens * cross_target).sum(dim=-1)

    similarity = local_tokens @ global_tokens.transpose(1, 2)
    positives = weights > 0
    top1 = similarity.argmax(dim=-1, keepdim=True)
    hit1 = positives.gather(dim=-1, index=top1).squeeze(-1).float()
    chance = positives.float().mean(dim=-1)

    positive_scores = similarity.masked_fill(~positives, -torch.inf).amax(dim=-1)
    best_positive_rank = 1 + (similarity > positive_scores.unsqueeze(-1)).sum(dim=-1)
    reciprocal_rank = best_positive_rank.float().reciprocal()

    return {
        "true_cosine": true_cosine.mean(dim=1),
        "shifted_cosine": shifted_cosine.mean(dim=1),
        "cross_cosine": cross_cosine.mean(dim=1),
        "true_minus_shifted": (true_cosine - shifted_cosine).mean(dim=1),
        "true_minus_cross": (true_cosine - cross_cosine).mean(dim=1),
        "hit1": hit1.mean(dim=1),
        "chance_hit1": chance.mean(dim=1),
        "mrr": reciprocal_rank.mean(dim=1),
    }


def _bootstrap_summary(values: Tensor, *, reps: int, seed: int) -> dict[str, float | int]:
    values = values.float().cpu()
    if values.numel() < 2:
        raise ValueError("At least two images are required for bootstrap inference")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(values.numel(), (reps, values.numel()), generator=generator)
    means = values[indices].mean(dim=1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=True)),
        "ci95_low": float(torch.quantile(means, 0.025)),
        "ci95_high": float(torch.quantile(means, 0.975)),
        "n_images": int(values.numel()),
    }


def _bootstrap_ratio(
    numerator: Tensor,
    denominator: Tensor,
    *,
    reps: int,
    seed: int,
) -> dict[str, float | int]:
    numerator, denominator = numerator.float().cpu(), denominator.float().cpu()
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(numerator.numel(), (reps, numerator.numel()), generator=generator)
    ratios = numerator[indices].mean(dim=1) / denominator[indices].mean(dim=1).clamp_min(1e-12)
    return {
        "ratio_of_means": float(numerator.mean() / denominator.mean().clamp_min(1e-12)),
        "ci95_low": float(torch.quantile(ratios, 0.025)),
        "ci95_high": float(torch.quantile(ratios, 0.975)),
        "n_images": int(numerator.numel()),
    }


def _summarize_subset(
    records: dict[str, list[Tensor]],
    mask: Tensor,
    *,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, (name, chunks) in enumerate(records.items()):
        values = torch.cat(chunks)[mask]
        result[name] = _bootstrap_summary(values, reps=reps, seed=seed + offset)
    hit = torch.cat(records["hit1"])[mask]
    chance = torch.cat(records["chance_hit1"])[mask]
    result["hit1_enrichment"] = _bootstrap_ratio(hit, chance, reps=reps, seed=seed + 100)
    return result


def _parse_layers(raw: str) -> list[int]:
    layers = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not layers or min(layers) < 0 or len(layers) != len(set(layers)):
        raise argparse.ArgumentTypeError("layers must be unique, non-negative block indices")
    return layers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--dataset", default=None, help="Override train.dataset_path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--records-output",
        type=Path,
        help="Optional per-image metric JSON for paired diagnostic comparisons.",
    )
    parser.add_argument("--layers", type=_parse_layers, default=_parse_layers("5,11,17,23"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--global-size", type=int, default=256)
    parser.add_argument("--local-size", type=int, default=112)
    parser.add_argument("--global-area-min", type=float, default=0.5)
    parser.add_argument("--local-area-min", type=float, default=0.05)
    parser.add_argument("--local-area-max", type=float, default=0.32)
    parser.add_argument("--photometric-policy", choices=("none", "bio_safe"), default="none")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--device", default="cuda:7")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch_size < 2:
        raise ValueError("batch-size must be >=2 for the cross-image control")
    if args.batches < 1 or args.bootstrap_reps < 100:
        raise ValueError("batches must be positive and bootstrap-reps must be >=100")
    if not (0 < args.local_area_min < args.local_area_max <= 1):
        raise ValueError("local crop area range must satisfy 0 < min < max <= 1")
    if args.workers != 0:
        raise ValueError("workers must be 0 so the isolated crop RNG defines one auditable stream")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(args.train_config))
    dataset_path = str(args.dataset or cfg.train.dataset_path)
    patch_size = int(cfg.student.patch_size)
    if args.global_size % patch_size or args.local_size % patch_size:
        raise ValueError("global-size and local-size must be divisible by the model patch size")

    crop_spec = NestedCropSpec(
        global_size=args.global_size,
        local_size=args.local_size,
        global_area_min=args.global_area_min,
        local_area_min=args.local_area_min,
        local_area_max=args.local_area_max,
        photometric_policy=args.photometric_policy,
        crop_seed=args.seed + 101,
        mean=tuple(float(x) for x in cfg.crops.rgb_mean),
        std=tuple(float(x) for x in cfg.crops.rgb_std),
    )
    dataset = make_dataset(
        dataset_str=dataset_path,
        transform=NestedCropTransform(crop_spec),
        target_transform=lambda _: (),
        target_channels=int(cfg.student.in_chans),
        wds_shuffle_buffer=1,
        wds_resample_seed=args.seed,
        wds_deterministic_resampling=True,
    )
    loader = make_data_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        seed=args.seed,
        drop_last=True,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        prefetch_factor=1 if args.workers > 0 else None,
        collate_fn=_collate_nested,
    )
    backbone = load_dinov3_backbone(
        args.checkpoint,
        args.train_config,
        device=device,
        freeze=True,
    ).eval()
    depth = len(backbone.blocks)
    if max(args.layers) >= depth:
        raise ValueError(f"Requested layer {max(args.layers)} but backbone depth is {depth}")

    records: dict[int, dict[str, list[Tensor]]] = {
        layer: {
            name: []
            for name in (
                "true_cosine",
                "shifted_cosine",
                "cross_cosine",
                "true_minus_shifted",
                "true_minus_cross",
                "hit1",
                "chance_hit1",
                "mrr",
            )
        }
        for layer in args.layers
    }
    areas: list[Tensor] = []
    keys: list[str] = []
    shift_generator = torch.Generator().manual_seed(args.seed + 17)
    local_grid, global_grid = args.local_size // patch_size, args.global_size // patch_size

    for batch_index, batch in enumerate(loader):
        if batch_index >= args.batches:
            break
        global_view = batch["global"].to(device, non_blocking=True)
        local_view = batch["local"].to(device, non_blocking=True)
        weights = _overlap_weights(batch["box_xyxy"].to(device), local_grid, global_grid)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            global_layers = backbone.get_intermediate_layers(global_view, n=args.layers)
            local_layers = backbone.get_intermediate_layers(local_view, n=args.layers)

        for layer, local_tokens, global_tokens in zip(args.layers, local_layers, global_layers):
            metrics = _image_metrics(local_tokens, global_tokens, weights, shift_generator)
            for name, values in metrics.items():
                records[layer][name].append(values.cpu())
        areas.append(batch["local_area"].cpu())
        keys.extend(batch["key"])
        LOGGER.info("processed batch %d/%d (%d images)", batch_index + 1, args.batches, len(keys))

    if not areas:
        raise RuntimeError("The data loader yielded no batches")
    all_areas = torch.cat(areas)
    area_edges = [args.local_area_min, 0.10, 0.20, args.local_area_max + 1e-6]
    area_edges = sorted(set(max(args.local_area_min, min(args.local_area_max + 1e-6, x)) for x in area_edges))

    layer_results: dict[str, Any] = {}
    for layer_offset, layer in enumerate(args.layers):
        layer_key = f"block_{layer + 1}"
        layer_results[layer_key] = {
            "all": _summarize_subset(
                records[layer],
                torch.ones_like(all_areas, dtype=torch.bool),
                reps=args.bootstrap_reps,
                seed=args.seed + 1000 * layer_offset,
            ),
            "area_bins": {},
        }
        for bin_index, (low, high) in enumerate(zip(area_edges[:-1], area_edges[1:])):
            mask = (all_areas >= low) & (all_areas < high)
            if int(mask.sum()) < 2:
                continue
            label = f"[{low:.2f},{min(high, args.local_area_max):.2f}{']' if high > args.local_area_max else ')'}"
            layer_results[layer_key]["area_bins"][label] = _summarize_subset(
                records[layer],
                mask,
                reps=args.bootstrap_reps,
                seed=args.seed + 1000 * layer_offset + 100 * (bin_index + 1),
            )

    final = layer_results[f"block_{args.layers[-1] + 1}"]["all"]
    supported_layers = sum(
        result["all"]["true_minus_shifted"]["ci95_low"] > 0
        and result["all"]["hit1_enrichment"]["ci95_low"] > 1
        for result in layer_results.values()
    )
    gate = {
        "preregistered_before_measurement": True,
        "criteria": {
            "final_true_minus_shifted_mean_at_least": 0.03,
            "final_true_minus_shifted_ci95_low_above": 0.0,
            "final_hit1_enrichment_mean_at_least": 3.0,
            "supporting_layers_required": min(3, len(args.layers)),
            "supporting_layer_definition": "margin CI95 low > 0 and enrichment CI95 low > 1",
        },
        "observed_supporting_layers": supported_layers,
    }
    gate["pass"] = bool(
        final["true_minus_shifted"]["mean"] >= 0.03
        and final["true_minus_shifted"]["ci95_low"] > 0
        and final["hit1_enrichment"]["ratio_of_means"] >= 3.0
        and supported_layers >= min(3, len(args.layers))
    )

    output = {
        "diagnostic": "frozen_nested_local_global_spatial_signal_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "train_config": str(Path(args.train_config).resolve()),
        "dataset": dataset_path,
        "seed": args.seed,
        "n_images": int(all_areas.numel()),
        "n_unique_keys": len(set(keys)),
        "patch_size": patch_size,
        "global_grid": global_grid,
        "local_grid": local_grid,
        "crop_spec": crop_spec.__dict__,
        "bootstrap_reps": args.bootstrap_reps,
        "controls": {
            "within_image": "independent nonzero toroidal shift per image and layer",
            "cross_image": "batch roll by one at the true coordinates",
        },
        "layers": layer_results,
        "decision_gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.records_output is not None:
        record_payload = {
            "diagnostic": output["diagnostic"],
            "checkpoint": output["checkpoint"],
            "seed": args.seed,
            "keys": keys,
            "local_area": all_areas.tolist(),
            "layers": {
                f"block_{layer + 1}": {
                    name: torch.cat(chunks).tolist()
                    for name, chunks in layer_records.items()
                }
                for layer, layer_records in records.items()
            },
        }
        args.records_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.records_output.with_name(f".{args.records_output.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(record_payload, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.records_output)
    LOGGER.info("wrote %s", args.output)
    print(json.dumps({"output": str(args.output), "decision_gate": gate}, indent=2))


if __name__ == "__main__":
    main()
