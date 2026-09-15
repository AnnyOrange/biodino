#!/usr/bin/env python3
"""Measure label-free CLS relation drift from a frozen checkpoint anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor

from dinov3.configs import get_default_config
from dinov3.data import make_dataset
from dinov3.data.loaders import make_data_loader
from dinov3.eval.bio_segmentation.model_utils import load_dinov3_backbone


LOGGER = logging.getLogger("frozen_anchor_relation_drift")


@dataclass(frozen=True)
class Arm:
    name: str
    checkpoint: Path
    config: Path


class CleanTwoCropTransform:
    """Generate two deterministic clean square crops from each training image."""

    def __init__(
        self,
        *,
        size: int,
        area_min: float,
        seed: int,
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> None:
        self.size = size
        self.area_min = area_min
        self.generator = torch.Generator().manual_seed(seed)
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def _crop(self, image: Tensor) -> Tensor:
        _, height, width = image.shape
        area = float(
            torch.empty(()).uniform_(self.area_min, 1.0, generator=self.generator)
        )
        side = max(2, min(height, width, int(round(math.sqrt(area) * min(height, width)))))
        top = int(torch.randint(height - side + 1, (), generator=self.generator).item())
        left = int(torch.randint(width - side + 1, (), generator=self.generator).item())
        crop = image[:, top : top + side, left : left + side].float()
        crop = F.interpolate(
            crop.unsqueeze(0),
            size=(self.size, self.size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        return (crop - self.mean) / self.std

    def __call__(self, image: Tensor) -> dict[str, Any]:
        if image.ndim != 3 or image.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"Expected CHW image with {self.mean.shape[0]} channels, got {tuple(image.shape)}"
            )
        return {"views": torch.stack((self._crop(image), self._crop(image)))}


def collate(samples: list[tuple[dict[str, Any], tuple]]) -> dict[str, Any]:
    views = [sample[0] for sample in samples]
    return {
        "views": torch.stack([view["views"] for view in views]),
        "keys": [str(view.get("__key__", "")) for view in views],
        "urls": [str(view.get("__url__", "")) for view in views],
    }


def parse_arm(raw: str) -> Arm:
    parts = raw.split("=", 1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("arm must be NAME=CHECKPOINT,CONFIG")
    files = parts[1].split(",", 1)
    if len(files) != 2 or not all(files):
        raise argparse.ArgumentTypeError("arm must be NAME=CHECKPOINT,CONFIG")
    return Arm(parts[0], Path(files[0]).resolve(), Path(files[1]).resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(
    arm: Arm,
    *,
    loader: Any,
    batches: int,
    device: torch.device,
) -> tuple[Tensor, list[str]]:
    backbone = load_dinov3_backbone(
        str(arm.checkpoint), str(arm.config), device=device, freeze=True
    ).eval()
    chunks: list[Tensor] = []
    keys: list[str] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= batches:
            break
        images = batch["views"].transpose(0, 1).flatten(0, 1).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = backbone(images, is_training=True)["x_norm_clstoken"]
        batch_size = len(batch["keys"])
        chunks.append(output.unflatten(0, (2, batch_size)).float().cpu())
        keys.extend(batch["keys"])
        LOGGER.info("%s batch %d/%d (%d images)", arm.name, batch_index + 1, batches, len(keys))
    del backbone
    torch.cuda.empty_cache()
    if len(chunks) != batches:
        raise RuntimeError(f"{arm.name} yielded {len(chunks)} batches, expected {batches}")
    return torch.cat(chunks, dim=1), keys


def bootstrap_mean(values: np.ndarray, *, reps: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, 1000):
        stop = min(start + 1000, reps)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(means, (0.025, 0.975))],
        "n_images": len(values),
    }


def relation_metrics(features: Tensor, anchor: Tensor) -> tuple[dict[str, float], dict[str, list[float]]]:
    features = F.normalize(features.float(), dim=-1)
    anchor = F.normalize(anchor.float(), dim=-1)
    relation = features @ features.transpose(-1, -2)
    anchor_relation = anchor @ anchor.transpose(-1, -2)
    difference = relation - anchor_relation
    n_images = features.shape[1]
    offdiag = ~torch.eye(n_images, dtype=torch.bool).unsqueeze(0)
    row_mse = (difference.square() * offdiag).sum(dim=(0, 2)) / (2 * (n_images - 1))

    masked_anchor = anchor_relation.masked_fill(~offdiag, -torch.inf)
    masked_relation = relation.masked_fill(~offdiag, -torch.inf)
    anchor_top1 = masked_anchor.argmax(dim=-1)
    top1 = masked_relation.argmax(dim=-1)
    top1_per_image = (top1 == anchor_top1).float().mean(dim=0)
    anchor_top5 = masked_anchor.topk(k=min(5, n_images - 1), dim=-1).indices
    top5 = masked_relation.topk(k=min(5, n_images - 1), dim=-1).indices
    top5_overlap = (
        (top5.unsqueeze(-1) == anchor_top5.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)
    ).mean(dim=0)
    summary = {
        "relation_mse_including_diagonal": float(difference.square().mean()),
        "relation_mse_offdiagonal": float(
            difference.square().masked_select(offdiag.expand_as(difference)).mean()
        ),
        "anchor_top1_neighbor_agreement": float(top1_per_image.mean()),
        "anchor_top5_neighbor_overlap": float(top5_overlap.mean()),
    }
    records = {
        "row_relation_mse_offdiagonal": row_mse.tolist(),
        "anchor_top1_neighbor_agreement": top1_per_image.tolist(),
        "anchor_top5_neighbor_overlap": top5_overlap.tolist(),
    }
    return summary, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True, type=parse_arm)
    parser.add_argument("--arm", action="append", required=True, type=parse_arm)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--area-min", type=float, default=0.32)
    parser.add_argument("--bootstrap-reps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch_size < 2 or args.batches < 1:
        raise ValueError("batch-size must be >=2 and batches must be positive")
    if not 0 < args.area_min <= 1:
        raise ValueError("area-min must be in (0, 1]")
    arms = [args.anchor, *args.arm]
    if len({arm.name for arm in arms}) != len(arms):
        raise ValueError("arm names must be unique")
    for arm in arms:
        if not arm.checkpoint.is_file() or not arm.config.is_file():
            raise FileNotFoundError(f"Missing input for {arm.name}: {arm}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(args.anchor.config))
    dataset_path = str(args.dataset or cfg.train.dataset_path)
    device = torch.device(args.device)
    features: dict[str, Tensor] = {}
    ordered_keys: list[str] | None = None
    for arm in arms:
        # Rebuild the loader so every backbone receives identical resampling and
        # crop RNG streams even if model construction consumes global RNG state.
        transform = CleanTwoCropTransform(
            size=args.crop_size,
            area_min=args.area_min,
            seed=args.seed + 101,
            mean=tuple(float(value) for value in cfg.crops.rgb_mean),
            std=tuple(float(value) for value in cfg.crops.rgb_std),
        )
        dataset = make_dataset(
            dataset_str=dataset_path,
            transform=transform,
            target_transform=lambda _: (),
            target_channels=int(cfg.student.in_chans),
            wds_shuffle_buffer=1,
            wds_resample_seed=args.seed,
            wds_deterministic_resampling=True,
        )
        loader = make_data_loader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            seed=args.seed,
            drop_last=True,
            persistent_workers=False,
            pin_memory=True,
            prefetch_factor=None,
            collate_fn=collate,
        )
        arm_features, keys = encode(arm, loader=loader, batches=args.batches, device=device)
        if ordered_keys is None:
            ordered_keys = keys
        elif keys != ordered_keys:
            raise RuntimeError(f"ordered image keys differ for {arm.name}")
        features[arm.name] = arm_features
    assert ordered_keys is not None
    if len(ordered_keys) != args.batch_size * args.batches:
        raise RuntimeError("unexpected image count")
    if len(set(ordered_keys)) != len(ordered_keys):
        raise RuntimeError("diagnostic image keys are not unique")

    anchor_features = features[args.anchor.name]
    summaries: dict[str, Any] = {}
    records: dict[str, Any] = {}
    for index, arm in enumerate(arms):
        summary, arm_records = relation_metrics(features[arm.name], anchor_features)
        row_mse = np.asarray(arm_records["row_relation_mse_offdiagonal"], dtype=np.float64)
        summary["row_relation_mse_bootstrap"] = bootstrap_mean(
            row_mse, reps=args.bootstrap_reps, seed=args.seed + index
        )
        summaries[arm.name] = summary
        records[arm.name] = arm_records

    relation_gate: dict[str, Any] | None = None
    required_names = {"control", "official_a7807", "dual"}
    if required_names <= summaries.keys():
        dual = summaries["dual"]["relation_mse_including_diagonal"]
        official = summaries["official_a7807"]["relation_mse_including_diagonal"]
        control = summaries["control"]["relation_mse_including_diagonal"]
        relation_gate = {
            "locked_before_dual_endpoint": True,
            "criteria": {
                "dual_over_official_a7807_at_most": 0.8,
                "dual_over_control_at_most": 1.1,
            },
            "observed": {
                "dual_over_official_a7807": dual / official,
                "dual_over_control": dual / control,
            },
            "pass": dual <= 0.8 * official and dual <= 1.1 * control,
        }

    payload = {
        "status": "VALID_COMPLETE_LABEL_FREE_RELATION_DIAGNOSTIC",
        "diagnostic": "frozen_anchor_clean_two_view_cls_relation_drift_v1",
        "dataset": dataset_path,
        "seed": args.seed,
        "n_images": len(ordered_keys),
        "n_unique_keys": len(set(ordered_keys)),
        "crop_size": args.crop_size,
        "crop_area_range": [args.area_min, 1.0],
        "views": 2,
        "anchor": args.anchor.name,
        "summaries": summaries,
        "decision_gate": relation_gate,
        "inputs": {
            arm.name: {
                "checkpoint": str(arm.checkpoint),
                "checkpoint_sha256": sha256(arm.checkpoint),
                "config": str(arm.config),
                "config_sha256": sha256(arm.config),
            }
            for arm in arms
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    if args.records_output:
        record_payload = {
            "diagnostic": payload["diagnostic"],
            "keys": ordered_keys,
            "arms": records,
        }
        args.records_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.records_output.with_name(
            f".{args.records_output.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(record_payload, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.records_output)
    print(json.dumps({"output": str(args.output), "decision_gate": relation_gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
