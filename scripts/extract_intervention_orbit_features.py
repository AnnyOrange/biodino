#!/usr/bin/env python3
"""Extract matched mature-HS6 features over an image-only acquisition orbit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dinov3.eval.bio_classification.common import create_linear_input
from dinov3.eval.bio_frozen_eval.intervention_stable_relations import ALL_VIEWS, make_intervention_views
from dinov3.eval.bio_segmentation.model_utils import load_dinov3_backbone


class PathDataset(Dataset):
    def __init__(self, paths: np.ndarray):
        self.paths = np.asarray(paths).astype(str)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            return image.convert("RGB"), path


def collate_images(batch):
    images, paths = zip(*batch)
    return list(images), list(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--autocast-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.batch_size <= 0 or args.num_workers < 0 or args.torch_num_threads <= 0:
        raise ValueError("Expected positive batch/threads and non-negative workers")
    for path in (args.input_npz, args.checkpoint, args.train_config):
        if not path.exists():
            raise FileNotFoundError(path)

    with np.load(args.input_npz, allow_pickle=False) as payload:
        if "paths" not in payload.files or "labels" not in payload.files:
            raise ValueError("input NPZ must contain paths and labels")
        paths = np.asarray(payload["paths"]).astype(str)
        labels = np.asarray(payload["labels"])
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        paths = paths[: args.max_samples]
        labels = labels[: args.max_samples]
    if len(paths) <= 20 or labels.shape[0] != len(paths):
        raise ValueError("Expected more than 20 aligned paths and labels")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first={missing[0]}")

    torch.set_num_threads(args.torch_num_threads)
    torch.set_num_interop_threads(min(4, args.torch_num_threads))
    device = torch.device(args.device)
    backbone = load_dinov3_backbone(
        str(args.checkpoint),
        str(args.train_config),
        device=device,
        freeze=True,
    ).eval()
    spatial_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.autocast_dtype
    ]
    loader = DataLoader(
        PathDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_images,
        pin_memory=True,
    )
    output_features: dict[str, dict[str, list[np.ndarray]]] = {
        view: {"nlb2_avg": [], "nlb2_cls": []} for view in ALL_VIEWS
    }
    seen_paths: list[str] = []

    with torch.inference_mode():
        for batch_index, (images, batch_paths) in enumerate(loader, 1):
            pixels = torch.stack([spatial_transform(image) for image in images])
            views = make_intervention_views(pixels)
            for view_name in ALL_VIEWS:
                normalized = ((views[view_name] - mean) / std).to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda" and amp_dtype != torch.float32,
                ):
                    tokens = backbone.get_intermediate_layers(
                        normalized,
                        n=2,
                        reshape=False,
                        return_class_token=True,
                    )
                    for readout, use_avgpool in (("nlb2_avg", True), ("nlb2_cls", False)):
                        encoded = create_linear_input(tokens, use_n_blocks=2, use_avgpool=use_avgpool)
                        encoded = F.normalize(encoded.float(), dim=-1)
                        output_features[view_name][readout].append(
                            encoded.cpu().numpy().astype(np.float16)
                        )
            seen_paths.extend(batch_paths)
            if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
                print(f"[orbit-features] {len(seen_paths)}/{len(paths)}", flush=True)

    if seen_paths != paths.tolist():
        raise RuntimeError("DataLoader changed path order")
    arrays = {}
    for readout in ("nlb2_avg", "nlb2_cls"):
        arrays[f"features_{readout}"] = np.stack(
            [np.concatenate(output_features[view][readout], axis=0) for view in ALL_VIEWS]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **arrays,
        paths=paths,
        labels=labels,
        view_names=np.asarray(ALL_VIEWS),
        construction_view_count=np.asarray(4, dtype=np.int64),
        input_npz=np.asarray(str(args.input_npz)),
        checkpoint=np.asarray(str(args.checkpoint)),
        train_config=np.asarray(str(args.train_config)),
        resize_crop=np.asarray([256, 224], dtype=np.int64),
        normalization_mean=np.asarray(IMAGENET_DEFAULT_MEAN, dtype=np.float32),
        normalization_std=np.asarray(IMAGENET_DEFAULT_STD, dtype=np.float32),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(paths),
                "views": list(ALL_VIEWS),
                "nlb2_avg_dim": int(arrays["features_nlb2_avg"].shape[-1]),
                "nlb2_cls_dim": int(arrays["features_nlb2_cls"].shape[-1]),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
