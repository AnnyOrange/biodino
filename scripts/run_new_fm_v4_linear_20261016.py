#!/usr/bin/env python3
"""Evaluate 2026 frozen image encoders on the existing v4 linear-probe datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
BENCH = Path("/mnt/huawei_deepcad/benchmark_model")
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(BENCH)]

import run_external_fm_linear_probe as probe  # noqa: E402
from benchmark_eval.encoders import ModelSpec  # noqa: E402
from dinov3.eval.bio_frozen_eval.run_classification import (  # noqa: E402
    resolve_dataset_resize_size,
    resolve_image_size,
)


MODELS = {
    "distillpath": BENCH / "Pathology_FM/DistillPath-KS16-Virchow2",
    "morphem": BENCH / "Cell_Bioimage_FM/MorphEm",
    "genbio_pathfm": BENCH / "Pathology_FM/GenBio-PathFM",
    "openpath": BENCH / "Pathology_FM/OpenPath",
}
V4 = json.loads((ROOT / "Evaluation Rules/protocol_v4.json").read_text())
V4_LINEAR_DATASETS = tuple(dict.fromkeys(
    dataset
    for section in ("tier_a", "tier_b", "union_extension")
    for task in ("classification", "regression")
    for dataset in V4[section].get(task, [])
))


class NewEncoder:
    def __init__(self, name: str, device: str, batch_size: int):
        self.name = name
        self.device = torch.device(device)
        self.batch_size = batch_size
        path = MODELS[name]
        self.crop_size = 224
        self.resize_size = 256
        if name == "distillpath":
            import timm
            from safetensors.torch import load_file

            self.model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
            self.model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
            self.mean = (0.5, 0.5, 0.5)
            self.std = (0.5, 0.5, 0.5)
        elif name == "morphem":
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(path, trust_remote_code=True, local_files_only=True)
            self.mean = self.std = None
        elif name == "genbio_pathfm":
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(path, trust_remote_code=True, local_files_only=True)
            self.mean = (0.697, 0.575, 0.728)
            self.std = (0.188, 0.240, 0.187)
        else:
            sys.path.insert(0, str(BENCH / "Pathology_FM/OpenPath-source/OpenPath"))
            import dinov2.models.vision_transformer as vits

            checkpoint = torch.load(path / "teacher_checkpoint.pth", map_location="cpu", weights_only=False)
            state = {key[len("backbone."):]: value for key, value in checkpoint["teacher"].items() if key.startswith("backbone.")}
            self.model = vits.vit_giant2(
                patch_size=14, img_size=224, block_chunks=4, num_register_tokens=4,
                ffn_layer="swiglufused", init_values=1e-5,
                interpolate_antialias=True, interpolate_offset=0.0,
            )
            self.model.load_state_dict(state, strict=True)
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def set_dataset(self, dataset: str) -> None:
        self.crop_size = resolve_image_size(dataset, "best", 224)
        self.resize_size = resolve_dataset_resize_size(dataset, self.crop_size, 0)

    def _prepare(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image.convert("RGB"))
        else:
            image = image.detach().float().clamp(0, 1)
            if image.shape[0] < 3:
                image = torch.cat((image, image[-1:].expand(3 - image.shape[0], -1, -1)))
            else:
                image = image[:3]
        height, width = image.shape[-2:]
        scale = self.resize_size / min(height, width)
        new_height = max(self.crop_size, round(height * scale))
        new_width = max(self.crop_size, round(width * scale))
        image = F.interpolate(image[None], size=(new_height, new_width), mode="bilinear", align_corners=False, antialias=True)[0]
        top = (new_height - self.crop_size) // 2
        left = (new_width - self.crop_size) // 2
        image = image[:, top:top + self.crop_size, left:left + self.crop_size]
        if self.crop_size != 224:
            image = F.interpolate(image[None], size=(224, 224), mode="bilinear", align_corners=False, antialias=True)[0]
        return image

    @torch.inference_mode()
    def encode_images(self, images: list) -> np.ndarray:
        chunks = []
        microbatch = min(self.batch_size, 2 if self.name in ("genbio_pathfm", "openpath") else 32)
        for start in range(0, len(images), microbatch):
            batch = torch.stack([self._prepare(image) for image in images[start:start + microbatch]]).to(self.device)
            if self.name == "morphem":
                features = []
                for channel in range(3):
                    gray = batch[:, channel:channel + 1]
                    gray = (gray - gray.mean((2, 3), keepdim=True)) / gray.std((2, 3), keepdim=True).clamp_min(1e-7)
                    features.append(self.model.forward_features(gray)["x_norm_clstoken"].float())
                feature = torch.cat(features, dim=1)
            else:
                mean = torch.tensor(self.mean, device=self.device)[None, :, None, None]
                std = torch.tensor(self.std, device=self.device)[None, :, None, None]
                batch = (batch - mean) / std
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    if self.name == "openpath":
                        feature = self.model.forward_features(batch)["x_norm_clstoken"]
                    else:
                        feature = self.model(batch)
            chunks.append(F.normalize(feature.float(), dim=1).cpu().numpy())
        return np.concatenate(chunks).astype(np.float16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--datasets", nargs="+", default=V4_LINEAR_DATASETS, choices=V4_LINEAR_DATASETS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    weight = MODELS[args.model] / ("teacher_checkpoint.pth" if args.model == "openpath" else "model.safetensors")
    if not weight.is_file():
        raise FileNotFoundError(weight)
    probe.MODEL_REGISTRY[args.model] = ModelSpec(args.model, "2026-frozen-encoder", MODELS[args.model])
    output = args.output_root / args.model / "linear"
    output.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(
        model=args.model, output_dir=str(output), feature_root=None,
        benchmark_root="/mnt/huawei_deepcad/benchmark", max_samples=None,
        max_per_class=None, batch_size=args.batch_size, num_workers=args.num_workers,
        overwrite_features=False, overwrite_results=False, no_save_features=False,
        save_paths=False, train_fraction=0.8, seed=0,
    )
    encoder = NewEncoder(args.model, "cuda", args.batch_size)
    failures = []
    for dataset in args.datasets:
        encoder.set_dataset(dataset)
        summary = output / "summary.csv"
        if probe._completed(summary, args.model, dataset, probe._split_label(settings, dataset)):
            print(f"[skip] {args.model}/{dataset}", flush=True)
            continue
        try:
            row = probe.evaluate_dataset(settings, encoder, dataset)
            row.update(
                protocol_id="bio-eval-union-v4",
                scope="OBSERVATIONAL" if args.model == "morphem" else "FROZEN_PROBE",
                preprocessing="dataset-best-crop/floating-input/model-published-normalization/224-input",
                crop_size=encoder.crop_size,
                resize_size=encoder.resize_size,
            )
            probe._append_csv(summary, row)
            (output / f"{dataset}.json").write_text(json.dumps(row, indent=2) + "\n")
            print(f"[complete] {args.model}/{dataset}", flush=True)
        except Exception as error:
            failures.append(dict(dataset=dataset, error=f"{type(error).__name__}: {error}"))
            print(f"[failed] {args.model}/{dataset}: {error}", flush=True)
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
