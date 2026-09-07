#!/usr/bin/env python3
"""Score Fig. 3 Panel B with ImageNet/DINOv2 baselines on the same pairs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT_DIR = REPO_ROOT / "outputs/04_figures/fig3_representation_20260812"
DEFAULT_DINOV3_OFFICIAL_WEIGHTS = (
    REPO_ROOT / "outputs/torch_cache/hub/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
EXTERNAL_DEPS = REPO_ROOT / "outputs/python_deps/fig3_ext"
BENCHMARK_MODEL_ROOT = Path("/mnt/huawei_deepcad/benchmark_model")


def compact_label(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    text = str(value).strip()
    return text if text and text.lower() != "nan" else default


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


class Fig3ImageDataset:
    def __init__(self, manifest: pd.DataFrame, transform):
        self.rows = manifest.to_dict(orient="records")
        self.transform = transform
        self._wsi_cache: dict[tuple[str, int], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        sample_id = compact_label(row["sample_id"])
        if compact_label(row.get("sample_type")) == "acrobat_tile":
            image = self._read_acrobat_tile(row)
        else:
            with Image.open(compact_label(row["image_path"])) as img:
                image = ImageOps.exif_transpose(img).convert("RGB")
        if self.transform is None:
            return image, sample_id
        return self.transform(image), sample_id

    def _read_acrobat_tile(self, row: dict[str, Any]) -> Image.Image:
        import tifffile

        from scripts.make_fig3_representation_draft import crop_tissue_tile

        path = compact_label(row["image_path"])
        page_idx = int(row.get("tile_page") or 0)
        key = (path, page_idx)
        if key not in self._wsi_cache:
            with tifffile.TiffFile(path) as tif:
                self._wsi_cache[key] = tif.pages[page_idx].asarray()
        return crop_tissue_tile(self._wsi_cache[key], row)


def collate(batch):
    import torch

    images, sample_ids = zip(*batch)
    if images and isinstance(images[0], Image.Image):
        return list(images), list(sample_ids)
    return torch.stack(list(images), dim=0), list(sample_ids)


def make_encoder(model_name: str, device: str):
    import torch
    from torch import nn
    from torchvision import models, transforms

    key = model_name.lower()
    if key == "imagenet_resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
        transform = weights.transforms(crop_size=224, resize_size=256)
    elif key == "random_resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=None)
        model.fc = nn.Identity()
        transform = weights.transforms(crop_size=224, resize_size=256)
    elif key == "dinov2_vitb14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    elif key in {"dinov2_local", "dinov2"}:
        for path in (
            EXTERNAL_DEPS,
            BENCHMARK_MODEL_ROOT / "_vendor/external_gapfill_py311",
            BENCHMARK_MODEL_ROOT,
            REPO_ROOT,
        ):
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from scripts.run_external_fm_linear_probe import ExternalEncoderAdapter

        model = ExternalEncoderAdapter("dinov2", device, batch_size=16)
        transform = None
    elif key in {"dinov3_official_vitl16", "official_dinov3_vitl16"}:
        from dinov3.data.transforms import make_classification_eval_transform
        from dinov3.eval.bio_classification.common import LinearFeatureModel
        from dinov3.hub import backbones

        if not DEFAULT_DINOV3_OFFICIAL_WEIGHTS.exists():
            raise FileNotFoundError(f"Official DINOv3 weights missing: {DEFAULT_DINOV3_OFFICIAL_WEIGHTS}")
        backbone = backbones.dinov3_vitl16(
            pretrained=True,
            weights=str(DEFAULT_DINOV3_OFFICIAL_WEIGHTS),
            check_hash=False,
        )
        model = LinearFeatureModel(
            backbone,
            n_last_blocks=1,
            use_avgpool=True,
            autocast_dtype=torch.bfloat16,
        )
        transform = make_classification_eval_transform(resize_size=256, crop_size=224)
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "parameters"):
        for param in model.parameters():
            param.requires_grad_(False)
    return model, transform


def extract_baseline_features(
    manifest: pd.DataFrame,
    model_name: str,
    out_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> tuple[np.ndarray, list[str]]:
    cache = out_dir / f"features_{model_name}.npz"
    if cache.exists() and not overwrite:
        pack = np.load(cache, allow_pickle=True)
        return pack["features"].astype(np.float32), [str(x) for x in pack["paths"]]

    import torch
    from torch.utils.data import DataLoader

    print(f"[fig3-baseline] loading {model_name}", flush=True)
    model, transform = make_encoder(model_name, device=device)
    dataset = Fig3ImageDataset(manifest, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    features: list[np.ndarray] = []
    paths: list[str] = []
    with torch.inference_mode():
        for i, (images, sample_ids) in enumerate(loader, 1):
            if hasattr(model, "encode_images"):
                feat = torch.from_numpy(model.encode_images(images).astype(np.float32))
            else:
                images = images.to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=device.startswith("cuda"), dtype=torch.bfloat16):
                    feat = model(images)
            if isinstance(feat, dict):
                feat = feat.get("x_norm_clstoken", next(iter(feat.values())))
            feat = feat.float()
            if feat.ndim > 2:
                feat = feat.flatten(1)
            feat = torch.nn.functional.normalize(feat, dim=1)
            features.append(feat.cpu().numpy().astype(np.float16))
            paths.extend(sample_ids)
            if i == 1 or i % 20 == 0 or i == len(loader):
                print(f"[fig3-baseline] {model_name}: {len(paths)}/{len(dataset)}", flush=True)
    arr = np.concatenate(features, axis=0).astype(np.float32)
    np.savez(cache, features=arr.astype(np.float16), paths=np.asarray(paths), model=model_name)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return arr, paths


def compute_pair_scores(pair_df: pd.DataFrame, features: np.ndarray, paths: list[str], model_name: str) -> pd.DataFrame:
    id_to_idx = {sample_id: i for i, sample_id in enumerate(paths)}
    x = l2_normalize(features)
    records: list[dict[str, Any]] = []
    for _, row in pair_df.iterrows():
        a = compact_label(row["anchor_id"])
        b = compact_label(row["other_id"])
        if a not in id_to_idx or b not in id_to_idx:
            continue
        rec = row.to_dict()
        rec["model"] = model_name
        rec["cosine_similarity"] = float(np.dot(x[id_to_idx[a]], x[id_to_idx[b]]))
        records.append(rec)
    return pd.DataFrame(records)


def bootstrap_alignment(pair_scores: pd.DataFrame, seed: int, rounds: int = 1000) -> dict[str, float]:
    pos = pair_scores[pair_scores["pair_type"].eq("same_biology_cross_modality")]
    neg = pair_scores[pair_scores["pair_type"].eq("different_biology_same_modality")]
    score = float(pos["cosine_similarity"].mean() - neg["cosine_similarity"].mean())
    entities = sorted(set(pair_scores["entity_id"].astype(str)))
    if len(entities) < 2:
        return {"score": score, "ci_low": score, "ci_high": score}
    rng = np.random.default_rng(seed)
    by_entity = {e: pair_scores[pair_scores["entity_id"].astype(str).eq(e)] for e in entities}
    vals = []
    for _ in range(rounds):
        draw = rng.choice(entities, size=len(entities), replace=True)
        boot = pd.concat([by_entity[e] for e in draw], axis=0)
        bpos = boot[boot["pair_type"].eq("same_biology_cross_modality")]
        bneg = boot[boot["pair_type"].eq("different_biology_same_modality")]
        if len(bpos) and len(bneg):
            vals.append(float(bpos["cosine_similarity"].mean() - bneg["cosine_similarity"].mean()))
    if not vals:
        return {"score": score, "ci_low": score, "ci_high": score}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"score": score, "ci_low": float(lo), "ci_high": float(hi)}


def summarize(model_scores: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for model, sub in model_scores.groupby("model", sort=False):
        pos = sub[sub["pair_type"].eq("same_biology_cross_modality")]["cosine_similarity"]
        neg = sub[sub["pair_type"].eq("different_biology_same_modality")]["cosine_similarity"]
        align = bootstrap_alignment(sub, seed=seed)
        rows.append(
            {
                "model": model,
                "n_pairs": int(len(sub)),
                "same_biology_cross_modality_mean": float(pos.mean()),
                "different_biology_same_modality_mean": float(neg.mean()),
                "alignment_score": align["score"],
                "ci_low": align["ci_low"],
                "ci_high": align["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def load_biodino_scores(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "panel_b_pair_scores.csv"
    df = pd.read_csv(path)
    df["model"] = "Biodino H+"
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--models", default="imagenet_resnet50,dinov2_local,dinov3_official_vitl16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--overwrite-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    manifest = pd.read_csv(out_dir / "feature_manifest.csv")
    pair_df = pd.read_csv(out_dir / "panel_b_test_pairs.csv")
    needed_ids = set(pair_df["anchor_id"].astype(str)).union(set(pair_df["other_id"].astype(str)))
    manifest = manifest[manifest["sample_id"].astype(str).isin(needed_ids)].reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("No manifest rows match Panel B pair ids.")

    all_scores = [load_biodino_scores(out_dir)]
    for model_name in [x.strip() for x in args.models.split(",") if x.strip()]:
        features, paths = extract_baseline_features(
            manifest=manifest,
            model_name=model_name,
            out_dir=out_dir,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite_features,
        )
        scores = compute_pair_scores(pair_df, features, paths, model_name)
        print(f"[fig3-baseline] {model_name}: scored {len(scores)} pairs", flush=True)
        all_scores.append(scores)

    model_scores = pd.concat(all_scores, axis=0, ignore_index=True)
    model_scores.to_csv(out_dir / "panel_b_model_pair_scores.csv", index=False)
    summary = summarize(model_scores, seed=args.seed)
    summary.to_csv(out_dir / "panel_b_model_alignment.csv", index=False)
    payload: dict[str, Any] = {
        "summary": summary.to_dict(orient="records"),
        "pair_scores": str(out_dir / "panel_b_model_pair_scores.csv"),
        "model_alignment": str(out_dir / "panel_b_model_alignment.csv"),
    }
    (out_dir / "panel_b_model_alignment.json").write_text(json.dumps(payload, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
