#!/usr/bin/env python3
"""Render a qualitative spatial-feature panel for Fig. 3.

The panel uses one tissue image, extracts the final-layer patch tokens from the
frozen H+ checkpoint, projects patch features to RGB with PCA, and writes a raw
crop plus a feature-PCA map. It is intentionally separate from the main Fig. 3
assembler because it needs to load the 840M-parameter checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from PIL import Image, ImageFile, ImageOps

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT_DIR = REPO_ROOT / "outputs/04_figures/fig3_representation_20260812"
DEFAULT_CKPT = (
    REPO_ROOT
    / "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/ckpt/100/checkpoint.pth"
)
DEFAULT_CONFIG = REPO_ROOT / "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/config.yaml"
DEFAULT_IMAGE = Path(
    "/mnt/huawei_deepcad/benchmark/Representation/CIMA/extracted/"
    "lung-lesion_2/scale-25pc/29-041-Izd2-w35-He-les2.jpg"
)


def resize_center_crop_pil(img: Image.Image, resize_size: int, crop_size: int) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    width, height = img.size
    short = max(1, min(width, height))
    scale = float(resize_size) / float(short)
    new_w = max(crop_size, int(round(width * scale)))
    new_h = max(crop_size, int(round(height * scale)))
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    left = max(0, (new_w - crop_size) // 2)
    top = max(0, (new_h - crop_size) // 2)
    return img.crop((left, top, left + crop_size, top + crop_size))


def percentile_rgb(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(y, dtype=np.float32)
    for c in range(y.shape[-1]):
        lo, hi = np.percentile(y[..., c], [1.0, 99.0])
        out[..., c] = np.clip((y[..., c] - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def pca_to_rgb(tokens: np.ndarray, patch_h: int, patch_w: int, out_size: int) -> Image.Image:
    from sklearn.decomposition import PCA

    x = np.asarray(tokens, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    rgb = PCA(n_components=3, random_state=0).fit_transform(x)
    rgb = percentile_rgb(rgb.reshape(patch_h, patch_w, 3))
    return Image.fromarray(rgb, mode="RGB").resize((out_size, out_size), Image.Resampling.NEAREST)


def remove_low_order_spatial_trend(tokens: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:patch_h, 0:patch_w].astype(np.float32)
    xx = (xx.reshape(-1) / max(patch_w - 1, 1)) * 2.0 - 1.0
    yy = (yy.reshape(-1) / max(patch_h - 1, 1)) * 2.0 - 1.0
    design = np.stack([np.ones_like(xx), xx, yy, xx * yy, xx * xx, yy * yy], axis=1)
    coef, *_ = np.linalg.lstsq(design, tokens.astype(np.float32), rcond=None)
    residual = tokens.astype(np.float32) - design @ coef
    residual -= residual.mean(axis=0, keepdims=True)
    residual /= np.clip(np.linalg.norm(residual, axis=1, keepdims=True), 1e-6, None)
    return residual.astype(np.float32)


def clusters_to_rgb(tokens: np.ndarray, patch_h: int, patch_w: int, out_size: int) -> Image.Image:
    from sklearn.cluster import KMeans

    palette = np.array(
        [
            [36, 91, 126],
            [42, 157, 143],
            [233, 196, 106],
            [244, 162, 97],
            [231, 111, 81],
        ],
        dtype=np.uint8,
    )
    n_clusters = min(len(palette), max(2, tokens.shape[0] // 16))
    labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(tokens)
    rgb = palette[labels % len(palette)].reshape(patch_h, patch_w, 3)
    return Image.fromarray(rgb, mode="RGB").resize((out_size, out_size), Image.Resampling.NEAREST)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--n-last-blocks", type=int, default=1)
    parser.add_argument("--autocast-dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--channel-policy", default="auto")
    parser.add_argument("--channel-tta-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.image.exists():
        raise FileNotFoundError(f"Image is missing: {args.image}")

    import torch

    from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder, parse_autocast_dtype

    print(f"[fig3-spatial] loading encoder: {args.checkpoint}", flush=True)
    encoder = Dinov3CkptEncoder(
        checkpoint=args.checkpoint,
        train_config=args.train_config,
        device=args.device,
        n_last_blocks=args.n_last_blocks,
        use_avgpool=True,
        autocast_dtype=parse_autocast_dtype(args.autocast_dtype),
        image_size=args.image_size,
        resize_size=args.resize_size,
        channel_policy=args.channel_policy,
        channel_tta_samples=args.channel_tta_samples,
        channel_policy_seed=args.seed,
    )

    with Image.open(args.image) as img:
        crop = resize_center_crop_pil(img, args.resize_size, args.image_size)
    raw_path = args.output_dir / "panel_e_raw_crop.png"
    crop.save(raw_path)

    x = encoder.transform(crop).unsqueeze(0).to(encoder.device, non_blocking=True)
    dtype = parse_autocast_dtype(args.autocast_dtype)
    print("[fig3-spatial] extracting patch tokens", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=dtype):
        outputs = encoder.model.backbone.get_intermediate_layers(
            x,
            n=1,
            reshape=True,
            return_class_token=False,
        )
    fmap = outputs[0].float().cpu().numpy()[0]
    patch_h, patch_w = int(fmap.shape[1]), int(fmap.shape[2])
    tokens = np.moveaxis(fmap, 0, -1).reshape(patch_h * patch_w, fmap.shape[0])
    feature_img = pca_to_rgb(tokens, patch_h, patch_w, args.image_size)
    feature_path = args.output_dir / "panel_e_feature_pca.png"
    feature_img.save(feature_path)
    residual_tokens = remove_low_order_spatial_trend(tokens, patch_h, patch_w)
    residual_path = args.output_dir / "panel_e_feature_pca_detrended.png"
    pca_to_rgb(residual_tokens, patch_h, patch_w, args.image_size).save(residual_path)
    cluster_path = args.output_dir / "panel_e_feature_clusters.png"
    clusters_to_rgb(residual_tokens, patch_h, patch_w, args.image_size).save(cluster_path)

    meta = {
        "image": str(args.image),
        "raw_crop": str(raw_path),
        "feature_pca": str(feature_path),
        "feature_pca_detrended": str(residual_path),
        "feature_clusters": str(cluster_path),
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "patch_grid": [patch_h, patch_w],
        "feature_dim": int(fmap.shape[0]),
    }
    (args.output_dir / "panel_e_spatial_metadata.json").write_text(json.dumps(meta, indent=2))
    torch.cuda.empty_cache()
    print(f"[fig3-spatial] wrote {raw_path} and {feature_path}", flush=True)


if __name__ == "__main__":
    main()
