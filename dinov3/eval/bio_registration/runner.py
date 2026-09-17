"""Reproducible preflight and pair-level native registration smoke entrypoint."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import socket
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

from .core import apply_affine, fit_correspondences, patch_coordinates, register_descriptors, registration_metrics
from .datasets import aligned_landmarks, anhir_pairs, cima_pairs, grouped_partition, preflight_pairs


Image.MAX_IMAGE_PIXELS = None


def resized_image(path, max_side=1024, multiple=16):
    """Full field of view, aspect-preserving resize with no center crop/padding."""
    if isinstance(path, Image.Image):
        image = path.convert("RGB")
    else:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
    original_hw = (image.height, image.width)
    scale = min(1., max_side / max(original_hw))
    size = tuple(max(multiple, round(dimension * scale / multiple) * multiple)
                 for dimension in (image.width, image.height))
    image = image.resize(size, Image.Resampling.BILINEAR)
    return image, original_hw


def extract_dinov3_descriptors(backbone, path, layers=1, device="cuda", max_side=1024):
    """Reuse segmentation's spatial DINO forward, preserving original XY frame.

    Input may be a path or a PIL image (for physically calibrated volume slices).
    Concatenate patch descriptors from specified layers, not CLS image vectors.
    Last vs 4-even is architecture-dependent and provided by the frozen registry.
    """
    import torch
    from dinov3.eval.bio_segmentation.feature_extractor import _backbone_spatial_features
    patch_size = backbone.patch_size
    if isinstance(patch_size, (tuple, list)):
        if patch_size[0] != patch_size[1]:
            raise ValueError("Non-square patch size not yet supported")
        patch_size = patch_size[0]
    image, original_hw = resized_image(path, max_side, int(patch_size))
    tensor = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float().div(255)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
    with torch.inference_mode(), autocast:
        outputs = _backbone_spatial_features(backbone, tensor, layers,
                                             multichannel=False, channel_policy="auto")
        features = torch.cat(outputs, dim=1)[0].float().cpu()
    _, gh, gw = features.shape
    return features.permute(1, 2, 0).reshape(-1, features.shape[0]).numpy(), \
        patch_coordinates((gh, gw), original_hw), original_hw


def evaluate_pair(pair, source_features, target_features, source_coords, target_coords,
                  ratio=.9, threshold_fraction=.005, seed=0, matching_backend="scipy", device="cpu"):
    source_landmarks, target_landmarks = aligned_landmarks(pair)
    with Image.open(pair.target_image) as image:
        target_diagonal = float(np.hypot(image.width, image.height))
    # BIRL officially gives a supplied cover-table diagonal precedence over
    # a newly measured target-image diagonal. Preserve ANHIR's provided values.
    reference_diagonal = pair.reference_diagonal or target_diagonal
    started = time.monotonic()
    fitted = register_descriptors(source_features, target_features, source_coords, target_coords,
        ratio=ratio, threshold=threshold_fraction * target_diagonal, seed=seed,
        matching_backend=matching_backend, device=device)
    warped = apply_affine(source_landmarks, fitted.matrix)
    metrics = registration_metrics(source_landmarks, target_landmarks, warped, reference_diagonal)
    metrics.update({"reference_diagonal": reference_diagonal,
                    "diagonal_source": "official_cover" if pair.reference_diagonal else "target_image"})
    return {"pair_id": pair.pair_id, "group": pair.group,
            "success": fitted.success, "reason": fitted.reason,
            "matrix": fitted.matrix.tolist(), "matches": fitted.matches, "inliers": fitted.inliers,
            "matching_fit_seconds": time.monotonic() - started,
            "parameters": {"ratio": ratio, "threshold_fraction": threshold_fraction, "seed": seed},
            "metrics": metrics}


def sift_pair_smoke(pair, max_side=1024, seed=0):
    """Actual image-based CPU smoke, explicitly NOT an hs6 model result."""
    started = time.monotonic()
    data = []
    for path in (pair.source_image, pair.target_image):
        image, original_hw = resized_image(path, max_side)
        array = np.asarray(image)
        keypoints, descriptors = cv2.SIFT_create(nfeatures=2500).detectAndCompute(
            cv2.cvtColor(array, cv2.COLOR_RGB2GRAY), None)
        if descriptors is None:
            descriptors = np.empty((0, 128), np.float32)
        coords = np.asarray([keypoint.pt for keypoint in keypoints], dtype=float).reshape(-1, 2)
        coords *= np.array([original_hw[1] / image.width, original_hw[0] / image.height])
        data.append((descriptors, coords))
    result = evaluate_pair(pair, data[0][0], data[1][0], data[0][1], data[1][1], seed=seed)
    result.update({"model": "OpenCV-SIFT-CPU-SMOKE-NOT-HS6", "total_seconds": time.monotonic() - started})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ANHIR", "CIMA"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", default="scale-25pc")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--train-config")
    parser.add_argument("--layers", default="last")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    pairs = anhir_pairs(args.root) if args.dataset == "ANHIR" else cima_pairs(args.root, args.scale)
    if not pairs:
        raise ValueError("No registration pairs")
    excluded = [p.group for p in pairs if p.group.startswith(("lung-", "mammary-"))] if args.dataset == "ANHIR" else []
    partitions = grouped_partition(pairs, args.seed, exclude_development_groups=excluded)
    report = {"dataset": args.dataset, "hostname": socket.gethostname(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "preflight": preflight_pairs(pairs), "partition": partitions,
              "pair_manifest": [{**asdict(pair), "source_image": str(pair.source_image),
                  "target_image": str(pair.target_image), "source_landmarks": str(pair.source_landmarks),
                  "target_landmarks": str(pair.target_landmarks)} for pair in pairs]}
    if args.smoke:
        pair = pairs[args.pair_index]
        if args.checkpoint:
            if not args.train_config:
                parser.error("--train-config required for DINO")
            from dinov3.eval.bio_segmentation.model_utils import load_dinov3_backbone
            backbone = load_dinov3_backbone(args.checkpoint, args.train_config, device="cuda", freeze=True)
            depth = len(backbone.blocks)
            layers = 1 if args.layers == "last" else [int(x) for x in args.layers.split(",")]
            if layers != 1 and (not layers or layers != sorted(set(layers)) or min(layers) < 0 or max(layers) >= depth):
                raise ValueError("Invalid explicit layer indices")
            source, source_xy, _ = extract_dinov3_descriptors(backbone, pair.source_image, layers)
            target, target_xy, _ = extract_dinov3_descriptors(backbone, pair.target_image, layers)
            report["smoke"] = evaluate_pair(pair, source, target, source_xy, target_xy, seed=args.seed)
            report["smoke"].update({"model": args.checkpoint, "layers": layers})
        else:
            report["smoke"] = sift_pair_smoke(pair, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "preflight": report["preflight"], "smoke": report.get("smoke")}))


if __name__ == "__main__":
    main()
