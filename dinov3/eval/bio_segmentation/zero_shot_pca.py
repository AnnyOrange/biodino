"""
Zero-Shot PCA feature visualization for biological cell segmentation.

Extracts DINOv3 patch-level features, projects them to 3-D PCA, and
renders them as an RGB image.  Well-separated colors between cell body
and background indicate strong feature representations.

K-Means (K=2) unsupervised IoU is computed as a quantitative proxy.

Usage:
    python -m dinov3.eval.bio_segmentation.zero_shot_pca \\
        --dataset    cellpose \\
        --data-path  /data/Cellpose \\
        --checkpoint /ckpts/dinov3_vitl16.pth \\
        --output-dir ./outputs/pca

    python -m dinov3.eval.bio_segmentation.zero_shot_pca \\
        --dataset    csc \\
        --data-path  /data/CSC \\
        --checkpoint /ckpts/dinov3_vitl16.pth \\
        --output-dir ./outputs/pca \\
        --split      tune
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .datasets import DATASET_REGISTRY
from .model_utils import load_dinov3_backbone
from .preprocessing import apply_preprocessing, resize_to_patch_multiple
from .visualization import (
    compute_kmeans_patch_iou,
    compute_pca_with_foreground,
    extract_patch_features,
    save_comparison_figure,
    save_pca_visualization,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bio_segmentation.zero_shot_pca')


# ============================================================================
# Dataset-specific image loading for PCA (no fixed resize, no augment)
# ============================================================================

def _load_image_for_pca(
    DatasetClass,
    img_path: str,
    mode: str,
    patch_size: int,
) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
    """
    Load, preprocess and pad one image to patch-size multiples.

    Returns:
        tensor:         [1, 3, H', W'] float32 ready for the backbone.
        original_size:  (H, W) before padding.
        preprocessed:   (H', W', 3) float32 numpy array for visualization.
    """
    # Instantiate a temporary single-item dataset just to reuse load_image
    tmp = object.__new__(DatasetClass)
    img = tmp.load_image(img_path)

    original_size = img.shape[:2]

    # Ensure 3-channel
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    img = apply_preprocessing(img, mode=mode)
    img = resize_to_patch_multiple(img, patch_size)
    preprocessed = img.copy()

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, original_size, preprocessed


def _load_mask_for_pca(DatasetClass, mask_path: str, patch_size: int) -> np.ndarray:
    """Load and binarize a mask, then pad to patch-size multiples."""
    tmp = object.__new__(DatasetClass)
    mask = tmp.load_mask(mask_path).astype(np.float32)
    return resize_to_patch_multiple(mask, patch_size)


# ============================================================================
# Core PCA runner
# ============================================================================

def run_pca_visualization(
    backbone: nn.Module,
    DatasetClass,
    img_paths: List[str],
    mask_paths: List[str],
    output_dir: str,
    modes: List[str],
    device: torch.device,
    max_samples: int = 20,
    kmeans_n_init: int = 10,
    kmeans_seed: int = 0,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    n = min(len(img_paths), max_samples)
    logger.info(f"Running PCA visualization on {n} images...")

    kmeans_ious: Dict[str, List[float]] = {m: [] for m in modes}

    for idx in tqdm(range(n), desc='Processing'):
        mask = _load_mask_for_pca(DatasetClass, mask_paths[idx], backbone.patch_size)
        results: Dict[str, Dict] = {}
        current_k_ious: Dict[str, float] = {}

        for mode in modes:
            img_tensor, _, preprocessed = _load_image_for_pca(
                DatasetClass, img_paths[idx], mode, backbone.patch_size
            )
            features, patch_grid = extract_patch_features(backbone, img_tensor, device)

            try:
                k_iou = compute_kmeans_patch_iou(
                    features, patch_grid, mask,
                    n_clusters=2, n_init=kmeans_n_init, random_state=kmeans_seed,
                )
                kmeans_ious[mode].append(k_iou)
                current_k_ious[mode] = k_iou
            except Exception as exc:
                logger.warning(f"K-Means failed idx={idx} mode={mode}: {exc}")
                current_k_ious[mode] = 0.0

            pca_image, fg_mask = compute_pca_with_foreground(features, patch_grid, mask=mask)
            results[mode] = {
                'preprocessed': preprocessed,
                'pca_image': pca_image,
                'fg_mask_patches': fg_mask,
            }

            mode_dir = os.path.join(output_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)
            save_pca_visualization(
                preprocessed, pca_image, mask,
                os.path.join(mode_dir, f'pca_{idx:03d}.png'),
                mode, fg_mask,
            )

        save_comparison_figure(
            results,
            os.path.join(output_dir, f'comparison_{idx:03d}.png'),
            mask=mask,
            kmeans_ious=current_k_ious or None,
        )

    # Summarize K-Means mIoU
    summary: Dict[str, Dict] = {}
    for mode in modes:
        vals = kmeans_ious[mode]
        summary[mode] = {
            'miou': float(np.mean(vals)) if vals else 0.0,
            'std':  float(np.std(vals))  if vals else 0.0,
            'n':    len(vals),
        }

    logger.info('=' * 60)
    logger.info('K-Means unsupervised mIoU (patch-level) summary')
    for mode in modes:
        s = summary[mode]
        logger.info(f"  {mode}: mIoU={s['miou']:.4f} ± {s['std']:.4f} (n={s['n']})")
    if summary:
        best_mode = max(summary, key=lambda m: summary[m]['miou'])
        logger.info(f"  Best: {best_mode} ({summary[best_mode]['miou']:.4f})")

    with open(os.path.join(output_dir, 'kmeans_miou.json'), 'w') as f:
        json.dump({'kmeans': {'n_clusters': 2, 'n_init': kmeans_n_init, 'seed': kmeans_seed},
                   'by_mode': summary, 'per_image_iou': kmeans_ious}, f, indent=2)

    logger.info(f"PCA visualization complete. Results: {output_dir}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bio-segmentation zero-shot PCA visualization')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_REGISTRY),
                        help='Dataset name (e.g. cellpose, csc)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Dataset root directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='DCP ckpt dir or consolidated .pth (teacher/model/flat)')
    parser.add_argument(
        '--train-config',
        type=str,
        required=True,
        help='Training YAML merged with ssl_default_config; must match checkpoint.',
    )

    parser.add_argument('--split', type=str, default=None,
                        help='Dataset split (default: test for cellpose, tune for csc)')
    parser.add_argument('--modes', nargs='+', default=['minmax', 'percentile', 'hybrid'])
    parser.add_argument('--max-samples', type=int, default=20)
    parser.add_argument('--kmeans-n-init', type=int, default=20)
    parser.add_argument('--kmeans-seed', type=int, default=0)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{args.dataset}_pca_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    DatasetClass, get_paths = DATASET_REGISTRY[args.dataset]

    default_split = {'cellpose': 'test', 'csc': 'tune'}
    split = args.split or default_split.get(args.dataset, 'test')

    img_paths, mask_paths = get_paths(args.data_path, split)
    logger.info(f"Found {len(img_paths)} image-mask pairs")

    backbone = load_dinov3_backbone(args.checkpoint, args.train_config, device=device, freeze=True)

    run_pca_visualization(
        backbone=backbone,
        DatasetClass=DatasetClass,
        img_paths=img_paths,
        mask_paths=mask_paths,
        output_dir=output_dir,
        modes=args.modes,
        device=device,
        max_samples=args.max_samples,
        kmeans_n_init=args.kmeans_n_init,
        kmeans_seed=args.kmeans_seed,
    )


if __name__ == '__main__':
    main()
