"""
Visualization utilities for bio-segmentation evaluation.

Covers two use cases:
  1. Linear probe prediction visualization (input / GT / prediction / error map).
  2. Zero-shot PCA feature visualization with optional foreground masking
     and K-Means unsupervised IoU evaluation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# Linear probe prediction visualization
# ============================================================================

def save_prediction_visualization(
    img: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
    save_path: str,
    mode: str,
) -> None:
    """
    Save a 4-panel figure: input | ground truth | prediction | error map.

    Args:
        img:  [3, H, W] input image tensor.
        mask: [H, W] ground-truth label tensor.
        pred: [2, H, W] logit tensor.
        save_path: output file path.
        mode: preprocessing mode label shown in title.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    img_np = img[0].cpu().numpy()
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title(f'Input ({mode})')
    axes[0].axis('off')

    mask_np = mask.cpu().numpy()
    axes[1].imshow(mask_np, cmap='tab20', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    pred_np = torch.argmax(pred, dim=0).cpu().numpy()
    axes[2].imshow(pred_np, cmap='tab20', vmin=0, vmax=1)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    diff = (pred_np != mask_np).astype(np.float32)
    axes[3].imshow(diff, cmap='Reds')
    axes[3].set_title('Error Map')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# PCA feature extraction
# ============================================================================

@torch.no_grad()
def extract_patch_features(
    model: nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Extract last-layer patch features from a DINOv3 backbone.

    Args:
        model:      DINOv3 backbone.
        img_tensor: [1, C, H, W] input tensor.
        device:     target device.

    Returns:
        features:   [N_patches, embed_dim] on CPU (float32).
        patch_grid: (H_patches, W_patches).
    """
    model.eval()
    img_tensor = img_tensor.to(device)
    _, _, h, w = img_tensor.shape
    h_p, w_p = h // model.patch_size, w // model.patch_size

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        features = model.get_intermediate_layers(
            img_tensor, n=1, reshape=True, norm=True
        )[0]  # [1, embed_dim, H_p, W_p]

    features = features.squeeze(0).permute(1, 2, 0).reshape(-1, features.shape[1])
    return features.float().cpu(), (h_p, w_p)


# ============================================================================
# PCA computation
# ============================================================================

def compute_pca_visualization(
    features: torch.Tensor,
    patch_grid: Tuple[int, int],
    n_components: int = 3,
) -> np.ndarray:
    """
    Project patch features to 3-D PCA and map to an RGB image.

    Returns:
        ndarray [H_p, W_p, 3], values in [0, 1].
    """
    from sklearn.decomposition import PCA
    h_p, w_p = patch_grid
    pca = PCA(n_components=n_components, whiten=True)
    projected = pca.fit_transform(features.numpy()).reshape(h_p, w_p, n_components)
    projected = torch.sigmoid(torch.from_numpy(projected) * 2.0)
    return projected.numpy()


def compute_pca_with_foreground(
    features: torch.Tensor,
    patch_grid: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
    n_components: int = 3,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    PCA visualization optionally fitted only on foreground patches.

    Args:
        features:   [N_patches, embed_dim].
        patch_grid: (H_p, W_p).
        mask:       optional binary foreground mask [H, W] (float in [0, 1]).
        n_components: PCA components.

    Returns:
        pca_image:       [H_p, W_p, 3] in [0, 1].
        fg_mask_patches: [H_p, W_p] bool array or None.
    """
    from sklearn.decomposition import PCA
    h_p, w_p = patch_grid
    fg_mask_patches = None

    if mask is not None:
        mask_resized = cv2.resize(
            mask.astype(np.float32), (w_p, h_p), interpolation=cv2.INTER_AREA
        )
        fg_flat = (mask_resized > 0.5).reshape(-1)
        fg_features = features[fg_flat]

        pca = PCA(n_components=n_components, whiten=True)
        if len(fg_features) > n_components:
            pca.fit(fg_features.numpy())
            projected = pca.transform(features.numpy())
        else:
            projected = pca.fit_transform(features.numpy())
        fg_mask_patches = fg_flat.reshape(h_p, w_p)
    else:
        pca = PCA(n_components=n_components, whiten=True)
        projected = pca.fit_transform(features.numpy())

    projected = projected.reshape(h_p, w_p, n_components)
    projected = torch.sigmoid(torch.from_numpy(projected) * 2.0)
    return projected.numpy(), fg_mask_patches


# ============================================================================
# Metric helpers
# ============================================================================

def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Binary IoU between two boolean arrays."""
    pred, target = pred.astype(bool), target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(intersection / (union + 1e-6))


def compute_pca_mask_correlation(
    pca_component: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """
    Pearson correlation and Otsu-threshold IoU between a PCA component and a mask.

    Returns:
        {'correlation': float, 'iou': float}
    """
    pca_flat, mask_flat = pca_component.flatten(), mask.flatten()

    if pca_flat.std() == 0 or mask_flat.std() == 0:
        correlation = 0.0
    else:
        correlation = float(abs(np.corrcoef(pca_flat, mask_flat)[0, 1]))
        if np.isnan(correlation):
            correlation = 0.0

    try:
        pca_8bit = ((pca_component - pca_component.min()) /
                    (pca_component.max() - pca_component.min() + 1e-8) * 255).astype(np.uint8)
        _, binary = cv2.threshold(pca_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = binary > 0
        iou = compute_iou(binary, mask > 0)
        if iou < 0.5:
            iou = compute_iou(~binary, mask > 0)
    except Exception:
        iou = 0.0

    return {'correlation': correlation, 'iou': iou}


def compute_kmeans_patch_iou(
    features: torch.Tensor,
    patch_grid: Tuple[int, int],
    mask: np.ndarray,
    n_clusters: int = 2,
    n_init: int = 10,
    random_state: int = 0,
) -> float:
    """
    K-Means (K=2) clustering at patch level; returns the best foreground IoU.

    Args:
        features:     [N_patches, embed_dim] CPU tensor.
        patch_grid:   (H_p, W_p).
        mask:         [H, W] binary mask.
        n_clusters:   number of clusters (default 2: bg / fg).
        n_init:       number of K-Means restarts.
        random_state: reproducibility seed.

    Returns:
        Best IoU between either cluster label and the downsampled GT mask.
    """
    from sklearn.cluster import KMeans
    h_p, w_p = patch_grid

    mask_p = cv2.resize(
        mask.astype(np.float32), (w_p, h_p), interpolation=cv2.INTER_AREA
    )
    mask_p = (mask_p > 0.5)

    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = kmeans.fit_predict(features.numpy().astype(np.float32)).reshape(h_p, w_p)

    return max(compute_iou(labels == 1, mask_p), compute_iou(labels == 0, mask_p))


# ============================================================================
# Figure saving
# ============================================================================

def save_pca_visualization(
    original_img: np.ndarray,
    pca_image: np.ndarray,
    mask: Optional[np.ndarray],
    save_path: str,
    mode: str,
    fg_mask_patches: Optional[np.ndarray] = None,
) -> None:
    """Save a single PCA visualization: preprocessed / PCA / upsampled / GT mask."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_cols = 4 if mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f'Preprocessed ({mode})')
    axes[0].axis('off')

    axes[1].imshow(pca_image)
    axes[1].set_title('PCA Features (RGB)')
    axes[1].axis('off')

    pca_up = cv2.resize(
        pca_image, (original_img.shape[1], original_img.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    axes[2].imshow(pca_up)
    axes[2].set_title('PCA Upsampled')
    axes[2].axis('off')

    if mask is not None:
        axes[3].imshow(mask, cmap='tab20')
        axes[3].set_title('Ground Truth Mask')
        axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_comparison_figure(
    results: Dict[str, Dict],
    save_path: str,
    mask: Optional[np.ndarray] = None,
    kmeans_ious: Optional[Dict[str, float]] = None,
) -> None:
    """
    Save a 3-row comparison figure for multiple preprocessing modes.

    Row 0: preprocessed image  (+ K-Means IoU in title if available)
    Row 1: PCA RGB features
    Row 2: PCA component-1 heatmap  (+ correlation / IoU vs mask if available)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    modes = list(results.keys())
    mode_labels = {'minmax': 'A: Min-Max', 'percentile': 'B: Percentile', 'hybrid': 'C: Hybrid'}

    fig, axes = plt.subplots(3, len(modes), figsize=(5 * len(modes), 12))

    for i, mode in enumerate(modes):
        pca_image = results[mode]['pca_image']
        h_p, w_p = pca_image.shape[:2]

        title = f"{mode_labels.get(mode, mode)}\nPreprocessed"
        if kmeans_ious and mode in kmeans_ious:
            title += f"\nK-Means IoU: {kmeans_ious[mode]:.3f}"
        axes[0, i].imshow(results[mode]['preprocessed'], cmap='gray')
        axes[0, i].set_title(title, fontsize=11)
        axes[0, i].axis('off')

        axes[1, i].imshow(pca_image)
        axes[1, i].set_title(f'PCA Features ({w_p}×{h_p} patches)', fontsize=11)
        axes[1, i].axis('off')

        comp1 = pca_image[:, :, 0]
        im = axes[2, i].imshow(comp1, cmap='hot')
        if mask is not None:
            mask_p = cv2.resize(
                mask.astype(np.float32), (w_p, h_p), interpolation=cv2.INTER_AREA
            )
            score = compute_pca_mask_correlation(comp1, mask_p)
            comp_title = f'Comp 1 | Corr: {score["correlation"]:.3f} | IoU: {score["iou"]:.3f}'
        else:
            comp_title = 'PCA Component 1'
        axes[2, i].set_title(comp_title, fontsize=10)
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)

    plt.suptitle('Preprocessing Method Comparison - PCA Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
