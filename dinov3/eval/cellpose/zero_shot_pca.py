#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Cellpose 16-bit Zero-Shot PCA 可视化

原理：
- 取 DINOv3 输出的最后一个 Layer 的所有 Patch Tokens
- 对其进行 PCA 降维到 3 维
- 将这 3 维特征映射为 RGB 图像

评估标准：
- 如果细胞主体呈现一种颜色，背景呈现另一种颜色，且边缘锐利，
  说明该预处理方案下的特征提取非常成功

使用方法:
    cd /mnt/huawei_deepcad/dinov3
    python -m dinov3.eval.cellpose.zero_shot_pca \
        --checkpoint /mnt/deepcad_nfs/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --model-size l \
        --data-path /mnt/deepcad_nfs/0-large-model-dataset/11-Cellpose \
        --output-dir dinov3/outputs/cellpose_pca
"""

import argparse
import json
import logging
import os
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cellpose_pca")


# ============================================================================
# 预处理方法 (与 linear_probe.py 共享)
# ============================================================================

def apply_preprocessing_single_channel(img: np.ndarray, mode: str = 'hybrid') -> np.ndarray:
    """
    对单通道 16-bit 图像应用预处理
    """
    img = img.astype(np.float32)
    
    # 如果通道全为0，直接返回0
    if img.max() == 0:
        return img
    
    if mode == 'minmax':
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
    elif mode == 'percentile':
        p_low, p_high = np.percentile(img, [0.3, 99.7])
        if p_high <= p_low:
            return img / (img.max() + 1e-8)
        img = np.clip(img, p_low, p_high)
        return (img - p_low) / (p_high - p_low + 1e-8)
    elif mode == 'hybrid':
        p_high = np.percentile(img, 99.9)
        if p_high == 0:
            return img
        img = np.clip(img, 0, p_high)
        return img / (p_high + 1e-8)
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")


def apply_preprocessing(img: np.ndarray, mode: str = 'hybrid') -> np.ndarray:
    """
    对 16-bit 图像应用预处理
    
    支持单通道和多通道图像，多通道时对每个通道独立处理
    
    Args:
        img: 16-bit 图像，可以是 (H, W) 或 (H, W, C)
        mode: 预处理模式
    
    Returns:
        归一化后的 0-1 图像，保持原始通道数
    """
    if len(img.shape) == 2:
        # 单通道
        return apply_preprocessing_single_channel(img, mode)
    elif len(img.shape) == 3:
        # 多通道：对每个通道独立处理
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            result[:, :, c] = apply_preprocessing_single_channel(img[:, :, c], mode)
        return result
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")


def resize_to_patch_multiple(img: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """
    将图像尺寸调整为 patch_size 的倍数
    """
    h, w = img.shape[:2]
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    
    if len(img.shape) == 3:
        return cv2.resize(img, (new_w, new_h))
    else:
        return cv2.resize(img, (new_w, new_h))


def load_and_preprocess_image(
    img_path: str,
    mode: str = 'hybrid',
    patch_size: int = 16,
) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
    """
    加载并预处理 16-bit 图像
    
    Cellpose 图像是 16-bit 三通道 RGB：
    - 通道0 (Blue): 通常为空
    - 通道1 (Green): 细胞质信号
    - 通道2 (Red): 细胞核信号
    
    每个通道独立进行预处理。
    
    Returns:
        tensor: 预处理后的图像 tensor [1, 3, H, W]
        original_size: 原始图像尺寸 (H, W)
        preprocessed_np: 预处理后的 numpy 数组用于可视化 (H, W, 3)
    """
    # 加载图像
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    original_size = img.shape[:2]
    
    # cv2 读取的是 BGR 格式，转换为 RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 应用预处理（对每个通道独立处理）
    img = apply_preprocessing(img, mode=mode)
    
    # 调整为 patch_size 的倍数
    img = resize_to_patch_multiple(img, patch_size)
    
    # 如果是单通道，扩展为三通道
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    
    preprocessed_np = img.copy()
    
    # 转换为 tensor [1, 3, H, W]
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    
    return tensor, original_size, preprocessed_np


# ============================================================================
# PCA 可视化
# ============================================================================

@torch.no_grad()
def extract_patch_features(
    model: nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    提取 patch 特征
    
    Args:
        model: DINOv3 backbone
        img_tensor: 输入图像 [1, 3, H, W]
        device: 设备
    
    Returns:
        features: patch 特征 [N_patches, embed_dim]
        patch_grid: patch 网格尺寸 (H_patches, W_patches)
    """
    model.eval()
    img_tensor = img_tensor.to(device)
    
    _, _, h, w = img_tensor.shape
    patch_size = model.patch_size
    h_patches = h // patch_size
    w_patches = w // patch_size
    
    # 提取最后一层特征
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        features = model.get_intermediate_layers(
            img_tensor, n=1, reshape=True, norm=True
        )[0]  # [1, embed_dim, H_patches, W_patches]
    
    # 重塑为 [N_patches, embed_dim]
    features = features.squeeze(0).permute(1, 2, 0)  # [H, W, embed_dim]
    features = features.reshape(-1, features.shape[-1])  # [N_patches, embed_dim]
    
    return features.float().cpu(), (h_patches, w_patches)


def compute_pca_visualization(
    features: torch.Tensor,
    patch_grid: Tuple[int, int],
    n_components: int = 3,
) -> np.ndarray:
    """
    计算 PCA 可视化
    
    Args:
        features: patch 特征 [N_patches, embed_dim]
        patch_grid: patch 网格尺寸 (H_patches, W_patches)
        n_components: PCA 组件数
    
    Returns:
        pca_image: PCA 可视化图像 [H_patches, W_patches, 3]
    """
    h_patches, w_patches = patch_grid
    
    # 执行 PCA
    pca = PCA(n_components=n_components, whiten=True)
    projected = pca.fit_transform(features.numpy())
    
    # 重塑为图像
    projected = projected.reshape(h_patches, w_patches, n_components)
    
    # 使用 sigmoid 映射到 [0, 1] 并增强颜色
    projected = torch.from_numpy(projected)
    projected = torch.sigmoid(projected * 2.0)  # 乘以 2 增强对比度
    
    return projected.numpy()


def compute_pca_with_foreground(
    features: torch.Tensor,
    patch_grid: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
    n_components: int = 3,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    计算 PCA 可视化（可选择只在前景上拟合 PCA）
    
    Args:
        features: patch 特征 [N_patches, embed_dim]
        patch_grid: patch 网格尺寸 (H_patches, W_patches)
        mask: 可选的前景掩码 [H, W]
        n_components: PCA 组件数
    
    Returns:
        pca_image: PCA 可视化图像 [H_patches, W_patches, 3]
        fg_mask_patches: patch 级别的前景掩码（如果提供了 mask）
    """
    h_patches, w_patches = patch_grid
    
    fg_mask_patches = None
    if mask is not None:
        # 将 mask 下采样到 patch 级别
        mask_resized = cv2.resize(
            mask.astype(np.float32),
            (w_patches, h_patches),
            interpolation=cv2.INTER_AREA
        )
        fg_mask_patches = (mask_resized > 0.5).reshape(-1)
        
        # 只用前景 patches 拟合 PCA
        fg_features = features[fg_mask_patches]
        if len(fg_features) > n_components:
            pca = PCA(n_components=n_components, whiten=True)
            pca.fit(fg_features.numpy())
            # 但对所有 patches 进行变换
            projected = pca.transform(features.numpy())
        else:
            # 前景 patches 太少，使用所有 patches
            pca = PCA(n_components=n_components, whiten=True)
            projected = pca.fit_transform(features.numpy())
        
        fg_mask_patches = fg_mask_patches.reshape(h_patches, w_patches)
    else:
        pca = PCA(n_components=n_components, whiten=True)
        projected = pca.fit_transform(features.numpy())
    
    # 重塑为图像
    projected = projected.reshape(h_patches, w_patches, n_components)
    projected = torch.from_numpy(projected)
    projected = torch.sigmoid(projected * 2.0)
    
    return projected.numpy(), fg_mask_patches


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算二值 IoU
    pred/target: bool 数组
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(intersection / (union + 1e-6))


def compute_pca_mask_correlation(
    pca_component: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """
    计算 PCA 第一主成分与 mask 的相关性评分
    
    返回：
    - correlation: 皮尔逊相关系数（越高越好）
    - iou: 二值化后的 IoU（使用 Otsu 阈值）
    """
    pca_flat = pca_component.flatten()
    mask_flat = mask.flatten()
    
    # 皮尔逊相关系数
    if pca_flat.std() == 0 or mask_flat.std() == 0:
        correlation = 0.0
    else:
        correlation = np.corrcoef(pca_flat, mask_flat)[0, 1]
    
    # 如果相关性为负，说明 PCA 特征与 mask 反向（背景更亮）
    # 取绝对值，因为我们只关心区分度
    correlation = abs(correlation) if not np.isnan(correlation) else 0.0
    
    # 使用 OpenCV Otsu 阈值二值化 PCA 并计算 IoU
    try:
        # 转换为 8-bit 图像用于 Otsu
        pca_8bit = ((pca_component - pca_component.min()) / 
                    (pca_component.max() - pca_component.min() + 1e-8) * 255).astype(np.uint8)
        _, pca_binary = cv2.threshold(pca_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pca_binary = pca_binary > 0
        
        # 计算 IoU
        intersection = np.logical_and(pca_binary, mask > 0).sum()
        union = np.logical_or(pca_binary, mask > 0).sum()
        iou = intersection / (union + 1e-8)
        
        # 如果 IoU 小于 0.5，尝试反向（说明前景背景反了）
        if iou < 0.5:
            pca_binary = ~pca_binary
            intersection = np.logical_and(pca_binary, mask > 0).sum()
            union = np.logical_or(pca_binary, mask > 0).sum()
            iou = intersection / (union + 1e-8)
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
    无监督 K-Means (K=2) 在 patch 级别计算 IoU，并处理簇编号不确定的问题：
    - 尝试“簇1是前景”和“簇0是前景”两种映射，取更大 IoU。

    Args:
        features: [N_patches, embed_dim] (CPU torch tensor)
        patch_grid: (H_patches, W_patches)
        mask: [H, W] 的二值 mask（0/1 或 float）
    """
    h_patches, w_patches = patch_grid

    # GT mask 下采样到 patch 分辨率（与 features 对齐）
    mask_patch = cv2.resize(
        mask.astype(np.float32),
        (w_patches, h_patches),
        interpolation=cv2.INTER_AREA,
    )
    mask_patch = (mask_patch > 0.5)

    # K-Means 聚类
    feats = features.numpy().astype(np.float32, copy=False)
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = kmeans.fit_predict(feats)  # [N_patches]
    pred_patch = labels.reshape(h_patches, w_patches)

    # 簇-前景对齐（取最优）
    iou_case1 = compute_iou(pred_patch == 1, mask_patch)
    iou_case2 = compute_iou(pred_patch == 0, mask_patch)
    return max(iou_case1, iou_case2)


def save_pca_visualization(
    original_img: np.ndarray,
    pca_image: np.ndarray,
    mask: Optional[np.ndarray],
    save_path: str,
    mode: str,
    fg_mask_patches: Optional[np.ndarray] = None,
):
    """
    保存 PCA 可视化结果
    """
    n_cols = 4 if mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    # 原始预处理图像
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f'Preprocessed ({mode})')
    axes[0].axis('off')
    
    # PCA 可视化
    axes[1].imshow(pca_image)
    axes[1].set_title('PCA Features (RGB)')
    axes[1].axis('off')
    
    # 上采样的 PCA 图像（与原图尺寸一致）
    pca_upsampled = cv2.resize(
        pca_image,
        (original_img.shape[1], original_img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    axes[2].imshow(pca_upsampled)
    axes[2].set_title('PCA Upsampled')
    axes[2].axis('off')
    
    if mask is not None:
        # Ground truth mask
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
):
    """
    保存三种预处理方法的对比图
    
    Args:
        results: 每种预处理方法的结果字典
        save_path: 保存路径
        mask: Ground truth mask [H, W]
        kmeans_ious: 每种预处理方法的 K-Means IoU（当前图的）
    """
    modes = list(results.keys())
    n_modes = len(modes)
    
    # 3 行：原图、PCA Features、Comp 1 热力图
    fig, axes = plt.subplots(3, n_modes, figsize=(5 * n_modes, 12))
    
    mode_titles = {
        'minmax': 'A: Min-Max',
        'percentile': 'B: Percentile',
        'hybrid': 'C: Hybrid',
    }
    
    for i, mode in enumerate(modes):
        result = results[mode]
        pca_image = result['pca_image']  # [H_patches, W_patches, 3]
        h_patches, w_patches = pca_image.shape[:2]
        
        # Row 1: 预处理图像
        axes[0, i].imshow(result['preprocessed'], cmap='gray')
        title = f"{mode_titles.get(mode, mode)}\nPreprocessed"
        # 如果有 K-Means IoU，添加到标题
        if kmeans_ious is not None and mode in kmeans_ious:
            title += f"\nK-Means IoU: {kmeans_ious[mode]:.3f}"
        axes[0, i].set_title(title, fontsize=11)
        axes[0, i].axis('off')
        
        # Row 2: PCA Features（显示 patch 尺寸）
        axes[1, i].imshow(pca_image)
        axes[1, i].set_title(f'PCA Features ({w_patches}×{h_patches} patches)', fontsize=11)
        axes[1, i].axis('off')
        
        # Row 3: Comp 1 热力图（如果有 mask，显示 corr 和 iou）
        comp1 = pca_image[:, :, 0]
        im = axes[2, i].imshow(comp1, cmap='hot')
        
        if mask is not None:
            # 将 mask 下采样到 patch 分辨率
            mask_patch = cv2.resize(
                mask.astype(np.float32),
                (w_patches, h_patches),
                interpolation=cv2.INTER_AREA
            )
            # 计算 corr 和 iou
            score = compute_pca_mask_correlation(comp1, mask_patch)
            title = f'Comp 1 | Corr: {score["correlation"]:.3f} | IoU: {score["iou"]:.3f}'
        else:
            title = 'PCA Component 1'
        axes[2, i].set_title(title, fontsize=10)
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
    
    plt.suptitle('Preprocessing Method Comparison - PCA Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# 模型加载
# ============================================================================

def load_dinov3_backbone(
    checkpoint_path: str,
    model_size: str = 'l',
    device: torch.device = torch.device('cuda'),
) -> nn.Module:
    """加载 DINOv3 backbone"""
    from dinov3.hub.backbones import dinov3_vitl16, dinov3_vit7b16
    
    logger.info(f"加载 DINOv3 {model_size.upper()} 模型...")
    
    if model_size.lower() == 'l':
        model = dinov3_vitl16(pretrained=False)
    elif model_size.lower() == '7b':
        model = dinov3_vit7b16(pretrained=False)
    else:
        raise ValueError(f"不支持的模型大小: {model_size}")
    
    # 加载 checkpoint
    logger.info(f"从 {checkpoint_path} 加载权重...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('backbone.'):
            k = k[9:]
        new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    logger.info(f"模型加载完成! embed_dim={model.embed_dim}, patch_size={model.patch_size}")
    
    return model


# ============================================================================
# 主函数
# ============================================================================

def run_pca_visualization(
    backbone: nn.Module,
    img_paths: List[str],
    mask_paths: List[str],
    output_dir: str,
    modes: List[str] = ['minmax', 'percentile', 'hybrid'],
    device: torch.device = torch.device('cuda'),
    max_samples: int = 20,
    kmeans_n_init: int = 10,
    kmeans_seed: int = 0,
):
    """
    运行 PCA 可视化实验
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 限制样本数量
    n_samples = min(len(img_paths), max_samples)
    
    logger.info(f"对 {n_samples} 张图像运行 PCA 可视化...")

    # 记录每种预处理的 K-Means IoU（最后取平均得到 mIoU）
    kmeans_ious: Dict[str, List[float]] = {m: [] for m in modes}
    
    for idx in tqdm(range(n_samples), desc="Processing"):
        img_path = img_paths[idx]
        mask_path = mask_paths[idx]
        
        # 加载 mask
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.float32)
        
        # 对每种预处理方法
        results = {}
        current_kmeans_ious: Dict[str, float] = {}  # 当前图的 K-Means IoU
        
        for mode in modes:
            # 加载并预处理图像
            img_tensor, original_size, preprocessed_np = load_and_preprocess_image(
                img_path, mode=mode, patch_size=backbone.patch_size
            )
            
            # 调整 mask 尺寸
            mask_resized = resize_to_patch_multiple(mask, backbone.patch_size)
            
            # 提取特征
            features, patch_grid = extract_patch_features(backbone, img_tensor, device)

            # K-Means 无监督 IoU（patch 级别）
            try:
                k_iou = compute_kmeans_patch_iou(
                    features=features,
                    patch_grid=patch_grid,
                    mask=mask_resized,
                    n_clusters=2,
                    n_init=kmeans_n_init,
                    random_state=kmeans_seed,
                )
                kmeans_ious[mode].append(k_iou)
                current_kmeans_ious[mode] = k_iou
            except Exception as e:
                logger.warning(f"K-Means 评估失败: idx={idx}, mode={mode}, err={e}")
                current_kmeans_ious[mode] = 0.0
            
            # 计算 PCA（在前景上拟合）
            pca_image, fg_mask_patches = compute_pca_with_foreground(
                features, patch_grid, mask=mask_resized
            )
            
            results[mode] = {
                'preprocessed': preprocessed_np,
                'pca_image': pca_image,
                'fg_mask_patches': fg_mask_patches,
            }
            
            # 保存单个结果
            mode_dir = os.path.join(output_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)
            
            save_pca_visualization(
                preprocessed_np, pca_image, mask_resized,
                os.path.join(mode_dir, f'pca_{idx:03d}.png'),
                mode, fg_mask_patches
            )
        
        # 保存对比图（传递 mask 和 K-Means IoU）
        save_comparison_figure(
            results,
            os.path.join(output_dir, f'comparison_{idx:03d}.png'),
            mask=mask_resized,
            kmeans_ious=current_kmeans_ious if current_kmeans_ious else None
        )

    # 保存/打印 K-Means mIoU 汇总
    kmeans_summary: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        vals = kmeans_ious.get(mode, [])
        if len(vals) == 0:
            kmeans_summary[mode] = {"miou": 0.0, "std": 0.0, "n": 0}
        else:
            kmeans_summary[mode] = {
                "miou": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": int(len(vals)),
            }

    try:
        best_mode = max(kmeans_summary.keys(), key=lambda m: kmeans_summary[m]["miou"])
        logger.info("=" * 60)
        logger.info("K-Means 无监督 mIoU (patch-level) 汇总")
        for mode in modes:
            s = kmeans_summary[mode]
            logger.info(f"- {mode}: mIoU={s['miou']:.4f} ± {s['std']:.4f} (n={s['n']})")
        logger.info(f"Best (K-Means mIoU): {best_mode} ({kmeans_summary[best_mode]['miou']:.4f})")
    except Exception:
        pass

    with open(os.path.join(output_dir, "kmeans_miou.json"), "w") as f:
        json.dump(
            {
                "kmeans": {
                    "n_clusters": 2,
                    "n_init": kmeans_n_init,
                    "seed": kmeans_seed,
                    "patch_level": True,
                },
                "by_mode": kmeans_summary,
                "per_image_iou": kmeans_ious,
            },
            f,
            indent=2,
        )

    logger.info(f"PCA 可视化完成! 结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Cellpose 16-bit Zero-Shot PCA Visualization')
    
    parser.add_argument('--data-path', type=str, required=True,
                        help='Cellpose 数据集根目录')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='DINOv3 checkpoint 路径')
    parser.add_argument('--model-size', type=str, default='7b',
                        choices=['l', '7b'], help='模型大小')
    parser.add_argument('--modes', type=str, nargs='+',
                        default=['minmax', 'percentile', 'hybrid'],
                        help='要测试的预处理模式')
    parser.add_argument('--max-samples', type=int, default=20,
                        help='最大样本数量')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test'], help='使用的数据集分割')
    parser.add_argument('--kmeans-n-init', type=int, default=20,
                        help='K-Means 的 n_init（重复初始化次数）')
    parser.add_argument('--kmeans-seed', type=int, default=0,
                        help='K-Means 的 random_state（复现用）')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'pca_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 获取数据路径
    split_dir = os.path.join(args.data_path, args.split, args.split)
    if not os.path.exists(split_dir):
        split_dir = os.path.join(args.data_path, args.split)
    
    img_paths = sorted(glob(os.path.join(split_dir, '*_img.png')))
    mask_paths = sorted(glob(os.path.join(split_dir, '*_masks.png')))
    
    logger.info(f"找到 {len(img_paths)} 张图像")
    
    # 加载模型
    backbone = load_dinov3_backbone(
        args.checkpoint,
        model_size=args.model_size,
        device=device
    )
    
    # 运行 PCA 可视化
    run_pca_visualization(
        backbone=backbone,
        img_paths=img_paths,
        mask_paths=mask_paths,
        output_dir=output_dir,
        modes=args.modes,
        device=device,
        max_samples=args.max_samples,
        kmeans_n_init=args.kmeans_n_init,
        kmeans_seed=args.kmeans_seed,
    )
    
    logger.info("=" * 60)
    logger.info("PCA 可视化分析")
    logger.info("=" * 60)
    logger.info("""
评估标准：
- 查看 PCA 可视化图像
- 如果细胞主体呈现一种颜色，背景呈现另一种颜色，且边缘锐利，
  说明该预处理方案下的特征提取非常成功
  
""")


if __name__ == '__main__':
    main()

