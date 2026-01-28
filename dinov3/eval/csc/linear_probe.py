# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
CSC (Cell Segmentation Challenge) Linear Probing 验证脚本

用于对比三种预处理方法对 DINOv3 特征提取效果的影响：
- A组 (minmax): 全局 Min-Max 归一化
- B组 (percentile): 截断 0.3% 和 99.7% 百分位
- C组 (hybrid): 截断 99.9% 后 Min-Max

使用方法:
    python -m dinov3.eval.csc.linear_probe \
        --checkpoint /mnt/deepcad_nfs/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --model-size l \
        --data-path /mnt/deepcad_nfs/0-large-model-dataset/56-CSC \
        --output-dir dinov3/outputs/csc_linear_probe \
        --epochs 10 \
        --batch-size 8
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from functools import partial
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 导入 CSC 数据加载工具
try:
    from .data_utils import (
        load_image_multi_format,
        load_mask_tiff,
        instance_mask_to_binary,
        get_csc_dataset_paths,
    )
except ImportError:
    from data_utils import (
        load_image_multi_format,
        load_mask_tiff,
        instance_mask_to_binary,
        get_csc_dataset_paths,
    )

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("csc_linear_probe")


# ============================================================================
# 预处理方法
# ============================================================================

def apply_preprocessing_single_channel(img: np.ndarray, mode: str = 'hybrid') -> np.ndarray:
    """
    对单通道 16-bit 图像应用预处理
    
    Args:
        img: 单通道 16-bit numpy array (H, W)
        mode: 预处理模式
    
    Returns:
        归一化后的 0-1 浮点数组
    """
    img = img.astype(np.float32)
    
    # 如果通道全为0，直接返回0
    if img.max() == 0:
        return img
    
    if mode == 'minmax':
        # A组: 全局 Min-Max 归一化
        # 问题：极亮噪点会压缩有效信号
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
        
    elif mode == 'percentile':
        # B组: 截断 0.3% 和 99.7% 百分位
        # 特点：能有效排除噪点，但可能截断过多高亮区域
        p_low, p_high = np.percentile(img, [0.3, 99.7])
        if p_high <= p_low:
            return img / (img.max() + 1e-8)
        img = np.clip(img, p_low, p_high)
        return (img - p_low) / (p_high - p_low + 1e-8)
        
    elif mode == 'hybrid':
        # C组: 截断 99.9% 后 Min-Max (假设背景在0附近)
        # 特点：只剔除最极端离群噪点，最大程度保留动态范围
        p_high = np.percentile(img, 99.9)
        if p_high == 0:
            return img
        img = np.clip(img, 0, p_high)
        return img / (p_high + 1e-8)
    
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")


def apply_preprocessing(img: np.ndarray, mode: str = 'hybrid') -> np.ndarray:
    """
    对图像应用预处理（支持 8-bit 和 16-bit）
    
    CSC 图像格式多样，可以是单通道或多通道，8-bit 或 16-bit。
    
    每个通道独立进行预处理。
    
    Args:
        img: 16-bit numpy array，可以是 (H, W) 或 (H, W, C)
        mode: 预处理模式 ('minmax', 'percentile', 'hybrid')
    
    Returns:
        归一化后的 0-1 浮点数组，保持原始通道数
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


# ============================================================================
# 数据集定义
# ============================================================================

def get_size_multiple_of_patch(original_size: Tuple[int, int], patch_size: int = 16) -> Tuple[int, int]:
    """
    将尺寸调整为 patch_size 的倍数
    
    Args:
        original_size: 原始尺寸 (H, W)
        patch_size: patch 大小
    
    Returns:
        调整后的尺寸 (H', W')，均为 patch_size 的倍数
    """
    h, w = original_size
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    return (new_h, new_w)


class CSCLinearProbeDataset(Dataset):
    """
    CSC (Cell Segmentation Challenge) 图像分割数据集
    
    支持读取多种格式的图像（tif, tiff, png, bmp）和位深（8bit, 16bit）
    """
    
    def __init__(
        self,
        img_paths: List[str],
        mask_paths: List[str],
        mode: str = 'hybrid',
        size: Optional[Tuple[int, int]] = None,
        patch_size: int = 16,
        augment: bool = False,
    ):
        """
        Args:
            img_paths: 图像文件路径列表
            mask_paths: 掩码文件路径列表
            mode: 预处理模式
            size: 输出图像尺寸。如果为 None，则自动调整为 patch_size 的倍数
            patch_size: ViT patch 大小，用于调整图像尺寸
            augment: 是否进行数据增强
        """
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.mode = mode
        self.size = size
        self.patch_size = patch_size
        self.augment = augment
        
        assert len(img_paths) == len(mask_paths), \
            f"图像数量 ({len(img_paths)}) 与掩码数量 ({len(mask_paths)}) 不匹配"
        
        # 如果没有指定 size，从第一张图像获取尺寸
        if self.size is None:
            sample_img = load_image_multi_format(self.img_paths[0])
            if sample_img is not None:
                h, w = sample_img.shape[:2]
                self.size = get_size_multiple_of_patch((h, w), patch_size)
                logger.info(f"自动设置图像尺寸: 原始 ({h}, {w}) -> 调整为 {self.size}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 加载图像（支持多种格式）
        img = load_image_multi_format(self.img_paths[idx])
        
        # 2. 处理通道格式
        if len(img.shape) == 2:
            # 单通道图像，扩展为三通道
            img = np.stack([img] * 3, axis=-1)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            # RGBA -> RGB
            img = img[:, :, :3]
        
        # 3. 应用预处理方案（对每个通道独立处理）
        img = apply_preprocessing(img, mode=self.mode)
        
        # 4. 调整尺寸 (size 是 (H, W)，cv2.resize 需要 (W, H))
        img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        
        # 5. 加载 Mask（TIFF instance mask）并转换为二值 (0:背景, 1:细胞)
        instance_mask = load_mask_tiff(self.mask_paths[idx])
        mask = instance_mask_to_binary(instance_mask)
        mask = cv2.resize(mask.astype(np.float32), (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        
        # 6. 简单数据增强
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
        
        # 7. 转换为 tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long()
        
        return img_tensor, mask_tensor


def get_dataset_paths(data_root: str, split: str = 'train') -> Tuple[List[str], List[str]]:
    """
    获取 CSC 数据集的图像和掩码路径（使用 CSC 数据加载函数）
    
    Args:
        data_root: 数据集根目录
        split: 'train' 或 'tune'
    
    Returns:
        (img_paths, mask_paths)
    """
    return get_csc_dataset_paths(data_root, split=split)


# ============================================================================
# 模型定义
# ============================================================================

class DINOv3LinearSegmenter(nn.Module):
    """
    基于冻结的 DINOv3 backbone 的线性分割模型
    
    只训练一个 1x1 卷积层 (线性分类器)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,
        patch_size: int = 16,
        use_multi_scale: bool = False,
        dropout: float = 0.1,
    ):
        """
        Args:
            backbone: 冻结的 DINOv3 模型
            num_classes: 分类数量 (2 = 背景 + 细胞)
            patch_size: ViT patch 大小
            use_multi_scale: 是否使用多尺度特征
            dropout: Dropout 比例
        """
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.use_multi_scale = use_multi_scale
        
        # 冻结 backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # 获取特征维度
        self.embed_dim = backbone.embed_dim
        
        # 根据是否使用多尺度特征调整输入通道数
        if use_multi_scale:
            # 使用最后4层特征
            self.n_layers = 4
            in_channels = self.embed_dim * self.n_layers
        else:
            # 只使用最后一层
            self.n_layers = 1
            in_channels = self.embed_dim
        
        # 线性分类头
        self.head = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )
        
        # 初始化
        nn.init.normal_(self.head[2].weight, mean=0, std=0.01)
        nn.init.constant_(self.head[2].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, 3, H, W]
        
        Returns:
            logits: [B, num_classes, H, W]
        """
        b, c, h, w = x.shape
        
        # 确保尺寸是 patch_size 的倍数
        feat_h, feat_w = h // self.patch_size, w // self.patch_size
        
        # 提取特征 (冻结模式)
        with torch.no_grad():
            if self.use_multi_scale:
                # 获取多层特征
                features = self.backbone.get_intermediate_layers(
                    x, n=self.n_layers, reshape=True
                )
                # 将所有层特征插值到相同尺寸并拼接
                feat_list = []
                for feat in features:
                    if feat.shape[2:] != (feat_h, feat_w):
                        feat = F.interpolate(feat, size=(feat_h, feat_w), mode='bilinear', align_corners=False)
                    feat_list.append(feat)
                features = torch.cat(feat_list, dim=1)
            else:
                # 只获取最后一层
                features = self.backbone.get_intermediate_layers(
                    x, n=1, reshape=True
                )[0]
        
        # 通过分类头
        logits = self.head(features.float())
        
        # 插值回原始分辨率
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        
        return logits


# ============================================================================
# 评估指标
# ============================================================================

def calculate_miou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = 255,
) -> Tuple[float, Dict[str, float]]:
    """
    计算平均交并比 (mIoU)
    
    Args:
        pred: 预测 logits [B, C, H, W]
        target: 真实标签 [B, H, W]
        num_classes: 类别数量
        ignore_index: 忽略的标签值
    
    Returns:
        (mIoU, {class_name: IoU})
    """
    pred = torch.argmax(pred, dim=1)  # [B, H, W]
    
    ious = {}
    class_names = ['background', 'cell']
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        # 排除忽略区域
        valid = (target != ignore_index)
        pred_cls = pred_cls & valid
        target_cls = target_cls & valid
        
        intersection = (pred_cls & target_cls).sum().float().item()
        union = (pred_cls | target_cls).sum().float().item()
        
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        
        ious[class_names[cls] if cls < len(class_names) else f'class_{cls}'] = iou
    
    miou = np.nanmean(list(ious.values()))
    return miou, ious


def calculate_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算 Dice 系数 (仅针对前景类)"""
    pred = torch.argmax(pred, dim=1)
    pred_fg = (pred == 1).float()
    target_fg = (target == 1).float()
    
    intersection = (pred_fg * target_fg).sum()
    union = pred_fg.sum() + target_fg.sum()
    
    if union == 0:
        return 1.0
    
    return (2 * intersection / union).item()


# ============================================================================
# 可视化
# ============================================================================

def save_prediction_visualization(
    img: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
    save_path: str,
    mode: str,
):
    """
    保存预测可视化结果
    
    Args:
        img: 输入图像 [3, H, W]
        mask: 真实掩码 [H, W]
        pred: 预测 logits [2, H, W]
        save_path: 保存路径
        mode: 预处理模式
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 原图
    img_np = img[0].cpu().numpy()  # 取第一个通道
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title(f'Input ({mode})')
    axes[0].axis('off')
    
    # 真实掩码
    mask_np = mask.cpu().numpy()
    axes[1].imshow(mask_np, cmap='tab20', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # 预测掩码
    pred_np = torch.argmax(pred, dim=0).cpu().numpy()
    axes[2].imshow(pred_np, cmap='tab20', vmin=0, vmax=1)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # 差异图
    diff = (pred_np != mask_np).astype(np.float32)
    axes[3].imshow(diff, cmap='Reds')
    axes[3].set_title('Error Map')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 训练循环
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()
    # 但确保 backbone 保持 eval 模式
    model.backbone.eval()
    
    total_loss = 0.0
    total_miou = 0.0
    total_dice = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # 前向传播
        logits = model(imgs)
        loss = criterion(logits, masks)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算指标
        with torch.no_grad():
            miou, _ = calculate_miou(logits, masks)
            dice = calculate_dice(logits, masks)
        
        total_loss += loss.item()
        total_miou += miou
        total_dice += dice
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mIoU': f'{miou:.4f}',
            'Dice': f'{dice:.4f}',
        })
    
    return {
        'loss': total_loss / num_batches,
        'mIoU': total_miou / num_batches,
        'Dice': total_dice / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    save_vis: bool = False,
    vis_dir: Optional[str] = None,
    mode: str = 'hybrid',
    max_vis: int = 10,
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    
    total_loss = 0.0
    all_ious = {'background': [], 'cell': []}
    all_dice = []
    num_batches = 0
    vis_count = 0
    
    for imgs, masks in tqdm(dataloader, desc='Evaluating'):
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        logits = model(imgs)
        loss = criterion(logits, masks)
        
        miou, class_ious = calculate_miou(logits, masks)
        dice = calculate_dice(logits, masks)
        
        total_loss += loss.item()
        for cls, iou in class_ious.items():
            if not np.isnan(iou):
                all_ious[cls].append(iou)
        all_dice.append(dice)
        num_batches += 1
        
        # 保存可视化
        if save_vis and vis_dir and vis_count < max_vis:
            for i in range(min(imgs.size(0), max_vis - vis_count)):
                save_path = os.path.join(vis_dir, f'{mode}_sample_{vis_count}.png')
                save_prediction_visualization(
                    imgs[i], masks[i], logits[i],
                    save_path, mode
                )
                vis_count += 1
    
    return {
        'loss': total_loss / num_batches,
        'mIoU': np.mean([np.mean(v) for v in all_ious.values() if v]),
        'IoU_background': np.mean(all_ious['background']) if all_ious['background'] else 0,
        'IoU_cell': np.mean(all_ious['cell']) if all_ious['cell'] else 0,
        'Dice': np.mean(all_dice),
    }


# ============================================================================
# 模型加载
# ============================================================================

def load_dinov3_backbone(
    checkpoint_path: str,
    model_size: str = 'l',
    device: torch.device = torch.device('cuda'),
) -> nn.Module:
    """
    加载 DINOv3 backbone
    
    Args:
        checkpoint_path: checkpoint 文件路径
        model_size: 模型大小 ('l' 或 '7b')
        device: 设备
    
    Returns:
        加载好权重的模型
    """
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
    
    # 处理可能的 state_dict 包装
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # 移除可能的前缀
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
    
    logger.info(f"模型加载完成! embed_dim={model.embed_dim}, patch_size={model.patch_size}")
    
    return model


# ============================================================================
# 主实验流程
# ============================================================================

def run_experiment(
    mode: str,
    backbone: nn.Module,
    train_img_paths: List[str],
    train_mask_paths: List[str],
    test_img_paths: List[str],
    test_mask_paths: List[str],
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device('cuda'),
    num_workers: int = 4,
    use_multi_scale: bool = False,
) -> Dict[str, float]:
    """
    运行单个预处理模式的实验
    
    Args:
        mode: 预处理模式
        backbone: DINOv3 backbone
        其他参数...
    
    Returns:
        最终测试结果
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"开始实验: {mode.upper()}")
    logger.info(f"{'='*60}")
    
    # 创建输出目录
    exp_dir = os.path.join(output_dir, mode)
    vis_dir = os.path.join(exp_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 创建数据集
    train_dataset = CSCLinearProbeDataset(
        train_img_paths, train_mask_paths,
        mode=mode, size=(224, 224), augment=True
    )
    test_dataset = CSCLinearProbeDataset(
        test_img_paths, test_mask_paths,
        mode=mode, size=(224, 224), augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = DINOv3LinearSegmenter(
        backbone=backbone,
        num_classes=2,
        patch_size=backbone.patch_size,
        use_multi_scale=use_multi_scale,
    ).to(device)
    
    # 只训练分类头
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.CrossEntropyLoss()
    
    # 训练记录
    history = {'train': [], 'test': []}
    best_miou = 0.0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        scheduler.step()
        
        # 评估
        test_metrics = evaluate(
            model, test_loader, criterion, device,
            save_vis=(epoch == epochs),  # 最后一个 epoch 保存可视化
            vis_dir=vis_dir,
            mode=mode,
        )
        
        history['train'].append(train_metrics)
        history['test'].append(test_metrics)
        
        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f}, mIoU: {train_metrics['mIoU']:.4f} | "
            f"Test Loss: {test_metrics['loss']:.4f}, mIoU: {test_metrics['mIoU']:.4f}, "
            f"Dice: {test_metrics['Dice']:.4f}"
        )
        
        # 保存最佳模型
        if test_metrics['mIoU'] > best_miou:
            best_miou = test_metrics['mIoU']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': test_metrics,
            }, os.path.join(exp_dir, 'best_model.pth'))
    
    # 保存训练历史
    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # 最终结果
    final_results = {
        'mode': mode,
        'best_epoch': best_epoch,
        'best_mIoU': best_miou,
        **history['test'][-1],
    }
    
    logger.info(f"[{mode.upper()}] 最佳 mIoU: {best_miou:.4f} (Epoch {best_epoch})")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='CSC Linear Probing')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, required=True,
                        help='CSC 数据集根目录')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='DINOv3 checkpoint 路径')
    parser.add_argument('--model-size', type=str, default='l',
                        choices=['l', '7b'], help='模型大小')
    parser.add_argument('--use-multi-scale', action='store_true',
                        help='使用多尺度特征')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练 epoch 数')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    
    # 实验参数
    parser.add_argument('--modes', type=str, nargs='+',
                        default=['minmax', 'percentile', 'hybrid'],
                        help='要运行的预处理模式')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 保存配置
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 加载数据路径（CSC 使用 train 和 tune 作为训练和验证集）
    train_img_paths, train_mask_paths = get_dataset_paths(args.data_path, 'train')
    test_img_paths, test_mask_paths = get_dataset_paths(args.data_path, 'tune')
    
    # 加载 backbone
    backbone = load_dinov3_backbone(
        args.checkpoint,
        model_size=args.model_size,
        device=device
    )
    
    # 运行所有实验
    all_results = {}
    
    for mode in args.modes:
        results = run_experiment(
            mode=mode,
            backbone=backbone,
            train_img_paths=train_img_paths,
            train_mask_paths=train_mask_paths,
            test_img_paths=test_img_paths,
            test_mask_paths=test_mask_paths,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            num_workers=args.num_workers,
            use_multi_scale=args.use_multi_scale,
        )
        all_results[mode] = results
    
    # 汇总结果
    logger.info(f"\n{'='*60}")
    logger.info("实验结果汇总")
    logger.info(f"{'='*60}")
    
    summary = []
    for mode, results in all_results.items():
        summary.append({
            'mode': mode,
            'mIoU': results['best_mIoU'],
            'Dice': results['Dice'],
            'IoU_cell': results['IoU_cell'],
            'IoU_background': results['IoU_background'],
        })
        logger.info(
            f"[{mode.upper():12s}] mIoU: {results['best_mIoU']:.4f}, "
            f"Dice: {results['Dice']:.4f}, "
            f"Cell IoU: {results['IoU_cell']:.4f}"
        )
    
    # 保存汇总结果
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 找出最佳模式
    best_mode = max(summary, key=lambda x: x['mIoU'])
    logger.info(f"\n最佳预处理模式: {best_mode['mode'].upper()}, mIoU: {best_mode['mIoU']:.4f}")
    
    logger.info(f"\n实验完成! 结果保存在: {output_dir}")
    
    return all_results


if __name__ == '__main__':
    main()

