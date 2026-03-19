"""
Unified Linear Probe for biological cell segmentation.

Trains a frozen DINOv3 backbone + a single linear segmentation head on any
dataset registered in DATASET_REGISTRY.  Compares three preprocessing modes
(minmax / percentile / hybrid) and reports mIoU / Dice per mode.

Usage:
    python -m dinov3.eval.bio_segmentation.linear_probe \\
        --dataset    cellpose \\
        --data-path  /data/Cellpose \\
        --checkpoint /ckpts/dinov3_vitl16.pth \\
        --output-dir ./outputs/linear_probe \\
        --model-size l \\
        --epochs 10

    python -m dinov3.eval.bio_segmentation.linear_probe \\
        --dataset    csc \\
        --data-path  /data/CSC \\
        --checkpoint /ckpts/dinov3_vitl16.pth \\
        --output-dir ./outputs/linear_probe \\
        --train-split train --eval-split tune
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import DATASET_REGISTRY
from .model_utils import load_dinov3_backbone
from .visualization import save_prediction_visualization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bio_segmentation.linear_probe')


# ============================================================================
# Model
# ============================================================================

class DINOv3LinearSegmenter(nn.Module):
    """
    Frozen DINOv3 backbone + lightweight linear segmentation head (1×1 conv).

    Only the head is trained; the backbone stays in eval() throughout.
    When use_multi_scale=True the last 4 layers are concatenated, otherwise
    only the final layer is used.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,
        use_multi_scale: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.patch_size = backbone.patch_size
        self.use_multi_scale = use_multi_scale
        self.n_layers = 4 if use_multi_scale else 1
        in_channels = backbone.embed_dim * self.n_layers

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.head = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )
        nn.init.normal_(self.head[2].weight, mean=0, std=0.01)
        nn.init.constant_(self.head[2].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        feat_h, feat_w = h // self.patch_size, w // self.patch_size

        with torch.no_grad():
            if self.use_multi_scale:
                feats = self.backbone.get_intermediate_layers(x, n=self.n_layers, reshape=True)
                feats = [
                    F.interpolate(f, (feat_h, feat_w), mode='bilinear', align_corners=False)
                    if f.shape[2:] != (feat_h, feat_w) else f
                    for f in feats
                ]
                features = torch.cat(feats, dim=1)
            else:
                features = self.backbone.get_intermediate_layers(x, n=1, reshape=True)[0]

        logits = self.head(features.float())
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)


# ============================================================================
# Metrics
# ============================================================================

def calculate_miou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = 255,
) -> Tuple[float, Dict[str, float]]:
    pred_cls = torch.argmax(pred, dim=1)
    class_names = ['background', 'cell']
    ious = {}
    for cls in range(num_classes):
        valid = target != ignore_index
        p = (pred_cls == cls) & valid
        t = (target == cls) & valid
        intersection = (p & t).sum().float().item()
        union = (p | t).sum().float().item()
        ious[class_names[cls] if cls < len(class_names) else f'class_{cls}'] = (
            float('nan') if union == 0 else intersection / union
        )
    return float(np.nanmean(list(ious.values()))), ious


def calculate_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_fg = (torch.argmax(pred, dim=1) == 1).float()
    target_fg = (target == 1).float()
    intersection = (pred_fg * target_fg).sum()
    union = pred_fg.sum() + target_fg.sum()
    return 1.0 if union == 0 else (2 * intersection / union).item()


# ============================================================================
# Training / evaluation loops
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    model.backbone.eval()

    total_loss = total_miou = total_dice = 0.0
    n = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = criterion(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            miou, _ = calculate_miou(logits, masks)
            dice = calculate_dice(logits, masks)
        total_loss += loss.item()
        total_miou += miou
        total_dice += dice
        n += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}', mIoU=f'{miou:.4f}', Dice=f'{dice:.4f}')

    return {'loss': total_loss / n, 'mIoU': total_miou / n, 'Dice': total_dice / n}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    save_vis: bool = False,
    vis_dir: Optional[str] = None,
    mode: str = 'hybrid',
    max_vis: int = 10,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_ious: Dict[str, List[float]] = {'background': [], 'cell': []}
    all_dice: List[float] = []
    n = vis_count = 0

    for imgs, masks in tqdm(loader, desc='Evaluating'):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = criterion(logits, masks)
        miou, class_ious = calculate_miou(logits, masks)
        dice = calculate_dice(logits, masks)

        total_loss += loss.item()
        for cls, iou in class_ious.items():
            if not np.isnan(iou):
                all_ious.setdefault(cls, []).append(iou)
        all_dice.append(dice)
        n += 1

        if save_vis and vis_dir and vis_count < max_vis:
            for i in range(min(imgs.size(0), max_vis - vis_count)):
                save_prediction_visualization(
                    imgs[i], masks[i], logits[i],
                    os.path.join(vis_dir, f'{mode}_sample_{vis_count}.png'),
                    mode,
                )
                vis_count += 1

    return {
        'loss': total_loss / n,
        'mIoU': float(np.mean([np.mean(v) for v in all_ious.values() if v])),
        'IoU_background': float(np.mean(all_ious['background'])) if all_ious['background'] else 0.0,
        'IoU_cell': float(np.mean(all_ious['cell'])) if all_ious['cell'] else 0.0,
        'Dice': float(np.mean(all_dice)),
    }


# ============================================================================
# Single-mode experiment
# ============================================================================

def run_experiment(
    mode: str,
    backbone: nn.Module,
    DatasetClass,
    train_img_paths: List[str],
    train_mask_paths: List[str],
    eval_img_paths: List[str],
    eval_mask_paths: List[str],
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device('cuda'),
    num_workers: int = 4,
    use_multi_scale: bool = False,
) -> Dict[str, float]:
    logger.info(f"\n{'='*60}\nStarting experiment: {mode.upper()}\n{'='*60}")

    exp_dir = os.path.join(output_dir, mode)
    vis_dir = os.path.join(exp_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    train_ds = DatasetClass(train_img_paths, train_mask_paths, mode=mode, size=(224, 224), augment=True)
    eval_ds  = DatasetClass(eval_img_paths,  eval_mask_paths,  mode=mode, size=(224, 224), augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    model = DINOv3LinearSegmenter(
        backbone, num_classes=2, use_multi_scale=use_multi_scale
    ).to(device)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List] = {'train': [], 'eval': []}
    best_miou = best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()
        eval_m  = evaluate(
            model, eval_loader, criterion, device,
            save_vis=(epoch == epochs), vis_dir=vis_dir, mode=mode,
        )
        history['train'].append(train_m)
        history['eval'].append(eval_m)

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss={train_m['loss']:.4f} mIoU={train_m['mIoU']:.4f} | "
            f"Eval Loss={eval_m['loss']:.4f} mIoU={eval_m['mIoU']:.4f} Dice={eval_m['Dice']:.4f}"
        )

        if eval_m['mIoU'] > best_miou:
            best_miou, best_epoch = eval_m['mIoU'], epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': eval_m,
            }, os.path.join(exp_dir, 'best_model.pth'))

    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"[{mode.upper()}] Best mIoU: {best_miou:.4f} (Epoch {best_epoch})")
    return {'mode': mode, 'best_epoch': best_epoch, 'best_mIoU': best_miou, **history['eval'][-1]}


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bio-segmentation linear probing')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_REGISTRY),
                        help='Dataset name (e.g. cellpose, csc)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Dataset root directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='DINOv3 checkpoint path')
    parser.add_argument('--model-size', type=str, default='l', choices=['l', '7b'],
                        help='Backbone size')
    parser.add_argument('--use-multi-scale', action='store_true',
                        help='Concatenate last 4 layers instead of last 1')

    parser.add_argument('--train-split', type=str, default='train',
                        help='Split name used for training')
    parser.add_argument('--eval-split', type=str, default=None,
                        help='Split name used for evaluation (defaults: cellpose=test, csc=tune)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--modes', nargs='+', default=['minmax', 'percentile', 'hybrid'],
                        help='Preprocessing modes to compare')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{args.dataset}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    DatasetClass, get_paths = DATASET_REGISTRY[args.dataset]

    # Default eval split per dataset
    default_eval = {'cellpose': 'test', 'csc': 'tune'}
    eval_split = args.eval_split or default_eval.get(args.dataset, 'test')

    train_imgs, train_masks = get_paths(args.data_path, args.train_split)
    eval_imgs,  eval_masks  = get_paths(args.data_path, eval_split)

    backbone = load_dinov3_backbone(args.checkpoint, args.model_size, device)

    all_results = {}
    for mode in args.modes:
        all_results[mode] = run_experiment(
            mode=mode,
            backbone=backbone,
            DatasetClass=DatasetClass,
            train_img_paths=train_imgs,
            train_mask_paths=train_masks,
            eval_img_paths=eval_imgs,
            eval_mask_paths=eval_masks,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            num_workers=args.num_workers,
            use_multi_scale=args.use_multi_scale,
        )

    logger.info(f"\n{'='*60}\nResults summary\n{'='*60}")
    summary = []
    for mode, r in all_results.items():
        summary.append(r)
        logger.info(
            f"[{mode.upper():12s}] mIoU={r['best_mIoU']:.4f}  "
            f"Dice={r['Dice']:.4f}  Cell-IoU={r['IoU_cell']:.4f}"
        )

    best = max(summary, key=lambda x: x['mIoU'])
    logger.info(f"\nBest preprocessing: {best['mode'].upper()}  mIoU={best['mIoU']:.4f}")

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Done. Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
