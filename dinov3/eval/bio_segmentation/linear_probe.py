"""
Unified Linear Probe for biological cell segmentation.

Two operation modes:
    ONLINE mode (default):
        Frozen DINOv3 backbone computes features every iteration.
        Slower but requires no pre-computation step.

    CACHED mode (--use-cached-features):
        Reads pre-extracted .npz feature files (from feature_extractor.py).
        Backbone is NOT loaded; training is extremely fast.

Metrics reported (both modes):
    Semantic  : mIoU, mDice, mPrecision, mRecall   per class
    Instance  : AJI, AP@0.5, AP (COCO), bPQ (+ mPQ when multi-class)

Usage – ONLINE:
    python -m dinov3.eval.bio_segmentation.linear_probe \\
        --dataset    monuseg \\
        --data-root  /data1/xuzijing/dataset/monuseg/extracted \\
        --checkpoint /data1/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \\
        --output-dir ./outputs/linear_probe/monuseg \\
        --model-size l --epochs 20 --batch-size 4

Usage – CACHED:
    python -m dinov3.eval.bio_segmentation.linear_probe \\
        --dataset    monuseg \\
        --use-cached-features \\
        --train-cache ./cache/monuseg/monuseg_train_l_last1_s512.npz \\
        --val-cache   ./cache/monuseg/monuseg_val_l_last1_s512.npz \\
        --test-cache  ./cache/monuseg/monuseg_test_l_last1_s512.npz \\
        --output-dir  ./outputs/linear_probe/monuseg \\
        --epochs 50 --batch-size 64
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
from scipy.ndimage import label as scipy_label
from skimage.measure import label as sk_label
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from .metrics import (
    accumulate_instance_metrics,
    accumulate_semantic_metrics,
    compute_aji,
    compute_ap,
    compute_pq,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bio_seg.linear_probe')


# ============================================================================
# Model: Conv head on top of pre-extracted or live-extracted features
# ============================================================================

class LinearSegHead(nn.Module):
    """
    Lightweight 1×1 convolutional segmentation head.

    Input  : [B, in_channels, H_p, W_p]  (patch-level features)
    Output : [B, num_classes, H, W]       (upsampled to original image size)
    """

    def __init__(
        self,
        in_channels:  int,
        num_classes:  int,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )
        nn.init.normal_(self.head[2].weight, mean=0, std=0.01)
        nn.init.constant_(self.head[2].bias, 0)

    def forward(self, x: torch.Tensor, out_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        logits = self.head(x.float())
        if out_size is not None and logits.shape[2:] != out_size:
            logits = F.interpolate(logits, size=out_size, mode='bilinear', align_corners=False)
        return logits


class OnlineLinearSegmenter(nn.Module):
    """
    Frozen DINOv3 backbone + LinearSegHead (online mode).
    Only the head parameters are trainable.
    """

    def __init__(
        self,
        backbone:     nn.Module,
        num_classes:  int = 2,
        n_layers:     int = 4,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.n_layers = n_layers
        self.patch_size = backbone.patch_size
        in_channels = backbone.embed_dim * n_layers

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.head = LinearSegHead(in_channels, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        with torch.no_grad():
            feats_list = self.backbone.get_intermediate_layers(
                x, n=self.n_layers, reshape=True, return_class_token=False
            )
        feats  = torch.cat(feats_list, dim=1)   # [B, D, H_p, W_p]
        return self.head(feats, out_size=(h, w))


# ============================================================================
# Instance post-processing (semantic → instance via CC)
# ============================================================================

def semantic_to_instance(
    sem_pred: np.ndarray,
    fg_classes: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Convert a semantic class map to an instance map using connected components.

    Args:
        sem_pred  : (H, W) predicted class map.
        fg_classes: which class IDs to treat as foreground (default: all >0).

    Returns:
        (H, W) instance map (0 = background, 1..N = instance IDs).
    """
    if fg_classes is None:
        fg_mask = sem_pred > 0
    else:
        fg_mask = np.zeros_like(sem_pred, dtype=bool)
        for c in fg_classes:
            fg_mask |= (sem_pred == c)

    inst_map, _ = scipy_label(fg_mask)
    return inst_map.astype(np.int32)


# ============================================================================
# CachedFeatureDataset
# ============================================================================

class CachedFeatureDataset(Dataset):
    """Wraps pre-extracted feature arrays for efficient DataLoader usage."""

    def __init__(
        self,
        features:   np.ndarray,   # [N, D, H_p, W_p]
        sem_masks:  np.ndarray,   # [N, H, W]
        inst_maps:  Optional[np.ndarray] = None,  # [N, H, W]
    ):
        assert len(features) == len(sem_masks)
        self.features  = features
        self.sem_masks = sem_masks
        self.inst_maps = inst_maps

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        feat = torch.from_numpy(self.features[idx].astype(np.float32))
        sem  = torch.from_numpy(self.sem_masks[idx].astype(np.int64))
        if self.inst_maps is not None:
            inst = torch.from_numpy(self.inst_maps[idx].astype(np.int64))
            return feat, sem, inst
        return feat, sem


# ============================================================================
# Training helpers
# ============================================================================

def train_one_epoch_cached(
    head:       LinearSegHead,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
    epoch:      int,
    orig_size:  Tuple[int, int],
) -> float:
    """One epoch of linear head training on cached features."""
    head.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False)
    for batch in pbar:
        feat = batch[0].to(device)
        sem  = batch[1].to(device)

        logits = head(feat, out_size=orig_size)  # [B, C, H, W]
        loss   = criterion(logits, sem)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    return total_loss / max(n, 1)


def train_one_epoch_online(
    model:      OnlineLinearSegmenter,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
    epoch:      int,
) -> float:
    """One epoch of online linear probe training (backbone + head)."""
    model.head.train()
    model.backbone.eval()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False)
    for batch in pbar:
        imgs = batch[0].to(device)
        sem  = batch[1].to(device)

        logits = model(imgs)
        loss   = criterion(logits, sem)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    return total_loss / max(n, 1)


# ============================================================================
# Evaluation
# ============================================================================

@torch.inference_mode()
def evaluate_cached(
    head:          LinearSegHead,
    loader:        DataLoader,
    num_classes:   int,
    orig_size:     Tuple[int, int],
    device:        torch.device,
    class_names:   Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate on cached features; returns full metric dict."""
    head.eval()
    all_pred_sem:  List[np.ndarray] = []
    all_gt_sem:    List[np.ndarray] = []
    all_pred_inst: List[np.ndarray] = []
    all_gt_inst:   List[np.ndarray] = []
    has_inst = False

    for batch in tqdm(loader, desc='Eval', leave=False):
        feat = batch[0].to(device)
        sem  = batch[1]
        if len(batch) == 3:
            inst_gt = batch[2].numpy()
            has_inst = True
        else:
            inst_gt = None

        logits    = head(feat, out_size=orig_size)
        pred_sem  = logits.argmax(dim=1).cpu().numpy()

        for i in range(len(pred_sem)):
            all_pred_sem.append(pred_sem[i])
            all_gt_sem.append(sem[i].numpy())
            pred_inst = semantic_to_instance(pred_sem[i])
            all_pred_inst.append(pred_inst)
            if has_inst and inst_gt is not None:
                all_gt_inst.append(inst_gt[i])
            else:
                all_gt_inst.append(semantic_to_instance(sem[i].numpy()))

    # Semantic metrics
    sem_metrics = accumulate_semantic_metrics(
        all_pred_sem, all_gt_sem, num_classes=num_classes, class_names=class_names
    )
    # Instance metrics
    inst_metrics = accumulate_instance_metrics(all_pred_inst, all_gt_inst)

    return {**sem_metrics, **inst_metrics}


@torch.inference_mode()
def evaluate_online(
    model:       OnlineLinearSegmenter,
    loader:      DataLoader,
    num_classes: int,
    device:      torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate the full online model."""
    model.eval()
    all_pred_sem:  List[np.ndarray] = []
    all_gt_sem:    List[np.ndarray] = []
    all_pred_inst: List[np.ndarray] = []
    all_gt_inst:   List[np.ndarray] = []

    for batch in tqdm(loader, desc='Eval', leave=False):
        imgs = batch[0].to(device)
        sem  = batch[1]

        logits   = model(imgs)
        pred_sem = logits.argmax(dim=1).cpu().numpy()

        for i in range(len(pred_sem)):
            all_pred_sem.append(pred_sem[i])
            all_gt_sem.append(sem[i].numpy())
            pred_inst = semantic_to_instance(pred_sem[i])
            all_pred_inst.append(pred_inst)
            gt_inst   = semantic_to_instance(sem[i].numpy())
            all_gt_inst.append(gt_inst)

    sem_metrics  = accumulate_semantic_metrics(
        all_pred_sem, all_gt_sem, num_classes=num_classes, class_names=class_names
    )
    inst_metrics = accumulate_instance_metrics(all_pred_inst, all_gt_inst)
    return {**sem_metrics, **inst_metrics}


# ============================================================================
# Main training loop (cached mode)
# ============================================================================

def run_cached_linear_probe(
    train_cache:  str,
    val_cache:    str,
    test_cache:   Optional[str],
    output_dir:   str,
    num_classes:  int,
    class_names:  Optional[List[str]],
    epochs:       int = 50,
    lr:           float = 1e-3,
    batch_size:   int  = 64,
    weight_decay: float = 1e-4,
    dropout:      float = 0.1,
    ignore_index: int   = 255,
) -> Dict[str, float]:
    """Full cached linear probe training pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # Load caches
    # ------------------------------------------------------------------
    def _load(path):
        d = np.load(path)
        has_inst = d['inst_maps'].any()
        inst = d['inst_maps'] if has_inst else None
        return d['features'], d['sem_masks'].astype(np.int64), inst, \
               (int(d['orig_H']), int(d['orig_W']))

    logger.info("Loading train cache ...")
    tr_feat, tr_sem, tr_inst, orig_size = _load(train_cache)
    D = tr_feat.shape[1]

    logger.info("Loading val cache ...")
    val_feat, val_sem, val_inst, _ = _load(val_cache)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    head = LinearSegHead(D, num_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # DataLoaders
    tr_ds  = CachedFeatureDataset(tr_feat,  tr_sem,  tr_inst)
    val_ds = CachedFeatureDataset(val_feat, val_sem, val_inst)
    tr_loader  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    best_val_miou  = -1.0
    best_ckpt_path = os.path.join(output_dir, 'best_head.pth')
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch_cached(head, tr_loader, optimizer, criterion, device, epoch, orig_size)
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            val_metrics = evaluate_cached(
                head, val_loader, num_classes, orig_size, device, class_names
            )
            miou = val_metrics['mIoU']
            logger.info(
                f"Epoch {epoch:3d}/{epochs}  loss={loss:.4f}  "
                f"val_mIoU={miou:.4f}  val_mDice={val_metrics['mDice']:.4f}  "
                f"val_AJI={val_metrics['AJI']:.4f}  val_AP50={val_metrics['AP50']:.4f}"
            )
            if miou > best_val_miou:
                best_val_miou = miou
                torch.save(head.state_dict(), best_ckpt_path)

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    head.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    results = {'val': evaluate_cached(head, val_loader, num_classes, orig_size, device, class_names)}

    if test_cache is not None and os.path.exists(test_cache):
        logger.info("Loading test cache ...")
        te_feat, te_sem, te_inst, _ = _load(test_cache)
        te_ds     = CachedFeatureDataset(te_feat, te_sem, te_inst)
        te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        results['test'] = evaluate_cached(
            head, te_loader, num_classes, orig_size, device, class_names
        )

    # Save results
    out_json = os.path.join(output_dir, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {out_json}")

    _log_results(results)
    return results


# ============================================================================
# Main training loop (online mode)
# ============================================================================

def run_online_linear_probe(
    dataset_name: str,
    data_root:    str,
    checkpoint:   str,
    output_dir:   str,
    num_classes:  int,
    class_names:  Optional[List[str]],
    model_size:   str  = 'l',
    img_size:     int  = 448,
    n_layers:     int  = 4,
    epochs:       int  = 20,
    lr:           float = 1e-3,
    batch_size:   int  = 4,
    weight_decay: float = 1e-4,
    dropout:      float = 0.1,
    ignore_index: int   = 255,
    num_workers:  int   = 4,
) -> Dict[str, float]:
    """Full online linear probe training pipeline."""
    from .model_utils import load_dinov3_backbone
    from .feature_extractor import _build_dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = load_dinov3_backbone(checkpoint, model_size=model_size, device=device, freeze=True)
    model    = OnlineLinearSegmenter(backbone, num_classes, n_layers, dropout).to(device)

    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tr_ds  = _build_dataset(dataset_name, data_root, 'train', img_size)
    val_ds = _build_dataset(dataset_name, data_root, 'val',   img_size)

    tr_loader  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    best_val_miou  = -1.0
    best_ckpt_path = os.path.join(output_dir, 'best_head.pth')

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch_online(model, tr_loader, optimizer, criterion, device, epoch)
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            val_metrics = evaluate_online(model, val_loader, num_classes, device, class_names)
            miou = val_metrics['mIoU']
            logger.info(
                f"Epoch {epoch:3d}/{epochs}  loss={loss:.4f}  "
                f"val_mIoU={miou:.4f}  val_mDice={val_metrics['mDice']:.4f}  "
                f"val_AJI={val_metrics['AJI']:.4f}  val_AP50={val_metrics['AP50']:.4f}"
            )
            if miou > best_val_miou:
                best_val_miou = miou
                torch.save(model.head.state_dict(), best_ckpt_path)

    model.head.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    try:
        te_ds     = _build_dataset(dataset_name, data_root, 'test', img_size)
        te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_metrics = evaluate_online(model, te_loader, num_classes, device, class_names)
    except Exception as e:
        logger.warning(f"Test split not available: {e}")
        test_metrics = {}

    results = {
        'val':  evaluate_online(model, val_loader, num_classes, device, class_names),
        'test': test_metrics,
    }
    out_json = os.path.join(output_dir, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    _log_results(results)
    return results


# ============================================================================
# Logging helper
# ============================================================================

def _log_results(results: Dict):
    for split, metrics in results.items():
        if not metrics:
            continue
        logger.info(f"\n{'='*60}")
        logger.info(f"  Split: {split.upper()}")
        logger.info(f"{'='*60}")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                logger.info(f"  {k:<25s}: {v:.4f}")


# ============================================================================
# Dataset-specific configs
# ============================================================================

DATASET_CONFIGS = {
    'bbbc038':  {'num_classes': 2,  'class_names': ['background', 'cell']},
    'conic':    {'num_classes': 7,  'class_names': ['background', 'neutrophil', 'epithelial',
                                                     'lymphocyte', 'plasma_cell', 'eosinophil', 'connective']},
    'livecell': {'num_classes': 2,  'class_names': ['background', 'cell']},
    'monuseg':  {'num_classes': 2,  'class_names': ['background', 'nucleus']},
    'pannuke':  {'num_classes': 6,  'class_names': ['background', 'neoplastic', 'inflammatory',
                                                     'connective', 'dead', 'epithelial']},
    'tissuenet':{'num_classes': 2,  'class_names': ['background', 'cell']},
    # existing
    'cellpose': {'num_classes': 2,  'class_names': ['background', 'cell']},
    'csc':      {'num_classes': 2,  'class_names': ['background', 'cell']},
}


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bio-segmentation linear probe (online or cached mode)'
    )
    # Common
    parser.add_argument('--dataset',     required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name')
    parser.add_argument('--output-dir',  required=True)
    parser.add_argument('--epochs',      type=int,   default=20)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--batch-size',  type=int,   default=4)
    parser.add_argument('--weight-decay',type=float, default=1e-4)
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--ignore-index',type=int,   default=255)
    parser.add_argument('--num-workers', type=int,   default=4)

    # Cached mode
    parser.add_argument('--use-cached-features', action='store_true',
                        help='Use pre-extracted feature cache (fast mode)')
    parser.add_argument('--train-cache', default=None,
                        help='Path to train .npz cache (cached mode)')
    parser.add_argument('--val-cache',   default=None,
                        help='Path to val .npz cache (cached mode)')
    parser.add_argument('--test-cache',  default=None,
                        help='Path to test .npz cache (cached mode, optional)')

    # Online mode
    parser.add_argument('--data-root',   default=None,
                        help='Dataset root directory (online mode)')
    parser.add_argument('--checkpoint',  default=None,
                        help='DINOv3 checkpoint path (online mode)')
    parser.add_argument('--model-size',  default='l', choices=['l', '7b'])
    parser.add_argument('--img-size',    type=int, default=448)
    parser.add_argument('--n-layers',    type=int, default=4)

    args = parser.parse_args()

    cfg = DATASET_CONFIGS.get(args.dataset, {'num_classes': 2, 'class_names': None})
    num_classes = cfg['num_classes']
    class_names = cfg['class_names']

    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_cached_features:
        if args.train_cache is None or args.val_cache is None:
            parser.error('--train-cache and --val-cache required in cached mode.')
        run_cached_linear_probe(
            train_cache  = args.train_cache,
            val_cache    = args.val_cache,
            test_cache   = args.test_cache,
            output_dir   = args.output_dir,
            num_classes  = num_classes,
            class_names  = class_names,
            epochs       = args.epochs,
            lr           = args.lr,
            batch_size   = args.batch_size,
            weight_decay = args.weight_decay,
            dropout      = args.dropout,
            ignore_index = args.ignore_index,
        )
    else:
        if args.data_root is None or args.checkpoint is None:
            parser.error('--data-root and --checkpoint required in online mode.')
        run_online_linear_probe(
            dataset_name = args.dataset,
            data_root    = args.data_root,
            checkpoint   = args.checkpoint,
            output_dir   = args.output_dir,
            num_classes  = num_classes,
            class_names  = class_names,
            model_size   = args.model_size,
            img_size     = args.img_size,
            n_layers     = args.n_layers,
            epochs       = args.epochs,
            lr           = args.lr,
            batch_size   = args.batch_size,
            weight_decay = args.weight_decay,
            dropout      = args.dropout,
            ignore_index = args.ignore_index,
            num_workers  = args.num_workers,
        )


if __name__ == '__main__':
    main()
