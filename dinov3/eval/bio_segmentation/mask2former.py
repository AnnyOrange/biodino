"""
Mask2Former-based segmentation for biological microscopy images.

Uses the DINOv3 + Mask2Former architecture from eval/segmentation/models/__init__.py
(build_segmentation_decoder) with the 'm2f' decoder type.

Training strategy
-----------------
- DINOv3 ViT backbone : frozen via DINOv3_Adapter.__init__ (requires_grad_(False))
- DINOv3_Adapter layers (SPM, InteractionBlocks, up-conv, norms): TRAINED
- Mask2FormerHead (pixel decoder + transformer decoder): TRAINED
- Loss: Dice + focal (per matched query) + CE (all queries), with Hungarian
  matching on EVERY decoder layer (deep supervision / auxiliary losses).

At inference: convert predicted queries to instance/semantic maps via greedy
assignment ordered by confidence score.

Metrics
-------
  Semantic : mIoU, mDice, mPrecision, mRecall
  Instance : AJI, AP@0.5, AP (COCO), bPQ, mPQ (multi-class)

Usage
-----
    python -m dinov3.eval.bio_segmentation.mask2former \\
        --dataset    monuseg \\
        --data-root  /data1/xuzijing/dataset/monuseg/extracted \\
        --checkpoint /data1/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \\
        --output-dir ./outputs/mask2former/monuseg \\
        --model-size l \\
        --epochs 50 \\
        --batch-size 2 \\
        --lr 1e-4
"""

import argparse
import importlib.util
import json
import logging
import os
import random
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label as scipy_label
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import (
    accumulate_instance_metrics,
    accumulate_semantic_metrics,
)
from .model_utils import load_dinov3_backbone
from .feature_extractor import DATASET_DEFAULT_IMG_SIZES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bio_seg.mask2former')


# ============================================================================
# Environment checks
# ============================================================================

def _ensure_ms_deform_attn_available() -> None:
    """
    Fail fast if the official MultiScaleDeformableAttention CUDA extension is
    not installed.

    Why this is needed:
    - Official DINOv3 Mask2Former uses `MSDeformAttn` in the pixel decoder.
    - Its forward path has a PyTorch fallback, so inference may appear to work.
    - Its backward path requires the compiled CUDA extension and raises at the
      first `loss.backward()` if the extension is missing.

    This helper surfaces the real issue immediately, before we spend minutes
    loading a large backbone checkpoint.
    """
    if importlib.util.find_spec("MultiScaleDeformableAttention") is not None:
        return

    ops_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "segmentation",
        "models",
        "utils",
        "ops",
    )
    raise RuntimeError(
        "Mask2Former training requires the official "
        "`MultiScaleDeformableAttention` CUDA extension, but it is not "
        "installed in the current environment.\n\n"
        "This is NOT a bio_segmentation-specific bug: the requirement comes "
        "from official `dinov3/eval/segmentation/models/utils/ms_deform_attn.py`, "
        "whose backward() raises exactly this error when the extension is missing.\n\n"
        "Please compile it first in the same conda environment:\n"
        f"  cd {ops_dir}\n"
        "  python setup.py build install\n\n"
        "If compilation fails, check that CUDA toolkit / `nvcc` is available. "
        "Inference-only forward may work without this extension, but training "
        "(loss.backward) cannot."
    )


# ============================================================================
# Loss functions
# ============================================================================

def dice_loss(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Binary Dice loss between sigmoid(pred_mask) and gt_mask (0/1 float).
    Both tensors: [M, H, W] → returns scalar.
    """
    pred_p = pred_mask.sigmoid().flatten(1)
    gt_f   = gt_mask.float().flatten(1)
    num    = 2.0 * (pred_p * gt_f).sum(1)
    den    = pred_p.sum(1) + gt_f.sum(1)
    return (1.0 - (num + eps) / (den + eps)).mean()


def sigmoid_focal_loss(
    pred:  torch.Tensor,
    gt:    torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss; [M, H, W] → scalar."""
    p    = torch.sigmoid(pred)
    ce   = F.binary_cross_entropy_with_logits(pred, gt.float(), reduction='none')
    p_t  = p * gt + (1 - p) * (1 - gt)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * gt + (1 - alpha) * (1 - gt)
        loss    = alpha_t * loss
    return loss.mean()


# ============================================================================
# Hungarian matching
# ============================================================================

@torch.no_grad()
def hungarian_match(
    pred_logits: torch.Tensor,   # [Q, C+1]
    pred_masks:  torch.Tensor,   # [Q, H, W]
    gt_labels:   torch.Tensor,   # [K]
    gt_masks:    torch.Tensor,   # [K, H, W]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hungarian optimal matching between Q queries and K GT instances.

    Returns:
        query_indices : [M] matched query indices
        gt_indices    : [M] matched GT indices
    """
    Q = pred_logits.shape[0]
    K = gt_labels.shape[0]

    if K == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty

    # Classification cost
    cls_probs = pred_logits.softmax(-1)         # [Q, C+1]
    cls_cost  = -cls_probs[:, gt_labels]        # [Q, K]

    # Dice cost
    pred_sig = pred_masks.sigmoid().flatten(1)  # [Q, HW]
    gt_flat  = gt_masks.float().flatten(1)      # [K, HW]
    inter    = torch.einsum('qp,kp->qk', pred_sig, gt_flat)
    sum_pred = pred_sig.sum(1, keepdim=True)
    sum_gt   = gt_flat.sum(1, keepdim=True).T
    dice_c   = 1.0 - (2.0 * inter + 1.0) / (sum_pred + sum_gt + 1.0)  # [Q, K]

    # Focal cost (approximate: spatial mean)
    pred_exp = pred_masks.unsqueeze(1).expand(-1, K, -1, -1)   # [Q,K,H,W]
    gt_exp   = gt_masks.unsqueeze(0).expand(Q, -1, -1, -1).float()
    bce      = F.binary_cross_entropy_with_logits(pred_exp, gt_exp, reduction='none')
    p_t      = pred_exp.sigmoid() * gt_exp + (1 - pred_exp.sigmoid()) * (1 - gt_exp)
    focal_c  = (bce * (1 - p_t) ** 2.0).mean((-2, -1))          # [Q, K]

    cost = cls_cost + 2.0 * dice_c + focal_c
    row_idx, col_idx = linear_sum_assignment(cost.cpu().float().numpy())

    return (
        torch.as_tensor(row_idx, dtype=torch.long),
        torch.as_tensor(col_idx, dtype=torch.long),
    )


# ============================================================================
# Per-layer loss (called for both the final output and each aux layer)
# ============================================================================

def _loss_single_layer(
    pred_logits: torch.Tensor,          # [B, Q, C+1]
    pred_masks:  torch.Tensor,          # [B, Q, H_p, W_p]
    gt_labels:   List[torch.Tensor],    # B × [K_i]
    gt_masks:    List[torch.Tensor],    # B × [K_i, H, W]
    num_classes: int,
    mask_weight: float = 5.0,
    cls_weight:  float = 2.0,
    dice_weight: float = 5.0,
) -> torch.Tensor:
    """
    Compute Hungarian-matched Mask2Former loss for one decoder layer.
    GT masks are bilinearly downsampled to match pred_masks resolution.
    """
    B, Q, H_p, W_p = pred_masks.shape
    device      = pred_logits.device
    no_obj_cls  = num_classes

    total = torch.tensor(0.0, device=device)

    for b in range(B):
        pl = pred_logits[b]            # [Q, C+1]
        pm = pred_masks[b]             # [Q, H_p, W_p]
        gl = gt_labels[b].to(device)  # [K]
        gm = gt_masks[b].to(device)   # [K, H, W]

        if gl.numel() == 0:
            tgt = torch.full((Q,), no_obj_cls, dtype=torch.long, device=device)
            total += cls_weight * F.cross_entropy(pl, tgt)
            continue

        # Down-sample GT masks to prediction resolution
        gm_down = F.interpolate(
            gm.float().unsqueeze(0), size=(H_p, W_p), mode='nearest'
        ).squeeze(0)                    # [K, H_p, W_p]

        q_idx, g_idx = hungarian_match(pl.detach(), pm.detach(), gl, gm_down)

        # Classification loss: matched → correct class, others → no-object
        cls_tgt = torch.full((Q,), no_obj_cls, dtype=torch.long, device=device)
        cls_tgt[q_idx] = gl[g_idx]
        no_obj_w              = torch.ones(num_classes + 1, device=device)
        no_obj_w[no_obj_cls]  = 0.1
        total += cls_weight * F.cross_entropy(pl, cls_tgt, weight=no_obj_w)

        # Mask losses for matched pairs only
        if len(q_idx) > 0:
            matched_pm = pm[q_idx]        # [M, H_p, W_p]
            matched_gm = gm_down[g_idx]   # [M, H_p, W_p]
            total += dice_weight  * dice_loss(matched_pm, matched_gm)
            total += mask_weight  * sigmoid_focal_loss(matched_pm, matched_gm)

    return total / B


def compute_m2f_loss(
    output:      Dict,                  # full model output dict (includes aux_outputs)
    gt_labels:   List[torch.Tensor],
    gt_masks:    List[torch.Tensor],
    num_classes: int,
    mask_weight: float = 5.0,
    cls_weight:  float = 2.0,
    dice_weight: float = 5.0,
) -> torch.Tensor:
    """
    Total Mask2Former loss = final-layer loss + sum of auxiliary losses
    (one per intermediate decoder layer = deep supervision).

    Using equal weight for all layers following the original paper.
    """
    final_loss = _loss_single_layer(
        output['pred_logits'], output['pred_masks'],
        gt_labels, gt_masks, num_classes,
        mask_weight, cls_weight, dice_weight,
    )

    # Auxiliary losses (one per intermediate decoder layer, 9 layers total)
    aux_loss = torch.tensor(0.0, device=output['pred_logits'].device)
    for aux in output.get('aux_outputs', []):
        aux_loss = aux_loss + _loss_single_layer(
            aux['pred_logits'], aux['pred_masks'],
            gt_labels, gt_masks, num_classes,
            mask_weight, cls_weight, dice_weight,
        )

    n_aux = len(output.get('aux_outputs', []))
    # Average over all layers (final + auxiliary)
    total = (final_loss + aux_loss) / (n_aux + 1)
    return total


# ============================================================================
# GT preparation: instance/semantic maps → list of binary masks + labels
# ============================================================================

def prepare_gt(
    sem_batch:  torch.Tensor,
    inst_batch: Optional[torch.Tensor],
    num_classes: int,
    max_instances: int = 100,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Convert per-sample semantic + instance maps into per-query GT format.

    If inst_batch is None, uses connected components on the semantic mask.

    Returns:
        gt_labels : B × [K_i] int64 class IDs
        gt_masks  : B × [K_i, H, W] float32 binary masks
    """
    B = sem_batch.shape[0]
    all_labels, all_masks = [], []

    for b in range(B):
        sem_np  = sem_batch[b].numpy()
        inst_np = inst_batch[b].numpy() if inst_batch is not None else None

        labels_list: List[int] = []
        masks_list:  List[np.ndarray] = []

        if inst_np is not None:
            inst_ids = np.unique(inst_np[inst_np > 0])
            for iid in inst_ids[:max_instances]:
                region  = (inst_np == iid)
                cls_vals = sem_np[region]
                pos_vals = cls_vals[cls_vals > 0]
                cls_id   = int(np.bincount(pos_vals).argmax()) if len(pos_vals) > 0 else 1
                labels_list.append(cls_id)
                masks_list.append(region.astype(np.float32))
        else:
            for cls_id in range(1, num_classes):
                binary = (sem_np == cls_id).astype(np.uint8)
                if binary.sum() == 0:
                    continue
                inst_tmp, n = scipy_label(binary)
                for iid in range(1, min(n + 1, max_instances)):
                    region = (inst_tmp == iid)
                    if region.sum() < 4:
                        continue
                    labels_list.append(cls_id)
                    masks_list.append(region.astype(np.float32))

        if labels_list:
            all_labels.append(torch.tensor(labels_list, dtype=torch.long))
            all_masks.append(torch.from_numpy(np.stack(masks_list, axis=0)))
        else:
            all_labels.append(torch.zeros(0, dtype=torch.long))
            all_masks.append(torch.zeros(0, sem_batch.shape[1], sem_batch.shape[2]))

    return all_labels, all_masks


# ============================================================================
# Inference: queries → semantic + instance maps
# ============================================================================

@torch.inference_mode()
def queries_to_maps(
    pred_logits: torch.Tensor,   # [Q, C+1]
    pred_masks:  torch.Tensor,   # [Q, H, W]  (upsampled to image size)
    num_classes: int,
    mask_threshold:  float = 0.5,
    score_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Mask2Former query outputs to (semantic_map, instance_map).

    Returns:
        sem_map  : (H, W) int64  class IDs    (0 = background)
        inst_map : (H, W) int64  instance IDs (0 = background)
    """
    no_obj_cls = num_classes
    H, W       = pred_masks.shape[1:]
    scores     = pred_logits.softmax(-1)
    cls_ids    = scores.argmax(-1)
    cls_scores = scores.max(-1).values

    sem_map  = np.zeros((H, W), dtype=np.int64)
    inst_map = np.zeros((H, W), dtype=np.int64)
    inst_id  = 1

    # Process queries in descending confidence order
    order = cls_scores.argsort(descending=True)
    for qi in order:
        cls = int(cls_ids[qi])
        if cls == no_obj_cls:
            continue
        if float(cls_scores[qi]) < score_threshold:
            break

        binary = (pred_masks[qi].sigmoid() > mask_threshold).cpu().numpy()
        # Only assign pixels not yet claimed by a higher-confidence query
        unset = (inst_map == 0) & binary
        if unset.sum() < 4:
            continue

        sem_map[unset]  = cls
        inst_map[unset] = inst_id
        inst_id += 1

    return sem_map, inst_map


# ============================================================================
# DataLoader collation helpers
# ============================================================================

def _random_crop_collate_fn(
    batch: list,
    crop_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for Mask2Former training with native-resolution datasets.

    Each item in the batch is a 3-tuple (img [3,H,W], sem [H,W], inst [H,W]).
    Because images may have different (and possibly very large) spatial
    dimensions, this collate fn applies a *synchronised* random crop of
    crop_size × crop_size to every item before stacking into a batch tensor.

    If an image is *smaller* than crop_size in either dimension it is first
    zero-padded so that the crop always yields exactly crop_size × crop_size.

    This is the Mask2Former analogue of the RandomResizeCrop transform used
    in the official DINOv3 segmentation training.
    """
    imgs  = [item[0] for item in batch]   # list of [3, Hi, Wi]
    sems  = [item[1] for item in batch]   # list of [Hi, Wi]
    insts = [
        item[2] if len(item) > 2 else torch.zeros_like(item[1])
        for item in batch
    ]                                      # list of [Hi, Wi]

    out_imgs, out_sems, out_insts = [], [], []
    for img, sem, inst in zip(imgs, sems, insts):
        _, H, W = img.shape

        # Pad if smaller than crop_size
        pad_h = max(0, crop_size - H)
        pad_w = max(0, crop_size - W)
        if pad_h > 0 or pad_w > 0:
            # F.pad format: (left, right, top, bottom)
            img  = F.pad(img,  (0, pad_w, 0, pad_h), value=0.0)
            sem  = F.pad(sem.unsqueeze(0), (0, pad_w, 0, pad_h), value=0).squeeze(0)
            inst = F.pad(inst.unsqueeze(0), (0, pad_w, 0, pad_h), value=0).squeeze(0)

        _, H2, W2 = img.shape
        top  = random.randint(0, H2 - crop_size)
        left = random.randint(0, W2 - crop_size)
        img  = img[:,  top:top + crop_size, left:left + crop_size]
        sem  = sem[    top:top + crop_size, left:left + crop_size]
        inst = inst[   top:top + crop_size, left:left + crop_size]

        out_imgs.append(img)
        out_sems.append(sem)
        out_insts.append(inst)

    return torch.stack(out_imgs), torch.stack(out_sems), torch.stack(out_insts)


# ============================================================================
# Training loop
# ============================================================================

def train_one_epoch(
    backbone_adapter: nn.Module,
    head:             nn.Module,
    loader:           DataLoader,
    optimizer:        torch.optim.Optimizer,
    device:           torch.device,
    epoch:            int,
    num_classes:      int,
) -> float:
    """
    One training epoch.

    The DINOv3 ViT backbone is frozen via requires_grad_(False) inside
    DINOv3_Adapter.__init__. The adapter's own layers (SPM, InteractionBlocks,
    up-conv, SyncBatchNorm) are trainable and put in train() mode here.

    IMPORTANT: do NOT wrap backbone_adapter(imgs) in torch.no_grad() —
    the adapter's trainable layers need gradient flow.  The ViT backbone call
    inside the adapter is already internally wrapped in torch.no_grad().
    """
    # Adapter in train mode → BN accumulates running stats; ViT weights stay frozen
    backbone_adapter.train()
    head.train()

    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc=f'M2F Epoch {epoch}', leave=False)

    for batch in pbar:
        imgs  = batch[0].to(device)       # [B, 3, H, W]
        sem   = batch[1]                  # [B, H, W]  CPU
        inst  = batch[2] if len(batch) == 3 else None

        # -------------------------------------------------------------------
        # Forward pass through adapter + M2F head.
        #
        # Do NOT use torch.no_grad() here — the adapter's trainable layers
        # (SPM, InteractionBlocks, norms) require gradient computation.
        # The ViT backbone inside DINOv3_Adapter is called with torch.no_grad()
        # internally (see dinov3_adapter.py line ~423), so its frozen weights
        # never accumulate gradients.
        # -------------------------------------------------------------------
        features = backbone_adapter(imgs)    # dict {"1"…"4"}: multi-scale features
        output   = head(features)            # {"pred_logits", "pred_masks", "aux_outputs"}

        # Prepare GT
        gt_labels, gt_masks = prepare_gt(sem, inst, num_classes)

        # Loss = final-layer loss + aux losses from all 9 intermediate layers
        loss = compute_m2f_loss(output, gt_labels, gt_masks, num_classes)

        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilise early training of adapter layers
        torch.nn.utils.clip_grad_norm_(
            list(backbone_adapter.parameters()) + list(head.parameters()),
            max_norm=0.1,
        )
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / max(n, 1)


# ============================================================================
# Evaluation loop
# ============================================================================

def _m2f_whole_inference(
    backbone_adapter: nn.Module,
    head:             nn.Module,
    imgs:             torch.Tensor,    # [B, 3, H, W]  already on device
    img_size:         Tuple[int, int],
    crop_size:        int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run Mask2Former on a batch with 'whole' inference.
    Resizes input to crop_size × crop_size (official DINOv3 default: 512)
    then upsamples predictions back to img_size.
    Returns pred_logits [B,Q,C+1] and pred_masks [B,Q,H,W].
    """
    imgs_resized = F.interpolate(
        imgs.float(), size=(crop_size, crop_size),
        mode='bilinear', align_corners=False,
    )
    with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
        features = backbone_adapter(imgs_resized)
        output   = head.predict(features, rescale_to=img_size)
    return output['pred_logits'], output['pred_masks']


def _m2f_slide_inference(
    backbone_adapter: nn.Module,
    head:             nn.Module,
    img:              torch.Tensor,    # [1, 3, H, W]  single image on device
    img_size:         Tuple[int, int],
    crop_size:        int = 512,
    stride:           int = 341,
    num_classes:      int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run Mask2Former with sliding-window inference on a single large image.
    Mirrors eval/segmentation/inference.py::slide_inference.
    Returns pred_logits [1,Q,C+1] and pred_masks [1,Q,H,W] (at img_size).
    """
    from dinov3.eval.segmentation.inference import slide_inference as _slide

    assert img.shape[0] == 1, "slide inference requires batch_size=1"
    H, W    = img.shape[2], img.shape[3]
    Q       = 100                       # Mask2Former default num_queries
    C_plus1 = num_classes + 1

    h_crop = min(crop_size, H)
    w_crop = min(crop_size, W)
    h_stride = stride
    w_stride = stride
    h_grids = max(H - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(W - w_crop + w_stride - 1, 0) // w_stride + 1

    # Accumulate mask logit maps [1, Q, H, W] via overlap-average
    acc_masks  = img.new_zeros((1, Q, H, W), dtype=torch.float32).cpu()
    count_mat  = img.new_zeros((1, 1, H, W), dtype=torch.float32).cpu()
    # For class logits we take the crop with highest foreground score
    best_logits = None
    best_score  = -1.0

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, H); y1 = max(y2 - h_crop, 0)
            x2 = min(x1 + w_crop, W); x1 = max(x2 - w_crop, 0)

            crop = img[:, :, y1:y2, x1:x2]
            with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
                features   = backbone_adapter(crop)
                crop_out   = head.predict(features, rescale_to=(y2 - y1, x2 - x1))

            crop_masks  = crop_out['pred_masks'].float().cpu()    # [1,Q,h,w]
            crop_logits = crop_out['pred_logits'].float()          # [1,Q,C+1]

            padded = F.pad(crop_masks,
                           (x1, W - x2, y1, H - y2))              # [1,Q,H,W]
            acc_masks += padded
            count_mat[:, :, y1:y2, x1:x2] += 1.0

            fg_score = float(crop_logits.softmax(-1)[..., :-1].max())
            if fg_score > best_score:
                best_score  = fg_score
                best_logits = crop_logits

            del crop, crop_masks, crop_out

    avg_masks = acc_masks / count_mat.clamp(min=1.0)
    # Upsample mask maps to target img_size
    avg_masks = F.interpolate(avg_masks, size=img_size, mode='bilinear', align_corners=False)
    return best_logits.to(img.device), avg_masks.to(img.device)


@torch.inference_mode()
def evaluate_m2f(
    backbone_adapter: nn.Module,
    head:             nn.Module,
    loader:           DataLoader,
    num_classes:      int,
    device:           torch.device,
    class_names:      Optional[List[str]] = None,
    inference_mode:   str = 'whole',
    crop_size:        int = 512,
    stride:           int = 341,
) -> Dict[str, float]:
    """
    Evaluate Mask2Former.

    The DataLoader must use batch_size=1 (required for both slide inference
    and for handling variable-size native-resolution images).

    For each image, the output resolution is automatically inferred from the
    image's actual spatial shape — no fixed img_size parameter is needed.

    inference_mode:
        'whole' — resize full image to crop_size × crop_size, run once,
                  then upsample predictions back to native resolution.
                  Matches official DINOv3 eval/segmentation/inference.py.
        'slide' — sliding-window with crop_size / stride, overlap-average.
                  Use for large images (e.g. MoNuSeg 1000 × 1000).
    crop_size:
        Window (or whole-image) size.  Must be a multiple of 16.
        Official default: 512.
    stride:
        Sliding window stride.  Official default: 341 (≈ 2/3 of crop_size).
    """
    assert inference_mode in ('whole', 'slide'), \
        f"inference_mode must be 'whole' or 'slide', got '{inference_mode}'"

    backbone_adapter.eval()
    head.eval()

    all_pred_sem:  List[np.ndarray] = []
    all_gt_sem:    List[np.ndarray] = []
    all_pred_inst: List[np.ndarray] = []
    all_gt_inst:   List[np.ndarray] = []

    for batch in tqdm(loader, desc=f'M2F Eval [{inference_mode}]', leave=False):
        imgs  = batch[0].to(device)    # [1, 3, H, W]  (batch_size=1)
        sem   = batch[1]               # [1, H, W]  CPU
        inst  = batch[2] if len(batch) == 3 else None

        # Derive the target output size from the actual image shape.
        # With native-resolution datasets (size=None), H and W reflect the
        # true image dimensions, so predictions are returned at native res.
        H, W = imgs.shape[2], imgs.shape[3]
        img_size_actual = (H, W)

        if inference_mode == 'slide':
            assert imgs.shape[0] == 1, \
                "Eval DataLoader must use batch_size=1 for slide inference"
            logits_b, masks_b = _m2f_slide_inference(
                backbone_adapter, head, imgs, img_size_actual,
                crop_size=crop_size, stride=stride, num_classes=num_classes,
            )
            p_sem, p_inst = queries_to_maps(logits_b[0], masks_b[0], num_classes)
            g_sem  = sem[0].numpy()
            g_inst = (
                inst[0].numpy() if inst is not None
                else np.where(g_sem > 0, scipy_label(g_sem > 0)[0], 0).astype(np.int32)
            )
            all_pred_sem.append(p_sem);  all_gt_sem.append(g_sem)
            all_pred_inst.append(p_inst.astype(np.int32))
            all_gt_inst.append(g_inst.astype(np.int32))
        else:
            # Whole inference: each image processed independently to handle
            # variable native sizes correctly.
            for b in range(imgs.shape[0]):
                img_b = imgs[b:b+1]                  # [1, 3, H, W]
                H_b, W_b = img_b.shape[2], img_b.shape[3]
                logits_b, masks_b = _m2f_whole_inference(
                    backbone_adapter, head, img_b,
                    img_size=(H_b, W_b), crop_size=crop_size,
                )
                p_sem, p_inst = queries_to_maps(logits_b[0], masks_b[0], num_classes)
                g_sem  = sem[b].numpy()
                g_inst = (
                    inst[b].numpy() if inst is not None
                    else np.where(g_sem > 0, scipy_label(g_sem > 0)[0], 0).astype(np.int32)
                )
                all_pred_sem.append(p_sem);  all_gt_sem.append(g_sem)
                all_pred_inst.append(p_inst.astype(np.int32))
                all_gt_inst.append(g_inst.astype(np.int32))

    sem_metrics  = accumulate_semantic_metrics(
        all_pred_sem, all_gt_sem, num_classes=num_classes, class_names=class_names
    )
    inst_metrics = accumulate_instance_metrics(all_pred_inst, all_gt_inst)
    return {**sem_metrics, **inst_metrics}


# ============================================================================
# Main pipeline
# ============================================================================

def run_mask2former(
    dataset_name:     str,
    data_root:        str,
    checkpoint:       str,
    output_dir:       str,
    num_classes:      int,
    class_names:      Optional[List[str]],
    model_size:       str   = 'l',
    img_size:         int   = 0,       # 0 → use per-dataset canonical size
    crop_size:        int   = 512,     # inference crop size (official default: 512)
    stride:           int   = 341,     # slide stride (official default: 341)
    inference_mode:   str   = 'whole', # 'whole' or 'slide'
    epochs:           int   = 50,
    lr:               float = 1e-4,
    adapter_lr:       float = 1e-5,
    batch_size:       int   = 2,
    weight_decay:     float = 1e-4,
    num_workers:      int   = 2,
) -> Dict:
    """
    Full Mask2Former training + evaluation pipeline.

    Trainable parameters
    --------------------
    The DINOv3_Adapter wraps the frozen ViT backbone with randomly-initialised
    adapter layers (SpatialPriorModule, InteractionBlocks, up-conv,
    SyncBatchNorm).  These adapter layers **must** be trained — they form the
    multi-scale FPN-like bridge between the ViT and the M2F decoder.
    A smaller learning rate (``adapter_lr``) is used for these layers since
    the backbone features they adapt are already high-quality.
    """
    from dinov3.eval.segmentation.models import build_segmentation_decoder, BackboneLayersSet
    from .feature_extractor import _build_dataset, DATASET_DEFAULT_IMG_SIZES

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    # Official Mask2Former training needs the compiled CUDA op for
    # MultiScaleDeformableAttention backward.
    _ensure_ms_deform_attn_available()

    # ---- Resolve crop size for training / inference ----
    # img_size here controls the per-dataset canonical crop size used during
    # both training (random crop) and whole/slide inference.
    # It does NOT force a resize of the full image at load time (that is now
    # done via the collate_fn and inference helpers).
    if img_size == 0:
        img_size = DATASET_DEFAULT_IMG_SIZES.get(dataset_name, 512)
        logger.info(f"img_size (crop_size baseline) auto → {img_size}")
    else:
        img_size = (img_size // 16) * 16
    # Use img_size as the default crop_size if not explicitly set by caller
    if crop_size == 512 and img_size != 512:
        crop_size = img_size
        logger.info(f"crop_size set to img_size={crop_size}")

    # ---- Resolve inference mode ----
    # For large-image datasets (MoNuSeg 1000×1000, BBBC038 variable),
    # slide inference gives better results than whole-image downscale.
    # For small-patch datasets (CoNIC/PanNuke 256×256) whole is fine.
    if inference_mode == 'auto':
        inference_mode = 'slide' if img_size >= 512 else 'whole'
        logger.info(f"inference_mode auto → '{inference_mode}'")

    logger.info(
        f"img_size={img_size}  inference_mode={inference_mode}  "
        f"crop_size={crop_size}  stride={stride}"
    )

    # ---- Load backbone ----
    backbone = load_dinov3_backbone(
        checkpoint, model_size=model_size, device=device, freeze=True
    )

    # ---- Build DINOv3_Adapter + Mask2FormerHead ----
    seg_model = build_segmentation_decoder(
        backbone,
        backbone_out_layers=BackboneLayersSet.FOUR_EVEN_INTERVALS,
        decoder_type='m2f',
        hidden_dim=256,
        num_classes=num_classes,
    )
    # seg_model.segmentation_model = ModuleList([backbone_adapter, m2f_head])
    backbone_adapter = seg_model.segmentation_model[0].to(device)
    m2f_head         = seg_model.segmentation_model[1].to(device)

    # ---- Parameter groups ----
    # DINOv3_Adapter.__init__ already calls self.backbone.requires_grad_(False),
    # so filter(requires_grad) on the adapter selects ONLY the adapter's own
    # trainable layers (SPM, interactions, up-conv, norms, level_embed).
    # DO NOT manually call backbone_adapter.requires_grad_(False)!
    adapter_params = [p for p in backbone_adapter.parameters() if p.requires_grad]
    head_params    = list(m2f_head.parameters())

    n_adapter = sum(p.numel() for p in adapter_params)
    n_head    = sum(p.numel() for p in head_params)
    logger.info(f"Trainable params — adapter: {n_adapter:,}  head: {n_head:,}")

    optimizer = torch.optim.AdamW(
        [
            {'params': adapter_params, 'lr': adapter_lr},
            {'params': head_params,    'lr': lr},
        ],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ---- Datasets (native resolution — NO forced resize) ----
    # For Mask2Former, images are loaded at their native resolution so that:
    #   - Training: random crops (crop_size × crop_size) via collate_fn give
    #               the model realistic field-of-view without distortion.
    #   - Evaluation: slide inference operates on the full native image, making
    #                 the sliding window mechanism genuinely effective.
    tr_ds  = _build_dataset(dataset_name, data_root, 'train', img_size=None, augment=True)
    val_ds = _build_dataset(dataset_name, data_root, 'val',   img_size=None, augment=False)

    # Training: random-crop collate to produce crop_size × crop_size batches
    train_collate = partial(_random_crop_collate_fn, crop_size=crop_size)
    tr_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_collate,
    )
    # Evaluation: batch_size=1 because images have variable (native) sizes
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    best_miou = -1.0
    best_ckpt = os.path.join(output_dir, 'best_m2f.pth')

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            backbone_adapter, m2f_head, tr_loader,
            optimizer, device, epoch, num_classes,
        )
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            val_metrics = evaluate_m2f(
                backbone_adapter, m2f_head,
                val_loader, num_classes, device, class_names,
                inference_mode=inference_mode,
                crop_size=crop_size, stride=stride,
            )
            miou = val_metrics.get('mIoU', 0.0)
            logger.info(
                f"Epoch {epoch:3d}/{epochs}  loss={loss:.4f}  "
                f"val_mIoU={miou:.4f}  val_mDice={val_metrics.get('mDice', 0):.4f}  "
                f"val_AJI={val_metrics.get('AJI', 0):.4f}  "
                f"val_AP50={val_metrics.get('AP50', 0):.4f}"
            )
            if miou > best_miou:
                best_miou = miou
                torch.save(
                    {
                        'adapter': backbone_adapter.state_dict(),
                        'head':    m2f_head.state_dict(),
                        'epoch':   epoch,
                        'miou':    miou,
                    },
                    best_ckpt,
                )

    # ---- Final evaluation ----
    ckpt = torch.load(best_ckpt, map_location=device)
    backbone_adapter.load_state_dict(ckpt['adapter'])
    m2f_head.load_state_dict(ckpt['head'])
    logger.info(f"Loaded best checkpoint from epoch {ckpt['epoch']} (val_mIoU={ckpt['miou']:.4f})")

    eval_kwargs = dict(
        inference_mode=inference_mode,
        crop_size=crop_size,
        stride=stride,
    )
    results = {
        'val': evaluate_m2f(
            backbone_adapter, m2f_head,
            val_loader, num_classes, device, class_names, **eval_kwargs,
        )
    }
    try:
        te_ds     = _build_dataset(dataset_name, data_root, 'test', img_size=None, augment=False)
        te_loader = DataLoader(te_ds, batch_size=1, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
        results['test'] = evaluate_m2f(
            backbone_adapter, m2f_head,
            te_loader, num_classes, device, class_names, **eval_kwargs,
        )
    except Exception as e:
        logger.warning(f"Test split evaluation skipped: {e}")
        results['test'] = {}

    out_json = os.path.join(output_dir, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {out_json}")

    for split, metrics in results.items():
        if metrics:
            logger.info(f"\n{'='*60}\n  {split.upper()}\n{'='*60}")
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    logger.info(f"  {k:<25s}: {v:.4f}")

    return results


# ============================================================================
# CLI
# ============================================================================

DATASET_CONFIGS = {
    'bbbc038':   {'num_classes': 2, 'class_names': ['background', 'cell']},
    'conic':     {'num_classes': 7, 'class_names': ['background', 'neutrophil', 'epithelial',
                                                     'lymphocyte', 'plasma_cell', 'eosinophil',
                                                     'connective']},
    'livecell':  {'num_classes': 2, 'class_names': ['background', 'cell']},
    'monuseg':   {'num_classes': 2, 'class_names': ['background', 'nucleus']},
    'pannuke':   {'num_classes': 6, 'class_names': ['background', 'neoplastic', 'inflammatory',
                                                     'connective', 'dead', 'epithelial']},
    'tissuenet': {'num_classes': 2, 'class_names': ['background', 'cell']},
    'cellpose':  {'num_classes': 2, 'class_names': ['background', 'cell']},
    'csc':       {'num_classes': 2, 'class_names': ['background', 'cell']},
}


def main():
    parser = argparse.ArgumentParser(description='Mask2Former for bio-segmentation')
    parser.add_argument('--dataset',      required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--data-root',    required=True)
    parser.add_argument('--checkpoint',   required=True)
    parser.add_argument('--output-dir',   required=True)
    parser.add_argument('--model-size',      default='l', choices=['l', '7b'])
    parser.add_argument('--img-size',        type=int, default=0,
                        help='Training image size (0 = per-dataset canonical from '
                             'DATASET_DEFAULT_IMG_SIZES, e.g. 256 for CoNIC/PanNuke, '
                             '512 for MoNuSeg/LIVECell). Rounded to multiple of 16.')
    parser.add_argument('--inference-mode',  default='whole', choices=['whole', 'slide', 'auto'],
                        help='"whole": resize to crop_size then upsample (fast, official default). '
                             '"slide": sliding window (better for large images ≥512px). '
                             '"auto": whole for <512px, slide for ≥512px.')
    parser.add_argument('--crop-size',       type=int, default=512,
                        help='Crop size for whole/slide inference (must be mult of 16). '
                             'Official DINOv3 default: 512.')
    parser.add_argument('--stride',          type=int, default=341,
                        help='Stride for slide inference. Official default: 341 (≈2/3 crop_size).')
    parser.add_argument('--epochs',          type=int,   default=50)
    parser.add_argument('--lr',            type=float, default=1e-4,
                        help='Learning rate for Mask2FormerHead')
    parser.add_argument('--adapter-lr',   type=float, default=1e-5,
                        help='Learning rate for DINOv3_Adapter trainable layers '
                             '(SPM, InteractionBlocks, up-conv, norms)')
    parser.add_argument('--batch-size',   type=int,   default=2)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers',  type=int,   default=2)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    run_mask2former(
        dataset_name   = args.dataset,
        data_root      = args.data_root,
        checkpoint     = args.checkpoint,
        output_dir     = args.output_dir,
        num_classes    = cfg['num_classes'],
        class_names    = cfg['class_names'],
        model_size     = args.model_size,
        img_size       = args.img_size,
        crop_size      = args.crop_size,
        stride         = args.stride,
        inference_mode = args.inference_mode,
        epochs         = args.epochs,
        lr             = args.lr,
        adapter_lr     = args.adapter_lr,
        batch_size     = args.batch_size,
        weight_decay   = args.weight_decay,
        num_workers    = args.num_workers,
    )


if __name__ == '__main__':
    main()
