"""
Instance and panoptic segmentation metrics for bio-image evaluation.

Implements:
    AJI     Aggregated Jaccard Index
    AP@0.5  Average Precision at IoU=0.5
    AP      COCO-style AP@0.5:0.05:0.95
    PQ      Panoptic Quality  (SQ × RQ)
    bPQ     binary PQ  (treat all instances as one class)
    mPQ     mean PQ   (average per-class PQ, excluding background)

All functions accept NumPy integer arrays:
    pred_instance : (H, W) int  – predicted instance map  (0 = background)
    gt_instance   : (H, W) int  – ground-truth instance map (0 = background)

For multi-class datasets:
    pred_semantic : (H, W) int  – predicted class map
    gt_semantic   : (H, W) int  – ground-truth class map

Reference implementations:
    AJI  → Kumar et al. "A Dataset and a Technique for Generalized Nuclear
            Segmentation for Computational Pathology", TMI 2017
    PQ   → Kirillov et al. "Panoptic Segmentation", CVPR 2019
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Internal helpers
# ============================================================================

def _get_instance_ids(inst_map: np.ndarray) -> np.ndarray:
    """Return sorted unique non-zero instance IDs."""
    return np.unique(inst_map[inst_map > 0])


def _pairwise_iou(
    pred_map: np.ndarray,
    gt_map:   np.ndarray,
    pred_ids: np.ndarray,
    gt_ids:   np.ndarray,
) -> np.ndarray:
    """
    Compute IoU for every (gt_id, pred_id) pair.

    Returns:
        iou_matrix : [len(gt_ids), len(pred_ids)] float32
    """
    iou = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for gi, gid in enumerate(gt_ids):
        gt_mask = gt_map == gid
        for pi, pid in enumerate(pred_ids):
            p_mask = pred_map == pid
            inter  = (gt_mask & p_mask).sum()
            if inter == 0:
                continue
            union  = (gt_mask | p_mask).sum()
            iou[gi, pi] = inter / (union + 1e-8)
    return iou


# ============================================================================
# AJI (Aggregated Jaccard Index)
# ============================================================================

def compute_aji(
    pred_instance: np.ndarray,
    gt_instance:   np.ndarray,
) -> float:
    """
    Compute the Aggregated Jaccard Index (AJI).

    For each GT instance G_i, greedily assign the prediction P_σ(i) with the
    highest IoU (may be 0 / no match).  Unmatched predictions contribute their
    area to the denominator.

    Args:
        pred_instance : (H, W) predicted instance map (0 = background).
        gt_instance   : (H, W) ground-truth instance map (0 = background).

    Returns:
        AJI score in [0, 1].
    """
    pred_ids = _get_instance_ids(pred_instance)
    gt_ids   = _get_instance_ids(gt_instance)

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    iou_mat = _pairwise_iou(pred_instance, gt_instance, pred_ids, gt_ids)
    # Transpose so shape is [n_gt, n_pred]
    iou_mat = _pairwise_iou(pred_instance, gt_instance, pred_ids, gt_ids).T  # [n_gt, n_pred]

    numerator   = 0.0
    denominator = 0.0
    matched_pred_ids = set()

    for gi, gid in enumerate(gt_ids):
        gt_mask = gt_instance == gid
        gt_area = gt_mask.sum()

        # Best matching prediction
        best_pi  = int(np.argmax(iou_mat[gi]))
        best_iou = float(iou_mat[gi, best_pi])

        if best_iou > 0.0:
            pid      = pred_ids[best_pi]
            p_mask   = pred_instance == pid
            inter    = (gt_mask & p_mask).sum()
            union    = (gt_mask | p_mask).sum()
            numerator   += inter
            denominator += union
            matched_pred_ids.add(pid)
        else:
            # No matching prediction → denominator gets GT area
            numerator   += 0
            denominator += gt_area

    # Unmatched predictions contribute their area to the denominator
    for pid in pred_ids:
        if pid not in matched_pred_ids:
            denominator += (pred_instance == pid).sum()

    return float(numerator / (denominator + 1e-8))


# ============================================================================
# PQ / bPQ / mPQ
# ============================================================================

def _compute_pq_single_class(
    pred_instance: np.ndarray,
    gt_instance:   np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float, int, int, int]:
    """
    PQ for a single class (or binary).

    Returns:
        (pq, sq, rq, n_tp, n_fp, n_fn)
    """
    pred_ids = _get_instance_ids(pred_instance)
    gt_ids   = _get_instance_ids(gt_instance)

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    if len(gt_ids) == 0:
        return 0.0, 0.0, 0.0, 0, len(pred_ids), 0
    if len(pred_ids) == 0:
        return 0.0, 0.0, 0.0, 0, 0, len(gt_ids)

    iou_mat = _pairwise_iou(pred_instance, gt_instance, pred_ids, gt_ids).T  # [n_gt, n_pred]

    matched_gt   = set()
    matched_pred = set()
    iou_sum = 0.0

    # Greedy matching (highest IoU first)
    iou_flat  = iou_mat.flatten()
    order     = np.argsort(-iou_flat)
    for idx in order:
        gi, pi = divmod(int(idx), len(pred_ids))
        if iou_mat[gi, pi] < iou_threshold:
            break
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        iou_sum += iou_mat[gi, pi]

    n_tp = len(matched_gt)
    n_fp = len(pred_ids) - n_tp
    n_fn = len(gt_ids)   - n_tp

    rq = n_tp / (n_tp + 0.5 * n_fp + 0.5 * n_fn + 1e-8)
    sq = iou_sum / (n_tp + 1e-8) if n_tp > 0 else 0.0
    pq = sq * rq

    return float(pq), float(sq), float(rq), n_tp, n_fp, n_fn


def compute_pq(
    pred_instance: np.ndarray,
    gt_instance:   np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Binary Panoptic Quality (bPQ): treat all instances as one class.

    Returns:
        dict with keys 'pq', 'sq', 'rq', 'n_tp', 'n_fp', 'n_fn'.
    """
    pq, sq, rq, tp, fp, fn = _compute_pq_single_class(
        pred_instance, gt_instance, iou_threshold
    )
    return {'pq': pq, 'sq': sq, 'rq': rq, 'n_tp': tp, 'n_fp': fp, 'n_fn': fn}


def compute_multi_class_pq(
    pred_instance: np.ndarray,
    pred_semantic: np.ndarray,
    gt_instance:   np.ndarray,
    gt_semantic:   np.ndarray,
    num_classes:   int,
    ignore_bg:     bool = True,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Multi-class Panoptic Quality: compute PQ per class, then average.

    Instances are split by their semantic class (majority vote on GT).

    Args:
        pred_instance : (H, W) predicted instance map.
        pred_semantic : (H, W) predicted semantic class map.
        gt_instance   : (H, W) GT instance map.
        gt_semantic   : (H, W) GT semantic class map.
        num_classes   : total classes including background.
        ignore_bg     : if True, skip class 0 (background) in averaging.
        iou_threshold : matching IoU threshold.

    Returns:
        dict with 'mPQ', 'bPQ', and per-class 'pq_class_<c>' values.
    """
    results: Dict[str, float] = {}

    # ---------- binary PQ (bPQ) ----------
    pred_bin = (pred_instance > 0).astype(np.int64)
    gt_bin   = (gt_instance   > 0).astype(np.int64)
    # Re-label connected components for binary
    from skimage.measure import label as sk_label
    pred_bin_inst = sk_label(pred_bin, connectivity=2)
    gt_bin_inst   = sk_label(gt_bin,   connectivity=2)
    bpq_dict = compute_pq(pred_bin_inst, gt_bin_inst, iou_threshold)
    results['bPQ'] = bpq_dict['pq']
    results['bSQ'] = bpq_dict['sq']
    results['bRQ'] = bpq_dict['rq']

    # ---------- per-class PQ (mPQ) ----------
    start_cls = 1 if ignore_bg else 0
    class_pqs = []

    for cls in range(start_cls, num_classes):
        # Extract GT instances for this class
        gt_cls_mask   = gt_semantic == cls
        gt_cls_inst   = np.where(gt_cls_mask, gt_instance, 0)

        # Extract predicted instances for this class
        pred_cls_mask = pred_semantic == cls
        pred_cls_inst = np.where(pred_cls_mask, pred_instance, 0)

        pq, sq, rq, tp, fp, fn = _compute_pq_single_class(
            pred_cls_inst, gt_cls_inst, iou_threshold
        )
        results[f'pq_class_{cls}'] = pq
        results[f'sq_class_{cls}'] = sq
        results[f'rq_class_{cls}'] = rq
        class_pqs.append(pq)

    results['mPQ'] = float(np.nanmean(class_pqs)) if class_pqs else float('nan')
    return results


# ============================================================================
# AP (COCO-style)
# ============================================================================

def compute_ap(
    pred_instance: np.ndarray,
    gt_instance:   np.ndarray,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    COCO-style Average Precision.

    Since linear-probe outputs don't have confidence scores, we assign each
    predicted instance a score proportional to its area (larger = more confident).

    IoU thresholds: 0.5, 0.55, ..., 0.95  → AP@0.5:0.95  (COCO mAP)
    Also reports AP@0.5 and AP@0.75 separately.

    Args:
        pred_instance : (H, W) predicted instance map (0 = bg).
        gt_instance   : (H, W) ground-truth instance map (0 = bg).
        iou_thresholds: custom thresholds (default: COCO 0.5:0.05:0.95).

    Returns:
        dict with keys 'AP', 'AP50', 'AP75'.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]

    pred_ids = _get_instance_ids(pred_instance)
    gt_ids   = _get_instance_ids(gt_instance)

    n_gt = len(gt_ids)

    if n_gt == 0 and len(pred_ids) == 0:
        return {'AP': 1.0, 'AP50': 1.0, 'AP75': 1.0}
    if n_gt == 0:
        return {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0}
    if len(pred_ids) == 0:
        return {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0}

    # Sort predictions by area (descending → larger cells first)
    pred_areas = {pid: (pred_instance == pid).sum() for pid in pred_ids}
    pred_ids_sorted = sorted(pred_ids, key=lambda p: pred_areas[p], reverse=True)

    iou_mat = _pairwise_iou(pred_instance, gt_instance, pred_ids_sorted, gt_ids).T  # [n_gt, n_pred_sorted]

    ap_list: List[float] = []
    results: Dict[str, float] = {}

    for thresh in iou_thresholds:
        gt_matched = [False] * n_gt
        tp_list, fp_list = [], []

        for pi in range(len(pred_ids_sorted)):
            ious_for_pred = iou_mat[:, pi]     # IoU with each GT
            best_gi = int(np.argmax(ious_for_pred))
            best_iou = float(ious_for_pred[best_gi])

            if best_iou >= thresh and not gt_matched[best_gi]:
                tp_list.append(1)
                fp_list.append(0)
                gt_matched[best_gi] = True
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_cum = np.cumsum(tp_list).astype(np.float32)
        fp_cum = np.cumsum(fp_list).astype(np.float32)
        recalls    = tp_cum / n_gt
        precisions = tp_cum / (tp_cum + fp_cum + 1e-8)

        # 11-point interpolation (VOC style)
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            p = precisions[recalls >= r].max() if (recalls >= r).any() else 0.0
            ap += p / 11.0

        ap_list.append(ap)
        if abs(thresh - 0.50) < 0.001:
            results['AP50'] = float(ap)
        if abs(thresh - 0.75) < 0.001:
            results['AP75'] = float(ap)

    results['AP'] = float(np.mean(ap_list))
    results.setdefault('AP50', float('nan'))
    results.setdefault('AP75', float('nan'))
    return results


# ============================================================================
# Convenience: accumulate across images
# ============================================================================

def accumulate_instance_metrics(
    pred_instances: List[np.ndarray],
    gt_instances:   List[np.ndarray],
    pred_semantics: Optional[List[np.ndarray]] = None,
    gt_semantics:   Optional[List[np.ndarray]] = None,
    num_classes:    int = 2,
    iou_threshold:  float = 0.5,
) -> Dict[str, float]:
    """
    Compute AJI, AP, and PQ averaged over a list of images.

    Args:
        pred_instances : list of (H, W) predicted instance maps.
        gt_instances   : list of (H, W) GT instance maps.
        pred_semantics : list of (H, W) predicted class maps (for mPQ).
        gt_semantics   : list of (H, W) GT class maps (for mPQ).
        num_classes    : number of classes (incl. background); used only when semantics provided.
        iou_threshold  : matching threshold for PQ.

    Returns:
        Averaged dict with keys 'AJI', 'AP', 'AP50', 'AP75', 'bPQ', 'mPQ'.
    """
    aji_list, ap_list, ap50_list, ap75_list, bpq_list, mpq_list = [], [], [], [], [], []

    for i, (p_inst, g_inst) in enumerate(zip(pred_instances, gt_instances)):
        aji_list.append(compute_aji(p_inst, g_inst))

        ap_dict = compute_ap(p_inst, g_inst)
        ap_list.append(ap_dict['AP'])
        ap50_list.append(ap_dict['AP50'])
        ap75_list.append(ap_dict['AP75'])

        if pred_semantics is not None and gt_semantics is not None:
            pq_dict = compute_multi_class_pq(
                p_inst, pred_semantics[i],
                g_inst, gt_semantics[i],
                num_classes=num_classes,
                iou_threshold=iou_threshold,
            )
            bpq_list.append(pq_dict['bPQ'])
            mpq_list.append(pq_dict['mPQ'])
        else:
            pq_dict = compute_pq(p_inst, g_inst, iou_threshold)
            bpq_list.append(pq_dict['pq'])

    results: Dict[str, float] = {
        'AJI':  float(np.nanmean(aji_list)),
        'AP':   float(np.nanmean(ap_list)),
        'AP50': float(np.nanmean(ap50_list)),
        'AP75': float(np.nanmean(ap75_list)),
        'bPQ':  float(np.nanmean(bpq_list)),
    }
    if mpq_list:
        results['mPQ'] = float(np.nanmean(mpq_list))

    return results
