"""
Semantic segmentation metrics for bio-image evaluation.

Computes pixel-wise metrics between predicted class maps and ground-truth:
    - Dice (F1)          per class and macro-averaged
    - IoU (Jaccard)      per class and macro-averaged (mIoU)
    - Precision          per class
    - Recall             per class

All functions operate on NumPy arrays or PyTorch tensors and return plain dicts.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# ============================================================================
# Internal helpers
# ============================================================================

def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """
    Compute (num_classes × num_classes) confusion matrix.
    Rows = ground truth, Columns = predicted.
    """
    valid = (target != ignore_index)
    pred_v   = pred[valid].astype(np.int64)
    target_v = target[valid].astype(np.int64)

    # Clamp out-of-range predictions
    pred_v = np.clip(pred_v, 0, num_classes - 1)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (target_v, pred_v), 1)
    return cm


# ============================================================================
# Public API
# ============================================================================

def compute_semantic_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    class_names: Optional[List[str]] = None,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """
    Compute per-class and macro-averaged Dice, IoU, Precision, Recall.

    Args:
        pred        : (H, W) integer class map (predicted).
        target      : (H, W) integer class map (ground truth).
        num_classes : total number of classes (including background).
        class_names : optional list of class name strings.
        ignore_index: pixels with this label are ignored.

    Returns:
        Dictionary with keys:
            'mIoU', 'mDice', 'mPrecision', 'mRecall'
            'iou_<name>', 'dice_<name>', 'precision_<name>', 'recall_<name>'  (per class)
    """
    pred   = _to_numpy(pred)
    target = _to_numpy(target)

    if class_names is None:
        class_names = [f'class_{c}' for c in range(num_classes)]

    cm = _confusion_matrix(pred, target, num_classes, ignore_index)

    results: Dict[str, float] = {}
    ious, dices, precs, recs = [], [], [], []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp   # column c, all rows except c
        fn = cm[c, :].sum() - tp   # row c, all cols except c

        iou       = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else float('nan')
        dice      = 2 * tp / (2 * tp + fp + fn + 1e-8) if (2 * tp + fp + fn) > 0 else float('nan')
        precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else float('nan')
        recall    = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else float('nan')

        name = class_names[c] if c < len(class_names) else f'class_{c}'
        results[f'iou_{name}']       = float(iou)
        results[f'dice_{name}']      = float(dice)
        results[f'precision_{name}'] = float(precision)
        results[f'recall_{name}']    = float(recall)

        ious.append(iou)
        dices.append(dice)
        precs.append(precision)
        recs.append(recall)

    results['mIoU']       = float(np.nanmean(ious))
    results['mDice']      = float(np.nanmean(dices))
    results['mPrecision'] = float(np.nanmean(precs))
    results['mRecall']    = float(np.nanmean(recs))

    return results


def accumulate_semantic_metrics(
    preds: List[Union[np.ndarray, torch.Tensor]],
    targets: List[Union[np.ndarray, torch.Tensor]],
    num_classes: int,
    class_names: Optional[List[str]] = None,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """
    Compute dataset-level semantic metrics by accumulating a global confusion matrix.

    Args:
        preds   : list of (H, W) predicted class maps.
        targets : list of (H, W) ground-truth class maps.

    Returns:
        Same keys as ``compute_semantic_metrics``.
    """
    if class_names is None:
        class_names = [f'class_{c}' for c in range(num_classes)]

    global_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, target in zip(preds, targets):
        pred   = _to_numpy(pred)
        target = _to_numpy(target)
        global_cm += _confusion_matrix(pred, target, num_classes, ignore_index)

    results: Dict[str, float] = {}
    ious, dices, precs, recs = [], [], [], []

    for c in range(num_classes):
        tp = global_cm[c, c]
        fp = global_cm[:, c].sum() - tp
        fn = global_cm[c, :].sum() - tp

        iou       = float(tp / (tp + fp + fn + 1e-8)) if (tp + fp + fn) > 0 else float('nan')
        dice      = float(2 * tp / (2 * tp + fp + fn + 1e-8)) if (2 * tp + fp + fn) > 0 else float('nan')
        precision = float(tp / (tp + fp + 1e-8)) if (tp + fp) > 0 else float('nan')
        recall    = float(tp / (tp + fn + 1e-8)) if (tp + fn) > 0 else float('nan')

        name = class_names[c]
        results[f'iou_{name}']       = iou
        results[f'dice_{name}']      = dice
        results[f'precision_{name}'] = precision
        results[f'recall_{name}']    = recall

        ious.append(iou)
        dices.append(dice)
        precs.append(precision)
        recs.append(recall)

    results['mIoU']       = float(np.nanmean(ious))
    results['mDice']      = float(np.nanmean(dices))
    results['mPrecision'] = float(np.nanmean(precs))
    results['mRecall']    = float(np.nanmean(recs))

    return results
