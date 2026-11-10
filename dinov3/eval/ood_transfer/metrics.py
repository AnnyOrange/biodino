"""Native OOD task metrics, independent of the encoder implementation."""
from __future__ import annotations
import numpy as np

RSNA_THRESHOLDS = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75)


def rsna_image_precision(truth, boxes, scores, thresholds=RSNA_THRESHOLDS):
    """Kaggle RSNA rules: confidence-ordered, strict IoU > t, one-to-one.

    Empty truth AND empty submission is unscored. Empty truth with a prediction
    is zero. Coordinates are continuous xyxy (no inclusive-pixel +1).
    Source: kaggle.com/c/rsna-pneumonia-detection-challenge/overview/evaluation
    """
    truth = np.asarray(truth, dtype=np.float64).reshape(-1, 4)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores) != len(boxes):
        raise ValueError("Each predicted box must have one confidence")
    if not all(np.isfinite(x).all() for x in (truth, boxes, scores)):
        raise ValueError("Nonfinite detection input")
    if any(np.any(x[:, 2:] <= x[:, :2]) for x in (truth, boxes)):
        raise ValueError("Detection boxes must have positive area")
    if len(truth) == len(boxes) == 0:
        return None
    if not len(truth) or not len(boxes):
        return 0.0
    wh = np.maximum(0, np.minimum(boxes[:, None, 2:], truth[None, :, 2:]) -
                    np.maximum(boxes[:, None, :2], truth[None, :, :2]))
    intersection = wh.prod(axis=2)
    union = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)[:, None] + (
        truth[:, 2:] - truth[:, :2]).prod(axis=1)[None, :] - intersection
    iou = intersection / union
    values = []
    for threshold in thresholds:
        used = np.zeros(len(truth), dtype=bool)
        tp = 0
        for j in np.argsort(-scores, kind="stable"):
            available = np.where((iou[j] > threshold) & ~used)[0]
            if len(available):
                best = available[np.argmax(iou[j, available])]
                used[best] = True
                tp += 1
        values.append(tp / (len(boxes) + len(truth) - tp))
    return float(np.mean(values))


def structure_knn(features, labels, *, chunk_size=256, groups=None):
    """Top-1 structural-origin kNN; exclude self, optionally same micrograph.

    Operates on raw frozen embeddings, not a fitted UMAP. The optional grouped
    score is a separate robustness analysis; the primary score excludes self.
    Chunking bounds memory independently of the number of particles.
    """
    x = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(labels) or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Invalid structure features/labels")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Zero embedding cannot be cosine-normalized")
    x = x / norm
    groups = None if groups is None else np.asarray(groups)
    if groups is not None and len(groups) != len(x):
        raise ValueError("Group count mismatch")
    neighbors = np.empty(len(x), dtype=np.int64)
    for start in range(0, len(x), chunk_size):
        end = min(start + chunk_size, len(x))
        similarity = x[start:end] @ x.T
        similarity[np.arange(end-start), np.arange(start, end)] = -np.inf
        if groups is not None:
            similarity[groups[start:end, None] == groups[None, :]] = -np.inf
        if not np.isfinite(similarity).any(axis=1).all():
            raise ValueError("No independent reference neighbor for a query")
        neighbors[start:end] = similarity.argmax(axis=1)
    prediction = labels[neighbors]
    return {"top1_knn_accuracy": float(np.mean(prediction == labels)),
            "n_particles": len(labels), "n_structures": len(np.unique(labels))}, neighbors


def pxct_confidence_metrics(truth, prediction):
    """Four ordered expert confidence levels, with native two-category summary."""
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if truth.shape != prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("PXCT confidence arrays must be finite and aligned")
    if np.any((truth < 1) | (truth > 4)):
        raise ValueError("PXCT author ratings must lie in 1..4")
    rounded = lambda x: np.clip(np.floor(x + .5), 1, 4).astype(int)
    y, p = rounded(truth), rounded(prediction)
    confusion = np.zeros((4, 4), dtype=int)
    np.add.at(confusion, (y - 1, p - 1), 1)
    binary = np.zeros((2, 2), dtype=int)
    np.add.at(binary, ((y >= 3).astype(int), (p >= 3).astype(int)), 1)
    tn, fp, fn, tp = binary.ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0
    f2 = float(5*tp/(5*tp+fp+4*fn)) if 5*tp+fp+4*fn else 0.0
    return {"confidence_mae": float(np.mean(np.abs(truth-prediction))),
            "four_category_accuracy": float(np.mean(y == p)),
            "two_category_accuracy": float(np.mean((y >= 3) == (p >= 3))),
            "confusion_four_categories": confusion.tolist(),
            "confusion_two_categories": binary.tolist(), "precision": precision,
            "recall": recall, "f1": f1, "f2": f2, "n_objects": len(y)}
