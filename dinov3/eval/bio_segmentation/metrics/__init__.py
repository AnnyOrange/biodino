"""
Segmentation metrics for bio-image evaluation.

Semantic (pixel-level):
    compute_semantic_metrics      – per-class and macro Dice/IoU/Precision/Recall
    accumulate_semantic_metrics   – dataset-level (accumulates confusion matrix)

Instance / Panoptic:
    compute_aji                   – Aggregated Jaccard Index
    compute_ap                    – COCO-style AP@0.5:0.95, AP50, AP75
    compute_pq                    – binary PQ (bPQ)
    compute_multi_class_pq        – per-class PQ + mPQ + bPQ
    accumulate_instance_metrics   – average AJI/AP/PQ over a list of images
"""

from .semantic import (
    compute_semantic_metrics,
    accumulate_semantic_metrics,
)
from .instance import (
    compute_aji,
    compute_ap,
    compute_pq,
    compute_multi_class_pq,
    accumulate_instance_metrics,
)

__all__ = [
    'compute_semantic_metrics',
    'accumulate_semantic_metrics',
    'compute_aji',
    'compute_ap',
    'compute_pq',
    'compute_multi_class_pq',
    'accumulate_instance_metrics',
]
