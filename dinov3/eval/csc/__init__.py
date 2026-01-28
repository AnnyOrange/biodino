"""
CSC (Cell Segmentation Challenge) 数据集评估模块

支持 zero-shot PCA 可视化和 linear probing 评估
"""

from .linear_probe import main as linear_probe_main
from .zero_shot_pca import main as zero_shot_pca_main

__all__ = ['linear_probe_main', 'zero_shot_pca_main']

