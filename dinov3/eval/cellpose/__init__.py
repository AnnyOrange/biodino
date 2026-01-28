# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Cellpose 16-bit 图像预处理验证模块

用于验证不同预处理方法对 DINOv3 特征提取效果的影响。

提供两种评估方式：
1. Linear Probing: 冻结 backbone，训练线性分类头，通过 mIoU 评估
2. Zero-Shot PCA: 无需训练，通过 PCA 可视化评估特征质量

使用方法:
    # 检查环境
    python -m dinov3.eval.cellpose.test_setup
    
    # 运行 Linear Probing 实验
    python -m dinov3.eval.cellpose.run_experiment
    
    # 运行 Zero-Shot PCA 可视化
    python -m dinov3.eval.cellpose.zero_shot_pca \
        --checkpoint /mnt/deepcad_nfs/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --model-size l \
        --data-path /mnt/deepcad_nfs/0-large-model-dataset/11-Cellpose \
        --output-dir dinov3/outputs/cellpose_pca
    
    # 可视化 Linear Probing 结果
    python -m dinov3.eval.cellpose.visualize_results <results_dir>
"""

