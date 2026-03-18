# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档 — 极速空块过滤模块。

"""
极速空块过滤 (Fast Quality Check)。

采用零拷贝降采样（stride=16）将计算量骤降 256 倍，
随后通过灰度阈值检测有效组织覆盖率和方差，
替代原有 float64 转换 + np.percentile 的高开销方案。

输入约定：array 为 (C, H, W) 格式的 ndarray。
"""

import logging

import numpy as np

from .config import QUALITY_DS_FACTOR, QUALITY_MIN_COVERAGE, QUALITY_MIN_STD

logger = logging.getLogger("dinov3")


def passes_quality_check(
    array: np.ndarray,
    ds_factor: int = QUALITY_DS_FACTOR,
) -> bool:
    """
    极速空块过滤入口。

    三阶段检测：
    1. 全常量快速拒绝（零拷贝降采样后判断）。
    2. 有效组织覆盖率低于阈值则拒绝。
    3. 有效像素标准差过低则拒绝（排除纯色死区）。

    Args:
        array: (C, H, W) 格式的裁切 patch。
        ds_factor: 降采样步长（默认 16，计算量降 256 倍）。

    Returns:
        True 表示通过质量检测，允许打包。
    """
    # 零拷贝降采样：仅对空间维度 H, W 步进采样
    thumb = array[:, ::ds_factor, ::ds_factor]  # (C, H', W')

    # 阶段 1：全常量快速拒绝
    if np.all(thumb == thumb.flat[0]):
        logger.debug("空块拒绝: 全常量 patch")
        return False

    # 跨通道取均值得到灰度缩略图
    gray = thumb.mean(axis=0)  # (H', W')

    # 阶段 2：有效组织覆盖率检测
    tissue_mask = _build_tissue_mask(gray)
    coverage = float(tissue_mask.mean())
    if coverage < QUALITY_MIN_COVERAGE:
        logger.debug(f"空块拒绝: 覆盖率 {coverage:.4f} < {QUALITY_MIN_COVERAGE}")
        return False

    # 阶段 3：有效像素标准差检测（排除纯色死区）
    valid_pixels = gray[tissue_mask]
    if len(valid_pixels) > 0 and float(valid_pixels.std()) < QUALITY_MIN_STD:
        logger.debug(f"空块拒绝: 有效像素 std={valid_pixels.std():.2f}")
        return False

    return True


def _build_tissue_mask(gray: np.ndarray) -> np.ndarray:
    """
    基于动态灰度范围计算有效组织掩膜。

    将灰度值低于 6% 分位（近黑背景）和高于 92% 分位（饱和白）
    排除在外，仅保留中间区域作为有效组织。
    比例与 uint8 下 15/255 ≈ 0.06、235/255 ≈ 0.92 等效，
    对 int16 / uint16 / float 等多种 dtype 均适用。

    Args:
        gray: (H', W') 灰度缩略图。

    Returns:
        (H', W') 布尔掩膜，True 为有效组织像素。
    """
    lo = float(gray.min())
    hi = float(gray.max())
    span = hi - lo

    # 动态范围极小时视为空块（不含足够信息）
    if span < 1e-6:
        return np.zeros_like(gray, dtype=bool)

    bg_low = lo + 0.06 * span
    bg_high = lo + 0.92 * span
    return (gray > bg_low) & (gray < bg_high)  # (H', W') bool
