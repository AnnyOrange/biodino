# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
多通道 TIFF 图像解码器模块。

用于从 WebDataset 字节流中解码多通道 TIFF 图像。
"""

import io
import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger("dinov3")


def decode_tiff_bytes(tiff_bytes: bytes) -> Optional[Tensor]:
    """
    从字节流解码 TIFF 图像为 PyTorch Tensor。

    Args:
        tiff_bytes: TIFF 图像的原始字节数据。

    Returns:
        解码后的 Tensor，形状为 (C, H, W)，dtype 为 float32。
        如果解码失败，返回 None。

    Raises:
        无显式抛出异常，错误时返回 None 并记录 Warning。
    """
    try:
        import tifffile
    except ImportError:
        logger.error("tifffile 未安装，请运行: pip install tifffile")
        return None

    try:
        with io.BytesIO(tiff_bytes) as buffer:
            image_array = tifffile.imread(buffer)
    except Exception as e:
        logger.warning(f"TIFF 解码失败: {type(e).__name__}: {e}")
        return None

    return _normalize_tiff_array(image_array)


def _normalize_tiff_array(image_array: np.ndarray) -> Optional[Tensor]:
    """
    将 TIFF numpy 数组标准化为 (C, H, W) 格式的 Tensor。

    Args:
        image_array: 从 tifffile 读取的 numpy 数组。

    Returns:
        标准化后的 Tensor，形状为 (C, H, W)，dtype 为 float32。
    """
    if image_array is None:
        return None

    # 处理不同的维度情况
    if image_array.ndim == 2:
        # 灰度图: (H, W) -> (1, H, W)
        image_array = image_array[np.newaxis, :, :]
    elif image_array.ndim == 3:
        # 多通道: 判断通道维度位置
        image_array = _ensure_channel_first(image_array)
    else:
        logger.warning(f"不支持的 TIFF 维度: {image_array.ndim}")
        return None

    # 转换为 float32 并归一化到 [0, 1]
    tensor = _to_float_tensor(image_array)
    return tensor  # Shape: (C, H, W)


def _ensure_channel_first(image_array: np.ndarray) -> np.ndarray:
    """
    确保数组为 channel-first 格式 (C, H, W)。

    Args:
        image_array: 3D numpy 数组，可能是 (H, W, C) 或 (C, H, W)。

    Returns:
        Channel-first 格式的数组 (C, H, W)。
    """
    # 启发式判断：通道数通常 < 16，空间维度通常 > 16
    if image_array.shape[2] < image_array.shape[0]:
        # 当前是 (H, W, C)，需要转置
        return image_array.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    # 已经是 (C, H, W)
    return image_array


def _to_float_tensor(image_array: np.ndarray) -> Tensor:
    """
    将 numpy 数组转换为归一化的 float32 Tensor。

    Args:
        image_array: numpy 数组，shape 为 (C, H, W)。

    Returns:
        归一化到 [0, 1] 的 float32 Tensor。
    """
    # 根据 dtype 确定归一化因子
    dtype_info = _get_dtype_normalization_info(image_array.dtype)
    max_val = dtype_info

    # 转换为 float32
    tensor = torch.from_numpy(image_array.astype(np.float32))

    # 归一化到 [0, 1]
    if max_val > 1.0:
        tensor = tensor / max_val

    return tensor  # Shape: (C, H, W), dtype: float32


def _get_dtype_normalization_info(dtype: np.dtype) -> float:
    """
    获取 numpy dtype 的归一化因子。

    Args:
        dtype: numpy 数据类型。

    Returns:
        该数据类型的最大值，用于归一化。
    """
    dtype_max_map = {
        np.uint8: 255.0,
        np.uint16: 65535.0,
        np.uint32: 4294967295.0,
        np.float32: 1.0,
        np.float64: 1.0,
    }
    return dtype_max_map.get(dtype.type, 255.0)


def create_tiff_decoder() -> callable:
    """
    创建用于 WebDataset 的 TIFF 解码器函数。

    Returns:
        可被 WebDataset map_dict 使用的解码器函数。

    Example:
        >>> decoder = create_tiff_decoder()
        >>> pipeline = wds.DataPipeline(...).map_dict(tiff=decoder)
    """
    def decoder(tiff_bytes: bytes) -> Optional[Tensor]:
        return decode_tiff_bytes(tiff_bytes)

    return decoder

