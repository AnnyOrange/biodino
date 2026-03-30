# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
WebDataset 图像解码器模块。

用于从 WebDataset 字节流中解码多通道 TIFF/NPY 图像。
"""

import io
import logging
from typing import Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger("dinov3")


def decode_tiff_bytes(tiff_bytes: bytes, target_channels: Optional[int] = None) -> Optional[Tensor]:
    """
    从字节流解码 TIFF 图像为 PyTorch Tensor。

    Args:
        tiff_bytes: TIFF 图像的原始字节数据。
        target_channels: 目标通道数；为 None 时保持原始通道数。

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

    return _normalize_image_array(image_array, target_channels=target_channels)


def decode_npy_bytes(npy_bytes: bytes, target_channels: Optional[int] = None) -> Optional[Tensor]:
    """
    从字节流解码 NPY 数组为 PyTorch Tensor。

    Args:
        npy_bytes: NPY 数组的原始字节数据。
        target_channels: 目标通道数；为 None 时保持原始通道数。

    Returns:
        解码后的 Tensor，形状为 (C, H, W)，dtype 为 float32。
        如果解码失败，返回 None。
    """
    try:
        with io.BytesIO(npy_bytes) as buffer:
            image_array = np.load(buffer, allow_pickle=False)
    except Exception as e:
        logger.warning(f"NPY 解码失败: {type(e).__name__}: {e}")
        return None

    return _normalize_image_array(image_array, target_channels=target_channels)


def _normalize_image_array(image_array: np.ndarray, target_channels: Optional[int] = None) -> Optional[Tensor]:
    """
    将图像数组标准化为 (C, H, W) 格式的 Tensor。

    Args:
        image_array: 从 TIFF/NPY 读取的 numpy 数组。
        target_channels: 目标通道数；为 None 时保持原始通道数。

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
        logger.warning(f"不支持的图像维度: {image_array.ndim}")
        return None

    if target_channels is not None:
        image_array = _ensure_target_channels(image_array, target_channels)

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
        return np.ascontiguousarray(image_array.transpose(2, 0, 1))  # (H, W, C) -> (C, H, W)
    # 已经是 (C, H, W)
    return np.ascontiguousarray(image_array)


def _ensure_target_channels(image_array: np.ndarray, target_channels: int) -> np.ndarray:
    """
    将 (C, H, W) 数组调整为 (target_channels, H, W)。

    规则：
      - 1 通道：复制到目标通道数
      - 通道数等于目标：原样返回
      - 通道数大于目标：截断前 target_channels 个通道
      - 其他小于目标的情况：循环填充
    """
    channels = image_array.shape[0]

    if channels == target_channels:
        return image_array

    if channels == 1:
        return np.repeat(image_array, target_channels, axis=0)

    if channels > target_channels:
        return image_array[:target_channels, :, :]

    repeats = (target_channels + channels - 1) // channels
    tiled = np.tile(image_array, (repeats, 1, 1))
    return tiled[:target_channels, :, :]


def _to_float_tensor(image_array: np.ndarray) -> Tensor:
    """
    将 numpy 数组转换为归一化的 float32 Tensor。

    Args:
        image_array: numpy 数组，shape 为 (C, H, W)。

    Returns:
        归一化到 [0, 1] 的 float32 Tensor。
    """
    if np.issubdtype(image_array.dtype, np.floating):
        array = np.clip(image_array, 0.0, 1.0).astype(np.float32, copy=False)
        return torch.from_numpy(array)

    if np.issubdtype(image_array.dtype, np.unsignedinteger):
        max_val = float(np.iinfo(image_array.dtype).max)
        array = image_array.astype(np.float32) / max_val
        return torch.from_numpy(array)

    # 其他整型（如 int16）使用 min-max 归一化，避免错误地按 255 缩放。
    array = image_array.astype(np.float32)
    min_val = float(array.min())
    max_val = float(array.max())
    if max_val > min_val:
        array = (array - min_val) / (max_val - min_val)
    else:
        array = np.zeros_like(array, dtype=np.float32)
    return torch.from_numpy(array)


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

