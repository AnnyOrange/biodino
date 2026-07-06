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
import random
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


def decode_packed_sample(
    sample: dict,
    target_channels: int = 8,
) -> Optional[Tensor]:
    """Decode a packed multi-channel sample produced by ``data/repackage``.

    Sample keys follow the pattern ``ch<N>.tif`` (N is 1-indexed).
    Channels present in the sample are decoded and placed at position
    ``ch_num - 1`` in the output tensor.  Missing channels remain **zero**,
    so the model always receives a fixed-size ``(target_channels, H, W)``
    tensor regardless of how many channels a given sample has.

    Args:
        sample: Raw WebDataset sample dict with ``ch*.tif`` and ``meta.json``.
        target_channels: Output tensor channel count.  Channels numbered
            above this value are silently ignored.

    Returns:
        Float32 Tensor ``(target_channels, H, W)`` normalised to ``[0, 1]``,
        or ``None`` if no valid channel could be decoded.
    """
    import re

    ch_key_re = re.compile(r"^ch(\d+)\.tiff?$", re.IGNORECASE)

    # Collect channel bytes keyed by 1-indexed channel number
    channel_bytes: dict[int, bytes] = {}
    for key, value in sample.items():
        m = ch_key_re.match(key)
        if m and isinstance(value, (bytes, bytearray)):
            channel_bytes[int(m.group(1))] = value

    if not channel_bytes:
        logger.warning(
            "packed sample %s has no ch*.tif keys — keys: %s",
            sample.get("__key__", "?"),
            list(sample.keys()),
        )
        return None

    # Decode each channel; derive H/W from first successful decode
    decoded: dict[int, Tensor] = {}
    h: Optional[int] = None
    w: Optional[int] = None

    for ch_num in sorted(channel_bytes):
        if ch_num < 1 or ch_num > target_channels:
            continue
        tensor = decode_tiff_bytes(channel_bytes[ch_num], target_channels=1)
        if tensor is None:
            logger.debug("packed sample: failed to decode ch%d", ch_num)
            continue
        decoded[ch_num] = tensor  # (1, H, W)
        if h is None:
            _, h, w = tensor.shape

    if not decoded or h is None:
        logger.warning(
            "packed sample %s: all channels failed to decode",
            sample.get("__key__", "?"),
        )
        return None

    result = torch.zeros(target_channels, h, w, dtype=torch.float32)
    for ch_num, tensor in decoded.items():
        result[ch_num - 1] = tensor[0]

    return result


def _robust_per_channel(channel: Tensor, p_low: float, p_high: float) -> Tensor:
    """Percentile clip + rescale a single ``(H, W)`` channel to ``[0, 1]``.

    Robust alternative to the dtype-max scaling in ``_to_float_tensor``: instead
    of dividing by the dtype maximum (which leaves 16-bit microscopy dim and
    low-contrast), clip to the ``[p_low, p_high]`` percentiles of *this* channel
    and rescale that window to ``[0, 1]``.  Percentiles are invariant to the
    prior monotonic dtype scaling, so this composes correctly on top of
    ``decode_tiff_bytes`` output without touching ``_to_float_tensor``.

    A (near-)constant channel (``p_high <= p_low``) maps to all zeros.
    """
    flat = channel.flatten().to(torch.float32)
    lo = torch.quantile(flat, p_low / 100.0)
    hi = torch.quantile(flat, p_high / 100.0)
    if not torch.isfinite(lo) or not torch.isfinite(hi) or hi <= lo:
        return torch.zeros_like(channel, dtype=torch.float32)
    out = (channel.to(torch.float32) - lo) / (hi - lo)
    return out.clamp_(0.0, 1.0)


def decode_packed_sample_robust(
    sample: dict,
    target_channels: int = 8,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> Optional[Tensor]:
    """Robust-normalization variant of :func:`decode_packed_sample`.

    Differences from :func:`decode_packed_sample` (which is left UNCHANGED and
    remains the default ``packwds:`` behavior):

    * each present channel is normalized with :func:`_robust_per_channel`
      (per-channel percentile clip + rescale) instead of dtype-max division;
    * a sample with exactly ONE present channel is **replicated** across all
      ``target_channels`` (a grayscale source becomes ``(gray, gray, gray)``
      for an RGB model instead of ``(gray, 0, 0)``).

    Samples with two or more present channels keep the original placement
    (channel ``N`` → index ``N-1``; absent channels stay zero).

    Selected by the ``packwds_robust:`` dataset prefix.
    """
    import re

    ch_key_re = re.compile(r"^ch(\d+)\.tiff?$", re.IGNORECASE)

    channel_bytes: dict[int, bytes] = {}
    for key, value in sample.items():
        m = ch_key_re.match(key)
        if m and isinstance(value, (bytes, bytearray)):
            channel_bytes[int(m.group(1))] = value

    if not channel_bytes:
        logger.warning(
            "packed_robust sample %s has no ch*.tif keys — keys: %s",
            sample.get("__key__", "?"),
            list(sample.keys()),
        )
        return None

    decoded: dict[int, Tensor] = {}
    h: Optional[int] = None
    w: Optional[int] = None
    for ch_num in sorted(channel_bytes):
        if ch_num < 1 or ch_num > target_channels:
            continue
        tensor = decode_tiff_bytes(channel_bytes[ch_num], target_channels=1)
        if tensor is None:
            logger.debug("packed_robust sample: failed to decode ch%d", ch_num)
            continue
        decoded[ch_num] = _robust_per_channel(tensor[0], p_low, p_high)  # (H, W)
        if h is None:
            _, h, w = tensor.shape

    if not decoded or h is None:
        logger.warning(
            "packed_robust sample %s: all channels failed to decode",
            sample.get("__key__", "?"),
        )
        return None

    result = torch.zeros(target_channels, h, w, dtype=torch.float32)
    if len(decoded) == 1:
        # Single present channel → replicate across all target channels.
        only_channel = next(iter(decoded.values()))
        result[:] = only_channel.unsqueeze(0)
    else:
        for ch_num, channel in decoded.items():
            result[ch_num - 1] = channel

    return result


def decode_packed_channelvit_sample(
    sample: dict,
    *,
    max_channels: int,
    sample_channels: Optional[int] = None,
    min_channels: int = 1,
) -> Optional[dict[str, Tensor]]:
    """Decode a packed sample for true ChannelViT training.

    Unlike ``decode_packed_sample``, this function never pads missing channels
    with zeros and never copies channels. It can filter samples with too few
    channels, then optionally samples a fixed-size subset from the channels
    actually present in the sample, returning both the image tensor and the
    corresponding 0-based channel ids.
    """
    import re

    if max_channels <= 0:
        raise ValueError(f"max_channels must be positive, got {max_channels}")
    if sample_channels is not None and sample_channels <= 0:
        raise ValueError(f"sample_channels must be positive, got {sample_channels}")
    if sample_channels is not None and sample_channels > max_channels:
        raise ValueError(
            f"sample_channels ({sample_channels}) must be <= max_channels ({max_channels})"
        )
    if min_channels <= 0:
        raise ValueError(f"min_channels must be positive, got {min_channels}")
    if min_channels > max_channels:
        raise ValueError(f"min_channels ({min_channels}) must be <= max_channels ({max_channels})")
    if sample_channels is not None and sample_channels < min_channels:
        raise ValueError(
            f"sample_channels ({sample_channels}) must be >= min_channels ({min_channels})"
        )

    ch_key_re = re.compile(r"^ch(\d+)\.tiff?$", re.IGNORECASE)
    channel_bytes: dict[int, bytes] = {}
    for key, value in sample.items():
        m = ch_key_re.match(key)
        if m and isinstance(value, (bytes, bytearray)):
            ch_num = int(m.group(1))
            if 1 <= ch_num <= max_channels:
                channel_bytes[ch_num] = value

    if not channel_bytes:
        logger.debug(
            "packed ChannelViT sample %s skipped: no channels present",
            sample.get("__key__", "?"),
        )
        return None

    available_channels = sorted(channel_bytes.keys())
    if len(available_channels) < min_channels:
        return None
    if sample_channels is not None and len(available_channels) > sample_channels:
        selected_channels = sorted(random.sample(available_channels, sample_channels))
    else:
        selected_channels = available_channels
    decoded: list[Tensor] = []
    channel_ids: list[int] = []
    expected_hw: tuple[int, int] | None = None

    for ch_num in selected_channels:
        tensor = decode_tiff_bytes(channel_bytes[ch_num], target_channels=1)
        if tensor is None:
            logger.debug("packed ChannelViT sample: failed to decode ch%d", ch_num)
            continue
        _, h, w = tensor.shape
        if expected_hw is None:
            expected_hw = (h, w)
        elif expected_hw != (h, w):
            logger.warning(
                "packed ChannelViT sample %s skipped: channel spatial mismatch %s vs %s",
                sample.get("__key__", "?"),
                expected_hw,
                (h, w),
            )
            return None
        decoded.append(tensor[0])
        channel_ids.append(ch_num - 1)

    if not decoded:
        return None

    return {
        "image": torch.stack(decoded, dim=0).to(torch.float32),
        "channel_ids": torch.tensor(channel_ids, dtype=torch.long),
    }


def decode_packed_channelvit_sample_robust(
    sample: dict,
    *,
    max_channels: int,
    sample_channels: Optional[int] = None,
    min_channels: int = 1,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> Optional[dict[str, Tensor]]:
    """Robust-normalization variant of ``decode_packed_channelvit_sample``.

    This keeps the true multi-channel ChannelViT/DualRoute contract (only real
    channels are returned, with their channel ids) while applying the #4
    percentile normalization independently to each selected channel.
    """
    import re

    if max_channels <= 0:
        raise ValueError(f"max_channels must be positive, got {max_channels}")
    if sample_channels is not None and sample_channels <= 0:
        raise ValueError(f"sample_channels must be positive, got {sample_channels}")
    if sample_channels is not None and sample_channels > max_channels:
        raise ValueError(
            f"sample_channels ({sample_channels}) must be <= max_channels ({max_channels})"
        )
    if min_channels <= 0:
        raise ValueError(f"min_channels must be positive, got {min_channels}")
    if min_channels > max_channels:
        raise ValueError(f"min_channels ({min_channels}) must be <= max_channels ({max_channels})")
    if sample_channels is not None and sample_channels < min_channels:
        raise ValueError(
            f"sample_channels ({sample_channels}) must be >= min_channels ({min_channels})"
        )

    ch_key_re = re.compile(r"^ch(\d+)\.tiff?$", re.IGNORECASE)
    channel_bytes: dict[int, bytes] = {}
    for key, value in sample.items():
        m = ch_key_re.match(key)
        if m and isinstance(value, (bytes, bytearray)):
            ch_num = int(m.group(1))
            if 1 <= ch_num <= max_channels:
                channel_bytes[ch_num] = value

    if not channel_bytes:
        logger.debug(
            "packed ChannelViT robust sample %s skipped: no channels present",
            sample.get("__key__", "?"),
        )
        return None

    available_channels = sorted(channel_bytes.keys())
    if len(available_channels) < min_channels:
        return None
    if sample_channels is not None and len(available_channels) > sample_channels:
        selected_channels = sorted(random.sample(available_channels, sample_channels))
    else:
        selected_channels = available_channels

    decoded: list[Tensor] = []
    channel_ids: list[int] = []
    expected_hw: tuple[int, int] | None = None

    for ch_num in selected_channels:
        tensor = decode_tiff_bytes(channel_bytes[ch_num], target_channels=1)
        if tensor is None:
            logger.debug("packed ChannelViT robust sample: failed to decode ch%d", ch_num)
            continue
        _, h, w = tensor.shape
        if expected_hw is None:
            expected_hw = (h, w)
        elif expected_hw != (h, w):
            logger.warning(
                "packed ChannelViT robust sample %s skipped: channel spatial mismatch %s vs %s",
                sample.get("__key__", "?"),
                expected_hw,
                (h, w),
            )
            return None
        decoded.append(_robust_per_channel(tensor[0], p_low, p_high))
        channel_ids.append(ch_num - 1)

    if not decoded:
        return None

    return {
        "image": torch.stack(decoded, dim=0).to(torch.float32),
        "channel_ids": torch.tensor(channel_ids, dtype=torch.long),
    }


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
