"""
全局生物医学图像 I/O 防腐层。

规则：
  - 所有生物任务的 Dataset 必须通过此模块读取图像，禁止在各自 Dataset 内写死读取逻辑。
  - 本模块屏蔽格式（png/jpg/tif）、位深（8-bit/16-bit）、通道数的差异，
    统一返回形状为 [C, H, W]、数值范围在 [0, 1]、dtype=float32 的 torch.Tensor。
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# tifffile 是可选依赖，仅对 .tif/.tiff 启用
try:
    import tifffile  # type: ignore

    _TIFFFILE_AVAILABLE = True
except ImportError:
    _TIFFFILE_AVAILABLE = False
    logger.warning(
        "tifffile 未安装，.tif/.tiff 文件将回退到 cv2.IMREAD_UNCHANGED 读取。"
        "如需完整多通道 TIFF 支持，请执行：pip install tifffile"
    )

_TIFF_SUFFIXES = {".tif", ".tiff"}


def _read_raw_array(file_path: str) -> np.ndarray:
    """
    读取图像原始 numpy 数组，保留完整位深与通道数。

    返回：
        np.ndarray，shape 为 (H, W) 或 (H, W, C)，dtype 为原始类型（uint8/uint16/float32 等）。
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in _TIFF_SUFFIXES and _TIFFFILE_AVAILABLE:
        # tifffile 可正确处理 16-bit、多通道荧光图等特殊 TIFF
        arr = tifffile.imread(str(path))
        # tifffile 读出的多通道 TIFF 可能是 (C, H, W)，需转为 (H, W, C)
        if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))
    else:
        # cv2.IMREAD_UNCHANGED 保留 16-bit 及 Alpha 通道
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"cv2 无法读取图像：{file_path}")
        # cv2 读出的彩色图为 BGR，转为 RGB
        if arr.ndim == 3:
            if arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            elif arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)

    return arr


def _normalize_to_float32(arr: np.ndarray) -> np.ndarray:
    """
    将任意位深的数组归一化到 [0, 1] float32。

    策略：
      - float 类型：假设已在 [0, 1]，clip 后直接转换。
      - uint8：除以 255。
      - uint16：除以 65535。
      - 其他无符号整型（如 uint32）：除以该类型最大值。
      - 有符号整型或其他：先转 float32，再 min-max 归一化（避免除零）。
    """
    if np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0

    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0

    if np.issubdtype(arr.dtype, np.unsignedinteger):
        max_val = np.iinfo(arr.dtype).max
        return arr.astype(np.float32) / float(max_val)

    # 有符号整型或其他：min-max 归一化
    arr_f = arr.astype(np.float32)
    lo, hi = arr_f.min(), arr_f.max()
    if hi > lo:
        return (arr_f - lo) / (hi - lo)
    return np.zeros_like(arr_f)


def _ensure_channels(arr: np.ndarray, target_channels: int) -> np.ndarray:
    """
    将 (H, W) 或 (H, W, C) 的数组调整为 (H, W, target_channels)。

    通道处理规则：
      - 灰度 (H, W) → 复制为 target_channels 通道。
      - 单通道 (H, W, 1) → 同上。
      - RGB (H, W, 3) 且 target_channels==3 → 直接使用。
      - 多通道 > target_channels → 取前 target_channels 通道。
      - 通道数 < target_channels（且非 1）→ 循环填充到目标通道数。
    """
    if arr.ndim == 2:
        # 灰度图：扩展到 (H, W, 1)
        arr = arr[:, :, np.newaxis]

    h, w, c = arr.shape

    if c == target_channels:
        return arr

    if c == 1:
        # 单通道复制
        return np.repeat(arr, target_channels, axis=2)

    if c > target_channels:
        return arr[:, :, :target_channels]

    # c < target_channels（且 c > 1）：循环填充
    repeats = (target_channels + c - 1) // c
    arr = np.tile(arr, (1, 1, repeats))[:, :, :target_channels]
    return arr


def read_bio_image(
    file_path: str,
    target_channels: int = 3,
    normalize: bool = True,
) -> torch.Tensor:
    """
    读取生物医学图像，返回标准化后的 Tensor。

    Args:
        file_path:       图像文件路径，支持 .png、.jpg、.tif、.tiff 等格式。
        target_channels: 目标通道数，默认为 3（RGB）以适配 ViT 模型。
        normalize:       是否将像素值归一化到 [0, 1]。默认 True。
                         设为 False 时返回原始 float32 数组对应的 Tensor。

    Returns:
        torch.Tensor，shape 为 [target_channels, H, W]，dtype=float32，值域 [0, 1]。
    """
    arr = _read_raw_array(file_path)

    if normalize:
        arr = _normalize_to_float32(arr)
    else:
        arr = arr.astype(np.float32)

    arr = _ensure_channels(arr, target_channels)  # (H, W, C)

    # (H, W, C) → (C, H, W)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor  # [C, H, W], float32, [0, 1]


def read_bio_image_as_numpy(
    file_path: str,
    target_channels: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """
    与 read_bio_image 相同，但返回 (H, W, C) 的 numpy 数组。

    ``normalize=True``：float32，值域 ``[0, 1]``。
    ``normalize=False``：保留 ``_read_raw_array`` 的位深（如 uint8 / uint16），
    便于调用方再用 ``_normalize_to_float32`` 做统一的 dtype→[0,1] 映射。
    若磁盘上已是浮点类型，则转为 float32 但不改数值范围。
    """
    arr = _read_raw_array(file_path)

    if normalize:
        arr = _normalize_to_float32(arr)
    else:
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32, copy=False)

    arr = _ensure_channels(arr, target_channels)  # (H, W, C)
    return arr
