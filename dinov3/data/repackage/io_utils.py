"""TIFF / JSON I/O and array normalisation helpers."""

import io
import json
import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("repackage.io")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_tiff_bytes(tiff_bytes: bytes) -> Optional[np.ndarray]:
    """Decode a TIFF byte-string into a numpy array (original dtype)."""
    import tifffile

    try:
        with io.BytesIO(tiff_bytes) as buf:
            return tifffile.imread(buf)
    except Exception as exc:
        logger.warning("TIFF decode failed: %s: %s", type(exc).__name__, exc)
        return None


def read_json_bytes(raw: bytes) -> Optional[Dict[str, Any]]:
    """Decode a JSON byte-string into a Python dict."""
    try:
        return json.loads(raw)
    except Exception as exc:
        logger.warning("JSON decode failed: %s: %s", type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def encode_tiff_uint16(array_2d: np.ndarray) -> bytes:
    """Encode a single-channel 2-D array as a uint16 TIFF byte-string.

    If the input is floating-point it is rescaled to [0, 65535].
    """
    import tifffile

    if np.issubdtype(array_2d.dtype, np.floating):
        lo, hi = float(array_2d.min()), float(array_2d.max())
        if hi > lo:
            scaled = (array_2d - lo) / (hi - lo) * 65535.0
        else:
            scaled = np.zeros_like(array_2d, dtype=np.float64)
        array_2d = np.clip(scaled, 0, 65535).astype(np.uint16)
    elif array_2d.dtype != np.uint16:
        array_2d = array_2d.astype(np.uint16)

    buf = io.BytesIO()
    tifffile.imwrite(buf, array_2d, compression=None)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def ensure_channel_first(image: np.ndarray) -> np.ndarray:
    """Normalise an image to (C, H, W) layout.

    * 2-D (H, W)       → (1, H, W)
    * 3-D channel-last  → transposed to (C, H, W)
    * 3-D channel-first → returned as-is
    """
    if image.ndim == 2:
        return image[np.newaxis, :, :]
    if image.ndim == 3:
        if image.shape[2] < image.shape[0]:
            return np.ascontiguousarray(image.transpose(2, 0, 1))
        return np.ascontiguousarray(image)
    raise ValueError(f"Unsupported image ndim={image.ndim}, shape={image.shape}")
