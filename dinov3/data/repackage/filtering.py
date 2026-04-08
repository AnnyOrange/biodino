"""Variance-based background patch filtering.

Discards crops whose reference-channel pixel variance falls below a
configurable threshold, removing pure-black, pure-white, or
near-structureless tiles.
"""

import logging
from typing import Literal, Tuple

import numpy as np

logger = logging.getLogger("repackage.filter")


def passes_variance_filter(
    patch: np.ndarray,
    reference_channel: int,
    variance_threshold: float,
    missing_ref_policy: Literal["fallback_first_available", "skip_sample"],
) -> Tuple[bool, float]:
    """Check whether *patch* passes the variance gate.

    Args:
        patch: (C, H, W) array — full multi-channel crop.
        reference_channel: 1-indexed channel used for the variance test.
        variance_threshold: Minimum acceptable pixel variance.
        missing_ref_policy: Action when reference channel is absent.

    Returns:
        ``(passed, variance_value)`` where *variance_value* is the
        computed variance (or -1.0 when the sample is skipped).
    """
    n_channels = patch.shape[0]
    ref_idx = reference_channel - 1

    if ref_idx < 0 or ref_idx >= n_channels:
        if missing_ref_policy == "skip_sample":
            logger.debug(
                "Reference ch%d absent (%d channels) — skipping",
                reference_channel,
                n_channels,
            )
            return False, -1.0
        ref_idx = 0
        logger.debug(
            "Reference ch%d absent — falling back to ch1 (index 0)",
            reference_channel,
        )

    ref_plane = patch[ref_idx].astype(np.float64)
    var = float(ref_plane.var())

    if var < variance_threshold:
        logger.debug("Rejected: variance %.4f < %.4f", var, variance_threshold)
        return False, var

    return True, var
