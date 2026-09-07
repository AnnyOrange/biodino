# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Utilities for preserving EMA time constants across batch sizes."""

from __future__ import annotations

import numpy as np


def scale_ema_momentum_schedule(
    schedule,
    *,
    effective_batch_size: int,
    reference_batch_size: int | None,
) -> tuple[np.ndarray, float]:
    """Rescale an EMA schedule from updates to processed-sample time.

    If one optimizer update consumes ``k`` times as many samples, an EMA decay
    ``rho`` becomes ``rho**k``.  Applying the power to every schedule value,
    rather than only its initial value, also preserves the time constant while
    a cosine momentum schedule changes during training.
    """

    values = np.asarray(schedule, dtype=np.float64)
    if reference_batch_size is None:
        return values, 1.0
    if reference_batch_size <= 0:
        raise ValueError(f"reference_batch_size must be positive, got {reference_batch_size}")
    if effective_batch_size <= 0:
        raise ValueError(f"effective_batch_size must be positive, got {effective_batch_size}")
    if np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError("EMA momentum values must lie in (0, 1]")

    exponent = float(effective_batch_size) / float(reference_batch_size)
    return np.power(values, exponent), exponent
