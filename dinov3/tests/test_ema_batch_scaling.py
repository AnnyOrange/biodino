# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import numpy as np
import pytest

from dinov3.train.ema_scaling import scale_ema_momentum_schedule


def test_scales_each_momentum_value_by_batch_ratio():
    schedule = np.asarray([0.992, 0.996, 1.0])
    scaled, exponent = scale_ema_momentum_schedule(
        schedule,
        effective_batch_size=4096,
        reference_batch_size=1024,
    )

    assert exponent == 4.0
    np.testing.assert_allclose(scaled, schedule**4)
    assert scaled[0] == pytest.approx(0.968381956096)
    assert scaled[-1] == 1.0


def test_null_reference_preserves_legacy_schedule():
    schedule = np.asarray([0.992, 0.999, 1.0])
    scaled, exponent = scale_ema_momentum_schedule(
        schedule,
        effective_batch_size=4096,
        reference_batch_size=None,
    )

    np.testing.assert_array_equal(scaled, schedule)
    assert exponent == 1.0


def test_supports_noninteger_batch_ratio():
    schedule = np.asarray([0.992, 0.999, 1.0])
    scaled, exponent = scale_ema_momentum_schedule(
        schedule,
        effective_batch_size=3840,
        reference_batch_size=1024,
    )

    assert exponent == 3.75
    np.testing.assert_allclose(scaled, schedule**3.75)


@pytest.mark.parametrize("reference_batch_size", [0, -1])
def test_rejects_invalid_reference_batch(reference_batch_size):
    with pytest.raises(ValueError, match="reference_batch_size must be positive"):
        scale_ema_momentum_schedule(
            np.asarray([0.992, 1.0]),
            effective_batch_size=4096,
            reference_batch_size=reference_batch_size,
        )
