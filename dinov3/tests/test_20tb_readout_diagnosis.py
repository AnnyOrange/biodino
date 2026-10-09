import numpy as np

from scripts.diagnose_20tb_readout_20261009 import recovery_row, split_indices


def test_allen_validation_is_fov_disjoint():
    labels = np.tile(np.arange(3), 30)
    groups = np.repeat(np.arange(30), 3).astype(str)
    fit, val = split_indices(labels, groups, "chammi-allen-task2", 20261009)
    assert not set(groups[fit]) & set(groups[val])
    assert len(fit) + len(val) == len(labels)


def test_recovery_uses_fit_only_and_detects_readout_amplification():
    rng = np.random.default_rng(4)
    current = rng.normal(size=(120, 3)).astype(np.float32)
    anchor = current @ np.diag([2.0, 1.0, 0.5]).astype(np.float32)
    fit, val = np.arange(90), np.arange(90, 120)
    result = recovery_row("synthetic", current, anchor, fit, val, "cpu")
    assert result["normalized_mse"] < 0.01
    assert result["decoder_norm"] > 1.5
    assert result["decoder_condition"] > 1.5
