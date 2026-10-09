from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from dinov3.eval.bio_detection.run_coexistence_detection import PairedPatchModel, UnitPatchModel
from dinov3.eval.bio_segmentation.coexistence_dense import _concat


class _FixedPatches(nn.Module):
    def __init__(self, patches):
        super().__init__()
        self.register_buffer('patches', torch.tensor(patches, dtype=torch.float32))

    def forward(self, images):
        return self.patches.expand(images.shape[0], -1, -1)


def test_spatial_concat_patch_wise_equal_energy():
    early = np.array([[[[3, 0]], [[4, 2]]]], dtype=np.float16)
    late = np.array([[[[0, 7]], [[5, 24]]]], dtype=np.float16)
    fused = _concat(early, late).astype(np.float32)
    assert fused.shape == (1, 4, 1, 2)
    np.testing.assert_allclose(np.linalg.norm(fused[:, :2], axis=1), 2**-.5, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(fused[:, 2:], axis=1), 2**-.5, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(fused, axis=1), 1, atol=1e-3)


def test_spatial_concat_rejects_zero_patch():
    early = np.zeros((1, 2, 1, 1), dtype=np.float16)
    with pytest.raises(ValueError, match='Zero-length'):
        _concat(early, np.ones_like(early))


def test_detection_paired_patch_map_equal_energy():
    first = _FixedPatches([[[3, 4], [0, 1]]])
    second = _FixedPatches([[[0, 5], [8, 6]]])
    fused = PairedPatchModel(first, second)(torch.ones((2, 1)))
    assert fused.shape == (2, 2, 4)
    torch.testing.assert_close(fused.norm(dim=-1), torch.ones((2, 2)))
    torch.testing.assert_close(fused[..., :2].norm(dim=-1), torch.full((2, 2), 2**-.5))


def test_detection_single_control_is_unit_patches():
    model = UnitPatchModel(_FixedPatches([[[3, 4], [0, 10]]]))
    output = model(torch.ones((2, 1)))
    assert output.shape == (2, 2, 2)
    torch.testing.assert_close(output.norm(dim=-1), torch.ones((2, 2)))
