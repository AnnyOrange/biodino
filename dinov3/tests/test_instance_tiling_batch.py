import numpy as np
import pytest
import torch

from dinov3.eval.bio_segmentation.instance_seg.tiling import sliding_window_predict


class _PointwiseModel(torch.nn.Module):
    def forward(self, image):
        base = image[:, :1]
        return {
            "np": torch.cat((base, -base), dim=1),
            "hv": torch.cat((2.0 * base, 3.0 * base), dim=1),
            "tp": None,
        }


@pytest.mark.parametrize("blend_mode", ["uniform", "gaussian"])
def test_tile_batching_matches_single_tile_inference(blend_mode):
    generator = torch.Generator().manual_seed(7)
    image = torch.rand((3, 301, 355), generator=generator)
    model = _PointwiseModel()
    single = sliding_window_predict(
        model, image, crop_size=128, stride=91, patch_size=16,
        blend_mode=blend_mode, tile_batch_size=1,
    )
    batched = sliding_window_predict(
        model, image, crop_size=128, stride=91, patch_size=16,
        blend_mode=blend_mode, tile_batch_size=8,
    )
    np.testing.assert_allclose(batched["np"], single["np"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(batched["hv"], single["hv"], rtol=1e-6, atol=1e-6)


def test_tile_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        sliding_window_predict(_PointwiseModel(), torch.zeros(3, 32, 32), tile_batch_size=0)
