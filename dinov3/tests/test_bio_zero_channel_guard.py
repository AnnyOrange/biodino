"""Opt-in preservation of zero-padded channels during 20TB bio-safe training."""

import random

import pytest
import torch

from dinov3.data.augmentations import (
    BioSafeIntensityJitter,
    ChannelAgnosticGaussianNoise,
    DataAugmentationDINO,
)


def test_intensity_offset_does_not_invent_a_missing_channel():
    random.seed(7)
    torch.manual_seed(7)
    image = torch.zeros(3, 32, 32)
    image[0] = .5
    transform = BioSafeIntensityJitter(brightness=0, contrast=0, gamma=None,
                                       offset=.03, p=1, preserve_zero_channels=True)
    for _ in range(50):
        result = transform(image)
        assert torch.count_nonzero(result[1:]) == 0
        assert torch.count_nonzero(result[0]) > 0


def test_noise_does_not_invent_a_missing_channel():
    torch.manual_seed(1)
    image = torch.zeros(3, 32, 32)
    image[0] = .5
    guarded = ChannelAgnosticGaussianNoise(p=1, preserve_zero_channels=True)
    unguarded = ChannelAgnosticGaussianNoise(p=1, preserve_zero_channels=False)
    assert torch.count_nonzero(guarded(image)[1:]) == 0
    assert torch.count_nonzero(unguarded(image)[1:]) > 0


def test_bio_safe_training_crops_keep_raw_zero_after_normalization():
    random.seed(8)
    torch.manual_seed(8)
    image = torch.zeros(3, 96, 96)
    image[0] = .5
    mean = (.02922119, .00776353, .04187588)
    std = (.11071338, .03956259, .10757116)
    augment = DataAugmentationDINO((.32, 1.), (.05, .32), 2,
                                   global_crops_size=32, local_crops_size=16,
                                   mean=mean, std=std, augmentation_policy="bio_safe",
                                   preserve_zero_channels=True)
    output = augment(image)
    for crop in output["global_crops"] + output["local_crops"]:
        assert torch.count_nonzero(crop[0]) > 0
        assert torch.allclose(crop[1], torch.full_like(crop[1], -mean[1] / std[1]), atol=1e-5)
        assert torch.allclose(crop[2], torch.full_like(crop[2], -mean[2] / std[2]), atol=1e-5)


@pytest.mark.parametrize("gpu_count,batch", [(4, 16), (4, 32)])
def test_global_batch_invariant(gpu_count, batch):
    assert 1024 % (gpu_count * batch) == 0
