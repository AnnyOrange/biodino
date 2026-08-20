import random

import numpy as np
import torch

from dinov3.data.loaders import DeterministicDataStream
from dinov3.data.wds_pipeline import WdsConfig, _make_sample_shuffle


class _RandomizedIterable:
    def __iter__(self):
        while True:
            yield (
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
            )


def _collect_with_model_noise(model_noise: bool):
    stream = DeterministicDataStream(_RandomizedIterable(), seed=29)
    iterator = iter(stream)
    values = []
    for _ in range(4):
        values.append(next(iterator))
        if model_noise:
            random.random()
            np.random.random()
            torch.rand(17)
    return values


def test_deterministic_data_stream_isolated_from_model_rng():
    assert _collect_with_model_noise(False) == _collect_with_model_noise(True)


class _FakeWebDataset:
    @staticmethod
    def shuffle(buffer_size, **kwargs):
        return buffer_size, kwargs


def test_controlled_wds_shuffle_uses_an_explicit_seed():
    config = WdsConfig(shard_urls=["one.tar"], shuffle_buffer=17, resample_seed=41, deterministic_resampling=True)
    assert _make_sample_shuffle(_FakeWebDataset, config) == (17, {"seed": 1_000_044})
