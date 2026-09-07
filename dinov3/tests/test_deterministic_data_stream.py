import random

import numpy as np
import torch

from dinov3.data.loaders import DeterministicDataStream, make_data_loader
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


class _FiniteWorkerRandomizedIterable(torch.utils.data.IterableDataset):
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        for index in range(4):
            yield torch.tensor(
                [
                    worker_id,
                    index,
                    random.random(),
                    float(np.random.random()),
                    float(torch.rand(())),
                ],
                dtype=torch.float64,
            )


def _collect_seeded_worker_batches(add_main_process_noise: bool):
    if add_main_process_noise:
        random.random()
        np.random.random()
        torch.rand(31)
    generator = torch.Generator().manual_seed(101)
    loader = make_data_loader(
        dataset=_FiniteWorkerRandomizedIterable(),
        batch_size=2,
        num_workers=2,
        sampler_type=None,
        drop_last=False,
        pin_memory=False,
        generator=generator,
    )
    return [batch.tolist() for batch in loader]


def test_explicit_generator_isolates_worker_rng_from_model_rng():
    assert _collect_seeded_worker_batches(False) == _collect_seeded_worker_batches(True)
