import numpy as np
import pytest

from dinov3.data.domain_reweighting import DoReMiDomainWeights
from dinov3.data.wds_pipeline import WeightedIterableDataset


class _ConstantDataset:
    def __init__(self, value):
        self.value = value

    def __iter__(self):
        while True:
            yield ({"value": self.value}, ())


def test_doremi_upweights_positive_excess_and_tracks_average():
    optimizer = DoReMiDomainWeights(["old", "new"], step_size=1.0, smoothing=0.0)
    first = optimizer.update([2.0, 1.0], [1.0, 1.0])
    second = optimizer.update([1.0, 1.0], [1.0, 1.0])

    assert first.weights[0] > first.weights[1]
    assert second.weights == pytest.approx(first.weights)
    assert optimizer.averaged_weights == pytest.approx(np.asarray(first.weights))


def test_doremi_clips_negative_excess_and_smooths():
    optimizer = DoReMiDomainWeights(["a", "b"], step_size=10.0, smoothing=0.2)
    result = optimizer.update([0.0, 3.0], [4.0, 0.0])

    assert result.excess_losses == pytest.approx((0.0, 3.0))
    assert result.weights[0] >= 0.1
    assert sum(result.weights) == pytest.approx(1.0)


def test_weighted_dataset_emits_domain_homogeneous_tagged_blocks():
    dataset = WeightedIterableDataset(
        [_ConstantDataset("a"), _ConstantDataset("b")],
        [1.0, 1.0],
        seed=7,
        names=["old", "new"],
        domain_block_size=4,
    )
    iterator = iter(dataset)
    samples = [next(iterator)[0] for _ in range(8)]

    for start in (0, 4):
        block = samples[start : start + 4]
        assert len({sample["mixture_domain_index"] for sample in block}) == 1
        assert len({sample["mixture_domain_name"] for sample in block}) == 1
        assert all(sample["value"] in {"a", "b"} for sample in block)


def test_weighted_dataset_rejects_invalid_block_size():
    with pytest.raises(ValueError, match="domain_block_size"):
        WeightedIterableDataset([_ConstantDataset("a")], [1.0], domain_block_size=0)
