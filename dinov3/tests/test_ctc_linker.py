import numpy as np

from dinov3.eval.bio_tracking.ctc_linker import (
    TrackLinker,
    consolidate_slices_3d,
    equivalent_diameters,
)


def _square(y: int, x: int, size: int = 4, label: int = 1) -> np.ndarray:
    mask = np.zeros((24, 24), dtype=np.int32)
    mask[y:y + size, x:x + size] = label
    return mask


def test_equivalent_diameters_support_2d_and_3d():
    two_d = np.zeros((8, 8), dtype=np.int32)
    two_d[1:5, 1:5] = 7
    assert np.allclose(equivalent_diameters(two_d), [2 * np.sqrt(16 / np.pi)])

    three_d = np.zeros((4, 4, 4), dtype=np.int32)
    three_d[:2, :2, :2] = 3
    assert np.allclose(equivalent_diameters(three_d), [2 * np.cbrt(3 * 8 / (4 * np.pi))])


def test_linker_continues_nearby_instance_and_starts_distant_one():
    linker = TrackLinker(diameter=8.0)
    first = _square(4, 4)
    second = _square(5, 5)
    second[16:20, 16:20] = 2
    out0 = linker.step(first)
    out1 = linker.step(second)
    assert set(np.unique(out0)) == {0, 1}
    assert out1[6, 6] == 1
    assert out1[17, 17] == 2
    assert linker.track_table().tolist() == [[1, 0, 1, 0], [2, 1, 1, 0]]


def test_linker_detects_division_before_hungarian_continuation():
    linker = TrackLinker(diameter=10.0, division_overlap=0.1)
    parent = _square(6, 6, size=8)
    children = np.zeros_like(parent)
    children[6:14, 6:10] = 3
    children[6:14, 10:14] = 9
    linker.step(parent)
    output = linker.step(children)
    assert set(np.unique(output)) == {0, 2, 3}
    assert linker.track_table().tolist() == [
        [1, 0, 0, 0],
        [2, 1, 1, 1],
        [3, 1, 1, 1],
    ]


def test_linker_rejects_zero_overlap_beyond_one_diameter():
    linker = TrackLinker(diameter=3.0)
    linker.step(_square(1, 1, size=2))
    output = linker.step(_square(18, 18, size=2))
    assert output[18, 18] == 2


def test_consolidate_slices_uses_cross_slice_26_connectivity():
    first = np.zeros((8, 8), dtype=np.int32)
    second = np.zeros_like(first)
    third = np.zeros_like(first)
    first[2:4, 2:4] = 1
    second[3:5, 3:5] = 8
    third[6:8, 6:8] = 2
    volume = consolidate_slices_3d([first, second, third])
    assert volume[0, 2, 2] == volume[1, 3, 3]
    assert volume[2, 6, 6] != volume[1, 3, 3]
