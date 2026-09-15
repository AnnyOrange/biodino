import numpy as np
import tifffile

from dinov3.eval.bio_tracking.ctc_2d import (
    CTC2DTrainDataset,
    normalized_image_tensor,
    percentile_scale,
)


def test_percentile_scale_is_finite_for_constant_image():
    scaled = percentile_scale(np.full((8, 8), 17, dtype=np.uint16))
    assert scaled.dtype == np.float32
    assert np.count_nonzero(scaled) == 0
    assert np.isfinite(normalized_image_tensor(scaled).numpy()).all()


def test_ctc_crop_is_deterministic_and_contains_foreground(tmp_path):
    image = np.arange(320 * 360, dtype=np.float32).reshape(320, 360)
    mask = np.zeros((320, 360), dtype=np.uint16)
    mask[300:305, 340:345] = 9
    image_path, mask_path = tmp_path / "t000.tif", tmp_path / "man_seg000.tif"
    tifffile.imwrite(image_path, image)
    tifffile.imwrite(mask_path, mask)
    dataset = CTC2DTrainDataset([{"image": str(image_path), "mask": str(mask_path)}], seed=4)

    first = dataset[0]
    second = dataset[0]
    assert all(np.array_equal(a.numpy(), b.numpy()) for a, b in zip(first, second))
    assert first[0].shape == (3, 256, 256)
    assert first[1].shape == (256, 256)
    assert int((first[1] > 0).sum()) > 0

    dataset.set_epoch(1)
    third = dataset[0]
    assert not np.array_equal(first[0].numpy(), third[0].numpy())
