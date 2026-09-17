import numpy as np
import pytest
from PIL import Image

from dinov3.eval.bio_frozen_eval.vgg_count import FOLDER, VGGCountDataset, dot_count, published_draws, ridge_development_sweep, ridge_draw_scores


def test_official_red_channel_count_not_rgb_mean_or_components(tmp_path):
    pixels = np.zeros((256, 256, 3), dtype=np.uint8)
    pixels[2, 2:4, 0] = 255
    path = tmp_path / "dots.png"
    Image.fromarray(pixels).save(path)
    assert dot_count(path) == 2
    pixels[2, 2, 1] = 255
    Image.fromarray(pixels).save(path)
    with pytest.raises(ValueError, match="binary red-channel"):
        dot_count(path)


def test_published_development_draws_preserve_test_and_unused():
    draws = published_draws(0)
    assert draws == published_draws(0)
    assert draws != published_draws(1)
    for draw in draws:
        assert len(draw["train"]) == len(draw["val"]) == 32
        assert len(draw["unused_development"]) == 36
        assert draw["test"] == [f"{i:03d}" for i in range(101, 201)]
        assert not set(draw["train"]) & set(draw["val"])
        assert set(draw["train"] + draw["val"] + draw["unused_development"]) == {f"{i:03d}" for i in range(1, 101)}
        order = np.random.default_rng(draw["repetition"]).permutation(100) + 1
        assert draw["train"] == [f"{i:03d}" for i in order[:32]]


def test_loader_full_development_bank_preserves_rgb_fov(tmp_path):
    root = tmp_path / FOLDER / "extracted"
    root.mkdir(parents=True)
    Image.fromarray(np.full((256, 256, 3), [1, 2, 3], dtype=np.uint8)).save(root / "001cell.png")
    manifest = {"records": [{"sample_id": "001", "path": "001cell.png", "target": 42,
                              "source_split": "development"}], "draws": published_draws(), "repetitions": 5}
    dataset = VGGCountDataset(tmp_path, manifest, "development")
    image, target, sample_id = dataset[0]
    assert len(dataset) == 1 and image.size == (256, 256)
    assert image.mode == "RGB" and list(np.asarray(image)[0, 0]) == [1, 2, 3]
    assert target == 42. and sample_id == "001"
    assert len(VGGCountDataset(tmp_path, manifest, "test")) == 0


def test_existing_ridge_sweep_never_uses_test_targets():
    features = np.arange(200, dtype=float).reshape(-1, 1)
    manifest = {"records": [{"sample_id": f"{i+1:03d}", "target": 2*i+3} for i in range(200)], "draws": published_draws()}
    first = ridge_development_sweep(features, manifest, alphas=(.1, 1., 10.))
    for row in manifest["records"][100:]:
        row["target"] = -1e6
    assert ridge_development_sweep(features, manifest, alphas=(.1, 1., 10.)) == first
    assert first["selected_alpha"] == .1
    assert len(first["sweep"]) == 3
    assert len(first["sweep"][0]["draw_scores"]) == 5
    assert not first["test_accessed"]


def test_ridge_rejects_incomplete_features_and_invalid_split():
    with pytest.raises(ValueError, match="exactly200"):
        ridge_draw_scores(np.zeros((199, 2)), {}, 1.)
    with pytest.raises(ValueError, match="validation or frozen test"):
        ridge_draw_scores(np.zeros((200, 2)), {}, 1., split="train")
