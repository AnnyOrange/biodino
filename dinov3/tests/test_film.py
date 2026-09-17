from collections import Counter

import numpy as np
import pytest
import tifffile
import torch

from dinov3.eval.bio_frozen_eval.film import (
    FILMAgeDataset,
    aggregate_group_features,
    grouped_folds,
    normalize_stack,
    read_stack,
    run_film_fold_probe,
)


def records():
    result = [{"group_id": f"D{age}_sample{sample}", "target": target,
               "sample_id": f"{age}/{sample}", "path": f"{age}/{sample}.tif"}
              for target, age in enumerate((2, 4, 6, 10))
              for sample in range(1, 7 if age == 6 else 8)]
    result.append({**result[14], "sample_id": "6/1-fov2", "path": "6/1-fov2.tif"})
    return result


def manifest():
    rows = records()
    folds = grouped_folds(rows)
    for fold in folds:
        ids = fold["train_indices"]
        fold["inner_folds"] = [
            {"train_indices": [ids[i] for i in f["train_indices"]],
             "val_indices": [ids[i] for i in f["test_indices"]]}
            for f in grouped_folds([rows[i] for i in ids])]
    return {"records": rows, "repetitions": [{"seed": 0, "folds": folds}]}


def test_folds_cover_every_stack_once_and_hold_both_d6_fovs():
    rows = records()
    folds = grouped_folds(rows)
    assert Counter(i for f in folds for i in f["test_indices"]) == Counter(range(28))
    for fold in folds:
        train = {rows[i]["group_id"] for i in fold["train_indices"]}
        test = {rows[i]["group_id"] for i in fold["test_indices"]}
        assert not train & test
        assert {rows[i]["target"] for i in fold["test_indices"]} == set(range(4))
    assert folds == grouped_folds(rows, seed=0)
    assert folds != grouped_folds(rows, seed=1)


def test_inner_folds_never_touch_outer_test():
    m = manifest()
    for fold in m["repetitions"][0]["folds"]:
        test = set(fold["test_indices"])
        for inner in fold["inner_folds"]:
            assert not test & set(inner["train_indices"] + inner["val_indices"])
            assert not set(inner["train_indices"]) & set(inner["val_indices"])


def test_conflicting_group_label_and_too_few_groups_rejected():
    rows = records()
    rows.append({**rows[0], "target": 1})
    with pytest.raises(ValueError, match="Conflicting group"):
        grouped_folds(rows)
    with pytest.raises(ValueError, match="too few groups"):
        grouped_folds(records()[:2])


def test_normalization_shared_across_planes_and_no_test_fitted_statistics():
    a = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    actual = normalize_stack(a)
    lo, hi = np.percentile(a, (1, 99))
    np.testing.assert_allclose(actual.numpy(), np.clip((a - lo) / (hi - lo), 0, 1))
    assert actual.dtype == torch.float32
    assert actual[0].max() < actual[2].min()
    with pytest.raises(ValueError, match="Non|finite"):
        normalize_stack(np.array([np.nan]))
    with pytest.raises(ValueError, match="dynamic range"):
        normalize_stack(np.ones((3, 2, 4)))


def test_existing_mean3_channel_policy_uses_all_spectral_bands_cpu():
    from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder

    encoder = Dinov3CkptEncoder.__new__(Dinov3CkptEncoder)
    stack = torch.arange(126, dtype=torch.float32).reshape(1, 126, 1, 1) / 125
    collapsed = encoder._collapse_to_three_channels_once(stack, torch.ones(1, 126, dtype=torch.bool), "mean3")
    assert collapsed.shape == (1, 3, 1, 1)
    torch.testing.assert_close(collapsed, stack.mean(dim=1, keepdim=True).expand(1, 3, 1, 1))


def test_read_stack_rejects_flattened_or_rgb_proxy(tmp_path):
    path = tmp_path / "bad.tif"
    tifffile.imwrite(path, np.zeros((3, 200, 200), dtype=np.float32), photometric="minisblack")
    with pytest.raises(ValueError, match="expected float32"):
        read_stack(path)


def test_group_feature_aggregation_equal_weights():
    rows = records()
    features = np.arange(28, dtype=float)[:, None]
    x, y, groups = aggregate_group_features(features, rows, list(range(28)))
    assert len(x) == len(y) == len(groups) == 27
    assert x[groups.index("D6_sample1"), 0] == (14 + 27) / 2
    with pytest.raises(ValueError, match="Finite features"):
        aggregate_group_features(features[:-1], rows, list(range(28)))


def test_existing_probe_end_to_end_cpu_on_group_features():
    m = manifest()
    features = np.eye(4, dtype=np.float32)[[r["target"] for r in m["records"]]]
    for fold in range(3):
        result = run_film_fold_probe(features, m, fold=fold)
        assert result["balanced_accuracy"] == 1
        assert result["n_train"] + result["n_test"] == 27
        assert not set(result["train_groups"]) & set(result["evaluation_groups"])
        inner = run_film_fold_probe(features, m, fold=fold, inner_fold=0)
        assert inner["balanced_accuracy"] == 1


def test_dataset_explicit_outer_and_inner_indices_and_invalid_val(tmp_path):
    m = manifest()
    train = FILMAgeDataset(tmp_path, m, split="train")
    test = FILMAgeDataset(tmp_path, m, split="test")
    val = FILMAgeDataset(tmp_path, m, split="val", inner_fold=0)
    assert len(train) + len(test) == 28
    assert len(val) < len(train)
    with pytest.raises(ValueError, match="unavailable"):
        FILMAgeDataset(tmp_path, m, split="val")
