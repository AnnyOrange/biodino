#!/usr/bin/env python3
"""Smoke test for the ADDITIVE multichannel eval pathway (keeps RGB path).

Checks:
  1. TissueNet loader: multichannel=True returns true [2,H,W]; default returns [3,H,W] (regression).
  2. dual-route stem routes a genuine 2-channel (low-corr) batch to the POOL path.
  3. feature_extractor._backbone_spatial_features:
       - multichannel=True (dualroute) -> features [B,D,h,w], finite, pool-routed.
       - multichannel=False -> RGB path features [B,D,h,w], finite (unchanged behaviour).
Run: /home/lxy/miniconda3/envs/dinov3/bin/python scripts/smoke_test_mc_eval.py
"""
import sys
import tempfile
import numpy as np
import torch

from dinov3.models.vision_transformer import DinoVisionTransformer
from dinov3.eval.bio_segmentation.feature_extractor import (
    _backbone_spatial_features, _build_channel_metadata,
)
from dinov3.eval.bio_segmentation.datasets.tissuenet import TissueNetDataset

IMG, P, D = 32, 16, 32
FAILED = []


def check(name, cond):
    print(("  ok  " if cond else "FAIL  ") + name)
    if not cond:
        FAILED.append(name)


def build_model():
    m = DinoVisionTransformer(img_size=IMG, patch_size=P, embed_dim=D, depth=2,
                              num_heads=2, n_storage_tokens=4, stem_type="dualroute")
    m.init_weights()
    m.eval()
    return m


def test_tissuenet_loader():
    print("[test_tissuenet_loader]")
    N, H, W = 3, 40, 40
    rng = np.random.RandomState(0)
    X = rng.rand(N, H, W, 2).astype(np.float32)            # 2 fluorescence channels
    y = rng.randint(0, 5, size=(N, H, W, 2)).astype(np.int32)
    with tempfile.NamedTemporaryFile(suffix="_train.npz", delete=False) as f:
        np.savez(f.name, X=X, y=y); path = f.name

    ds_rgb = TissueNetDataset(path, size=(IMG, IMG), do_normalize=True)
    ds_mc = TissueNetDataset(path, size=(IMG, IMG), do_normalize=True, multichannel=True)
    img_rgb = ds_rgb[0][0]
    img_mc = ds_mc[0][0]
    check("RGB mode -> [3,H,W]", tuple(img_rgb.shape) == (3, IMG, IMG))
    check("multichannel mode -> [2,H,W]", tuple(img_mc.shape) == (2, IMG, IMG))
    check("mc channels are NOT a duplicate (ch0 != ch1)", not torch.allclose(img_mc[0], img_mc[1]))
    check("mc finite", torch.isfinite(img_mc).all().item())


def test_routing_and_features(model):
    print("[test_routing_and_features]")
    torch.manual_seed(0)
    # genuine 2-channel batch: nuclear vs membrane = independent -> low corr -> pool route
    x2 = torch.randn(2, 2, IMG, IMG)
    valid2 = torch.ones(2, 2, dtype=torch.bool)
    is_rgb, _ = model.patch_embed._route_and_rgb_input(x2, valid2)
    check("2ch low-corr routes to POOL (is_rgb all False)", (~is_rgb).all().item())

    cid, cmask = _build_channel_metadata(x2)
    check("channel_ids shape (C,) long", tuple(cid.shape) == (2,) and cid.dtype == torch.long)
    check("channel_valid_mask shape (B,C) bool", tuple(cmask.shape) == (2, 2) and cmask.dtype == torch.bool)

    feats = _backbone_spatial_features(model, x2, n_layers=1, multichannel=True)
    f = torch.cat(feats, dim=1)
    hp = IMG // P
    check("mc feats shape [B,D,h,w]", tuple(f.shape) == (2, D, hp, hp))
    check("mc feats finite", torch.isfinite(f).all().item())

    # RGB path unchanged: 3ch correlated input still works via the default branch
    x3 = torch.randn(2, 3, IMG, IMG)
    feats_rgb = _backbone_spatial_features(model, x3, n_layers=1, multichannel=False)
    fr = torch.cat(feats_rgb, dim=1)
    check("RGB feats shape [B,D,h,w]", tuple(fr.shape) == (2, D, hp, hp))
    check("RGB feats finite", torch.isfinite(fr).all().item())


def main():
    model = build_model()
    test_tissuenet_loader()
    test_routing_and_features(model)
    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}"); sys.exit(1)
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
