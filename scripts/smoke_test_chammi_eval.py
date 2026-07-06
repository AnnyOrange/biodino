#!/usr/bin/env python3
"""Smoke test for CHAMMI true-multichannel frozen-eval plumbing.

This validates dataset registry entries, flattened-channel decoding, and the
encoder's tensor resize/crop/normalization path without loading a checkpoint.
"""
from __future__ import annotations

import argparse

from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder, ROBUST_MC_MEAN, ROBUST_MC_STD
from dinov3.eval.bio_frozen_eval.registry import CHAMMI_DATASETS, NATIVE_TEST_SPLIT_DATASETS, build_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    args = parser.parse_args()

    missing_native = sorted(CHAMMI_DATASETS - NATIVE_TEST_SPLIT_DATASETS)
    if missing_native:
        raise AssertionError(f"CHAMMI datasets missing native split registration: {missing_native}")

    enc = Dinov3CkptEncoder.__new__(Dinov3CkptEncoder)
    enc.image_size = args.image_size
    enc.resize_size = args.resize_size
    enc.mc_mean = ROBUST_MC_MEAN
    enc.mc_std = ROBUST_MC_STD

    for name in sorted(CHAMMI_DATASETS):
        train, task = build_dataset(name, "train", max_samples=1, max_per_class=None, benchmark_root=args.benchmark_root)
        test, _ = build_dataset(name, "test", max_samples=1, max_per_class=None, benchmark_root=args.benchmark_root)
        for split_name, dataset in [("train", train), ("test", test)]:
            image, label, path = dataset[0]
            if image.ndim != 3:
                raise AssertionError(f"{name}/{split_name}: expected C,H,W tensor, got {tuple(image.shape)}")
            crop = enc._resize_center_crop_tensor(image)
            norm = enc._normalize_tensor_batch(crop.unsqueeze(0))
            if crop.shape[-2:] != (args.image_size, args.image_size):
                raise AssertionError(f"{name}/{split_name}: bad crop shape {tuple(crop.shape)}")
            print(
                f"[ok] {name}/{split_name} task={task} "
                f"raw={tuple(image.shape)} crop={tuple(crop.shape)} "
                f"norm={tuple(norm.shape)} label={label} path={path}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
