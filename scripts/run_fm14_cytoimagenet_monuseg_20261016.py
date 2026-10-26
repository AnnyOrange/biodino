#!/usr/bin/env python3
"""Resource retry for missing FM14 CytoImageNet MoNuSeg 30/7/14 seeds."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BENCH = Path("/mnt/huawei_deepcad/benchmark_model")
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(BENCH)]
os.environ.setdefault("DINOV3_CODE_ROOT", str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/02_eval_runs/fm14_cytoimagenet_monuseg_20261016")
    args = parser.parse_args()
    if args.extract_batch_size not in (1, 2, 4, 8):
        parser.error("Extraction batch must be 1, 2, 4, or 8")
    np.random.seed(0)
    torch.manual_seed(0)

    from benchmark_eval.rules_features import RuleFMFeatures
    extractor = RuleFMFeatures("cytoimagenet", "cuda:0", layers="last")
    from dinov3.eval.bio_segmentation.datasets.monuseg import MoNuSegDataset
    from dinov3.eval.bio_segmentation.linear_probe import DATASET_CONFIGS, run_cached_linear_probe
    from dinov3.eval.bio_frozen_eval.external_fm_protocol import validate_best_head

    spec = importlib.util.spec_from_file_location("fm14_cytoimagenet_dense_cache", BENCH / "run_dense_probe_benchmark.py")
    cache = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cache)
    data_root = Path("/mnt/huawei_deepcad/benchmark/segmentation/monuseg/extracted")
    manifest_path = data_root / "monuseg2018_train30_extra7val_test14_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    samples = {part: [row for row in manifest["samples"] if row["split"] == part]
               for part in ("train", "val", "test")}
    counts = {part: len(rows) for part, rows in samples.items()}
    if counts != {"train": 30, "val": 7, "test": 14}:
        raise ValueError(f"Unexpected MoNuSeg membership: {counts}")
    ids = [row["id"] for rows in samples.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("MoNuSeg split membership overlaps")
    cache.OUT_ROOT = args.output_root
    cache.DenseFeatureExtractor = lambda *unused, **kwargs: extractor
    cache.build_bioseg_dataset = lambda dataset, part, size: MoNuSegDataset(
        [str(data_root / row["image"]) for row in samples[part]],
        [str(data_root / row["xml"]) for row in samples[part]],
        size=(size, size), resize_mode="pad", augment=False, do_normalize=False,
    )
    settings = SimpleNamespace(
        img_size=768, device="cuda:0", feature_canonical=True,
        overwrite_cache=False, extract_batch_size=args.extract_batch_size,
        num_workers=0, max_feature_side=0,
    )
    caches = {part: cache.extract_cache(settings, "cytoimagenet", "monuseg", part)
              for part in ("train", "val", "test")}
    extractor.close()
    del extractor
    torch.cuda.empty_cache()
    for budget in (20, 50):
        for seed in (0, 1, 2):
            output = args.output_root / f"E{budget}" / f"seed{seed}"
            result_path = output / "results.json"
            if result_path.exists():
                print(f"[skip] {result_path}", flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            recipe = dict(
                model="cytoimagenet", dataset="monuseg", split_protocol=manifest["protocol"],
                split_manifest=str(manifest_path), feature_layers="last", image_size=768,
                resize_mode="pad", feature_batch_size=args.extract_batch_size,
                feature_batch_resource_retry=True, probe_batch_size=32, epochs=budget,
                seed=seed, class_weight_mode="none", protocol_id="bio-eval-union-v4",
                status="OBSERVATIONAL",
            )
            (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
            run_cached_linear_probe(
                train_cache=str(caches["train"]), val_cache=str(caches["val"]),
                test_cache=str(caches["test"]), output_dir=str(output),
                **DATASET_CONFIGS["monuseg"], epochs=budget, lr=1e-3,
                batch_size=32, weight_decay=1e-4, dropout=0.1, num_workers=0,
                eval_every=1, seed=seed, class_weight_mode="none",
            )
            validate_best_head(json.loads(result_path.read_text())["_meta"])
            print(f"[complete] cytoimagenet/monuseg/E{budget}/seed{seed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
