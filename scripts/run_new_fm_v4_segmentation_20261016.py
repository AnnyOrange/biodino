#!/usr/bin/env python3
"""Run the existing v4 cached segmentation probe for a new frozen encoder."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
BENCH = Path("/mnt/huawei_deepcad/benchmark_model")
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(BENCH)]
os.environ.setdefault("DINOV3_CODE_ROOT", str(ROOT))

from run_new_fm_v4_detection_20261016 import NewSpatialBackbone  # noqa: E402
from dinov3.eval.bio_frozen_eval.external_fm_protocol import (  # noqa: E402
    PANNUKE_PROTOCOLS, SEGMENTATION, validate_best_head,
)


class NewDenseExtractor:
    patch_size = 16
    n_layers = 1

    def __init__(self, model: str):
        self.backbone = NewSpatialBackbone(model)

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        images = images[:, :3].cuda(non_blocking=True).float().clamp(0, 1)
        if images.shape[1] < 3:
            images = torch.cat((images, images[:, -1:].expand(-1, 3 - images.shape[1], -1, -1)), dim=1)
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        mean = images.new_tensor(self.backbone.input_mean).view(1, 3, 1, 1)
        std = images.new_tensor(self.backbone.input_std).view(1, 3, 1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.backbone.get_intermediate_layers((images - mean) / std, n=1, reshape=True)[0]

    def close(self) -> None:
        pass


def main() -> int:
    from unified_paper_gpu_admission import reserve_for_current_command
    reserve_for_current_command()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("distillpath", "morphem", "genbio_pathfm", "openpath"))
    parser.add_argument("--dataset", required=True, choices=SEGMENTATION)
    parser.add_argument("--split-protocol", choices=PANNUKE_PROTOCOLS)
    parser.add_argument("--budget", type=int, choices=(20, 50), default=20)
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--all-probes", action="store_true")
    parser.add_argument("--extract-test-only", action="store_true", help="Restore only frozen test cache; no downstream training")
    parser.add_argument("--extract-cache-only", action="store_true", help="Restore all frozen split caches; no downstream training")
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016")
    args = parser.parse_args()
    if args.dataset == "pannuke" and args.split_protocol is None:
        parser.error("PanNuke needs an explicit fold rotation")
    if args.dataset != "pannuke" and args.split_protocol is not None:
        parser.error("Only PanNuke accepts --split-protocol")

    lock_dir = args.output_root / args.model / "segmentation" / args.dataset / (args.split_protocol or "default")
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = (lock_dir / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)

    np.random.seed(0)
    torch.manual_seed(0)
    extractor = NewDenseExtractor(args.model)
    from dinov3.eval.bio_segmentation.feature_extractor import _build_dataset
    from dinov3.eval.bio_segmentation.linear_probe import DATASET_CONFIGS, run_cached_linear_probe
    from dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline import _resolve_data_root

    spec = importlib.util.spec_from_file_location("new_fm_dense_cache", BENCH / "run_dense_probe_benchmark.py")
    cache = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cache)

    image_size, resize_mode, _layers, weight_mode = SEGMENTATION[args.dataset]
    split = args.split_protocol or (
        "official-baseline-fold0-nested-v1" if args.dataset == "conic" else
        "monuseg2018-train30-extra7val-test14-v1" if args.dataset == "monuseg" else
        "formal-static-v1"
    )
    root = args.output_root / args.model / "segmentation" / args.dataset / split
    source_recipe_path = ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016" / args.model / "segmentation" / args.dataset / split / "E20/seed0/recipe.json"
    if args.extract_test_only or args.extract_cache_only:
        source_recipe = json.loads(source_recipe_path.read_text())
        assert source_recipe['model'] == args.model and source_recipe['dataset'] == args.dataset
        assert source_recipe['image_size'] == image_size and source_recipe['resize_mode'] == resize_mode
        args.extract_batch_size = int(source_recipe['extract_batch_size'])
        print(f"[restore-original-recipe] extraction batch={args.extract_batch_size} source={source_recipe_path}", flush=True)
    jobs = [(budget, seed) for budget in ((20, 50) if args.all_probes else (args.budget,))
            for seed in (range(5) if args.all_probes else (args.seed,))]
    pending = [(budget, seed) for budget, seed in jobs
               if not (root / f"E{budget}" / f"seed{seed}" / "results.json").exists()]
    if not pending and not (args.extract_test_only or args.extract_cache_only):
        print(f"[skip] {args.model}/{args.dataset}: all requested probes complete", flush=True)
        return 0
    data_root = _resolve_data_root(Path("/mnt/huawei_deepcad/benchmark/segmentation"), args.dataset)
    if args.dataset == "conic":
        from run_fm_dense_rules import _isolated_conic_indices
        from dinov3.eval.bio_segmentation.datasets.conic import CoNICDataset

        indices = _isolated_conic_indices(next(Path(data_root).rglob("patch_info.csv")))
        images_npy = next(Path(data_root).rglob("images.npy"))
        labels_npy = next(Path(data_root).rglob("labels.npy"))
        cache.build_bioseg_dataset = lambda dataset, part, size: CoNICDataset(
            str(images_npy), str(labels_npy), indices=indices[part], size=(size, size),
            resize_mode=resize_mode, augment=False, do_normalize=False,
        )
    elif args.dataset == "monuseg":
        from dinov3.eval.bio_segmentation.datasets.monuseg import MoNuSegDataset

        manifest = json.loads((Path(data_root) / "monuseg2018_train30_extra7val_test14_manifest.json").read_text())
        samples = {part: [row for row in manifest["samples"] if row["split"] == part]
                   for part in ("train", "val", "test")}
        counts = {part: len(rows) for part, rows in samples.items()}
        if counts != {"train": 30, "val": 7, "test": 14}:
            raise ValueError(f"Unexpected MoNuSeg 30/7/14 membership: {counts}")
        ids = [row["id"] for rows in samples.values() for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("MoNuSeg split membership overlaps")
        cache.build_bioseg_dataset = lambda dataset, part, size: MoNuSegDataset(
            [str(Path(data_root) / row["image"]) for row in samples[part]],
            [str(Path(data_root) / row["xml"]) for row in samples[part]],
            size=(size, size), resize_mode=resize_mode, augment=False, do_normalize=False,
        )
    else:
        cache.build_bioseg_dataset = lambda dataset, part, size: _build_dataset(
            dataset, str(data_root), part, size, resize_mode=resize_mode,
            augment=False, do_normalize=False, multichannel=False,
            dataset_split_protocol=split,
        )
    cache.OUT_ROOT = root
    cache.DenseFeatureExtractor = lambda *unused, **kwargs: extractor
    settings = SimpleNamespace(
        img_size=image_size, device="cuda:0", feature_canonical=True,
        overwrite_cache=False, extract_batch_size=args.extract_batch_size,
        num_workers=0, max_feature_side=0,
    )
    feature_grid = 224 // extractor.backbone.patch_size
    caches = {part: cache.extract_cache(settings, args.model, args.dataset, part) for part in (("test",) if args.extract_test_only else ("train", "val", "test"))}
    extractor.close()
    del extractor
    torch.cuda.empty_cache()
    if args.extract_test_only or args.extract_cache_only:
        print(f"[restored-test-cache] {caches['test']}", flush=True)
        return 0
    for budget, seed in pending:
        result_path = root / f"E{budget}" / f"seed{seed}" / "results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        recipe = dict(model=args.model, dataset=args.dataset, split_protocol=split,
                      image_size=image_size, resize_mode=resize_mode, feature_layers="final-patch-map",
                      feature_grid=f"224-input/{feature_grid}x{feature_grid}-common-grid", probe_budget=budget,
                      probe_seed=seed, extract_batch_size=args.extract_batch_size,
                      probe_batch_size=32, class_weight_mode=weight_mode,
                      protocol_id="bio-eval-union-v4", status="OBSERVATIONAL")
        (result_path.parent / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        run_cached_linear_probe(
            train_cache=str(caches["train"]), val_cache=str(caches["val"]),
            test_cache=str(caches["test"]), output_dir=str(result_path.parent),
            **DATASET_CONFIGS[args.dataset], epochs=budget, lr=1e-3,
            batch_size=32, weight_decay=1e-4, dropout=0.1, num_workers=0,
            eval_every=1, seed=seed, class_weight_mode=weight_mode,
        )
        result = json.loads(result_path.read_text())
        validate_best_head(result["_meta"])
        print(f"[complete] {args.model}/{args.dataset}/E{budget}/seed{seed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
