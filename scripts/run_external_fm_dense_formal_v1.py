#!/usr/bin/env python3
"""Evaluate one external FM on formal-v1 CoNIC or PanNuke dense probing.

This adapter deliberately refuses legacy split and hyperparameter choices.  It
uses the existing external-encoder implementations but replaces their dataset
builder with the approved source/fold-disjoint DINOv3 loader.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
BENCHMARK_MODEL_ROOT = Path("/mnt/huawei_deepcad/benchmark_model")
EXTERNAL_SCRIPT = BENCHMARK_MODEL_ROOT / "run_dense_probe_benchmark.py"
CONIC_PROTOCOL = "official-baseline-fold0-nested-v1"
PANNUKE_PROTOCOLS = (
    "pannuke-fold1-train-fold2-val-fold3-test",
    "pannuke-fold2-train-fold1-val-fold3-test",
    "pannuke-fold3-train-fold2-val-fold1-test",
)
MODELS = (
    "dinov2", "mae", "siglip2", "bioclip", "cytoself", "jump_cp",
    "cytoimagenet", "pe", "uni", "conch", "phikon2", "virchow2",
    "gigapath", "hoptimus0",
)


def _load_dense_module():
    sys.path[:0] = [
        str(REPO),
        str(BENCHMARK_MODEL_ROOT),
        str(BENCHMARK_MODEL_ROOT / "_vendor"),
        str(BENCHMARK_MODEL_ROOT / "_vendor" / "external_gapfill_py311"),
    ]
    spec = importlib.util.spec_from_file_location("formal_v1_external_dense", EXTERNAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {EXTERNAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _interpolate_positional_embedding(pos: torch.Tensor, token_count: int) -> torch.Tensor:
    """Resize a ViT absolute spatial position grid while preserving CLS tokens."""
    if pos.ndim == 2:
        pos = pos.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if pos.shape[1] == token_count:
        return pos.squeeze(0) if squeeze else pos
    old_spatial = int(round((pos.shape[1] - 1) ** 0.5))
    new_spatial = int(round((token_count - 1) ** 0.5))
    if old_spatial * old_spatial != pos.shape[1] - 1 or new_spatial * new_spatial != token_count - 1:
        raise ValueError(
            f"Cannot interpolate positional embedding {tuple(pos.shape)} to {token_count} tokens"
        )
    prefix, spatial = pos[:, :1], pos[:, 1:]
    spatial = spatial.reshape(1, old_spatial, old_spatial, pos.shape[-1]).permute(0, 3, 1, 2)
    spatial = F.interpolate(spatial.float(), (new_spatial, new_spatial), mode="bicubic", align_corners=False)
    spatial = spatial.to(pos.dtype).permute(0, 2, 3, 1).reshape(1, new_spatial * new_spatial, pos.shape[-1])
    resized = torch.cat([prefix, spatial], dim=1)
    return resized.squeeze(0) if squeeze else resized


def _install_formal_adapters(dense, dataset: str, split_protocol: str) -> None:
    from dinov3.eval.bio_segmentation.feature_extractor import _build_dataset

    data_root = Path("/mnt/huawei_deepcad/benchmark/segmentation")
    root_by_dataset = {
        "conic": data_root / "conic" / "extracted",
        "pannuke": data_root / "pannuke" / "extracted",
    }

    def formal_dataset_builder(dataset_name: str, split: str, img_size: int):
        if dataset_name != dataset:
            raise ValueError(f"This invocation is locked to dataset={dataset}")
        return _build_dataset(
            dataset_name,
            str(root_by_dataset[dataset_name]),
            split,
            img_size,
            resize_mode="stretch",
            augment=False,
            do_normalize=False,
            dataset_split_protocol=split_protocol,
        )

    dense.build_bioseg_dataset = formal_dataset_builder
    dense.DATASET_IMG_SIZE.update({"conic": 256, "pannuke": 256})

    parent = dense.DenseFeatureExtractor

    class FormalDenseFeatureExtractor(parent):
        @torch.inference_mode()
        def __call__(self, imgs_01: torch.Tensor) -> torch.Tensor:
            # Fixed-grid OpenCLIP towers need explicit positional interpolation
            # when the approved 256px input differs from their native grid.
            if self.spec.kind == "open_clip":
                x = self._norm(self._resize(imgs_01).to(self.device))
                visual = self.model.visual
                y = visual.conv1(x).reshape(x.shape[0], visual.conv1.out_channels, -1).permute(0, 2, 1)
                cls = visual.class_embedding.to(y.dtype).expand(y.shape[0], 1, -1)
                y = torch.cat([cls, y], dim=1)
                pos = _interpolate_positional_embedding(visual.positional_embedding, y.shape[1])
                y = visual.ln_pre(y + pos.to(y.dtype))
                y = visual.transformer(y.permute(1, 0, 2)).permute(1, 0, 2)
                return self._tokens_to_map(visual.ln_post(y)[:, 1:]).float()

            # The old adapter inferred ChannelViT's spatial grid from 224px.
            # Formal-v1 uses the actual 256px tensor instead.
            if self.spec.kind == "channelvit":
                x = self._norm(self._resize(imgs_01).to(self.device))
                xx = torch.zeros(
                    (x.shape[0], self.in_chans, x.shape[2], x.shape[3]),
                    dtype=x.dtype,
                    device=x.device,
                )
                xx[:, :3] = x
                channels = torch.arange(self.in_chans, device=x.device).repeat(x.shape[0], 1)
                outputs = self.model.get_intermediate_layers(xx, extra_tokens={"channels": channels}, n=1)
                tokens = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                spatial = (x.shape[-2] // self.patch_size) * (x.shape[-1] // self.patch_size)
                if tokens.shape[1] % spatial == 0:
                    token_channels = tokens.shape[1] // spatial
                    tokens = tokens.reshape(tokens.shape[0], token_channels, spatial, tokens.shape[2]).mean(1)
                return self._tokens_to_map(tokens).float()
            return super().__call__(imgs_01)

    dense.DenseFeatureExtractor = FormalDenseFeatureExtractor

    def formal_run_linear_probe(run_args, model_name: str, dataset_name: str, caches):
        out_dir = dense.OUT_ROOT / "linear_probe" / dataset_name / model_name
        result_path = out_dir / "results.json"
        if result_path.exists() and not run_args.overwrite_probe:
            return result_path
        cfg = dense.DATASET_CONFIGS[dataset_name]
        class_weight_mode = "sqrt_inverse" if dataset_name == "conic" else "none"
        script = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
from dinov3.eval.bio_segmentation.linear_probe import run_cached_linear_probe
run_cached_linear_probe(
    train_cache={str(caches['train'])!r},
    val_cache={str(caches['val'])!r},
    test_cache={str(caches['test'])!r},
    output_dir={str(out_dir)!r},
    num_classes={cfg['num_classes']},
    class_names={cfg['class_names']!r},
    epochs=50,
    lr=1e-3,
    batch_size=32,
    weight_decay=1e-4,
    dropout=0.1,
    num_workers=2,
    eval_every=50,
    seed=0,
    class_weight_mode={class_weight_mode!r},
    class_weight_beta=0.999,
)
"""
        out_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO) + os.pathsep + environment.get("PYTHONPATH", "")
        log_path = out_dir / "linear_probe.log"
        metric_python = os.environ["DENSE_PROBE_METRIC_PYTHON"]
        with log_path.open("w") as log:
            process = subprocess.run(
                [metric_python, "-c", script],
                cwd=REPO,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if process.returncode != 0:
            raise RuntimeError(f"linear probe failed; see {log_path}")
        return result_path

    dense.run_linear_probe = formal_run_linear_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--dataset", required=True, choices=("conic", "pannuke"))
    parser.add_argument("--split-protocol", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--metric-python", default="/home/inspur/anaconda3/envs/dinov3/bin/python")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    allowed = (CONIC_PROTOCOL,) if args.dataset == "conic" else PANNUKE_PROTOCOLS
    if args.split_protocol not in allowed:
        parser.error(f"split protocol must be one of {allowed} for {args.dataset}")

    output_root = Path(args.output_root).resolve() / args.dataset / args.split_protocol / args.model
    manifest = {
        "status": "PLANNED" if args.dry_run else "RUNNING",
        "protocol_id": "bio-eval-formal-v1",
        "model": args.model,
        "dataset": args.dataset,
        "split_protocol": args.split_protocol,
        "resolution": 256,
        "resize": "stretch",
        "feature_batch_size": 32,
        "feature_layers": "external-final-dense-map",
        "probe_batch_size": 32,
        "probe_epochs": 50,
        "probe_eval_every": 50,
        "class_weight": "sqrt_inverse" if args.dataset == "conic" else "none",
        "seed": 0,
        "challenge_test": False if args.dataset == "conic" else None,
        "external_script": str(EXTERNAL_SCRIPT),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "formal_v1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    os.environ["DENSE_PROBE_METRIC_PYTHON"] = args.metric_python
    dense = _load_dense_module()
    _install_formal_adapters(dense, args.dataset, args.split_protocol)
    dense.OUT_ROOT = output_root
    run_args = SimpleNamespace(
        img_size=256,
        overwrite_cache=False,
        feature_canonical=True,
        device="cuda",
        extract_batch_size=32,
        num_workers=2,
        max_feature_side=32,
        overwrite_probe=False,
        epochs=50,
        lr=1e-3,
        probe_batch_size=32,
        probe_num_workers=2,
        weight_decay=1e-4,
        dropout=0.1,
        eval_every=50,
        train_samples=None,
        train_fraction=None,
        seed=0,
    )
    try:
        caches = {
            split: dense.extract_cache(run_args, args.model, args.dataset, split)
            for split in ("train", "val", "test")
        }
        result_path = dense.run_linear_probe(run_args, args.model, args.dataset, caches)
        manifest.update({"status": "VALIDATION_PENDING", "result_path": str(result_path)})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[done] {args.model}/{args.dataset}/{args.split_protocol}: {result_path}", flush=True)
        return 0
    except Exception as error:
        manifest.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}"})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
