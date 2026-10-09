#!/usr/bin/env python3
"""Measure final-layer CLS-to-each-patch cosine on fixed BloodMNIST images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dinov3.data.transforms import make_classification_eval_transform
from dinov3.eval.bio_frozen_eval.registry import build_dataset
from dinov3.eval.bio_segmentation.model_utils import load_dinov3_backbone


REPO = Path("/mnt/huawei_deepcad/dinov3")
CHECKPOINT_ROOT = REPO / "outputs/02_eval_inputs/hs6_l_5t_every_05m_20260907"
TRAIN_CONFIG = REPO / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907/config.yaml"
REPORT_MANIFEST = REPO / "outputs/00_reports/hs6_l5_5tb_task_peak_curves_20260914/input_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=int, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    parser.add_argument("--benchmark-root", type=Path, default=Path("/mnt/huawei_deepcad/benchmark"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser.parse_args()


def atomic_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def fixed_indices(labels: np.ndarray, samples_per_class: int) -> np.ndarray:
    selected = []
    for label in sorted(np.unique(labels).tolist()):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < samples_per_class:
            raise ValueError(f"Class {label} has only {len(candidates)} samples")
        # Even spacing avoids relying on a random generator and covers the full
        # deterministic official-test ordering.
        positions = np.linspace(0, len(candidates) - 1, samples_per_class, dtype=int)
        selected.extend(candidates[positions].tolist())
    return np.asarray(selected, dtype=int)


def bootstrap_ci(values: np.ndarray, checkpoint: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(20260915 + checkpoint)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(REPORT_MANIFEST.read_text())
    checkpoints = args.checkpoints or [int(value) for value in manifest["checkpoints"]]
    dataset, task = build_dataset("bloodmnist", "test", None, None, benchmark_root=args.benchmark_root)
    if task != "classification":
        raise ValueError(task)
    labels = np.asarray(dataset.labels).astype(int)
    indices = fixed_indices(labels, args.samples_per_class)
    records = [dataset[int(index)] for index in indices]
    images = [record[0] for record in records]
    sample_ids = [record[2] for record in records]
    sample_hash = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
    transform = make_classification_eval_transform(resize_size=439, crop_size=384)
    transformed = [transform(image) for image in images]
    device = torch.device(args.device)
    rows = []
    for checkpoint in checkpoints:
        print(f"[checkpoint] {checkpoint}", flush=True)
        checkpoint_path = args.checkpoint_root / str(checkpoint) / "checkpoint.pth"
        backbone = load_dinov3_backbone(str(checkpoint_path), str(args.train_config), device=device, freeze=True)
        backbone.eval()
        per_image_patch = []
        per_image_mean_patch = []
        with torch.inference_mode():
            for start in range(0, len(transformed), args.batch_size):
                batch = torch.stack(transformed[start : start + args.batch_size]).to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = backbone.forward_features(batch)
                cls = output["x_norm_clstoken"].float()
                patches = output["x_norm_patchtokens"].float()
                patch_cosine = F.cosine_similarity(cls[:, None, :], patches, dim=-1).mean(dim=1)
                mean_patch_cosine = F.cosine_similarity(cls, patches.mean(dim=1), dim=-1)
                per_image_patch.append(patch_cosine.cpu().numpy())
                per_image_mean_patch.append(mean_patch_cosine.cpu().numpy())
                del batch, output, cls, patches
        patch_values = np.concatenate(per_image_patch)
        mean_patch_values = np.concatenate(per_image_mean_patch)
        low, high = bootstrap_ci(patch_values, checkpoint, args.bootstrap_replicates)
        rows.append(
            {
                "checkpoint": checkpoint,
                "image_visits": (checkpoint + 1) * 1024,
                "dataset": "bloodmnist",
                "split": "official-test",
                "n_images": len(indices),
                "samples_per_class": args.samples_per_class,
                "n_classes": len(np.unique(labels)),
                "image_size": 384,
                "resize_size": 439,
                "mean_cls_to_each_patch_cosine": float(patch_values.mean()),
                "image_bootstrap_ci_low": low,
                "image_bootstrap_ci_high": high,
                "mean_cls_to_mean_patch_cosine": float(mean_patch_values.mean()),
                "sample_id_sha256": sample_hash,
            }
        )
        atomic_csv(args.output / "cls_patch_cosine.csv", rows)
        del backbone
        torch.cuda.empty_cache()
    metadata = {
        "status": "VALID_COMPLETE" if len(rows) == len(checkpoints) else "PARTIAL",
        "formula": "mean over images of mean over final-layer patch tokens of cosine(CLS, patch)",
        "dataset": "BloodMNIST official test",
        "selection": f"{args.samples_per_class} evenly spaced samples per class in deterministic dataset order",
        "n_images": len(indices),
        "sample_id_sha256": sample_hash,
        "checkpoint_count": len(rows),
        "checkpoints": checkpoints,
        "image_bootstrap_replicates": args.bootstrap_replicates,
        "image_bootstrap_note": "Image-level uncertainty for the fixed diagnostic set; not checkpoint/probe-seed uncertainty.",
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
