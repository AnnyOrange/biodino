#!/usr/bin/env python3
"""Repeat new-FM v4 clustering seeds using one cached feature extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dinov3.eval.bio_frozen_eval import run_retrieval_clustering as retrieval
from run_new_fm_v4_linear_20261016 import MODELS, NewEncoder, ROOT


DATASETS = ("nct-crc-he-100", "nct-crc-he-1k", "crc-val-he-7k",
            "hpa-subcellular", "rxrx1-cross", "lc25000")


def run_dataset(model: str, dataset: str, path: Path, weight: Path, root: Path) -> None:
    output = root / dataset
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "last_result.json"
    seed0 = output / "seed0.json"
    if existing.is_file() and not seed0.is_file():
        payload = json.loads(existing.read_text())
        if payload.get("dataset") != dataset:
            raise ValueError(f"Unexpected existing result in {existing}")
        payload.update(seed=0, multiseed_protocol="v4-clustering-seed-repeat")
        seed0.write_text(json.dumps(payload, indent=2) + "\n")

    summary = root / "summary.csv"
    original_summary = root / "seed0_original_summary.csv"
    if summary.is_file() and not original_summary.is_file():
        summary.replace(original_summary)

    for seed in range(5):
        target = output / f"seed{seed}.json"
        seed_summary = output / f"seed{seed}_summary.csv"
        if seed and target.is_file() and not seed_summary.is_file():
            target.unlink()
        if target.is_file():
            print(f"[skip] {model}/{dataset}/seed{seed}", flush=True)
            continue
        status = retrieval.main([
            "--checkpoint", str(weight), "--train-config", str(path / "config.json"),
            "--benchmark-root", "/mnt/huawei_deepcad/benchmark",
            "--datasets", dataset, "--output-dir", str(root), "--model-name", model,
            "--device", "cuda:0", "--batch-size", "16", "--num-workers", "0",
            "--seed", str(seed), "--metric-device", "cpu",
        ])
        if status:
            raise RuntimeError(f"Retrieval runner failed for {model}/{dataset}/seed{seed}")
        if not summary.is_file():
            raise FileNotFoundError(f"Expected fresh summary after {model}/{dataset}/seed{seed}")
        summary.replace(seed_summary)
        payload = json.loads((root / "last_result.json").read_text())
        if payload.get("dataset") != dataset:
            raise ValueError(f"Unexpected fresh result for {model}/{dataset}/seed{seed}")
        payload.update(seed=seed, multiseed_protocol="v4-clustering-seed-repeat")
        target.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[complete] {model}/{dataset}/seed{seed}", flush=True)
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=DATASETS)
    group.add_argument("--datasets", nargs="+", choices=DATASETS)
    args = parser.parse_args()
    model = args.model
    path = MODELS[model]
    weight = path / ("teacher_checkpoint.pth" if model == "openpath" else "model.safetensors")
    if not weight.is_file():
        raise FileNotFoundError(weight)
    root = ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016" / model / "retrieval_exact"
    encoder = NewEncoder(model, "cuda:0", 16)
    retrieval.Dinov3CkptEncoder = lambda **kwargs: encoder
    for dataset in args.datasets or (args.dataset,):
        run_dataset(model, dataset, path, weight, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
