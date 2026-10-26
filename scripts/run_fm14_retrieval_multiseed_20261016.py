#!/usr/bin/env python3
"""Repeat FM14 v4 clustering seeds while caching frozen retrieval features once."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BENCH = Path("/mnt/huawei_deepcad/benchmark_model")
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(BENCH)]
os.environ.setdefault("DINOV3_CODE_ROOT", str(ROOT))

from benchmark_eval.encoders import MODEL_REGISTRY  # noqa: E402
from dinov3.eval.bio_frozen_eval import run_retrieval_clustering as retrieval  # noqa: E402
from run_fm_rules import StandardExternalEncoder  # noqa: E402


DATASETS = ("nct-crc-he-100", "nct-crc-he-1k", "crc-val-he-7k",
            "hpa-subcellular", "rxrx1-cross", "lc25000")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/02_eval_runs/fm14_retrieval_multiseed_20261016")
    args = parser.parse_args()
    output = args.output_root / args.model / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    encoder = StandardExternalEncoder(args.model, 224, 256)
    retrieval.Dinov3CkptEncoder = lambda **kwargs: encoder
    for seed in range(5):
        result_path = output / f"seed{seed}.json"
        if result_path.is_file():
            print(f"[skip] {args.model}/{args.dataset}/seed{seed}", flush=True)
            continue
        status = retrieval.main([
            "--checkpoint", str(MODEL_REGISTRY[args.model].path),
            "--train-config", "/external-not-used", "--benchmark-root", "/mnt/huawei_deepcad/benchmark",
            "--datasets", args.dataset, "--output-dir", str(output), "--model-name", args.model,
            "--device", "cuda:0", "--batch-size", "64", "--num-workers", "2",
            "--seed", str(seed), "--metric-device", "cpu",
        ])
        if status:
            raise RuntimeError(f"Retrieval runner failed for {args.model}/{args.dataset}/seed{seed}")
        payload = json.loads((output / "last_result.json").read_text())
        payload["seed"] = seed
        payload["multiseed_protocol"] = "fm14-v4-clustering-seed-repeat"
        result_path.write_text(json.dumps(payload, indent=2) + "\n")
        (output / "summary.csv").replace(output / f"seed{seed}_summary.csv")
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[complete] {args.model}/{args.dataset}/seed{seed}", flush=True)
    encoder.features.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
