#!/usr/bin/env python3
"""Run the exact FM14 v4 retrieval/clustering dataset protocols for new encoders."""

from __future__ import annotations

import sys
from pathlib import Path

from dinov3.eval.bio_frozen_eval import run_retrieval_clustering as retrieval
from run_new_fm_v4_linear_20261016 import MODELS, NewEncoder, ROOT


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
        raise SystemExit("usage: run_new_fm_v4_retrieval_exact_20261016.py MODEL [existing retrieval options]")
    model = sys.argv[1]
    path = MODELS[model]
    weight = path / ("teacher_checkpoint.pth" if model == "openpath" else "model.safetensors")
    if not weight.is_file():
        raise FileNotFoundError(weight)
    retrieval.Dinov3CkptEncoder = lambda **kwargs: NewEncoder(model, kwargs["device"], 16)
    output = ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016" / model / "retrieval_exact"
    return retrieval.main([
        "--checkpoint", str(weight), "--train-config", str(path / "config.json"),
        "--model-name", model, "--output-dir", str(output), "--device", "cuda",
        "--batch-size", "16", "--num-workers", "0", "--metric-device", "cpu",
        *sys.argv[2:],
    ])


if __name__ == "__main__":
    raise SystemExit(main())
