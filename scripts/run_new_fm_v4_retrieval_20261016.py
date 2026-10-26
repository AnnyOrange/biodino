#!/usr/bin/env python3
"""Run existing frozen retrieval/clustering protocols for 2026 encoders."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/mnt/huawei_deepcad/benchmark_model")
from benchmark_eval.encoders import ModelSpec

import run_external_fm_retrieval_clustering as retrieval
from run_new_fm_v4_linear_20261016 import MODELS, NewEncoder


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
        raise SystemExit("usage: run_new_fm_v4_retrieval_20261016.py MODEL [existing retrieval options]")
    model = sys.argv[1]
    path = MODELS[model]
    weight = path / ("teacher_checkpoint.pth" if model == "openpath" else "model.safetensors")
    if not weight.is_file():
        raise FileNotFoundError(weight)
    retrieval.MODEL_REGISTRY[model] = ModelSpec(model, "2026-frozen-encoder", path)
    retrieval.ExternalEncoderAdapter = NewEncoder
    sys.argv = [sys.argv[0], "--model", model, "--output-dir", str(
        Path(__file__).resolve().parents[1] / "outputs/02_eval_runs/new_fm_v4_matched_20261016" / model / "retrieval"
    ), *sys.argv[2:]]
    return retrieval.main()


if __name__ == "__main__":
    raise SystemExit(main())
