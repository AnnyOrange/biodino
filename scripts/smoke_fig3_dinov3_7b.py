#!/usr/bin/env python3
"""Check whether original DINOv3-7B/16 can run one BBBC048 image on a 24 GB GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.extract_fig3_panel_d_baseline_features import (
    DEFAULT_7B_WEIGHTS,
    make_encoder,
)
from dinov3.eval.bio_frozen_eval.registry import build_dataset

OUT_DIR = REPO_ROOT / "outputs/04_figures/fig3_representation_20260812"


def main() -> None:
    device = "cuda:0"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 7B smoke test.")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dataset, _ = build_dataset(
        "bbbc048-cellcycle", "train", None, None, benchmark_root="/mnt/huawei_deepcad/benchmark"
    )
    print(f"[panel-d-7b] loading {DEFAULT_7B_WEIGHTS}", flush=True)
    encoder = make_encoder("dinov3_official_vit7b16", device, Path(), DEFAULT_7B_WEIGHTS)
    image, label, path = dataset[0]
    print(f"[panel-d-7b] encoding {path}", flush=True)
    features = encoder.encode_images([image])
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    payload = {
        "status": "passed",
        "model": "original DINOv3-7B/16 LVD",
        "weights": str(DEFAULT_7B_WEIGHTS),
        "sample_path": str(path),
        "sample_label": int(label),
        "feature_shape": [int(x) for x in features.shape],
        "feature_dtype": str(features.dtype),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "panel_d_dinov3_7b_smoke.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
