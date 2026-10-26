#!/usr/bin/env python3
"""Repeat FM14 v4 detection proxies with independent probe seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BENCH = Path("/mnt/huawei_deepcad/benchmark_model")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(BENCH), str(BENCH / "_vendor")]
os.environ.setdefault("DINOV3_CODE_ROOT", str(ROOT))

from benchmark_eval.encoders import MODEL_REGISTRY  # noqa: E402
from dinov3.eval.bio_detection import center_probe  # noqa: E402
from run_fm_companion_rules import FMSpatialBackbone  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--dataset", required=True, choices=("bbbc038", "conic", "livecell"))
    parser.add_argument("--seed", required=True, type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, default=Path("/mnt/huawei_deepcad/benchmark"))
    args = parser.parse_args()

    output = args.output_root / args.model / args.dataset / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "component_result.json"
    if result_path.exists():
        print(f"[skip] {result_path}", flush=True)
        return 0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    center_probe.load_backbone = lambda **kwargs: FMSpatialBackbone(args.model, "imagenet")
    result = center_probe.run_bio_detection_eval(
        arch="vit_base",
        weights="",
        checkpoint=str(MODEL_REGISTRY[args.model].path),
        train_config=None,
        benchmark_root=str(args.benchmark_root),
        output_dir=str(output),
        dataset=args.dataset,
        batch_size=8,
        num_workers=2,
        image_size=224,
        epochs=5,
        lr=1e-3,
        autocast_dtype=torch.bfloat16,
        channel_policy="auto",
        max_samples_per_split=0,
        seed=args.seed,
        conic_split_protocol="official-baseline-fold0-nested-v1",
    )
    result.update(
        status="OBSERVATIONAL",
        aggregate_allowed=False,
        union_status="V4_DETECTION_PROXY_MULTISEED",
        model=args.model,
        probe_seed=args.seed,
        protocol_id="bio-eval-union-v4",
    )
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(result_path)
    print(json.dumps({"model": args.model, "dataset": args.dataset, "seed": args.seed, "output": str(result_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
