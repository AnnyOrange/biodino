#!/usr/bin/env python3
"""One real-volume smoke test for the native CTC 3-D prediction path."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "outputs/02_eval_runtime/imagecodecs_py311_wheel_2026.3.6", ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dinov3.eval.bio_segmentation.instance_seg.model import build_dino_hovernet  # noqa: E402
from dinov3.eval.bio_tracking.ctc_2d import atomic_json, sha256  # noqa: E402
from dinov3.eval.bio_tracking.ctc_linker import TrackLinker  # noqa: E402
from run_ctc_native_full_hs6 import _infer_frame  # noqa: E402


CHECKPOINT = ROOT / (
    "outputs/01_training_runs/"
    "HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907/"
    "eval/training_21959/teacher_checkpoint.pth"
)
CONFIG = ROOT / (
    "outputs/01_training_runs/"
    "HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907/"
    "config.yaml"
)
HEAD = ROOT / (
    "outputs/02_eval_runs/ctc_native_2d_l5_candidates_observation_v2_20260911/"
    "models/hs6_l_5tb_ck21959/folds/fold0/head_epoch50.pth"
)
IMAGE = ROOT / "outputs/02_eval_inputs/formal_v3/ctc_native_full/Fluo-N3DH-CHO/02/t000.tif"
OUTPUT = ROOT / "outputs/02_eval_runs/ctc_native_real3d_smoke_20260916"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    started = time.perf_counter()
    model = build_dino_hovernet(
        checkpoint=str(CHECKPOINT), train_config=str(CONFIG), layers=[23], num_types=0,
        freeze_backbone=True, feature_size=32, embed_proj=384,
        fusion_mode="bucket_concat", decoder_variant="current", device=device,
    )
    payload = torch.load(HEAD, map_location="cpu", weights_only=True)
    model.decoder.load_state_dict(payload["decoder"])
    local = _infer_frame(model, str(IMAGE), 3, device)
    tracked = TrackLinker(diameter=10.0).step(local)
    tifffile.imwrite(OUTPUT / "mask000.tif", tracked.astype(np.uint16), compression="zlib")
    report = {
        "status": "PASS",
        "test": "real Fluo-N3DH-CHO t000 slice-wise prediction and 26-connected consolidation",
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
        "head": str(HEAD), "head_sha256": sha256(HEAD),
        "input": str(IMAGE), "input_shape": list(tifffile.imread(IMAGE).shape),
        "prediction_shape": list(local.shape),
        "local_instances": int(local.max(initial=0)),
        "tracked_instances": int(tracked.max(initial=0)),
        "seconds": time.perf_counter() - started,
        "host": os.uname().nodename,
    }
    atomic_json(OUTPUT / "result.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
