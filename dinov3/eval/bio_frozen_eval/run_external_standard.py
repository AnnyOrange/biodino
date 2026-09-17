"""Pinned entrypoint for the reviewed implementation in benchmark_model."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("DINOV3_CODE_ROOT", str(ROOT))
sys.path.insert(0, "/mnt/huawei_deepcad/benchmark_model")
from run_fm_rules import StandardExternalEncoder, cls_patch_feature, main

if __name__ == "__main__":
    raise SystemExit(main())
