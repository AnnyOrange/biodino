#!/usr/bin/env python3
"""Run the verified MoNuSeg campaign worker with a higher GPU slot target."""

import argparse
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--target-per-gpu", type=int, required=True)
    parser.add_argument("--max-host-jobs", type=int, required=True)
    args = parser.parse_args()
    nvidia_roots = Path("/home").glob("*/miniconda3/envs/dinov3/lib/python*/site-packages/nvidia")
    nvidia_root = next((path for path in nvidia_roots
                        if (path / "cuda_cupti/lib/libcupti.so.12").is_file()
                        and (path / "cuda_runtime/lib/libcudart.so.12").is_file()), None)
    if nvidia_root is None:
        raise RuntimeError("Could not locate the CUDA CUPTI and runtime libraries on this host")
    library_dirs = [nvidia_root / name / "lib" for name in ("cuda_cupti", "cuda_runtime", "cuda_nvrtc")]
    library_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        [*(str(path) for path in library_dirs if path.is_dir()), *filter(None, library_paths)])
    sys.path.insert(0, str(args.snapshot / "scripts"))
    fleet = importlib.import_module("run_retest_fleet_20260918")
    fleet.queue.project_tests = lambda: {}
    worker_args = SimpleNamespace(
        output=args.output,
        host=args.host,
        gpus=args.gpus,
        target_per_gpu=args.target_per_gpu,
        max_host_jobs=args.max_host_jobs,
        max_global_jobs=400,
        task_family="segmentation",
        admission_guard=lambda gpu, actual, task: task["dataset"]["task"] == "segmentation",
    )
    os.umask(0o000)
    fleet.worker(worker_args)


if __name__ == "__main__":
    main()
