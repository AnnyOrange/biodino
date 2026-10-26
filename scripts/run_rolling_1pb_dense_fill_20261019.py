#!/usr/bin/env python3
"""Fill 3090 test cards with one extra low-resolution dense evaluation each."""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path


SNAPSHOT = Path("/mnt/huawei_deepcad/dinov3_20tb_online_snapshot_20260918")
OUTPUT = Path("/mnt/huawei_blm/rolling_1pb_eval_v4_20261019")
sys.path.insert(0, str(SNAPSHOT / "scripts"))
import run_retest_fleet_20260918 as fleet


def low_resolution_dense(_gpu: int, _actual: int, task: dict) -> bool:
    dataset = task["dataset"]
    return (dataset["task"] == "segmentation" and
            int(dataset["image_size"]) <= 256 and
            int(dataset.get("reserve_mib", 0)) <= 8192)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="3090-qi-rolling1pb-dense-fill")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 3, 4])
    args_cli = parser.parse_args()
    # The primary worker counts all host processes and permits only one dense
    # task per GPU. This worker observes its own slot; GPU free-memory checks
    # and the shared global task admission lock still apply.
    fleet.queue.project_tests = lambda: {}
    args = Namespace(output=OUTPUT, host=args_cli.host,
                     gpus=args_cli.gpus, target_per_gpu=1,
                     max_host_jobs=len(args_cli.gpus), max_global_jobs=24,
                     task_family="segmentation",
                     admission_guard=low_resolution_dense)
    fleet.worker(args)


if __name__ == "__main__":
    main()
