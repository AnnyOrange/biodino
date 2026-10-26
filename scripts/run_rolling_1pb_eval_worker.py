#!/usr/bin/env python3
"""Run the pinned 1PB evaluator with an isolated, capacity-aware scheduler."""

import argparse
import os
import sys
from pathlib import Path

SNAPSHOT_SCRIPTS = Path('/mnt/huawei_deepcad/dinov3_20tb_online_snapshot_20260918/scripts')
sys.path.insert(0, str(SNAPSHOT_SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rolling_1pb_eval_queue as queue
import run_retest_fleet_20260918 as fleet


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--max-dense-per-gpu',type=int,default=2)
    options, remaining = parser.parse_known_args()
    if options.max_dense_per_gpu < 1:
        parser.error('--max-dense-per-gpu must be positive')
    os.environ['ROLLING_1PB_MAX_DENSE_PER_GPU'] = str(options.max_dense_per_gpu)
    fleet.queue = queue
    sys.argv = [sys.argv[0], *remaining]
    fleet.main()


if __name__ == '__main__':
    main()
