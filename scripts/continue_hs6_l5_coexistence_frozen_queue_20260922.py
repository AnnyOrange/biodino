#!/usr/bin/env python3
"""Start the next bounded v4 lane only if the prior lane completed successfully."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
WORKER = Path('/mnt/huawei_deepcad/dinov3/scripts/run_hs6_l5_coexistence_frozen_queue_20260922.py')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior-log', required=True, choices=('queue_frozen_cpu19.log', 'queue_frozen_cpu2.log'))
    p.add_argument('--datasets', nargs='+', required=True)
    args = p.parse_args()
    before = ROOT / 'logs' / args.prior_log
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        content = before.read_text() if before.exists() else ''
        if 'QUEUE_COMPLETED all assigned READY datasets' in content:
            cmd = [sys.executable, '-B', str(WORKER), '--datasets', *args.datasets]
            print(f'[continue] prior={before}, command={cmd}', flush=True)
            return subprocess.call(cmd)
        if 'FAILED_PROTOCOL_OR_RESOURCE' in content or 'FAILED_FUSION' in content or 'QUEUE_TIME_LIMIT' in content:
            print(f'PRIOR_LANE_FAILURE: will not dispatch further tests: {before}', flush=True)
            return 1
        time.sleep(45)
    print(f'PRIOR_LANE_TIMEOUT: {before}', flush=True)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
