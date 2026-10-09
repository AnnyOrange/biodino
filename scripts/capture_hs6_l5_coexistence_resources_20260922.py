#!/usr/bin/env python3
"""Append one read-only 3090 cluster capacity/PID snapshot to campaign telemetry."""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
HOSTS = ('cpu1', 'cpu2', 'cpu5', 'cpu8', 'cpu9', 'cpu10', 'cpu11', 'cpu12', 'cpu15', 'cpu19', '3090-qi')
COMMAND = (
    'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; '
    'nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader; '
    'free -h; df -h /mnt/huawei_deepcad'
)


def capture(host):
    try:
        proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', host, COMMAND],
                              capture_output=True, text=True, timeout=20)
        return {'host': host, 'returncode': proc.returncode, 'output': proc.stdout,
                'error': proc.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'host': host, 'error': repr(exc)}


def main():
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(capture, HOSTS))
    entry = {'captured_utc': datetime.now(timezone.utc).isoformat(),
             'scope': 'coexistence v4 single-3090 and 3090-qi shared-host telemetry',
             'hosts': rows}
    output = ROOT / 'gpu_telemetry.jsonl'
    with output.open('a') as stream:
        stream.write(json.dumps(entry) + '\n')
    print(f'Saved {len(rows)} host snapshots to {output}; failed={sum(x.get("returncode", 1) != 0 for x in rows)}')


if __name__ == '__main__':
    main()
