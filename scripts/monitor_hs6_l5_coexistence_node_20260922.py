#!/usr/bin/env python3
"""Per-node 30-minute read-only GPU/RAM/NFS admission telemetry for one day."""
from __future__ import annotations

import json
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
CHECKS = {
    'gpu': ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'],
    'compute_pids': ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'],
    'ram': ['free', '-h'],
    'nfs': ['df', '-h', '/mnt/huawei_deepcad'],
}


def snapshot():
    output = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'hostname': socket.gethostname()}
    for name, command in CHECKS.items():
        try:
            r = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
            output[name] = {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            output[name] = {'error': repr(exc)}
    return output


def main():
    path = ROOT / 'node_telemetry' / f'{socket.gethostname()}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(49):
        with path.open('a') as f:
            f.write(json.dumps(snapshot()) + '\n')
        print(f'[{socket.gethostname()}] captured sample {i + 1}/49', flush=True)
        if i != 48:
            time.sleep(1800)


if __name__ == '__main__':
    main()
