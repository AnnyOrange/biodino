#!/usr/bin/env python3
"""Read-only, one-minute GPU/host-RAM threshold monitor on an evaluation node."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')


def sample(host_tag: str) -> dict:
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True, timeout=20)
    gpu = []
    for line in output.splitlines():
        index, used, total, util = (int(part.strip()) for part in line.split(','))
        gpu.append({'index': index, 'used_mib': used, 'total_mib': total,
                    'memory_fraction': round(used / total, 5), 'utilization_pct': util,
                    'over_80pct': used / total > .80})
    if len(gpu) != 8:
        raise RuntimeError(f'Expected 8 physical GPUs, got {len(gpu)}')
    meminfo = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemTotal:', 'MemAvailable:', 'SwapTotal:', 'SwapFree:')):
            key, val = line.split(':', 1)
            meminfo[key] = int(val.strip().split()[0])
    ram_fraction = 1 - meminfo['MemAvailable'] / meminfo['MemTotal']
    return {'utc': datetime.now(timezone.utc).isoformat(), 'host_tag': host_tag,
            'gpu': gpu, 'ram_effective_used_fraction': round(ram_fraction, 5),
            'ram_over_80pct': ram_fraction > .80, 'swap_total_kib': meminfo['SwapTotal'],
            'swap_used_kib': meminfo['SwapTotal'] - meminfo['SwapFree']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-tag', required=True, choices=('local5090', 'qi3090'))
    parser.add_argument('--hours', type=float, default=24)
    args = parser.parse_args()
    output = ROOT / 'node_telemetry' / f'80pct_{args.host_tag}.jsonl'
    if output.exists():
        raise FileExistsError(f'Possible duplicate monitor, refusing to overwrite {output}')
    deadline = time.monotonic() + args.hours * 3600
    with output.open('x') as stream:
        while time.monotonic() < deadline:
            try:
                point = sample(args.host_tag)
            except Exception as err:
                point = {'utc': datetime.now(timezone.utc).isoformat(), 'host_tag': args.host_tag,
                         'monitor_error': repr(err)}
            stream.write(json.dumps(point) + '\n')
            stream.flush()
            if point.get('ram_over_80pct') or any(g['over_80pct'] for g in point.get('gpu', [])):
                print(f'OVER_80_THRESHOLD {point["utc"]} {args.host_tag}', flush=True)
            time.sleep(60)


if __name__ == '__main__':
    main()
