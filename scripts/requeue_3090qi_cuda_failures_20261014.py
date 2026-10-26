#!/usr/bin/env python3
"""Archive only interrupted jobs and GPU7 CUDA-init failures from this worker."""
import argparse
import json
from pathlib import Path

ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/route2_all_checkpoints_v4_20261012/extensions')
CLAIMS = ROOT / 'claims'
ARCHIVE = ROOT / 'claims_recovery_20261014_gpu7'
OLD_ARCHIVE = ROOT / 'claims_recovery_20261014_old_interrupted'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    selected = []
    for status_file in CLAIMS.glob('*/status.json'):
        status = json.loads(status_file.read_text())
        host = status.get('host')
        task = status_file.parent.name
        state = status.get('state')
        if host == '3090-qi' and state == 'INTERRUPTED':
            selected.append((status_file.parent, OLD_ARCHIVE, 'old worker interrupted'))
        elif host == '3090-qi-healthy' and state == 'INTERRUPTED':
            selected.append((status_file.parent, ARCHIVE, state))
        elif host == '3090-qi-healthy' and state == 'FAILED' and status.get('gpu') == 7:
            log = ROOT / 'logs' / f'{task}.log'
            if log.exists() and 'No CUDA GPUs are available' in log.read_text(errors='replace'):
                selected.append((status_file.parent, ARCHIVE, 'GPU7 CUDA initialization failed'))
    for source, archive, reason in selected:
        target = archive / source.name
        if args.apply:
            archive.mkdir(exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            source.rename(target)
        print(f'{"ARCHIVED" if args.apply else "WOULD_ARCHIVE"} {source.name}: {reason}')
    print(f'total={len(selected)}')


if __name__ == '__main__':
    main()
