#!/usr/bin/env python3
"""Run frozen V4 extensions on CUDA-verified 3090-qi GPUs."""
import argparse
import runpy
import subprocess
import sys
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
GPUS = (0, 2, 3, 4, 5, 6)
sys.path.insert(0, str(REPO / 'scripts'))
import run_deepcad_method_20260927 as resource


def gpu_info():
    output = subprocess.check_output(
        ['nvidia-smi', '-i', ','.join(map(str, GPUS)),
         '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True)
    return {int(parts[0]): dict(uuid=parts[1].strip(), used=int(parts[2]),
                                total=int(parts[3]), util=int(parts[4]))
            for line in output.splitlines() if (parts := line.split(','))}


def gpu_processes():
    output = subprocess.check_output(
        ['nvidia-smi', '-i', ','.join(map(str, GPUS)),
         '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
         '--format=csv,noheader,nounits'], text=True)
    return [dict(uuid=parts[0].strip(), pid=int(parts[1]), memory=parts[2].strip())
            for line in output.splitlines() if len(parts := line.split(',')) == 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-jobs', type=int, default=8)
    parser.add_argument('--host', default='3090-qi-healthy')
    parser.add_argument('--gpus', type=int, nargs='+', default=list(GPUS))
    parser.add_argument('--family', default='all')
    args = parser.parse_args()
    if not set(args.gpus) <= set(GPUS):
        parser.error(f'GPU list must be a subset of CUDA-verified cards {GPUS}')
    resource.gpu_info = gpu_info
    resource.gpu_processes = gpu_processes
    if args.family == 'all':
        queue = runpy.run_path(str(REPO / 'scripts/queue_all_route2_checkpoints_v4_20261012.py'))
        queue['worker_full'](argparse.Namespace(host=args.host, gpus=args.gpus,
                                                 max_jobs=args.max_jobs))
    else:
        worker = runpy.run_path(str(REPO / 'scripts/run_v2_full_v4_20261007.py'))['worker']
        worker.__globals__.update(
            PAIRED=REPO / 'outputs/02_eval_runs/route2_all_checkpoints_v4_20261012/components',
            ROOT=REPO / 'outputs/02_eval_runs/route2_all_checkpoints_v4_20261012/extensions',
            MONUSRC=Path('/mnt/huawei_deepcad/dinov3_monuseg_train30val7_snapshot_20260929'))
        worker(argparse.Namespace(host=args.host, gpus=args.gpus,
                                  max_jobs=args.max_jobs, ram_reserve=48,
                                  family=args.family))


if __name__ == '__main__':
    main()
