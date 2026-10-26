#!/usr/bin/env python3
"""Join one prepared 100TB checkpoint's frozen MoNuSeg 30/7/14 queue."""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


REPO = Path('/mnt/huawei_deepcad/dinov3')
CONTROL = REPO / 'outputs/02_eval_runs/hs6_100tb_fullpool_v4_every500_20261013'
PYTHON = '/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'
SPLIT = REPO / 'outputs/02_eval_inputs/monuseg37_v4_20261012/manifest.json'
SPLIT_SHA256 = '6b71c7d02c3031558ceeb2b17eb57c1916520d55807249668696c061550688e7'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--share-gpu', action='store_true')
    args = parser.parse_args()
    campaign = CONTROL / f'campaigns/100tb_ck{args.step}'
    prepared = json.loads((campaign / 'prepared.json').read_text())
    if prepared['monuseg_counts'] != {'train': 30, 'val': 7, 'test': 14}:
        raise ValueError('MoNuSeg split differs from train30/val7/test14')
    env = os.environ.copy()
    env.update(DINOV3_MONUSEG_MANIFEST_FILE=str(SPLIT),
               DINOV3_MONUSEG_MANIFEST_SHA256=SPLIT_SHA256,
               DINOV3_MONUSEG_COUNTS=json.dumps(prepared['monuseg_counts']))
    if args.share_gpu:
        free = int(subprocess.check_output([
            'nvidia-smi', '-i', str(args.gpu), '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True).strip())
        if free < 20_000:
            raise RuntimeError(f'GPU {args.gpu} has only {free} MiB free; MoNuSeg needs 20,000 MiB')
        sys.path.insert(0, str(REPO / 'scripts'))
        import run_retest_fleet_20260918 as fleet

        audit = campaign / 'monuseg/scheduling_override.json'
        audit.write_text(json.dumps(dict(
            reason='Share GPU with existing small test process for MoNuSeg 30/7/14',
            host=platform.node(), gpu=args.gpu, free_mib_at_start=free,
            task_command_and_validation='unchanged frozen fleet worker'), indent=2) + '\n')
        os.environ.update(env)
        fleet.queue.project_tests = lambda: {}
        fleet.worker(SimpleNamespace(
            output=campaign / 'monuseg', host=f'{platform.node()}-100tb-monu-{args.step}-gpu{args.gpu}',
            gpus=[args.gpu], target_per_gpu=5, max_host_jobs=1,
            max_global_jobs=160, task_family='segmentation'))
        return
    command = [PYTHON, '-u', str(REPO / 'scripts/run_retest_fleet_20260918.py'),
               'worker', '--output', str(campaign / 'monuseg'),
               '--host', f'{platform.node()}-100tb-monu-{args.step}-gpu{args.gpu}',
               '--gpus', str(args.gpu), '--target-per-gpu', '5',
               '--max-host-jobs', '1', '--max-global-jobs', '160',
               '--task-family', 'segmentation']
    raise SystemExit(subprocess.call(command, cwd=REPO, env=env))


if __name__ == '__main__':
    main()
