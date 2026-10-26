#!/usr/bin/env python3
"""Run queued V4 extensions on 3090-qi using UUID-pinned CUDA devices."""
import argparse
import runpy
import subprocess
import sys
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
# CUDA skips the NVML-broken physical GPU1, so CUDA ordinals 1..6 map to
# nvidia-smi indices 2..7. Tasks are pinned by UUID to avoid ordinal drift.
CUDA_TO_NVML = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}
GPU_UUIDS = {
    0: 'GPU-22a32e7b-0afa-688c-13f9-e17b650b413f',
    1: 'GPU-11bdd5ba-3f5f-72d0-d51e-cc71ab995679',
    2: 'GPU-299e3bae-f266-1f85-1709-b99de36b3ad5',
    3: 'GPU-1a989a05-97bd-2ad6-9d2d-ceee2132b651',
    4: 'GPU-539bdd9d-ad06-4bec-ac0d-59de9d5e5e6e',
    5: 'GPU-dd128450-e584-0d79-b334-5c9ea37ec2a0',
    6: 'GPU-eaa93504-8a61-8c8b-29cd-eb4413b22042',
}
sys.path.insert(0, str(REPO / 'scripts'))
import run_deepcad_method_20260927 as resource


def query(gpu, fields):
    return subprocess.check_output(
        ['nvidia-smi', '-i', str(CUDA_TO_NVML[gpu]), *fields,
         '--format=csv,noheader,nounits'], text=True).strip()


def gpu_info():
    result = {}
    for cuda_gpu, nvml_gpu in CUDA_TO_NVML.items():
        line = query(cuda_gpu, ['--query-gpu=uuid,memory.used,memory.total,utilization.gpu'])
        uuid, used, total, util = [part.strip() for part in line.split(',')]
        expected_uuid = GPU_UUIDS[cuda_gpu].removeprefix('GPU-')
        if uuid.removeprefix('GPU-') != expected_uuid:
            raise RuntimeError(f'GPU mapping changed: CUDA {cuda_gpu} -> NVML {nvml_gpu}, got {uuid}')
        result[cuda_gpu] = dict(uuid=uuid, used=int(used), total=int(total), util=int(util))
    return result


def gpu_processes():
    rows = []
    for cuda_gpu in CUDA_TO_NVML:
        output = query(cuda_gpu, ['--query-compute-apps=gpu_uuid,pid,used_gpu_memory'])
        if not output or output == 'No running processes found':
            continue
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(',')]
            if len(parts) == 3:
                rows.append(dict(uuid=parts[0], pid=int(parts[1]), memory=parts[2]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-jobs', type=int, default=10)
    parser.add_argument('--host', default='3090-qi-remapped')
    parser.add_argument('--gpus', type=int, nargs='+', default=[3, 5])
    args = parser.parse_args()
    if not set(args.gpus) <= set(CUDA_TO_NVML):
        parser.error(f'GPU list must be a subset of CUDA ordinals {tuple(CUDA_TO_NVML)}')

    resource.gpu_info = gpu_info
    resource.gpu_processes = gpu_processes
    original_popen = subprocess.Popen

    def uuid_pinned_popen(*popen_args, **kwargs):
        env = kwargs.get('env')
        if env:
            visible = env.get('CUDA_VISIBLE_DEVICES')
            if visible and visible.isdecimal() and int(visible) in args.gpus:
                env = dict(env, CUDA_VISIBLE_DEVICES=GPU_UUIDS[int(visible)])
                kwargs['env'] = env
        return original_popen(*popen_args, **kwargs)

    subprocess.Popen = uuid_pinned_popen
    queue = runpy.run_path(str(REPO / 'scripts/queue_all_route2_checkpoints_v4_20261012.py'))
    queue['worker_full'](argparse.Namespace(host=args.host, gpus=args.gpus,
                                             max_jobs=args.max_jobs))


if __name__ == '__main__':
    main()
