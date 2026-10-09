#!/usr/bin/env python3
"""Run identity-locked X-ray OOD feature extraction on hxw GPU 0/1."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/data/hs6_5tb_v4_ood_20260924')
SOURCE = Path('/data/hs6_l_5tb_nogram_eval_20260921/bin/v4_monuseg_source_snapshot_20260924')
PYTHON = '/home/xzj/eval_envs/hs6_protocol_v2/bin/python'
BENCHMARK = Path('/data/benchmark')
ROOTS = {
    'hplus': Path('/data/hs6_hplus_5tb_eval_20260921'),
    'l': Path('/data/hs6_l_5tb_nogram_eval_20260921'),
}
HPOINTS = (0, 487, 975, 1463, 1951, 2439, 2927, 3415, 3903, 4391, 4879, 5367, 5855, 6343, 6831)
CUDA_LIBS = (
    '/home/xzj/miniconda3/envs/dinov3/lib/python3.11/site-packages/nvidia/cuda_runtime/lib',
    '/home/xzj/miniconda3/envs/dinov3/lib/python3.11/site-packages/nvidia/cuda_cupti/lib',
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def preflight() -> dict:
    sys.path.insert(0, str(SOURCE))
    import numpy as np
    from dinov3.eval.eval_ood.datasets import XrayTomogramSliceDataset, build_id_reference_dataset

    xray = XrayTomogramSliceDataset(BENCHMARK / 'ood', slices_per_volume=8,
                                    input_mode='three_slices', percentiles=(0.5, 99.5))
    if len(xray) != 992 or len({r.volume_id for r in xray.records}) != 124:
        raise RuntimeError(f'X-ray incomplete: slices={len(xray)} volumes={len({r.volume_id for r in xray.records})}')
    records = []
    for r in xray.records:
        records.append((r.volume_id, r.tomo_id, r.variant, r.z_index,
                        r.raw_shape_xyz, r.raw_path.stat().st_size))
    ids = build_id_reference_dataset(BENCHMARK, transform=None, max_samples=3000,
                                    dataset_names=('bloodmnist', 'bbbc048', 'cyclops'), seed=0)
    if len(ids) != 3000 or [len(x) for x in ids.datasets] != [1000, 1000, 1000]:
        raise RuntimeError('ID reference does not match the locked 1000/source selection')
    selected = []
    for item in ids.datasets:
        subset = item.dataset
        selected.append({'source': item.source, 'source_size': len(subset.dataset),
                         'indices': [int(i) for i in subset.indices]})
    files = ['dinov3/eval/eval_ood/datasets.py', 'dinov3/eval/eval_ood/dinov3_runner.py',
             'dinov3/eval/eval_ood/metrics.py', 'dinov3/eval/bio_frozen_eval/encoder.py']
    code = {name: sha((SOURCE / name).read_bytes()) for name in files}
    manifest = {
        'protocol': 'bio-eval-union-v4/xray-ood', 'input_root': str(BENCHMARK),
        'xray': {'n_volumes': 124, 'n_slices': 992, 'records_sha256': sha(canonical(records)),
                 'raw_files': sorted({r.raw_path.name: r.raw_path.stat().st_size for r in xray.records}.items())},
        'id': {'n_samples': 3000, 'selection_sha256': sha(canonical(selected)),
               'sources': [{k: v for k, v in s.items() if k != 'indices'} for s in selected]},
        'settings': {'batch_size': 64, 'seed': 0, 'workers': 2, 'dtype': 'bf16',
                     'resize': 256, 'crop': 224, 'readout': 'final_cls+final_patch_mean',
                     'xray_mode': 'three_slices', 'slices_per_volume': 8,
                     'percentiles': [0.5, 99.5], 'id_train_fraction': 0.7, 'knn_k': 10},
        'source_sha256': code,
    }
    BASE.mkdir(parents=True, exist_ok=True)
    path = BASE / 'xray_input_manifest.json'
    if path.exists():
        if canonical(json.loads(path.read_text())) != canonical(manifest):
            raise RuntimeError(f'Frozen X-ray input changed: {path}')
    else:
        path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return manifest


def points():
    lroot = ROOTS['l'] / 'adapters'
    lpoints = sorted(int(p.parent.name) for p in lroot.glob('*/checkpoint.pth')
                     if p.stat().st_size == 1401909871)
    if len(lpoints) != 25:
        raise RuntimeError(f'Expected 25 L points, got {len(lpoints)}')
    for name, seq in (('hplus', HPOINTS), ('l', lpoints)):
        for point in seq:
            checkpoint = ROOTS[name] / 'adapters' / str(point) / 'checkpoint.pth'
            if not checkpoint.exists():
                raise RuntimeError(f'Missing checkpoint: {checkpoint}')
            yield name, point


def command(campaign: str, point: int, phase: str) -> list[str]:
    root = ROOTS[campaign]
    return [PYTHON, '-u', '-m', 'dinov3.eval.eval_ood.dinov3_runner',
            '--model-name', f'{campaign}_{point}_xray', '--ckpt-root', str(root / 'adapters'),
            '--ckpt-iter', str(point), '--train-config', str(root / 'source/config.yaml'),
            '--output-dir', str(BASE / 'results'), '--benchmark-root', str(BENCHMARK),
            '--ood-root', str(BENCHMARK / 'ood'), '--tasks', 'xray', '--device', 'cuda:0',
            '--batch-size', '64', '--num-workers', '2', '--n-last-blocks', '1',
            '--autocast-dtype', 'bf16', '--resize-size', '256', '--crop-size', '224',
            '--xray-input-mode', 'three_slices', '--xray-slices-per-volume', '8',
            '--id-max-samples', '3000', '--id-datasets', 'bloodmnist', 'bbbc048',
            'cyclops', '--seed', '0', '--phase', phase]


def run(gpu: int, slot: int):
    manifest = preflight()
    key_list = list(points())
    for folder in ('claims', 'logs', 'failures'):
        (BASE / folder).mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(DINOV3_ROOT=str(SOURCE), PYTHONPATH=str(SOURCE),
               CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    env['LD_LIBRARY_PATH'] = ':'.join(CUDA_LIBS) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    print(f'preflight PASS gpu={gpu} slot={slot} xray={manifest["xray"]["n_slices"]} id={manifest["id"]["n_samples"]}', flush=True)
    while True:
        selected = None
        for campaign, point in key_list:
            key = f'{campaign}_{point}_xray'
            out = BASE / 'results' / key / str(point)
            if (out / 'features_complete.json').exists() or (BASE / 'failures' / f'{key}.json').exists():
                continue
            claim = BASE / 'claims' / key
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            selected = campaign, point, key, claim, out
            break
        if selected is None:
            print('no_unclaimed_xray_cells', flush=True)
            return
        campaign, point, key, claim, out = selected
        cmd = command(campaign, point, 'extract')
        (claim / 'command.json').write_text(json.dumps({'command': cmd, 'gpu': gpu,
                     'slot': slot, 'manifest_sha256': sha((BASE / 'xray_input_manifest.json').read_bytes()),
                     'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2) + '\n')
        print(f'START {key} gpu={gpu} slot={slot}', flush=True)
        with (BASE / 'logs' / f'{key}.log').open('w') as log:
            rc = subprocess.run(cmd, env=env, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc == 0 and (out / 'features_complete.json').exists():
            (claim / 'command.json').unlink()
            claim.rmdir()
            print(f'EXTRACTED {key}', flush=True)
        else:
            (BASE / 'failures' / f'{key}.json').write_text(json.dumps({'returncode': rc,
                        'log': str(BASE / 'logs' / f'{key}.log')}) + '\n')
            print(f'FAILED {key} rc={rc}', flush=True)
        time.sleep(1)


def metrics_worker(slot: int):
    preflight()
    for folder in ('metrics_claims', 'metric_failures'):
        (BASE / folder).mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(DINOV3_ROOT=str(SOURCE), PYTHONPATH=str(SOURCE),
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    while True:
        selected = None
        for campaign, point in points():
            key = f'{campaign}_{point}_xray'
            out = BASE / 'results' / key / str(point)
            if not (out / 'features_complete.json').exists() or (out / 'last_result.json').exists():
                continue
            if (BASE / 'metric_failures' / f'{key}.json').exists():
                continue
            claim = BASE / 'metrics_claims' / key
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            selected = campaign, point, key, claim, out
            break
        if selected is None:
            if all((BASE / 'results' / f'{c}_{p}_xray' / str(p) / 'last_result.json').exists()
                   or (BASE / 'failures' / f'{c}_{p}_xray.json').exists()
                   or (BASE / 'metric_failures' / f'{c}_{p}_xray.json').exists()
                   for c, p in points()):
                return
            time.sleep(5)
            continue
        campaign, point, key, claim, out = selected
        cmd = command(campaign, point, 'metrics')
        print(f'METRICS {key} slot={slot}', flush=True)
        with (BASE / 'logs' / f'{key}_metrics.log').open('w') as log:
            rc = subprocess.run(cmd, env=env, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc or not (out / 'last_result.json').exists():
            (BASE / 'metric_failures' / f'{key}.json').write_text(json.dumps({'returncode': rc}) + '\n')
        claim.rmdir()


def cryo_project_preflight(project: str) -> dict:
    if project not in {'10535', '11043', '11387', '11388'}:
        raise ValueError(project)
    sys.path.insert(0, str(SOURCE))
    from dinov3.eval.eval_ood.datasets import CryoParticleDataset
    project_root = BASE / 'project_roots' / project / 'cryo_em_foundation_model' / 'extracted'
    project_root.mkdir(parents=True, exist_ok=True)
    link = project_root / project
    actual = BENCHMARK / 'ood' / 'cryo_em_foundation_model' / 'extracted' / project
    if not link.exists() and not link.is_symlink():
        link.symlink_to(actual, target_is_directory=True)
    if link.resolve() != actual.resolve():
        raise RuntimeError(f'Wrong project link: {link}')
    ds = CryoParticleDataset(project_root.parents[1], max_projects=1,
                             max_particles_per_project=20000, percentiles=(0.5, 99.5),
                             invert=False, seed=0)
    if len(ds) != 20000 or {r.project_id for r in ds.records} != {project}:
        raise RuntimeError(f'Cryo project incomplete: {project} records={len(ds)}')
    records = [(r.project_id, r.cs_path.name, r.mrc_path.name, r.particle_index,
                r.class_id) for r in ds.records]
    manifest = {'protocol': 'bio-eval-union-v4/cryo-ood-project-feature-part',
                'project': project, 'n_records': len(records),
                'records_sha256': sha(canonical(records)),
                'settings': {'batch_size': 64, 'seed': 0, 'workers': 2,
                             'dtype': 'bf16', 'resize': 256, 'crop': 224,
                             'percentiles': [0.5, 99.5], 'invert': False,
                             'max_particles_per_project': 20000}}
    path = BASE / 'project_roots' / project / 'input_manifest.json'
    if path.exists():
        if canonical(json.loads(path.read_text())) != canonical(manifest):
            raise RuntimeError(f'Frozen Cryo project input changed: {path}')
    else:
        path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return manifest


def cryo_project_worker(project: str, gpu: int, slot: int):
    manifest = cryo_project_preflight(project)
    project_base = BASE / 'cryo_project_features' / project
    for folder in ('claims', 'logs', 'failures'):
        (project_base / folder).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(DINOV3_ROOT=str(SOURCE), PYTHONPATH=str(SOURCE),
               CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    env['LD_LIBRARY_PATH'] = ':'.join(CUDA_LIBS) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    print(f'cryo_preflight PASS project={project} records={manifest["n_records"]} gpu={gpu} slot={slot}', flush=True)
    for campaign, point in points():
        key = f'{campaign}_{point}_cryo'
        out = project_base / 'results' / key / str(point)
        if (out / 'features_complete.json').exists() or (project_base / 'failures' / f'{key}.json').exists():
            continue
        claim = project_base / 'claims' / key
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        xray_id = BASE / 'results' / f'{campaign}_{point}_xray' / str(point) / 'features/id_reference.npz'
        if not xray_id.is_file():
            claim.rmdir()
            continue
        feature_dir = out / 'features'
        feature_dir.mkdir(parents=True, exist_ok=True)
        id_link = feature_dir / 'id_reference.npz'
        if not id_link.exists():
            id_link.symlink_to(xray_id)
        cmd = command(campaign, point, 'extract')
        index = cmd.index('--output-dir')
        cmd[index + 1] = str(project_base / 'results')
        index = cmd.index('--model-name')
        cmd[index + 1] = key
        index = cmd.index('--ood-root')
        cmd[index + 1] = str(BASE / 'project_roots' / project)
        index = cmd.index('--tasks')
        cmd[index + 1] = 'cryo'
        cmd.extend(['--cryo-max-projects', '1', '--cryo-max-particles-per-project', '20000'])
        (claim / 'command.json').write_text(json.dumps({'command': cmd, 'gpu': gpu,
            'project_manifest_sha256': sha((BASE / 'project_roots' / project / 'input_manifest.json').read_bytes()),
            'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2) + '\n')
        print(f'START project={project} {key} gpu={gpu} slot={slot}', flush=True)
        with (project_base / 'logs' / f'{key}.log').open('w') as log:
            rc = subprocess.run(cmd, env=env, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc == 0 and (out / 'features_complete.json').exists():
            (claim / 'command.json').unlink()
            claim.rmdir()
            print(f'EXTRACTED project={project} {key}', flush=True)
        else:
            (project_base / 'failures' / f'{key}.json').write_text(json.dumps({'returncode': rc}) + '\n')
            print(f'FAILED project={project} {key} rc={rc}', flush=True)
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--metrics-worker', action='store_true')
    parser.add_argument('--cryo-project', choices=('10535', '11043', '11387', '11388'))
    parser.add_argument('--gpu', type=int, choices=(0, 1))
    parser.add_argument('--slot', type=int)
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(), indent=2))
    elif args.metrics_worker:
        metrics_worker(args.slot or 0)
    elif args.cryo_project:
        if args.gpu is None or args.slot is None:
            parser.error('--cryo-project requires --gpu and --slot')
        cryo_project_worker(args.cryo_project, args.gpu, args.slot)
    else:
        if args.gpu is None or args.slot is None:
            parser.error('--gpu and --slot required')
        run(args.gpu, args.slot)


if __name__ == '__main__':
    main()
