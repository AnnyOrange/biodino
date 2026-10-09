#!/usr/bin/env python3
"""Validate four Cryo project feature parts and compute formal v4 OOD scores."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path('/data/hs6_5tb_v4_ood_20260924')
SOURCE = Path('/data/hs6_l_5tb_nogram_eval_20260921/bin/v4_monuseg_source_snapshot_20260924')
BENCHMARK = Path('/data/benchmark')
PROJECTS = ('10535', '11043', '11387', '11388')
HPOINTS = (0, 487, 975, 1463, 1951, 2439, 2927, 3415, 3903, 4391, 4879, 5367, 5855, 6343, 6831)
ROOTS = {'hplus': Path('/data/hs6_hplus_5tb_eval_20260921'),
         'l': Path('/data/hs6_l_5tb_nogram_eval_20260921')}
sys.path.insert(0, str(SOURCE))
from dinov3.eval.eval_ood.datasets import CryoParticleDataset
from dinov3.eval.eval_ood.dinov3_runner import load_feature_cache
from dinov3.eval.eval_ood.metrics import id_vs_ood_knn


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def key_of_record(rec):
    return (rec.project_id, str(rec.cs_path.resolve()), str(rec.mrc_path.resolve()),
            int(rec.particle_index), int(rec.class_id))


def key_of_meta(meta):
    return (str(meta['project_id']), str(Path(meta['cs_path']).resolve()),
            str(Path(meta['path']).resolve()), int(meta['particle_index']),
            int(meta['class_id']))


def full_preflight() -> dict:
    ds = CryoParticleDataset(BENCHMARK / 'ood', max_particles_per_project=20000,
                             percentiles=(0.5, 99.5), invert=False, seed=0)
    counts = {project: sum(rec.project_id == project for rec in ds.records)
              for project in PROJECTS}
    if len(ds) != 80000 or counts != dict.fromkeys(PROJECTS, 20000):
        raise RuntimeError(f'Cryo input incomplete: n={len(ds)} counts={counts}')
    project_manifests = {}
    for project in PROJECTS:
        path = BASE / 'project_roots' / project / 'input_manifest.json'
        if not path.is_file():
            raise FileNotFoundError(path)
        project_manifests[project] = digest(path)
    records = [key_of_record(r) for r in ds.records]
    manifest = {
        'protocol': 'bio-eval-union-v4/cryo-ood',
        'n_records': len(records), 'project_counts': counts,
        'selected_records_sha256': hashlib.sha256(canonical(records)).hexdigest(),
        'project_manifest_sha256': project_manifests,
        'id_reference_manifest_sha256': digest(BASE / 'xray_input_manifest.json'),
        'settings': {'batch_size': 64, 'seed': 0, 'workers': 2,
                     'dtype': 'bf16', 'resize': 256, 'crop': 224,
                     'readout': 'final_cls+final_patch_mean',
                     'percentiles': [0.5, 99.5], 'invert': False,
                     'id_train_fraction': 0.7, 'knn_k': 10},
        'source_sha256': {name: digest(SOURCE / name) for name in
                          ('dinov3/eval/eval_ood/datasets.py',
                           'dinov3/eval/eval_ood/dinov3_runner.py',
                           'dinov3/eval/eval_ood/metrics.py',
                           'dinov3/eval/bio_frozen_eval/encoder.py')},
    }
    path = BASE / 'cryo_input_manifest.json'
    if path.exists():
        if canonical(json.loads(path.read_text())) != canonical(manifest):
            raise RuntimeError('Frozen full Cryo input changed')
    else:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def points():
    lpoints = sorted(int(p.parent.name) for p in (ROOTS['l'] / 'adapters').glob('*/checkpoint.pth')
                     if p.stat().st_size == 1401909871)
    if len(lpoints) != 25:
        raise RuntimeError('Expected 25 L checkpoints')
    for campaign, seq in (('hplus', HPOINTS), ('l', lpoints)):
        for point in seq:
            yield campaign, point


def finalize(campaign: str, point: int, manifest: dict):
    key = f'{campaign}_{point}_cryo'
    out = BASE / 'cryo_results' / key / str(point)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / 'last_result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get('status') == 'VALID_COMPLETE' and result.get('cryo_ood_ood_test') == 80000:
            return
        raise RuntimeError(f'Existing invalid result: {result_path}')
    xray_id = BASE / 'results' / f'{campaign}_{point}_xray' / str(point) / 'features/id_reference.npz'
    id_features, _labels, _metas = load_feature_cache(xray_id)
    if len(id_features) != 3000:
        raise RuntimeError(f'Wrong ID size for {key}')
    full_ds = CryoParticleDataset(BENCHMARK / 'ood', max_particles_per_project=20000,
                                  percentiles=(0.5, 99.5), invert=False, seed=0)
    expected = [key_of_record(r) for r in full_ds.records]
    features = []
    observed = []
    part_hashes = {}
    for project in PROJECTS:
        path = BASE / 'cryo_project_features' / project / 'results' / key / str(point) / 'features/cryo_raw_mpp20000.npz'
        marker = path.parent.parent / 'features_complete.json'
        if not marker.is_file() or not path.is_file():
            raise FileNotFoundError(f'Cryo part incomplete: {path}')
        part, _labels, metas = load_feature_cache(path)
        if len(part) != 20000 or len(metas) != 20000:
            raise RuntimeError(f'Wrong part length: {project} {key}')
        features.append(part)
        observed.extend(key_of_meta(m) for m in metas)
        part_hashes[project] = digest(path)
    if observed != expected:
        raise RuntimeError(f'Cryo part record order/identity differs from full lock: {key}')
    x = np.concatenate(features, axis=0)
    score = id_vs_ood_knn(id_features, x, k=10, train_fraction=0.7, seed=0)
    if score['id_bank'] != 2100 or score['id_test'] != 900 or score['ood_test'] != 80000:
        raise RuntimeError(f'Invalid metric counts: {score}')
    checkpoint = ROOTS[campaign] / 'adapters' / str(point) / 'checkpoint.pth'
    result = {
        'status': 'VALID_COMPLETE', 'protocol': 'bio-eval-union-v4/cryo-ood',
        'campaign': campaign, 'point': point, 'checkpoint': str(checkpoint),
        'checkpoint_sha256': digest(checkpoint),
        'train_config_sha256': digest(ROOTS[campaign] / 'source/config.yaml'),
        'input_manifest_sha256': digest(BASE / 'cryo_input_manifest.json'),
        'id_cache_sha256': digest(xray_id), 'project_feature_sha256': part_hashes,
        'n_projects': 4, 'n_ood': 80000,
        'cryo_ood_auroc': float(score['auroc']),
        'cryo_ood_average_precision': float(score['average_precision']),
        'cryo_ood_id_bank': int(score['id_bank']),
        'cryo_ood_id_test': int(score['id_test']),
        'cryo_ood_ood_test': int(score['ood_test']),
        'completed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    temporary = result_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    os.replace(temporary, result_path)
    print(f'VALID_COMPLETE {key} auroc={score["auroc"]:.6f}', flush=True)


def worker(slot: int):
    manifest = full_preflight()
    claims = BASE / 'cryo_results' / 'claims'
    failures = BASE / 'cryo_results' / 'failures'
    claims.mkdir(parents=True, exist_ok=True)
    failures.mkdir(parents=True, exist_ok=True)
    while True:
        selected = None
        for campaign, point in points():
            key = f'{campaign}_{point}_cryo'
            out = BASE / 'cryo_results' / key / str(point) / 'last_result.json'
            if out.exists() or (failures / f'{key}.json').exists():
                continue
            if any(not (BASE / 'cryo_project_features' / project / 'results' / key / str(point) / 'features_complete.json').exists()
                   for project in PROJECTS):
                continue
            claim = claims / key
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            selected = campaign, point, claim
            break
        if selected is None:
            if all((BASE / 'cryo_results' / f'{c}_{p}_cryo' / str(p) / 'last_result.json').exists()
                   or (failures / f'{c}_{p}_cryo.json').exists() for c, p in points()):
                return
            time.sleep(10)
            continue
        campaign, point, claim = selected
        try:
            finalize(campaign, point, manifest)
        except Exception as exc:
            key = f'{campaign}_{point}_cryo'
            (failures / f'{key}.json').write_text(json.dumps({'error': repr(exc)}) + '\n')
            print(f'FAILED {key}: {exc!r}', flush=True)
        claim.rmdir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(full_preflight(), indent=2))
    elif args.worker is not None:
        worker(args.worker)
    else:
        parser.error('Specify --preflight or --worker')


if __name__ == '__main__':
    main()
