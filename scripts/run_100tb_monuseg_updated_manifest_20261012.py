#!/usr/bin/env python3
"""Re-evaluate the 100TB checkpoint against the newer benchmark MoNuSeg manifest."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3')
BENCHMARK = Path('/mnt/huawei_deepcad/benchmark')
DATA = BENCHMARK / 'segmentation/monuseg/extracted'
MANIFEST = DATA / 'monuseg37_v4_20260929_superseded_manifest.json'
MANIFEST_SHA = '743d8fe6a7dec80dcc0c699cfc173349f83328c3a9fda8a9422fcf6727c66f68'
ORIGINAL_CAMPAIGN = ROOT / 'plot/fig2/data_quality/long_100tb_full_pool_partial_v4_id_20261012_retry1/100tb_ck7999/monuseg'
ORIGINAL_CELL = next((ORIGINAL_CAMPAIGN / 'cells').glob('*'))
BASE_SNAPSHOT = Path('/mnt/huawei_deepcad/dinov3_monuseg_train30val7_snapshot_20260929')
SNAPSHOT = Path('/mnt/huawei_deepcad/dinov3_monuseg37_updated_manifest_snapshot_20261012')
OUT = ROOT / 'plot/fig2/data_quality/long_100tb_full_pool_partial_v4_id_20261012_retry1/100tb_ck7999/monuseg_updated_manifest_20261012'
PYTHON = '/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prepare() -> dict:
    if sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError('The benchmark MoNuSeg manifest changed after inspection')
    source = json.loads(MANIFEST.read_text())
    counts = {split: sum(row['split'] == split for row in source['samples'])
              for split in ('train', 'val', 'test')}
    if counts != {'train': 30, 'val': 7, 'test': 14}:
        raise RuntimeError(f'Unexpected updated MoNuSeg split counts: {counts}')
    for sample in source['samples']:
        for field in ('image', 'xml'):
            path = DATA / sample[field]
            if not path.is_file() or sha(path) != sample[f'{field}_sha256']:
                raise RuntimeError(f'MoNuSeg source content mismatch: {path}')

    dataset_root = OUT / 'data/segmentation/monuseg/extracted'
    dataset_root.mkdir(parents=True, exist_ok=True)
    for name in ('MoNuSeg 2018 Training Data', 'MoNuSegTestData'):
        target = dataset_root / name
        expected_target = DATA / name
        if target.is_symlink() and target.resolve() == expected_target.resolve():
            target.unlink()
        if not target.exists():
            shutil.copytree(expected_target, target)
    normalized = dict(source)
    normalized['official_source_url'] = source['source']
    normalized['official_inventory_sha256'] = hashlib.sha256(
        json.dumps(source['samples'], sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()
    normalized_path = dataset_root / 'monuseg37_updated_manifest_20261012.json'
    normalized_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + '\n')
    normalized_sha = sha(normalized_path)

    if not SNAPSHOT.exists():
        shutil.copytree(BASE_SNAPSHOT, SNAPSHOT)
    lock_path = SNAPSHOT / 'Evaluation Rules/protocol_v3.json'
    lock = json.loads(lock_path.read_text())
    dense = lock['dense_splits']['monuseg']
    dense.update(protocol=source['protocol'], manifest_file=normalized_path.name,
                 manifest_sha256=normalized_sha, counts=counts,
                 formal_status='ISOLATED_UPDATED_MANIFEST_RETEST_20261012')
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + '\n')

    invocation = json.loads(next(ORIGINAL_CELL.glob('invocation_manifest.json')).read_text())
    command = list(invocation['command'])
    for flag, value in (
        ('--output-root', str(OUT / 'results')),
        ('--cache-root', str(OUT / 'cache')),
        ('--run-name', 'monuseg37_updated_manifest'),
    ):
        command[command.index(flag) + 1] = value
    command[command.index('--data-root-base') + 1] = str(OUT / 'data/segmentation')
    if command[command.index('--probe-epoch-grid') + 1:command.index('--probe-seeds')] != ['20', '50']:
        raise RuntimeError('Unexpected probe epoch grid in reference invocation')
    if command[command.index('--probe-seeds') + 1:command.index('--probe-eval-every')] != ['0', '1', '2']:
        raise RuntimeError('Unexpected probe seeds in reference invocation')

    OUT.mkdir(parents=True, exist_ok=True)
    record = {
        'state': 'PREPARED', 'host': '5090-zxr', 'gpu': 0,
        'checkpoint_sha256': invocation['checkpoint']['sha256'],
        'checkpoint_step': 7999,
        'source_manifest': str(MANIFEST), 'source_manifest_sha256': MANIFEST_SHA,
        'normalized_manifest': str(normalized_path), 'normalized_manifest_sha256': normalized_sha,
        'source_manifest_protocol': source['protocol'], 'counts': counts,
        'updated_split_identity_is_distinct_from_formal': True,
        'evaluator_snapshot': str(SNAPSHOT), 'evaluator_original_snapshot': str(BASE_SNAPSHOT),
        'invocation': command,
    }
    (OUT / 'run_manifest.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    return record


def run() -> None:
    record = prepare()
    env = {
        'DINOV3_CODE_ROOT': str(SNAPSHOT), 'DINOV3_ROOT': str(SNAPSHOT),
        'PYTHONPATH': f'{SNAPSHOT}:{SNAPSHOT / "evaluation_external"}',
        'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
    }
    command = record['invocation']
    remote = 'cd ' + shlex.quote(str(SNAPSHOT)) + ' && env ' + ' '.join(
        f'{key}={shlex.quote(value)}' for key, value in env.items()) + ' ' + shlex.join(command)
    log_path = OUT / 'run.log'
    with log_path.open('w') as log:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                 '5090-zxr', remote], text=True, stdout=log,
                                stderr=subprocess.STDOUT)
    record.update(state='PROCESS_EXITED', returncode=result.returncode)
    (OUT / 'run_manifest.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    if result.returncode:
        raise SystemExit(f'Evaluation failed; see {log_path}')

    expected = {(epoch, seed) for epoch in (20, 50) for seed in (0, 1, 2)}
    found = {}
    for path in (OUT / 'results').rglob('results.json'):
        row = json.loads(path.read_text())
        meta = row.get('_meta', {})
        if meta.get('used_train_samples') == 30 and meta.get('probe_batch_size') == 32:
            key = (meta.get('probe_epochs'), meta.get('seed'))
            found[key] = (float(row['test']['mDice']), str(path))
    if set(found) != expected:
        raise RuntimeError(f'Expected six updated-manifest results; found {sorted(found)}')
    summary = {
        'status': 'VALID_COMPLETE', 'split': 'updated-manifest-train30-val7-test14',
        'manifest_sha256': MANIFEST_SHA, 'checkpoint_sha256': record['checkpoint_sha256'],
        'checkpoint_step': 7999,
        'results': [{'epochs': epoch, 'seed': seed, 'test_mDice': found[(epoch, seed)][0],
                     'path': found[(epoch, seed)][1]}
                    for epoch, seed in sorted(expected)],
    }
    summary['mean_test_mDice_by_epochs'] = {
        str(epoch): sum(found[(epoch, seed)][0] for seed in (0, 1, 2)) / 3
        for epoch in (20, 50)
    }
    (OUT / 'validation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        print(json.dumps(prepare(), indent=2, ensure_ascii=False))
    else:
        run()
