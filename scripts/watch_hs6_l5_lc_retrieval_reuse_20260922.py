#!/usr/bin/env python3
"""Reuse identity-proven LC25000 full frozen banks for v4 within-set retrieval."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
SOURCE = ROOT / 'source_snapshot_retrieval_v3'
ROLES = {'E': 12687, 'M': 20007, 'L': 29279}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    sys.path.insert(0, str(SOURCE))
    from dinov3.eval.bio_frozen_eval.retrieval_clustering import build_retrieval_dataset

    deadline = time.monotonic() + 24 * 3600
    feature_files = {
        role: ROOT / 'frozen/lc25000' / role / 'features/lc25000' / f'hs6_l5_{role}_{step}.npz'
        for role, step in ROLES.items()
    }
    summaries = [ROOT / 'frozen/lc25000' / role / 'summary.csv' for role in ROLES]
    while time.monotonic() < deadline:
        if all(p.is_file() and p.stat().st_size > 0 for p in (*feature_files.values(), *summaries)):
            break
        time.sleep(45)
    else:
        print('TIMEOUT: LC25000 full E/M/L classification extraction not ready', flush=True)
        return 2
    if sha(ROOT / 'source_snapshot/dinov3/eval/bio_frozen_eval/encoder.py') != \
            sha(SOURCE / 'dinov3/eval/bio_frozen_eval/encoder.py'):
        raise ValueError('Frozen global encoder source differs between classification and retrieval snapshots')
    dataset, classes = build_retrieval_dataset('lc25000')
    source_paths = np.asarray([str(path) for path, _ in dataset.samples])
    source_labels = np.asarray([int(label) for _, label in dataset.samples])
    if len(source_paths) != 25000 or len(classes) != 5 or len(set(source_paths)) != 25000:
        raise ValueError('LC25000 full retrieval inventory is not 25000 unique five-class images')
    evidence = {'dataset': 'lc25000', 'protocol': 'full-within-set-leave-one-out',
                'reuse_condition': 'same source encoder, 224/256, BF16, auto channels, last1 CLS+patchmean; '
                                   'all ordered paths and labels verified',
                'source_paths_sha256': hashlib.sha256('\n'.join(source_paths).encode()).hexdigest(),
                'source_labels_sha256': hashlib.sha256(source_labels.tobytes()).hexdigest(),
                'linked_banks': {}}
    for role, file in feature_files.items():
        import csv
        with summaries[('E', 'M', 'L').index(role)].open(newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 1 or rows[0]['error'] or int(rows[0]['image_size']) != 224 or \
                int(rows[0]['resize_size']) != 256 or rows[0]['channel_policy'] != 'auto' or \
                int(rows[0]['batch_size']) != 64:
            raise ValueError(f'LC25000 original frozen geometry/precision baseline mismatch: {role}')
        with np.load(file, allow_pickle=False) as bank:
            if not np.array_equal(bank['paths'], source_paths) or \
                    not np.array_equal(bank['labels'].astype(int), source_labels) or \
                    bank['features'].shape != (25000, 2048):
                raise ValueError(f'LC25000 full retrieval vs classification bank mismatch: {role}')
        dest = ROOT / 'retrieval/lc25000' / role / 'features/lc25000' / f'hs6_l5_{role}_{ROLES[role]}.npz'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(f'Refusing to replace an existing retrieval feature: {dest}')
        dest.symlink_to(file.resolve())
        evidence['linked_banks'][role] = {'link': str(dest), 'source': str(file), 'sha256': sha(file)}
    manifest = ROOT / 'retrieval/lc25000/identity_verified_reuse.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(json.dumps(evidence, indent=2) + '\n')
    result = ROOT / 'fusion/retrieval_lc25000.json'
    if result.exists():
        raise FileExistsError(result)
    cmd = [sys.executable, '-B', '-m', 'dinov3.eval.bio_frozen_eval.run_coexistence_retrieval',
           '--dataset', 'lc25000', '--feature-root', str(ROOT / 'retrieval'), '--output', str(result)]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(SOURCE), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    print(f'[metrics] source={ROOT / "retrieval_v3_manifest.json"} provenance={manifest} command={cmd}', flush=True)
    return subprocess.call(cmd, cwd=SOURCE, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
