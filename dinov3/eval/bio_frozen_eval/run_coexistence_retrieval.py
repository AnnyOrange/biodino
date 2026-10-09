#!/usr/bin/env python3
"""Matched retrieval/clustering v4 probes for within-set leave-one-out banks.

PCA is explicitly unavailable without a distinct frozen calibration inventory;
never fit PCA on the query/gallery or evaluation clustering samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .coexistence import balanced_concat, load_feature_bank, verify_same_samples
from .retrieval_clustering import clustering_metrics, retrieval_metrics


ROLES = {'E': '12687', 'M': '20007', 'L': '29279'}
DATASETS = {'crc-val-he-7k', 'nct-crc-he-1k', 'nct-crc-he-100', 'lc25000'}


def digest(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    p.add_argument('--feature-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    banks = {}
    files = {}
    for role, step in ROLES.items():
        f = args.feature_root / args.dataset / role / 'features' / args.dataset / f'hs6_l5_{role}_{step}.npz'
        banks[role] = load_feature_bank(f)
        files[role] = {'path': str(f), 'sha256': digest(f), 'n': len(banks[role].labels)}
    verify_same_samples(*banks.values())
    arms = {
        **banks, 'E+L': balanced_concat(banks['E'], banks['L']),
        'M+L': balanced_concat(banks['M'], banks['L']),
    }
    scores = {}
    for arm, bank in arms.items():
        labels = np.asarray(bank.labels, dtype=int)
        print(f'[metrics] {args.dataset} {arm} n={len(labels)}', flush=True)
        scores[arm] = {
            **retrieval_metrics(bank.features, labels),
            **clustering_metrics(bank.features, labels, seed=0),
        }
    scores['PCA(E+L)->d'] = {
        'status': 'PCA_UNAVAILABLE_NO_DISJOINT_CALIBRATION_BANK',
        'reason': 'Within-set leave-one-out samples are all evaluation query/gallery; fitting on them leaks.'
    }
    output = {
        'dataset': args.dataset, 'task': 'retrieval_clustering',
        'protocol': 'within-set-leave-one-out', 'n': len(banks['E'].labels),
        'feature_files': files, 'results': scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=True) + '\n')
    print(f'[done] {args.output}', flush=True)


if __name__ == '__main__':
    main()
