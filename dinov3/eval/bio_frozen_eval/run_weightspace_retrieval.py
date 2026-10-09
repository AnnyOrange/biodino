#!/usr/bin/env python3
"""Identity-matched v4 retrieval/clustering for weight-space arms vs the E/M/L banks.

Within-set datasets: recompute cosine leave-one-out retrieval + KMeans clustering
for every single arm from the saved banks (same functions as the coexistence
runner); E/M/L must reproduce the coexistence fusion JSON; E+L/M+L are copied.
HPA: verify gallery/query/cluster sample identity against E, copy each arm's
pinned-evaluator summary rows, and compute E+L/M+L query->gallery retrieval and
all-41 clustering with the same balanced concatenation (ge10 subset not fused).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .coexistence import balanced_concat, load_feature_bank, verify_same_samples
from .retrieval_clustering import clustering_metrics, query_gallery_metrics, remap_labels, retrieval_metrics
from .run_coexistence_retrieval import digest
from .run_weightspace_classification import numeric_equal

BASE_ROLES = {'E': '12687', 'M': '20007', 'L': '29279'}
WITHIN = {'crc-val-he-7k', 'nct-crc-he-1k', 'nct-crc-he-100', 'lc25000'}


def model(role: str) -> str:
    return f'hs6_l5_{role}_{BASE_ROLES[role]}' if role in BASE_ROLES else f'hs6_l5_{role}'


def within_set(args, specs):
    banks, files = {}, {}
    for role, root in specs.items():
        f = root / args.dataset / role / 'features' / args.dataset / f'{model(role)}.npz'
        banks[role] = load_feature_bank(f)
        files[role] = {'path': str(f), 'sha256': digest(f), 'n': int(len(banks[role].labels))}
    verify_same_samples(*banks.values())
    scores = {}
    for role, bank in banks.items():
        labels = np.asarray(bank.labels, dtype=int)
        print(f'[metrics] {args.dataset} {role} n={len(labels)}', flush=True)
        scores[role] = {**retrieval_metrics(bank.features, labels), **clustering_metrics(bank.features, labels, seed=0)}
    fusion = json.loads(args.baseline_fusion.read_text())
    for role in BASE_ROLES:
        if fusion['feature_files'][role]['sha256'] != files[role]['sha256']:
            raise ValueError(f'Baseline bank {role} differs from the coexistence fusion input')
    reproduced = {role: numeric_equal(fusion['results'][role], scores[role]) for role in BASE_ROLES}
    for arm in ('E+L', 'M+L', 'PCA(E+L)->d'):
        scores[arm] = {**fusion['results'][arm], 'copied_from': str(args.baseline_fusion)}
    status = 'VALID_COMPLETE' if all(reproduced.values()) else 'BASELINE_REPRODUCTION_MISMATCH'
    if args.dataset == 'nct-crc-he-100' and status == 'VALID_COMPLETE':
        status = 'LOW_N'
    return {'dataset': args.dataset, 'task': 'retrieval_clustering', 'protocol': 'within-set-leave-one-out',
            'n': int(len(banks['E'].labels)), 'feature_files': files, 'results': scores,
            'baseline_reproduced': reproduced, 'status': status,
            'baseline_fusion': {'path': str(args.baseline_fusion), 'sha256': digest(args.baseline_fusion)}}


def hpa(args, specs):
    parts = ('gallery', 'query', 'cluster')
    banks, files, rows = {}, {}, {}
    for role, root in specs.items():
        banks[role] = {}
        files[role] = {}
        for part in parts:
            f = root / args.dataset / role / 'features' / args.dataset / f'{model(role)}_{part}.npz'
            banks[role][part] = load_feature_bank(f)
            files[role][part] = {'path': str(f), 'sha256': digest(f), 'n': int(len(banks[role][part].labels))}
        with (root / args.dataset / role / 'summary.csv').open(newline='') as fh:
            rows[role] = [r for r in csv.DictReader(fh) if r['dataset'] == args.dataset and not r['error']]
        if len(rows[role]) != 3:
            raise ValueError(f'Expected three HPA summary rows for {role}')
    for part in parts:
        verify_same_samples(*(banks[role][part] for role in specs))
    scores = {}
    numeric = ('recall_at_1', 'recall_at_5', 'recall_at_10', 'map_at_1', 'map_at_5', 'map_at_10', 'mrr',
               'cluster_accuracy', 'ari', 'nmi', 'silhouette_cosine', 'n_gallery', 'n_query', 'n_samples', 'n_classes')
    for role in specs:
        scores[role] = {}
        for r in rows[role]:
            key = f"{r['task']}:{r['protocol']}"
            scores[role][key] = {k: float(r[k]) for k in numeric if r.get(k) not in (None, '')}
    for arm, (a, b) in {'E+L': ('E', 'L'), 'M+L': ('M', 'L')}.items():
        g = balanced_concat(banks[a]['gallery'], banks[b]['gallery'])
        q = balanced_concat(banks[a]['query'], banks[b]['query'])
        c = balanced_concat(banks[a]['cluster'], banks[b]['cluster'])
        scores[arm] = {
            'retrieval:custom-v1-same-gene-query-gallery': {
                'n_gallery': int(len(g.labels)), 'n_query': int(len(q.labels)),
                **query_gallery_metrics(g.features, np.asarray(g.labels, dtype=int), q.features,
                                        np.asarray(q.labels, dtype=int), chunk_size=256, metric_device='cpu')},
            'clustering:custom-v1-single-location-all41': {
                'n_samples': int(len(c.labels)),
                **clustering_metrics(c.features, remap_labels(np.asarray(c.labels, dtype=int)), seed=0)},
            'clustering:custom-v1-single-location-ge10-34': {'status': 'NOT_FUSED_HERE'},
            'computed_by': 'run_weightspace_retrieval (balanced_concat of verified banks)'}
    return {'dataset': args.dataset, 'task': 'retrieval_clustering', 'protocol': 'locked-query-gallery + single-location clustering',
            'feature_files': files, 'results': scores, 'status': 'VALID_COMPLETE'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=sorted(WITHIN | {'hpa-subcellular'}))
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--baseline-fusion', type=Path, default=None)
    p.add_argument('--arm-root', type=Path, required=True)
    p.add_argument('--arms', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    specs = {role: args.baseline_root for role in BASE_ROLES}
    specs.update({arm: args.arm_root for arm in args.arms})
    if args.dataset in WITHIN:
        if args.baseline_fusion is None:
            p.error('--baseline-fusion is required for within-set datasets')
        out = within_set(args, specs)
    else:
        out = hpa(args, specs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=True) + '\n')
    print(f'[done] {args.output} status={out["status"]}', flush=True)


if __name__ == '__main__':
    main()
