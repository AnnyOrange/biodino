#!/usr/bin/env python3
"""Snapshot the full v4 inventory and currently available six-arm metrics."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
RULES = ROOT / 'source_snapshot' / 'Evaluation Rules/protocol_v4.json'
ARMS = ('E', 'M', 'L', 'E+L', 'M+L', 'PCA(E+L)->d')
SEGMENTATION_ARMS = {'E': 'hs6_l5_early_ampbf16_b32__best__last1__pad__s512_spformal_static_v1',
                     'M': 'hs6_l5_middle_ampbf16_b32__best__last1__pad__s512_spformal_static_v1',
                     'L': 'hs6_l5_late_ampbf16_b32__best__last1__pad__s512_spformal_static_v1',
                     'E+L': 'E+L', 'M+L': 'M+L', 'PCA(E+L)->d': 'PCA_E+L'}
STEPS = {'E': '12687', 'M': '20007', 'L': '29279'}


def family_datasets(spec, name):
    return sorted({dataset for section in ('tier_a', 'tier_b', 'union_extension')
                   for dataset in spec[section].get(name, [])})


def record(rows, family, dataset, arm, metrics=None, status='PENDING', **extra):
    rows.append({'family': family, 'dataset': dataset, 'arm': arm, 'status': status,
                 'metrics': metrics or {}, **extra})


def main():
    spec = json.loads(RULES.read_text())
    rows = []
    for family in ('classification', 'regression'):
        for dataset in family_datasets(spec, family):
            path = ROOT / 'fusion' / f'{dataset}.json'
            result = json.loads(path.read_text()) if path.exists() else None
            for arm in ARMS:
                score = result and result['results'].get(arm)
                status = ('PENDING' if score is None else score.get('status', 'COMPLETE'))
                if dataset == 'lc25000' and status == 'COMPLETE':
                    status = 'PROVISIONAL_LEGACY_ONLY'
                record(rows, family, dataset, arm, score if score and status in ('COMPLETE', 'PROVISIONAL_LEGACY_ONLY') else {},
                       status, source=str(path) if path.exists() else '')
    retrieval_datasets = family_datasets(spec, 'retrieval')
    for family in ('retrieval', 'clustering'):
        for dataset in retrieval_datasets:
            path = ROOT / 'fusion' / f'retrieval_{dataset}.json'
            result = json.loads(path.read_text()) if path.exists() else None
            for arm in ARMS:
                score = result and result['results'].get(arm)
                status = ('PENDING' if score is None else score.get('status', 'COMPLETE'))
                if dataset == 'nct-crc-he-100' and status == 'COMPLETE':
                    status = 'LOW_N'
                record(rows, family, dataset, arm, score if score and status in ('COMPLETE', 'LOW_N') else {},
                       status, source=str(path) if path.exists() else '')
    for dataset in family_datasets(spec, 'segmentation'):
        for budget in (20, 50):
            for seed in (0, 1, 2):
                for arm in ARMS:
                    if dataset == 'cellpose':
                        name = SEGMENTATION_ARMS[arm]
                        if arm in STEPS:
                            path = ROOT / 'segmentation/results' / name / f'budget{budget}/seed{seed}/cellpose' / STEPS[arm] / 'results.json'
                        else:
                            path = ROOT / 'segmentation/fused_results' / name / f'budget{budget}/seed{seed}/cellpose/results.json'
                    else:
                        path = None
                    parsed = json.loads(path.read_text()) if path and path.exists() else None
                    valid = bool(parsed and parsed.get('_meta', {}).get('test_evaluations') == 1
                                 and parsed.get('_meta', {}).get('probe_epochs') == budget
                                 and parsed.get('_meta', {}).get('seed') == seed and 'test' in parsed)
                    status = 'COMPLETE' if valid else 'BLOCKED_NOT_TESTED' if dataset == 'monuseg' else 'PENDING'
                    record(rows, 'segmentation', dataset, arm,
                           {'test_mIoU': parsed['test']['mIoU'], 'best_val_mIoU': parsed['_meta']['best_val_miou']}
                           if valid else {}, status, budget=budget, seed=seed, source=str(path) if path else '')
    for dataset in family_datasets(spec, 'detection_proxy'):
        for arm in ARMS:
            path = ROOT / 'detection' / dataset / arm / 'results_bio_detection.json'
            value = json.loads(path.read_text()) if path.exists() else None
            if value and 'test_patch_f1' in value:
                record(rows, 'detection_proxy', dataset, arm,
                       {'test_patch_f1': value['test_patch_f1'], 'val_patch_f1': value['val_patch_f1']},
                       'COMPLETE', source=str(path))
            else:
                record(rows, 'detection_proxy', dataset, arm, status='PENDING', source=str(path))
    for family in ('cell_tracking', 'ood'):
        for dataset in family_datasets(spec, family):
            for arm in ARMS:
                status = 'APPROVED_PENDING_FIXED_HEAD_AND_LINKER' if family == 'cell_tracking' else 'PENDING'
                record(rows, family, dataset, arm, status=status)
    counts = spec['expected_unique_dataset_counts']
    totals = {}
    for family, expected in counts.items():
        dataset_rows = {name: [r for r in rows if r['family'] == family and r['dataset'] == name]
                        for name in {r['dataset'] for r in rows if r['family'] == family}}
        totals[family] = {
            'v4_datasets': expected,
            'fully_reportable_six_arms': sum(bool(cells) and all(r['status'] == 'COMPLETE' for r in cells)
                                             for cells in dataset_rows.values()),
            'datasets_with_any_observation': sum(any(r['status'] in ('COMPLETE', 'LOW_N', 'PROVISIONAL_LEGACY_ONLY')
                                                     for r in cells) for cells in dataset_rows.values()),
        }
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = ROOT / 'progress_snapshots'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f'v4_inventory_{stamp}.json'
    out_csv = out_dir / f'v4_inventory_{stamp}.csv'
    if out_json.exists() or out_csv.exists():
        raise FileExistsError('Progress snapshot already exists for this UTC second')
    out_json.write_text(json.dumps({'protocol': spec['protocol_id'], 'captured_utc': stamp,
                                    'overall_v4_complete': False, 'expected': counts,
                                    'coverage': totals, 'rows': rows}, indent=2) + '\n')
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('family', 'dataset', 'arm', 'status', 'budget', 'seed', 'metrics_json', 'source'))
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row.get(k, '') for k in writer.fieldnames},
                             'metrics_json': json.dumps(row['metrics'], sort_keys=True)})
    print(f'{len(rows)} explicit status rows; saved {out_json} and {out_csv}')


if __name__ == '__main__':
    main()
