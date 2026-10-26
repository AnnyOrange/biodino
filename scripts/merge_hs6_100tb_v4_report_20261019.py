#!/usr/bin/env python3
"""Publish validated 100TB V4 checkpoint scores into the comparison report."""

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path


ROOT = Path('/mnt/huawei_deepcad/dinov3')
CONTROL = ROOT / 'outputs/02_eval_runs/hs6_100tb_fullpool_v4_every500_20261013'
FIGURE = ROOT / 'plot/fig2/route2_20tb_v4_20261008'
REPORT = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/100tb_fullpool_every500_v4_20261019'
CK7999 = REPORT / 'ck7999'
CK7999_FILES = ('status.json', 'v4_id_task_status.csv',
                'v4_id_validated_partial_cell_scores.csv', 'monuseg_updated_manifest_retest.csv')
STEM = 'route2_20tb_vs_5tb_nogram_vs_1tb_hs6_selected29_with_100tb_ck7999'
FAMILIES = {'classification', 'regression', 'retrieval', 'clustering', 'segmentation', 'overall'}


def load_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows, fields):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_text(path, content):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(content)
    os.replace(temporary, path)


def merge_ck7999():
    status = json.loads((CK7999 / 'status.json').read_text())
    if (status.get('checkpoint_step') != 7999 or status.get('shared_valid_complete') != 38
            or status.get('extension_valid_complete') != 9
            or status.get('monuseg') != 'VALID_COMPLETE'):
        raise RuntimeError('The original checkpoint 7999 report is not fully validated')
    return {name: hashlib.sha256((CK7999 / name).read_bytes()).hexdigest()
            for name in CK7999_FILES}


def refresh():
    inventory_path = CONTROL / 'checkpoint_inventory.csv'
    curve_path = FIGURE / f'{STEM}_curve.csv'
    inventory = load_csv(inventory_path)
    curve = load_csv(curve_path)
    complete = {int(row['checkpoint']) for row in inventory if row['complete'] == 'True'}
    rows = [row for row in curve if row['model'] == '100tb_fullpool_every500'
            and row['complete'] == 'True' and row['value']]
    counts = Counter(int(row['checkpoint']) for row in rows)
    if len(inventory) != 160 or set(counts) != complete or any(count != len(FAMILIES) for count in counts.values()):
        raise RuntimeError('Curve and inventory are out of sync; retry after the next summary refresh')
    if any({row['family'] for row in rows if int(row['checkpoint']) == step} != FAMILIES
           for step in complete):
        raise RuntimeError('A validated checkpoint is missing a selected-29 family score')
    for item in inventory:
        if item['complete'] == 'True' and int(item['checkpoint']) != 7999:
            status = Path(item['campaign']) / 'driver_status.json'
            if not status.is_file() or json.loads(status.read_text()).get('state') != 'COMPLETE':
                raise RuntimeError(f'Checkpoint {item["checkpoint"]} lacks a complete driver status')

    REPORT.mkdir(parents=True, exist_ok=True)
    ck7999_hashes = merge_ck7999()
    for row in rows:
        if int(row['checkpoint']) == 7999:
            row['source'] = str(CK7999 / 'v4_id_validated_partial_cell_scores.csv')
    save_csv(REPORT / 'checkpoint_inventory.csv', inventory, list(inventory[0]))
    fields = ['model', 'checkpoint', 'family', 'value', 'complete', 'source_protocol', 'source']
    save_csv(REPORT / 'selected29_complete_family_curve.csv', rows, fields)
    manifest = {
        'generated_unix': time.time(),
        'checkpoints_expected': 160,
        'checkpoints_saved': sum(row['saved'] == 'True' for row in inventory),
        'checkpoints_fully_validated': len(complete),
        'completed_steps': sorted(complete),
        'ck7999_detail': str(CK7999),
        'ck7999_file_sha256': ck7999_hashes,
        'campaign_root': str(CONTROL / 'campaigns'),
        'source_inventory': str(inventory_path),
        'source_curve': str(curve_path),
        'protocol': '38 shared ID tasks + 9 v4 extensions + MoNuSeg train30/val7/test14',
        'aggregation': 'Selected-29 dataset means per family; equal mean over five families',
        'scope': 'validated V4 components and selected-29 diagnostic; not the historical v3 shared-40 aggregate',
    }
    save_text(REPORT / 'manifest.json', json.dumps(manifest, indent=2) + '\n')
    save_text(REPORT / 'README.md',
              '# 100TB full-pool V4 checkpoint report\n\n'
              f'Fully validated checkpoints: {len(complete)}/160.\n\n'
              'Only complete checkpoints appear in `selected29_complete_family_curve.csv`. '
              'Checkpoint 7999 task details and validation evidence are consolidated in `ck7999/`. '
              'The inventory retains pending and not-yet-trained checkpoints. '
              'Each complete point requires 38 shared ID tasks, 9 V4 extensions, and '
              'MoNuSeg train30/val7/test14. Scores are selected-29 diagnostic values; '
              'they are not merged into the historical v3 shared-40 aggregate.\n\n'
              f'Source curve: `{curve_path}`.\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--interval', type=int, default=900)
    args = parser.parse_args()
    while True:
        try:
            print(json.dumps(refresh()), flush=True)
        except Exception as error:
            print(json.dumps({'time_unix': time.time(), 'error': str(error)}), flush=True)
            if not args.loop:
                raise
        if not args.loop:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
