#!/usr/bin/env python3
"""Snapshot H100 1TB Union-v4 results in the fixed selected-29 metric convention."""

import base64
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


HOST = 'suxin-8H100-1'
REMOTE = Path('/data_2/suxin/runs/HS6_L_1TB_40000_to80000_2xH100_DDP_20261019/eval/union_v4')
REPO = Path('/mnt/huawei_deepcad/dinov3')
SELECTED = REPO / 'plot/fig2/hs6_l5_5tb_selected29_gram_nogram_20260924/selected_29_datasets.csv'
OUT = REPO / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/1tb_h100_selected29_v4_20261020.json'
FAMILIES = ('classification', 'regression', 'retrieval', 'clustering', 'segmentation')
PANNUKE = ('pannuke-fold1-train-fold2-val-fold3-test',
           'pannuke-fold2-train-fold1-val-fold3-test',
           'pannuke-fold3-train-fold2-val-fold1-test')


def selected_sets():
    with SELECTED.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected = {family: sorted(row['dataset'] for row in rows if row['family'] == family)
                for family in FAMILIES}
    if [len(selected[family]) for family in FAMILIES] != [16, 2, 4, 2, 5]:
        raise ValueError('Selected-29 inventory changed')
    return selected


def remote_extract(selected):
    manifest = json.loads((REMOTE / 'manifest.json').read_text())
    points, pending = [], []
    for checkpoint in manifest['checkpoints']:
        values = defaultdict(dict)
        missing = []
        for family in FAMILIES:
            for dataset in selected[family]:
                task_family = 'retrieval_clustering' if family in ('retrieval', 'clustering') else family
                splits = PANNUKE if dataset == 'pannuke' else (None,)
                scores = []
                for split in splits:
                    task_id = task_family + '__' + dataset + ('__' + split if split else '')
                    state_file = REMOTE / f'states/ck{checkpoint}/{task_id}.json'
                    if not state_file.is_file():
                        break
                    state = json.loads(state_file.read_text())
                    if state.get('status') != 'VALID_COMPLETE':
                        break
                    output = REMOTE / f'ck{checkpoint}/{task_id}'
                    if family == 'segmentation':
                        by_seed = {}
                        for result_file in (output / 'results').rglob('results.json'):
                            parts = result_file.parts
                            if 'budget20' not in parts:
                                continue
                            result = json.loads(result_file.read_text())
                            seed = int(result['_meta']['seed'])
                            if seed in by_seed:
                                raise ValueError(f'Duplicate E20 seed: {checkpoint}/{task_id}/{seed}')
                            by_seed[seed] = float(result['test']['mDice'])
                        if set(by_seed) != {0, 1, 2}:
                            raise ValueError(f'Validated task lacks E20 seeds: {checkpoint}/{task_id}')
                        scores.append(statistics.mean(by_seed.values()))
                    else:
                        with (output / 'summary.csv').open(newline='') as stream:
                            results = [row for row in csv.DictReader(stream)
                                       if row.get('dataset') == dataset and not row.get('error')]
                        if family in ('classification', 'regression'):
                            if len(results) != 1:
                                raise ValueError(f'Expected one summary row: {checkpoint}/{task_id}')
                            field = 'macro_f1' if family == 'classification' else 'spearman'
                        else:
                            field = 'map_at_5' if family == 'retrieval' else 'nmi'
                            results = [row for row in results if
                                       row.get('task') in (family, 'retrieval_clustering') and row.get(field) and
                                       (family != 'clustering' or 'ge10' not in row.get('protocol', ''))]
                            if family == 'retrieval' and any(row.get('aggregation') == 'global' for row in results):
                                results = [row for row in results if row.get('aggregation') == 'global']
                            if len(results) != 1:
                                raise ValueError(f'Expected one {family} row: {checkpoint}/{task_id}')
                        scores.append(float(results[0][field]))
                if len(scores) != len(splits):
                    missing.append(f'{family}/{dataset}')
                else:
                    score = statistics.mean(scores)
                    if not math.isfinite(score):
                        raise ValueError(f'Nonfinite result: {checkpoint}/{family}/{dataset}')
                    values[family][dataset] = score
        if missing:
            pending.append(dict(checkpoint=checkpoint, missing=len(missing), examples=missing[:5]))
            continue
        means = {family: statistics.mean(values[family].values()) for family in FAMILIES}
        means['overall'] = statistics.mean(means.values())
        points.append(dict(checkpoint=checkpoint, values=means,
                           selected_cells={family: len(values[family]) for family in FAMILIES}))
    return dict(host=HOST, remote_root=str(REMOTE),
                protocol='Union-v4; selected-29 Macro-F1/Spearman/mAP@5/NMI/E20 mDice',
                remote_manifest_sha256=hashlib.sha256((REMOTE / 'manifest.json').read_bytes()).hexdigest(),
                source_checkpoint_15374_sha256=json.loads(
                    (REMOTE / 'checkpoints/15374/identity.json').read_text())['sha256'],
                selected29_sha256=hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
                points=points, pending=pending)


def main():
    if '--remote' in sys.argv:
        encoded = sys.argv[sys.argv.index('--remote') + 1]
        print(json.dumps(remote_extract(json.loads(base64.b64decode(encoded)))))
        return
    selected = selected_sets()
    encoded = base64.b64encode(json.dumps(selected).encode()).decode()
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST,
               f'python3 - --remote {encoded}']
    result = subprocess.run(command, input=Path(__file__).read_text(), text=True,
                            capture_output=True)
    if result.returncode:
        raise RuntimeError(f'H100 selected-29 extraction failed: {result.stderr.strip()}')
    snapshot = json.loads(result.stdout)
    snapshot['selected29_file_sha256'] = hashlib.sha256(SELECTED.read_bytes()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snapshot, indent=2) + '\n')
    print(f"H100 complete selected-29 checkpoints: {len(snapshot['points'])}; "
          f"latest: {snapshot['points'][-1]['checkpoint'] if snapshot['points'] else 'none'}")


if __name__ == '__main__':
    main()
