#!/usr/bin/env python3
"""Refresh provisional seed-0 native results without mixing training repeats."""
import argparse
import csv
import fcntl
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
SOURCE = REPORT / 'execution_jobs.csv'
TASKS = {
    ('cell_tracking', 'ctc'),
    ('nuclei_detection_segmentation', 'nucls_qc'),
    ('instance_segmentation', 'bbbc038-official'),
}
COLUMNS = ['dataset', 'task', 'model', 'checkpoint', 'fold', 'seed', 'metric',
           'value', 'scale', 'direction', 'status', 'reason', 'result']


def scores(task, payload):
    if task == 'cell_tracking':
        return [(payload['fold'], metric, payload['macro'][metric])
                for metric in ('TRA', 'DET', 'SEG')]
    if task == 'nuclei_detection_segmentation':
        return [(payload['fold'], metric, payload['metrics'][metric])
                for metric in ('bbox_ap_50_95', 'matched_mask_dice_pooled',
                               'detection_f1_iou50')]
    return [(stage, 'dsb_mean_precision', value['metrics']['dsb_mean_precision'])
            for stage, value in payload['stages'].items()]


def atomic_csv(path, rows, columns):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def refresh():
    source_rows = list(csv.DictReader(SOURCE.open(newline='')))
    jobs = [row for row in source_rows
            if row['required_for_current_protocol'] == 'True'
            and (row['task'], row['dataset']) in TASKS and row['seed'] == '0']
    rows = []
    counts = Counter()
    for job in jobs:
        result = Path(job['result'])
        base = {key: job[key] for key in ('dataset', 'task', 'model', 'checkpoint')}
        base.update(fold=job['fold'], seed=0, result=str(result))
        if result.is_file():
            try:
                payload = json.loads(result.read_text())
                values = scores(job['task'], payload)
            except (KeyError, ValueError, TypeError, OSError) as error:
                values = []
                reason = f'Existing result could not be parsed: {error}'
            else:
                counts[(job['dataset'], 'complete')] += 1
                for fold, metric, value in values:
                    rows.append({**base, 'fold': fold, 'metric': metric, 'value': value,
                                 'scale': '0..1', 'direction': 'higher',
                                 'status': 'seed0_complete', 'reason': ''})
                continue
        else:
            reason = job['reason']
        counts[(job['dataset'], 'incomplete')] += 1
        metric = {'ctc': 'TRA', 'nucls_qc': 'bbox_ap_50_95',
                  'bbbc038-official': 'dsb_mean_precision'}[job['dataset']]
        rows.append({**base, 'metric': metric, 'value': '', 'scale': '0..1',
                     'direction': 'higher', 'status': job['execution_status'],
                     'reason': reason})
    atomic_csv(REPORT / 'seed0_first_pass.csv', rows, COLUMNS)
    summary = [dict(dataset=dataset, task=task, required_seed0=sum(1 for j in jobs
                    if j['dataset'] == dataset), completed_seed0=counts[(dataset, 'complete')],
                    incomplete_seed0=counts[(dataset, 'incomplete')],
                    note='Provisional seed 0 only; folds remain separate; no five-seed mean or CI')
               for task, dataset in sorted(TASKS)]
    atomic_csv(REPORT / 'seed0_first_pass_summary.csv', summary,
               ['dataset', 'task', 'required_seed0', 'completed_seed0',
                'incomplete_seed0', 'note'])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    lock_path = REPORT / '.seed0_first_pass.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        while True:
            print(json.dumps({'updated': time.time(), 'summary': refresh()}), flush=True)
            if not args.watch:
                return
            time.sleep(300)


if __name__ == '__main__':
    main()
