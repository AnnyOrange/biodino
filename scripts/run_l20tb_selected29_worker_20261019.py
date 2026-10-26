#!/usr/bin/env python3
"""Locked selected-29 evaluator: E20, three seeds, all PanNuke rotations."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['DINOV3_CODE_ROOT'] = str(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import run_retest_fleet_20260918 as fleet


def selected_metric(family, result):
    field = {'classification': 'macro_f1', 'regression': 'spearman',
             'retrieval': 'map_at_5', 'clustering': 'nmi'}[family]
    if 'rows' in result:
        matches = [r for r in result['rows'] if r.get('task') == family
                   and (family != 'retrieval' or r.get('aggregation') == 'global')]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one global {family} result')
        result = matches[0]
    value = float(result[field])
    if not math.isfinite(value):
        raise ValueError('Nonfinite selected-29 metric')
    return value


def command(task, directory, benchmark):
    cmd = fleet.command(task, directory, benchmark)
    if task['dataset']['task'] == 'segmentation':
        start = cmd.index('--probe-epoch-grid') + 1
        end = cmd.index('--probe-seeds')
        cmd[start:end] = ['20']
    return cmd


def validate(task, directory, invocation):
    spec = task['dataset']
    if spec['task'] != 'segmentation':
        report = fleet.ORIGINAL_VALIDATE(task, directory, invocation)
        result = json.loads((directory / 'component_result.json').read_text())
        selected_metric(spec['task'], result)
        if spec['dataset'] in ('nct-crc-he-1k', 'crc-val-he-7k'):
            selected_metric('clustering', result)
        return report
    paths = sorted((directory / 'results').rglob('results.json'))
    if len(paths) != 3:
        raise ValueError('Exactly three E20 seed0/1/2 fits required per split')
    signatures = set()
    for path in paths:
        result = json.loads(path.read_text())
        meta = result['_meta']
        fleet.validate_best_head(meta)
        if meta['probe_batch_size'] != 32 or meta['probe_eval_every'] != 1:
            raise ValueError('Dense batch/validation mismatch')
        if meta['full_train_samples'] != spec['counts']['train'] or meta['used_train_samples'] != spec['counts']['train']:
            raise ValueError('Dense training count/subsampling mismatch')
        if not math.isfinite(float(result['test']['mDice'])):
            raise ValueError('Nonfinite dense primary metric')
        signatures.add((meta['probe_epochs'], meta['seed']))
    if signatures != {(20, seed) for seed in (0, 1, 2)}:
        raise ValueError('Dense budget/seed coverage mismatch')
    report = dict(status='VALID_COMPLETE', scope='SELECTED29_E20_COMPONENT',
                  full_v3_aggregate_allowed=False,
                  validator_commit=invocation['git_commit'],
                  source_snapshot_sha256=invocation['source_snapshot_sha256'],
                  input_fingerprint=fleet.queue.fingerprint(invocation),
                  expected_counts=spec['counts'],
                  result_sha256={str(p): fleet.queue.sha256(p) for p in paths})
    fleet.queue.save(directory / 'validation_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--host', required=True)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--target-per-gpu', type=int, default=1)
    parser.add_argument('--max-host-jobs', type=int, default=3)
    parser.add_argument('--max-global-jobs', type=int, default=6)
    args = parser.parse_args()
    args.task_family = 'mixed'
    fleet.queue.command = command
    fleet.queue.validate_cell = validate
    # Do not co-locate these feature-extraction jobs with a newly arrived user
    # process. The three selected A100s were idle at registration time.
    def guard(gpu, actual, task):
        used, total = fleet.queue.resources()[gpu]
        return actual == 0 and used < 1024
    args.admission_guard = guard
    fleet.queue.worker(args)


if __name__ == '__main__':
    main()
