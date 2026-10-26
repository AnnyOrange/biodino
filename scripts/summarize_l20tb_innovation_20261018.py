#!/usr/bin/env python3
"""Descriptive online screening against the fixed strong5TB reference.

Only complete, matching dataset inventories get an aggregate. These historical
screening test metrics do not automatically select or promote a 7B method.
"""
import json
import math
from pathlib import Path
import statistics
import time

from score_cooldown_arms_20261013 import parse_fleet

ROOT = Path('/mnt/huawei_deepcad/dinov3')
CONTROL = ROOT/'outputs/00_reports/l20tb_domain_7b_baseline_20261018'
REFERENCE = ROOT/'outputs/01_training_runs/cooldown_20261013/eval_root/5tb_peak19519/results'
FAMILIES = ('classification', 'regression', 'retrieval', 'clustering', 'segmentation')


def compare(reference, cells, complete):
    families = {}
    for family in (*FAMILIES, 'detection'):
        expected = {k for k in reference if k[0] == family}
        available = {k for k in cells if k in expected and math.isfinite(cells[k])}
        ready = bool(expected) and available == expected
        families[family] = dict(completed=len(available), expected=len(expected),
            delta_pp=100 * statistics.mean(cells[k]-reference[k] for k in expected) if ready else None)
    ready = complete and all(families[f]['delta_pp'] is not None for f in FAMILIES)
    return dict(families=families, complete=ready,
        five_family_delta_pp=statistics.mean(families[f]['delta_pp'] for f in FAMILIES) if ready else None,
        four_family_without_clustering_delta_pp=statistics.mean(
            families[f]['delta_pp'] for f in FAMILIES if f != 'clustering') if ready else None)


def summarize():
    cfg = json.loads((CONTROL/'L_eval_config.json').read_text())
    reference = parse_fleet(REFERENCE).get(19519, {})
    parsed = {}
    points = {}
    for target in cfg['targets']:
        arm, ck = target['arm'], target['checkpoint']
        root = Path(cfg['eval_root'])/arm/'results'
        if arm not in parsed:
            parsed[arm] = parse_fleet(root)
        done = len(list((root/'_state/done').glob(f'ckpt_{ck}__*.json')))
        terminal = len(list((root/'_state/terminal').glob(f'ckpt_{ck}__*.json')))
        points[f'{arm}/{ck}'] = dict(done_lanes=done, expected_lanes=11, terminal_lanes=terminal,
            **compare(reference, parsed[arm].get(ck, {}), complete=done == 11 and terminal == 0))
    report = dict(updated=time.time(), reference='5TB/19519', reference_metric_count=len(reference),
                  protocol='Historical screening45 (25 classification + 4 regression + 4 retrieval + 4 clustering + 8 segmentation); not selected29; 3 detection proxies reported separately',
                  aggregate_metric_count=45,
                  selected29_comparable=False,
                  scope_note='Existing parser summarizes four flat retrieval/clustering results; HPA and RxRx1 nested results are outside this historical 45-cell aggregate. Segmentation uses E50, whereas selected29 uses E20.',
                  promotion='No automatic promotion from screening test metrics', points=points)
    dest = CONTROL/'L_screening_summary.json'
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(report,indent=2)+'\n')
    tmp.replace(dest)
    return report


if __name__ == '__main__':
    print(json.dumps(summarize(),indent=2))
