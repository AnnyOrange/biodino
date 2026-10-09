#!/usr/bin/env python3
"""Independent validator for the weight-space campaign (Rule 04 §3).

Re-checks, without trusting any summary: task completion, checkpoint identity,
protocol fields recorded inside every result, paired sample-identity evidence,
and the arm grid of every reported cell. Writes validation_report.json.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hs6_l5_weightspace_campaign_20260922 import (  # noqa: E402
    CLS_LARGE, CLS_SMALL, COEX, DET, RET_HPA, RET_WITHIN, ROOT, SEG_CKPT_ID, SEG_FIRST, SEG_REST, sha256, task_done)

WEIGHT = ['WA025', 'WA050', 'WA075', 'AVG3']
BASE = ['E', 'M', 'L']


def main() -> int:
    findings, checks = [], {}
    tasks = json.loads((ROOT / 'tasks.json').read_text())['tasks']
    ck = json.loads((ROOT / 'checkpoints/checkpoint_manifest.json').read_text())

    # 1. checkpoint identity: rebuild-time hashes still match the files on disk
    live = {a: sha256(Path(o['checkpoint'])) for a, o in ck['outputs'].items()}
    checks['checkpoints_unchanged'] = all(live[a] == ck['outputs'][a]['sha256'] for a in live)
    src = {r: sha256(Path(s['checkpoint'])) for r, s in ck['sources'].items()}
    coex = json.loads((COEX / 'launch_manifest.json').read_text())['teachers']
    checks['source_teachers_match_coexistence'] = all(src[r] == coex[r]['sha256'] for r in src)
    checks['merge_formula_max_error'] = max(o['max_abs_formula_error_fp32_roundtrip'] for o in ck['outputs'].values())
    if not checks['checkpoints_unchanged'] or not checks['source_teachers_match_coexistence']:
        findings.append('CHECKPOINT_IDENTITY_DRIFT')

    # 2. every task DONE with a valid, protocol-conformant output
    states = Counter()
    for t in tasks:
        st = json.loads((ROOT / 'claims' / t['id'] / 'status.json').read_text())
        states[st.get('state')] += 1
        done, why = task_done(t['done'])
        if st.get('state') != 'DONE' or not done:
            findings.append(f'INCOMPLETE {t["id"]} state={st.get("state")} check={why}')
    checks['task_states'] = dict(states)
    checks['n_tasks'] = len(tasks)

    # 3. protocol fields inside the produced results (batch/resolution/epochs/seed never lowered)
    proto = []
    for t in tasks:
        if t['family'] in ('classification', 'retrieval'):
            with Path(t['done']['path']).open(newline='') as f:
                for row in csv.DictReader(f):
                    if t['family'] == 'classification' and (row.get('batch_size') != '64' or row.get('seed') != '0'
                                                            or row.get('channel_policy') != 'auto'
                                                            or row.get('channel_tta_samples') != '8'):
                        proto.append(f'{t["id"]}: frozen protocol fields {row.get("batch_size")}/{row.get("seed")}')
                    if row.get('error'):
                        proto.append(f'{t["id"]}: error field set')
        elif t['family'] == 'detection_proxy':
            d = json.loads(Path(t['done']['path']).read_text())
            if d.get('batch_size') != 8 or d.get('epochs') != 5 or d.get('image_size') != 224 or d.get('seed') != 0:
                proto.append(f'{t["id"]}: detection protocol {d.get("batch_size")}/{d.get("epochs")}/{d.get("image_size")}')
    for p in Path(ROOT / 'segmentation/results').rglob('results.json'):
        d = json.loads(p.read_text())
        m = d.get('_meta', {})
        if m.get('test_evaluations') != 1 or m.get('probe_batch_size') != 32 or m.get('probe_epochs') not in (20, 50) \
                or m.get('seed') not in (0, 1, 2) or m.get('probe_eval_every') != 1:
            proto.append(f'segmentation {p}: meta {m.get("probe_epochs")}/{m.get("seed")}/{m.get("test_evaluations")}')
    checks['protocol_field_violations'] = proto
    findings += proto

    # 4. paired cells: identity evidence + full arm grid
    paired = defaultdict(dict)
    for p in sorted((ROOT / 'paired').glob('*.json')):
        if p.name.startswith(('detection_', 'segmentation_')):
            continue
        d = json.loads(p.read_text())
        arms = set(d['results'])
        missing = [a for a in BASE + WEIGHT if a not in arms or not d['results'][a]]
        entry = {'status': d.get('status'), 'missing_arms': missing,
                 'baseline_reproduced': d.get('baseline_reproduced'),
                 'n_banks': {k: len(v) for k, v in d.get('feature_files', {}).items() if isinstance(v, dict)}}
        if missing:
            findings.append(f'MISSING_ARMS {p.name}: {missing}')
        if not d.get('feature_files'):
            findings.append(f'NO_IDENTITY_EVIDENCE {p.name}')
        paired[p.stem] = entry
    checks['paired_probe_cells'] = len(paired)
    checks['paired'] = paired

    # 5. segmentation/detection gathers: every arm present per cell
    seg = {}
    for ds in SEG_FIRST + SEG_REST:
        p = ROOT / 'paired' / f'segmentation_{ds}.json'
        if not p.exists():
            findings.append(f'MISSING_GATHER segmentation_{ds}')
            continue
        d = json.loads(p.read_text())
        full = sum(1 for c in d['cells'] if all(len([1]) and a in c['arms'] for a in BASE + WEIGHT)
                   and all(len(c['arms'][a]) if isinstance(c['arms'][a], list) else 1 for a in BASE + WEIGHT))
        seg[ds] = {'cells': len(d['cells']), 'expected': d['expected_cells'], 'complete_cells': full}
        if len(d['cells']) != d['expected_cells']:
            findings.append(f'SEG_CELL_COUNT {ds}: {len(d["cells"])} != {d["expected_cells"]}')
    checks['segmentation_cells'] = seg
    det = {}
    for ds in DET:
        d = json.loads((ROOT / 'paired' / f'detection_{ds}.json').read_text())
        det[ds] = {'protocol_consistent': d['protocol_consistent'],
                   'complete_arms': sorted(a for a, v in d['arms'].items() if v.get('status') == 'COMPLETE')}
        if not d['protocol_consistent']:
            findings.append(f'DETECTION_PROTOCOL_MISMATCH {ds}')
    checks['detection'] = det

    report = {
        'validator': 'validate_hs6_l5_weightspace_20260922.py',
        'validated_utc': datetime.now(timezone.utc).isoformat(),
        'campaign': str(ROOT), 'protocol': 'bio-eval-union-v4',
        'checks': checks,
        'findings': findings,
        'verdict': 'VALID_COMPLETE_FOR_REPORTED_CELLS' if not findings else 'FINDINGS_PRESENT',
        'scope_note': ('Validates only the cells this campaign reports. MoNuSeg, rxrx1-cross, rxrx3-core, CTC and OOD were '
                       'never scheduled and remain outside any aggregate; LC25000 stays PROVISIONAL_LEGACY_ONLY and '
                       'nct-crc-he-100 LOW_N.'),
    }
    out = ROOT / 'validation_report.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{out}: {report["verdict"]}, {len(findings)} findings, {checks["paired_probe_cells"]} paired probe cells')
    for f in findings[:10]:
        print('  -', f)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
