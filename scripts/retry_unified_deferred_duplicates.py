#!/usr/bin/env python3
"""Audit and retry local tasks whose exit 75 was only a duplicate claim."""
import csv
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/02_eval_runs/unified_paper'
REPORT = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
AUDITS = OUT / 'audits/duplicate_deferred_retry'


def dump(path, row):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(row, indent=2) + '\n')
    temporary.replace(path)


def main():
    AUDITS.mkdir(parents=True, exist_ok=True)
    retried = []
    skipped = {}
    for job in csv.DictReader((REPORT / 'execution_jobs.csv').open(newline='')):
        if (job['required_for_current_protocol'] != 'True' or
                job['execution_status'] != '阻塞' or
                job['dataset'] not in ('xray_chest', 'conic') or
                Path(job['queue']).name == 'remote_queue.json'):
            continue
        key = job['task_key']
        state = OUT / 'states' / f'{key}_attempts.json'
        audit = AUDITS / f'{key}.json'
        claim = OUT / 'locks' / f'{key}.claimdir'
        guard = OUT / 'locks' / f'execution_{key}.claimdir'
        if audit.exists() or Path(job['result']).exists() or claim.exists() or guard.exists():
            skipped[key] = 'already audited, completed, or running'
            continue
        previous = json.loads(state.read_text()) if state.exists() else {}
        log = Path(job['log'])
        marker = f'DEFER DUPLICATE TASK; no model loaded {key}'
        resumable = (key == 'cytoself__rsna_pneumonia__seed3' and
                     log.is_file() and
                     (OUT / 'results/cytoself/rsna_pneumonia/seed3/training/training_state.pth').is_file() and
                     'REAL TRAIN STEP cytoself RSNA_Pneumonia 3' in log.read_text(errors='replace'))
        if (previous.get('attempts') != 2 or previous.get('exit_code') != 75 or
                not log.is_file() or (marker not in log.read_text(errors='replace') and not resumable)):
            skipped[key] = 'not a verified duplicate-claim exit'
            continue
        dump(audit, {'task': key, 'previous_state': previous,
                     'original_log': str(log), 'reason': 'Resume same seed from retained epoch state' if resumable else marker,
                     'retried_at': time.time(), 'retry_limit': 1})
        dump(state, {**previous, 'attempts': 1, 'retry_audit': str(audit),
                     'retry_reason': 'Resume same seed from retained epoch state' if resumable else 'Duplicate claim ended before model loading; no downstream fit consumed',
                     'updated': time.time()})
        retried.append(key)
    print(json.dumps({'unblocked': retried, 'skipped': skipped}))


if __name__ == '__main__':
    main()
