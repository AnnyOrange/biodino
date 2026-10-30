#!/usr/bin/env python3
"""Allow one audited retry of local CTC jobs killed during tracking."""
import json
import time
from pathlib import Path


OUT = Path(__file__).resolve().parents[1] / 'outputs/02_eval_runs/unified_paper'
CASES = {
    'dinov2__ctc__fold2_seed0': -9,
    'mae__ctc__fold2_seed0': -9,
    'distillpath__ctc__fold2_seed0': -9,
    'biodino_1tb_hplus__ctc__fold2_seed0': -9,
    'biodino_1tb_splus__ctc__fold2_seed0': 75,
    'mae__ctc__fold2_seed1': -9,
    'conch__ctc__fold2_seed1': -9,
    'phikon2__ctc__fold2_seed1': -9,
    'virchow2__ctc__fold2_seed1': -9,
    'biodino_1tb_splus__ctc__fold2_seed1': -9,
    'biodino_1tb_hplus__ctc__fold2_seed1': -9,
    'biodino_20tb_early_lw5_best__ctc__fold2_seed1': -9,
    'biodino_20tb_nogram_best__ctc__fold2_seed1': -9,
    'bioclip__ctc__fold2_seed2': -9,
}


def dump(path, row):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(row, indent=2) + '\n')
    temporary.replace(path)


def main():
    audit_dir = OUT / 'audits/ctc_signal_retry'
    audit_dir.mkdir(parents=True, exist_ok=True)
    retried = []
    skipped = {}
    for key, exit_code in CASES.items():
        state = OUT / 'states' / f'{key}_attempts.json'
        audit = audit_dir / f'{key}.json'
        legacy_audit = OUT / 'audits/ctc_seed0_retry' / f'{key}.json'
        model, fit = key.split('__ctc__')
        fold, seed = fit.split('_')
        result = OUT / 'results' / model / 'ctc' / fold / seed / 'component_result.json'
        claim = OUT / 'locks' / f'{key}.claimdir'
        guard = OUT / 'locks' / f'execution_{key}.claimdir'
        if audit.exists() or legacy_audit.exists() or result.exists() or claim.exists() or guard.exists():
            skipped[key] = 'already audited, completed, or running'
            continue
        row = json.loads(state.read_text()) if state.exists() else {}
        if row.get('attempts') != 2 or row.get('exit_code') != exit_code:
            skipped[key] = 'attempt state differs'
            continue
        reason = ('duplicate execution claim; no model loaded' if exit_code == 75
                  else 'old process exited on signal 9 after partial CTC output; exact kill cause unconfirmed')
        dump(audit, {'task': key, 'previous_state': row, 'reason': reason,
                     'original_log': str(OUT / 'logs' / f'{key}.log'),
                     'retried_at': time.time(), 'retry_limit': 1})
        dump(state, {**row, 'attempts': 1, 'retry_audit': str(audit),
                     'retry_reason': reason, 'updated': time.time()})
        retried.append(key)
    print(json.dumps({'unblocked': retried, 'skipped': skipped}))


if __name__ == '__main__':
    main()
