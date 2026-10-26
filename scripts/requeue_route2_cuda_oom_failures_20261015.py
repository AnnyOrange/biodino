#!/usr/bin/env python3
"""Archive route2 all-checkpoint FAILED claims whose log shows a retryable CUDA resource error
(OOM, busy/unavailable device, CUBLAS alloc), so the queue re-runs them. Real evaluator failures
(e.g. CTC uint track-label limit) are left untouched. Dry-run unless --apply."""
import argparse, json, re
from pathlib import Path

ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/route2_all_checkpoints_v4_20261012/extensions')
ARCHIVE = ROOT / 'claims_recovery_20261015_cuda_oom'
RETRYABLE = re.compile(r'CUDA error: out of memory|CUDA out of memory|OutOfMemoryError|device\(s\) is/are busy or unavailable|CUBLAS_STATUS_ALLOC_FAILED|No CUDA GPUs are available')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--apply', action='store_true'); args = ap.parse_args()
    n = 0; by = {}
    for status_file in (ROOT / 'claims').glob('*/status.json'):
        try: s = json.loads(status_file.read_text())
        except Exception: continue
        if s.get('state') != 'FAILED': continue
        tid = status_file.parent.name
        log = ROOT / 'logs' / f'{tid}.log'
        if not log.exists(): continue
        tail = log.read_text(errors='replace')[-20000:]
        if not RETRYABLE.search(tail): continue
        fam = tid.split('__')[0]; by[fam] = by.get(fam, 0) + 1; n += 1
        if args.apply:
            ARCHIVE.mkdir(exist_ok=True)
            dest = ARCHIVE / tid; k = 1
            while dest.exists():  # the same cell can OOM again after an earlier requeue
                dest = ARCHIVE / f'{tid}.retry{k}'; k += 1
            status_file.parent.rename(dest)
            (ARCHIVE / 'logs').mkdir(exist_ok=True)
            log.rename(ARCHIVE / 'logs' / f'{dest.name}.log')
    print(('ARCHIVED' if args.apply else 'WOULD_ARCHIVE'), n, by)

if __name__ == '__main__':
    main()
