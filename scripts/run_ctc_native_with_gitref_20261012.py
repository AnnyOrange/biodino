#!/usr/bin/env python3
"""Run the admitted native CTC evaluator with a read-only Git ref lookup."""
from __future__ import annotations

import re
import sys
from pathlib import Path

METHOD = Path('/mnt/huawei_deepcad/dinov3_method_full_v4_snapshot_20260928')
METRICS_REPO = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runtime/py-ctcmetrics')
sys.path.insert(0, str(METHOD / 'scripts'))

import run_ctc_native_full_hs6 as native


def ctcmetrics_commit() -> str:
    git_dir = METRICS_REPO / '.git'
    head = (git_dir / 'HEAD').read_text().strip()
    if head.startswith('ref: '):
        ref = head[5:]
        ref_path = git_dir / ref
        if ref_path.is_file():
            commit = ref_path.read_text().strip()
        else:
            commit = ''
            for line in (git_dir / 'packed-refs').read_text().splitlines():
                fields = line.split()
                if len(fields) == 2 and fields[1] == ref:
                    commit = fields[0]
                    break
    else:
        commit = head
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise RuntimeError(f'Invalid py-ctcmetrics Git ref: {commit!r}')
    return commit


native.base.ctcmetrics_commit = ctcmetrics_commit
from run_method_native_ctc_20260928 import main

if __name__ == '__main__':
    main()
