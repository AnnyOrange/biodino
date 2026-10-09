#!/usr/bin/env python3
"""Fix exact NCT-CRC-HE-100 identity, class support and low-N warning."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
PARQUET = Path('/mnt/huawei_deepcad/benchmark/Retrieval_Clustering/NCT-CRC-HE/owkin_hf_parquet/data/nct_crc_he_100-00000-of-00001-25a54abad9e9e379.parquet')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    result = json.loads((ROOT / 'fusion/retrieval_nct-crc-he-100.json').read_text())
    path = Path(result['feature_files']['E']['path'])
    with np.load(path, allow_pickle=False) as bank:
        labels = np.asarray(bank['labels'], dtype=int)
        paths = np.asarray(bank['paths'])
    if len(labels) != len(paths) or len(set(paths)) != len(labels) or len(labels) != result['n']:
        raise ValueError('LOW_N parquet bank missing/duplicate identities')
    support = {str(label): count for label, count in sorted(Counter(labels).items())}
    report = {
        'dataset': 'nct-crc-he-100', 'protocol': result['protocol'],
        'status': 'LOW_N_DESCRIPTIVE_ONLY', 'usable_n': len(labels),
        'n_classes': len(support), 'class_support': support,
        'self_match_rule': 'excluded by retrieval_metrics diagonal -inf',
        'parquet_path': str(PARQUET), 'parquet_sha256': sha(PARQUET),
        'feature_banks': result['feature_files'],
        'warning': 'Only 99 usable images; paired per-class uncertainty needed before treating deltas as evidence.',
    }
    output = ROOT / 'nct100_low_n_audit.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'usable_n': len(labels), 'support': support, 'audit': str(output)}))


if __name__ == '__main__':
    main()
