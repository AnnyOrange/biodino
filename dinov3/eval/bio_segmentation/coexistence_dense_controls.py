#!/usr/bin/env python3
"""Normalize single Cellpose teacher patch maps to match E+L/M+L fusion preprocessing."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .coexistence_dense import _read_verified, _save


def unit_spatial(features: np.ndarray, block: int = 8) -> np.ndarray:
    if features.ndim != 4:
        raise ValueError(f'Expected [N,D,H,W] patch maps, found {features.shape}')
    result = np.empty_like(features, dtype=np.float16)
    for start in range(0, len(features), block):
        end = min(start + block, len(features))
        x = features[start:end].astype(np.float32)
        norms = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
        if np.any(norms <= 1e-12):
            raise ValueError(f'Zero-length single-teacher spatial patch in images {start}:{end}')
        result[start:end] = (x / norms).astype(np.float16)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('/mnt/huawei_deepcad/benchmark/segmentation/Cellpose'))
    args = p.parse_args()
    for split in ('train', 'val', 'test'):
        sources, evidence = _read_verified(args.input_root, split, args.data_root)
        for role in 'EML':
            out = args.output_root / f'unit_{role}' / f'{split}.npz'
            if out.exists():
                if not out.with_suffix('.identity.json').exists():
                    raise FileExistsError(f'Incomplete normalised control needs audit: {out}')
                print(f'[skip] existing normalized single: {out}', flush=True)
                continue
            normalized = unit_spatial(sources[role]['features'])
            _save(out, normalized, sources[role], {**evidence, 'unit_normalized_control': role})


if __name__ == '__main__':
    main()
