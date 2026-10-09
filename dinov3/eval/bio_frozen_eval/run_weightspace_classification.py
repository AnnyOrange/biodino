#!/usr/bin/env python3
"""Identity-matched v4 probes for weight-space arms against the E/M/L coexistence banks.

For one dataset this loads the coexistence E/M/L frozen banks and the
weight-space banks (WA025/WA050/WA075/AVG3), fails closed unless every bank has
identical ordered sample paths and labels, re-runs the locked v4 probe on every
single arm with exactly the split logic of run_coexistence_classification, and
requires the re-run E/M/L scores to reproduce the coexistence fusion JSON.
E+L, M+L and PCA(E+L)->d are copied from that JSON (same verified banks).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .coexistence import FeatureBank, load_feature_bank, verify_same_samples
from .datasets import random_indices, stratified_indices
from .group_keys import GROUP_SPLIT_DATASETS
from .make_group_splits import group_split_indices
from .probes import run_bbbc013_compound_oof_probe
from .registry import MEDMNIST_NAMES, NATIVE_TEST_SPLIT_DATASETS, build_dataset
from .run_classification import BEST_IMAGE_SIZE_BY_DATASET, feature_cache_stem, resolve_dataset_resize_size, split_protocol_for_dataset
from .run_coexistence_classification import _digest_file, _locked_lc_indices, _probe, _subset

BASE_ROLES = {'E': '12687', 'M': '20007', 'L': '29279'}
FUSION_ARMS = ('E+L', 'M+L', 'PCA(E+L)->d')


def bank_path(root: Path, dataset: str, role: str, model: str, split: str) -> Path:
    size = BEST_IMAGE_SIZE_BY_DATASET.get(dataset, 224)
    resize = resolve_dataset_resize_size(dataset, size, 0)
    suffix = f'_{split}' if split in {'train', 'test'} else ''
    return root / dataset / role / 'features' / dataset / f'{feature_cache_stem(model, size, resize)}{suffix}.npz'


def load_banks(specs: dict[str, tuple[Path, str]], dataset: str, split: str):
    banks, files = {}, {}
    for role, (root, model) in specs.items():
        path = bank_path(root, dataset, role, model, split)
        banks[role] = load_feature_bank(path)
        files[role] = {'path': str(path), 'sha256': _digest_file(path), 'n': int(len(banks[role].labels))}
    verify_same_samples(*banks.values())
    return banks, files


def evaluate(dataset: str, benchmark_root: Path, specs: dict[str, tuple[Path, str]]) -> dict:
    native = dataset in NATIVE_TEST_SPLIT_DATASETS
    _, task = build_dataset(dataset, 'train', None, None, benchmark_root=benchmark_root)
    if native:
        train, train_files = load_banks(specs, dataset, 'train')
        test, test_files = load_banks(specs, dataset, 'test')
        overlap = set(train['E'].paths.tolist()) & set(test['E'].paths.tolist())
        if overlap and dataset not in MEDMNIST_NAMES:
            raise ValueError(f'Official train/test sample paths overlap: {len(overlap)}')
        provenance = {'train': train_files, 'test': test_files}
    else:
        full, files = load_banks(specs, dataset, 'whole')
        provenance = {'whole': files}
        if dataset == 'lc25000':
            train_idx, test_idx, manifest_sha = _locked_lc_indices(full['E'])
            provenance['locked_lc_split'] = {'sha256': manifest_sha, 'status': 'PROVISIONAL_LEGACY_ONLY'}
        elif dataset == 'bbbc013':
            if task != 'regression':
                raise ValueError('BBBC013 must be regression')
            ds, _ = build_dataset(dataset, 'train', None, None, benchmark_root=benchmark_root)
            if not np.array_equal(np.asarray([str(s.image_path) for s in ds.samples]), full['E'].paths):
                raise ValueError('BBBC013 frozen rows do not match source sample order')
            results = {role: run_bbbc013_compound_oof_probe(b.features, b.labels, [str(p) for p in b.paths]).to_dict()
                       for role, b in full.items()}
            return {'dataset': dataset, 'task': task, 'split': split_protocol_for_dataset(dataset),
                    'feature_files': provenance, 'results': results}
        elif dataset in GROUP_SPLIT_DATASETS:
            ds, _ = build_dataset(dataset, 'train', None, None, benchmark_root=benchmark_root)
            source_paths = np.asarray([str(s.image_path if hasattr(s, 'image_path') else s[0]) for s in ds.samples])
            if not np.array_equal(source_paths, full['E'].paths):
                raise ValueError('Group split frozen rows do not match source sample order')
            train_idx, test_idx = group_split_indices(dataset, ds, benchmark_root)
            if len(ds) != len(full['E'].labels):
                raise ValueError('Group split dataset size and frozen bank differ')
        elif task == 'classification':
            train_idx, test_idx = stratified_indices(full['E'].labels.astype(int), 0.8, 0)
        else:
            train_idx, test_idx = random_indices(len(full['E'].labels), 0.8, 0)
        train = {role: _subset(b, train_idx) for role, b in full.items()}
        test = {role: _subset(b, test_idx) for role, b in full.items()}
    results = {}
    for role in specs:
        print(f'[probe] {dataset} {role} train={len(train[role].labels)} test={len(test[role].labels)}', flush=True)
        results[role] = _probe(task, train[role], test[role]).to_dict()
    return {'dataset': dataset, 'task': task,
            'split': 'PROVISIONAL_LEGACY_ONLY' if dataset == 'lc25000' else split_protocol_for_dataset(dataset),
            'feature_files': provenance, 'results': results}


def numeric_equal(a: dict, b: dict, tol: float = 1e-6) -> bool:
    """Exact reproduction up to float32 accumulation noise (silhouette differed by ~1e-7 at tol 1e-9)."""
    for key, value in a.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            other = b.get(key)
            if other is None or (isinstance(value, float) and math.isnan(value) and isinstance(other, float) and math.isnan(other)):
                continue
            if not isinstance(other, (int, float)) or abs(float(value) - float(other)) > tol * max(1.0, abs(float(value))):
                return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--baseline-root', type=Path, required=True, help='coexistence campaign frozen/ root')
    p.add_argument('--baseline-fusion', type=Path, required=True, help='coexistence fusion/<dataset>.json')
    p.add_argument('--arm-root', type=Path, required=True, help='weight-space campaign frozen/ root')
    p.add_argument('--arms', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--benchmark-root', type=Path, default=Path('/mnt/huawei_deepcad/benchmark'))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    specs = {role: (args.baseline_root, f'hs6_l5_{role}_{step}') for role, step in BASE_ROLES.items()}
    specs.update({arm: (args.arm_root, f'hs6_l5_{arm}') for arm in args.arms})
    result = evaluate(args.dataset, args.benchmark_root, specs)
    fusion = json.loads(args.baseline_fusion.read_text())
    # The fusion JSON must have been computed from the very banks we verified (sha256 per split/role).
    for split, files in result['feature_files'].items():
        if split == 'locked_lc_split':
            continue
        for role in BASE_ROLES:
            if fusion['feature_files'][split][role]['sha256'] != files[role]['sha256']:
                raise ValueError(f'Baseline bank {split}/{role} differs from the coexistence fusion input')
    reproduced = {role: numeric_equal(fusion['results'][role], result['results'][role]) for role in BASE_ROLES}
    for arm in FUSION_ARMS:
        result['results'][arm] = {**fusion['results'][arm], 'copied_from': str(args.baseline_fusion)}
    result['baseline_reproduced'] = reproduced
    result['status'] = ('VALID_COMPLETE' if all(reproduced.values()) else 'BASELINE_REPRODUCTION_MISMATCH')
    if args.dataset == 'lc25000':
        result['status'] = 'PROVISIONAL_LEGACY_ONLY' if all(reproduced.values()) else result['status']
    result['baseline_fusion'] = {'path': str(args.baseline_fusion), 'sha256': _digest_file(args.baseline_fusion)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + '\n')
    print(f'[done] {args.output} status={result["status"]}', flush=True)


if __name__ == '__main__':
    main()
