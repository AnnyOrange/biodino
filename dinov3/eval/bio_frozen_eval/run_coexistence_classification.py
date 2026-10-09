#!/usr/bin/env python3
"""Evaluate six identity-matched v4 frozen-probe arms from three saved banks.

This runner intentionally does not extract features or alter v4 split manifests.
Run the matching original checkpoint extractor with --save-paths first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .coexistence import FeatureBank, balanced_concat, fit_train_only_pca, load_feature_bank, verify_same_samples
from .datasets import random_indices, stratified_indices
from .group_keys import GROUP_SPLIT_DATASETS
from .make_group_splits import group_split_indices
from .probes import (
    run_bbbc013_compound_oof_probe,
    run_classification_probe_split,
    run_multilabel_classification_probe_split,
    run_regression_probe_split,
)
from .registry import MEDMNIST_NAMES, NATIVE_TEST_SPLIT_DATASETS, build_dataset
from .run_classification import (
    BEST_IMAGE_SIZE_BY_DATASET,
    feature_cache_stem,
    resolve_dataset_resize_size,
    split_protocol_for_dataset,
)


ROLES = {"E": "12687", "M": "20007", "L": "29279"}
LC_SPLIT = Path(
    '/mnt/huawei_deepcad/benchmark_model/benchmark_runs/'
    'hs6_5tb_protocol_union_nonseg_20260921/locked_inputs/lc25000_classification_split.json'
)


def _digest_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _subset(bank: FeatureBank, indices: np.ndarray) -> FeatureBank:
    return FeatureBank(bank.features[indices], bank.labels[indices], bank.paths[indices])


def _input_banks(root: Path, dataset: str, split: str) -> tuple[dict[str, FeatureBank], dict[str, dict]]:
    size = BEST_IMAGE_SIZE_BY_DATASET.get(dataset, 224)
    resize = resolve_dataset_resize_size(dataset, size, 0)
    banks: dict[str, FeatureBank] = {}
    provenance: dict[str, dict] = {}
    for role, update in ROLES.items():
        stem = feature_cache_stem(f"hs6_l5_{role}_{update}", size, resize)
        suffix = f"_{split}" if split in {"train", "test"} else ""
        path = root / dataset / role / "features" / dataset / f"{stem}{suffix}.npz"
        banks[role] = load_feature_bank(path)
        provenance[role] = {"path": str(path), "sha256": _digest_file(path), "n": len(banks[role].labels)}
    verify_same_samples(*banks.values())
    return banks, provenance


def _probe(task: str, train: FeatureBank, test: FeatureBank):
    if task == "classification":
        return run_classification_probe_split(train.features, train.labels, test.features, test.labels)
    if task == "multilabel_classification":
        return run_multilabel_classification_probe_split(train.features, train.labels, test.features, test.labels)
    if task == "regression":
        return run_regression_probe_split(train.features, train.labels, test.features, test.labels)
    raise ValueError(f"Unknown task {task}")


def _arms(banks: dict[str, FeatureBank]) -> dict[str, FeatureBank]:
    return {
        **banks,
        "E+L": balanced_concat(banks["E"], banks["L"]),
        "M+L": balanced_concat(banks["M"], banks["L"]),
    }


def _locked_lc_indices(bank: FeatureBank) -> tuple[np.ndarray, np.ndarray, str]:
    spec = json.loads(LC_SPLIT.read_text())
    if spec.get('protocol') != 'legacy-stratified-80-20-seed0':
        raise ValueError('Unexpected LC25000 provisional locked split protocol')
    lookup = {str(path): i for i, path in enumerate(bank.paths)}
    if len(lookup) != len(bank.paths):
        raise ValueError('Duplicated LC25000 feature paths')
    subsets = []
    for split in ('train', 'test'):
        rows = spec[split]
        indices = np.asarray([lookup[str(path)] for path, _ in rows], dtype=np.int64)
        if not np.array_equal(bank.labels[indices].astype(int), np.asarray([int(label) for _, label in rows])):
            raise ValueError(f'LC25000 locked {split} labels disagree with frozen feature bank')
        subsets.append(indices)
    if len(subsets[0]) != 20000 or len(subsets[1]) != 5000 or \
            len(set(subsets[0]) & set(subsets[1])) or len(set(np.r_[*subsets])) != len(bank.paths):
        raise ValueError('LC25000 locked split has missing, extra or overlapping samples')
    return subsets[0], subsets[1], _digest_file(LC_SPLIT)


def evaluate(root: Path, dataset: str, benchmark_root: Path) -> dict:
    native = dataset in NATIVE_TEST_SPLIT_DATASETS
    _, task = build_dataset(dataset, "train", None, None, benchmark_root=benchmark_root)
    if native:
        full_train, train_files = _input_banks(root, dataset, "train")
        full_test, test_files = _input_banks(root, dataset, "test")
        # These are different official partitions. Verify no sample identity overlaps.
        overlap = set(full_train["E"].paths.tolist()) & set(full_test["E"].paths.tolist())
        # MedMNIST NPZ loaders call both split-specific arrays "file.npz:idx";
        # the array key (train_images/test_images) is part of the sample identity.
        if overlap and dataset not in MEDMNIST_NAMES:
            raise ValueError(f"Official train/test sample paths overlap: {len(overlap)}")
        provenance = {"train": train_files, "test": test_files}
    else:
        full, files = _input_banks(root, dataset, "whole")
        provenance = {"whole": files}
        if dataset == "lc25000":
            train_idx, test_idx, manifest_sha = _locked_lc_indices(full['E'])
            provenance['locked_lc_split'] = {'path': str(LC_SPLIT), 'sha256': manifest_sha,
                                              'status': 'PROVISIONAL_LEGACY_ONLY'}
        elif dataset == "bbbc013":
            if task != "regression":
                raise ValueError("BBBC013 must be regression")
            ds, _ = build_dataset(dataset, "train", None, None, benchmark_root=benchmark_root)
            if not np.array_equal(np.asarray([str(s.image_path) for s in ds.samples]), full["E"].paths):
                raise ValueError("BBBC013 frozen rows do not match source sample order")
            results = {}
            for arm, bank in _arms(full).items():
                results[arm] = run_bbbc013_compound_oof_probe(
                    bank.features, bank.labels, [str(p) for p in bank.paths]
                ).to_dict()
            results["PCA(E+L)->d"] = {
                "status": "PCA_UNAVAILABLE_INSUFFICIENT_TRAIN_RANK",
                "reason": "BBBC013 OOF fits only 36 train samples per fold, fewer than d=2048",
            }
            return {"dataset": dataset, "task": task, "split": split_protocol_for_dataset(dataset),
                    "feature_files": provenance, "results": results}
        elif dataset in GROUP_SPLIT_DATASETS:
            ds, _ = build_dataset(dataset, "train", None, None, benchmark_root=benchmark_root)
            source_paths = np.asarray([
                str(s.image_path if hasattr(s, "image_path") else s[0]) for s in ds.samples
            ])
            if not np.array_equal(source_paths, full["E"].paths):
                raise ValueError("Group split frozen rows do not match source sample order")
            train_idx, test_idx = group_split_indices(dataset, ds, benchmark_root)
            if len(ds) != len(full["E"].labels):
                raise ValueError("Group split dataset size and frozen bank differ")
        elif dataset != 'lc25000' and task == "classification":
            train_idx, test_idx = stratified_indices(full["E"].labels.astype(int), 0.8, 0)
        elif dataset != 'lc25000':
            train_idx, test_idx = random_indices(len(full["E"].labels), 0.8, 0)
        full_train = {role: _subset(bank, train_idx) for role, bank in full.items()}
        full_test = {role: _subset(bank, test_idx) for role, bank in full.items()}

    train_arms = _arms(full_train)
    test_arms = _arms(full_test)
    results: dict[str, dict] = {}
    for arm in train_arms:
        print(f"[probe] {dataset} {arm} train={len(train_arms[arm].labels)} test={len(test_arms[arm].labels)}", flush=True)
        results[arm] = _probe(task, train_arms[arm], test_arms[arm]).to_dict()
    d = int(full_train["E"].features.shape[1])
    if len(train_arms["E+L"].labels) < d:
        results["PCA(E+L)->d"] = {
            "status": "PCA_UNAVAILABLE_INSUFFICIENT_TRAIN_RANK", "train_n": len(train_arms["E+L"].labels), "d": d
        }
    else:
        print(f"[pca] {dataset} train-only d={d}", flush=True)
        train_pca, test_pca = fit_train_only_pca(train_arms["E+L"], test_arms["E+L"], dimension=d)
        results["PCA(E+L)->d"] = _probe(task, train_pca, test_pca).to_dict()
    return {"dataset": dataset, "task": task, "split": (
                'PROVISIONAL_LEGACY_ONLY' if dataset == 'lc25000' else split_protocol_for_dataset(dataset)),
            "feature_files": provenance, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, default=Path("/mnt/huawei_deepcad/benchmark"))
    args = parser.parse_args()
    result = evaluate(args.feature_root, args.dataset, args.benchmark_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace an existing evaluation: {args.output}")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(f"[done] {args.output}", flush=True)


if __name__ == "__main__":
    main()
