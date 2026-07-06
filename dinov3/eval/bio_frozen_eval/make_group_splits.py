#!/usr/bin/env python3
"""Generate and load fixed, documented, leakage-safe (group-aware) train/test splits.

For benchmark datasets with NO official test split (cyclops-protein-loc,
bbbc048-cellcycle, midog25-atypical, bbbc013, bbbc005), a random crop-level split
leaks across train/test when crops share a source. We instead split by SOURCE group
(see :mod:`group_keys`) with a fixed seed, and commit the result to
``splits/<dataset>.json``. Both the in-repo probe and the external baseline harness
read these JSONs, so BioDINO and every baseline are scored on the exact same test set.

Generate (run once, commit the JSONs)::

    python -m dinov3.eval.bio_frozen_eval.make_group_splits \
        --benchmark-root /mnt/huawei_deepcad/benchmark

Eval-time use::

    from .make_group_splits import group_split_indices
    train_idx, test_idx = group_split_indices(name, dataset, benchmark_root)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .group_keys import GROUP_KEY_RULES, GROUP_SPLIT_DATASETS, group_keys
from .registry import build_dataset

SPLIT_DIR = Path(__file__).resolve().parent / "splits"
SEED = 0
TEST_SIZE = 0.2


def _partition(groups: list[str], seed: int, test_size: float):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(np.arange(len(groups)), groups=groups))
    train_groups = {groups[i] for i in train_idx}
    test_groups = {groups[i] for i in test_idx}
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(f"group leakage across split: {sorted(overlap)[:5]}…")
    return np.sort(train_idx), np.sort(test_idx), sorted(test_groups)


def build_split(dataset_name: str, benchmark_root, seed: int = SEED, test_size: float = TEST_SIZE) -> dict:
    dataset, task = build_dataset(dataset_name, "train", None, None, benchmark_root=benchmark_root)
    groups = group_keys(dataset_name, dataset, benchmark_root)
    train_idx, test_idx, test_groups = _partition(groups, seed, test_size)
    return {
        "dataset": dataset_name,
        "task": task,
        "seed": seed,
        "test_size": test_size,
        "group_key_rule": GROUP_KEY_RULES[dataset_name],
        "n_samples": int(len(groups)),
        "n_groups": int(len(set(groups))),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_groups": test_groups,
    }


def split_path(dataset_name: str) -> Path:
    return SPLIT_DIR / f"{dataset_name}.json"


def load_split(dataset_name: str) -> dict:
    p = split_path(dataset_name)
    if not p.exists():
        raise FileNotFoundError(
            f"No committed group split at {p}. Generate it with "
            f"`python -m dinov3.eval.bio_frozen_eval.make_group_splits`."
        )
    return json.loads(p.read_text())


def group_split_indices(dataset_name: str, dataset, benchmark_root=None) -> tuple[np.ndarray, np.ndarray]:
    """Train/test indices for ``dataset`` from the committed split, by group membership.

    Recomputes group keys (deterministic) and partitions by the JSON's ``test_groups``,
    so the split is stable regardless of sample ordering across harnesses.
    """
    spec = load_split(dataset_name)
    test_groups = set(spec["test_groups"])
    groups = group_keys(dataset_name, dataset, benchmark_root)
    test_idx = np.array([i for i, g in enumerate(groups) if g in test_groups], dtype=np.int64)
    train_idx = np.array([i for i, g in enumerate(groups) if g not in test_groups], dtype=np.int64)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"{dataset_name}: empty train/test after group split ({spec['n_train']}/{spec['n_test']} expected)")
    if len(groups) == int(spec.get("n_samples", -1)):
        expected = (int(spec["n_train"]), int(spec["n_test"]))
        observed = (int(len(train_idx)), int(len(test_idx)))
        if observed != expected:
            raise ValueError(
                f"{dataset_name}: committed group split is stale: observed train/test "
                f"{observed}, expected {expected}. Regenerate splits with "
                "`python -m dinov3.eval.bio_frozen_eval.make_group_splits`."
            )
    return train_idx, test_idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    ap.add_argument("--datasets", nargs="+", default=GROUP_SPLIT_DATASETS, choices=GROUP_SPLIT_DATASETS)
    ap.add_argument("--out-dir", default=str(SPLIT_DIR))
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        spec = build_split(name, args.benchmark_root)
        (out_dir / f"{name}.json").write_text(json.dumps(spec, indent=2, sort_keys=True))
        print(
            f"[group-split] {name}: {spec['n_samples']} samples, {spec['n_groups']} groups "
            f"-> train {spec['n_train']} / test {spec['n_test']} "
            f"({len(spec['test_groups'])} test groups)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
