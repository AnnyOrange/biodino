"""Frozen-eval dataset registry.

Full table used to live here; this copy covers the D-scale 10-shot fill
datasets (nct-crc-he, chammi-allen-task1/2) with the same official splits
as the existing last_result.json files.
"""
from __future__ import annotations

from pathlib import Path

from .datasets import CHAMMIClassificationDataset, ParquetClassificationDataset

ALL_DATASETS = (
    "nct-crc-he",
    "chammi-allen-task1",
    "chammi-allen-task2",
)
CLASSIFICATION_DATASETS = ALL_DATASETS
NATIVE_TEST_SPLIT_DATASETS = frozenset(ALL_DATASETS)
UNSUPPORTED_OFFICIAL_SPLIT_DATASETS = frozenset()


def _bench(benchmark_root) -> Path:
    return Path(benchmark_root) if benchmark_root is not None else Path("/mnt/huawei_deepcad/benchmark")


def build_dataset(
    name: str,
    split: str,
    max_samples: int | None,
    max_per_class: int | None,
    benchmark_root=None,
):
    root = _bench(benchmark_root)
    if name == "nct-crc-he":
        if split == "train":
            files = sorted(
                (root / "Retrieval_Clustering/NCT-CRC-HE/1aurent_hf_parquet/data").glob(
                    "NCT_CRC_HE_100K_NONORM-*.parquet"
                )
            )
        elif split == "test":
            files = sorted(
                (root / "Retrieval_Clustering/NCT-CRC-HE/owkin_hf_parquet/data").glob("crc_val_he_7k-*.parquet")
            )
        else:
            raise ValueError(f"nct-crc-he has no split={split}")
        if not files:
            raise FileNotFoundError(f"nct-crc-he parquet missing for split={split} under {root}")
        return (
            ParquetClassificationDataset(
                files, max_samples=max_samples, max_per_class=max_per_class
            ),
            "classification",
        )
    if name in {"chammi-allen-task1", "chammi-allen-task2"}:
        test_split = "Task_one" if name.endswith("task1") else "Task_two"
        split_name = "Train" if split == "train" else test_split
        dataset = CHAMMIClassificationDataset(
            root / "Classification/CHAMMI",
            "Allen",
            split_name,
            max_samples=max_samples,
            max_per_class=max_per_class,
        )
        return dataset, "classification"
    raise KeyError(f"dataset not in this registry copy: {name}")
