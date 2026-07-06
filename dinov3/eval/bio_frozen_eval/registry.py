# Dataset registry: name -> (Dataset, task).
#
# Vendored from `build_dataset` in `benchmark_model/run_benchmark.py` so the
# classification/regression/multilabel datasets resolve to exactly the same
# files/labels/tasks as the run that produced `benchmark_results_*.md`. The only
# change vs the reference is that the benchmark root is a parameter instead of a
# module-level constant.
from __future__ import annotations

import glob
from pathlib import Path

from .datasets import (
    BBBC005RegressionDataset,
    BBBC013RegressionDataset,
    CHAMMIClassificationDataset,
    CSVImageClassificationDataset,
    ImageFolderDataset,
    MappedImageFolderDataset,
    NPZClassificationDataset,
    NPZMultiLabelClassificationDataset,
    ParquetClassificationDataset,
)

DEFAULT_BENCHMARK_ROOT = Path("/mnt/huawei_deepcad/benchmark")

MEDMNIST_NAMES = {
    "bloodmnist", "pathmnist", "tissuemnist", "breastmnist",
    "organamnist", "organcmnist", "organsmnist", "dermamnist",
    "octmnist", "pneumoniamnist", "retinamnist", "chestmnist",
}

# Every dataset name understood by build_dataset, grouped by task. Mirrors the
# choices in the reference run_benchmark.py and is used for argparse validation.
# The original 15-dataset suite (reproduces benchmark_results_*.md byte-identically)
# plus added histopathology classification benchmarks (nct-crc-he, lc25000).
_BASE_CLASSIFICATION = {"cyclops-protein-loc", "bbbc048-cellcycle", "midog25-atypical"} | (MEDMNIST_NAMES - {"chestmnist"})
_CHAMMI_SPECS = {
    "chammi-allen-task1": ("Allen", "Task_one"),
    "chammi-allen-task2": ("Allen", "Task_two"),
    "chammi-hpa-task1": ("HPA", "Task_one"),
    "chammi-hpa-task2": ("HPA", "Task_two"),
    "chammi-hpa-task3": ("HPA", "Task_three"),
    "chammi-cp-task1": ("CP", "Task_one"),
    "chammi-cp-task2": ("CP", "Task_two"),
    "chammi-cp-task3": ("CP", "Task_three"),
    "chammi-cp-task4": ("CP", "Task_four"),
}
CHAMMI_DATASETS = set(_CHAMMI_SPECS)

_ADDED_CLASSIFICATION = {"nct-crc-he", "lc25000", "pcam"} | CHAMMI_DATASETS
CLASSIFICATION_DATASETS = sorted(_BASE_CLASSIFICATION | _ADDED_CLASSIFICATION)
MULTILABEL_DATASETS = ["chestmnist"]
REGRESSION_DATASETS = ["bbbc013", "bbbc005"]
ALL_DATASETS = sorted(set(CLASSIFICATION_DATASETS) | set(MULTILABEL_DATASETS) | set(REGRESSION_DATASETS))

# Datasets that ship a native (publication-standard) train/test split, evaluated
# with the explicit-split probes instead of an internal stratified 80/20.
# MedMNIST .npz files all carry official train/val/test arrays (chestmnist is the
# multilabel one) -> fit on official train, score on official test.
NATIVE_TEST_SPLIT_DATASETS = {"nct-crc-he", "pcam"} | MEDMNIST_NAMES | CHAMMI_DATASETS

# NCT-CRC-HE parquet locations (relative to benchmark_root):
#   train = NCT-CRC-HE-100K (NONORM, 100k tiles, 31 shards) -- the harder, stain-
#           unnormalised variant; test = CRC-VAL-HE-7K (7180 tiles, different patients).
_NCT_CRC_TRAIN_GLOB = "Retrieval_Clustering/NCT-CRC-HE/1aurent_hf_parquet/data/NCT_CRC_HE_100K_NONORM-*.parquet"
_NCT_CRC_TEST_GLOB = "Retrieval_Clustering/NCT-CRC-HE/owkin_hf_parquet/data/crc_val_he_7k-*.parquet"

# PatchCamelyon (binary metastasis classification; bool label). 1aurent/PatchCamelyon
# parquet mirror, native train/test split (262k train / 32.8k test).
_PCAM_TRAIN_GLOB = "Classification/PatchCamelyon/hf_parquet/data/train-*.parquet"
_PCAM_TEST_GLOB = "Classification/PatchCamelyon/hf_parquet/data/test-*.parquet"


def build_dataset(
    name: str,
    split: str,
    max_samples: int | None,
    max_per_class: int | None,
    benchmark_root: str | Path | None = None,
):
    """Return ``(dataset, task)`` for ``name``.

    ``task`` is one of ``classification`` / ``multilabel_classification`` /
    ``regression`` and selects the probe in :mod:`probes`.
    """
    root = Path(benchmark_root or DEFAULT_BENCHMARK_ROOT)
    if name == "cyclops-protein-loc":
        return ImageFolderDataset(root / "Classification/cyclops-protein-loc", max_per_class=max_per_class), "classification"
    if name == "bbbc048-cellcycle":
        classes = ["Anaphase", "G1", "G2", "Metaphase", "Prophase", "S", "Telophase"]
        return ImageFolderDataset(root / "Classification/BBBC048v1/CellCycle", max_per_class=max_per_class, class_names=classes), "classification"
    if name in MEDMNIST_NAMES:
        med_root = root / "Classification/MedMNIST"
        med_path = med_root / f"{name}.npz"
        if med_path.exists():
            if name == "chestmnist":
                return NPZMultiLabelClassificationDataset(med_path, split=split, max_samples=max_samples), "multilabel_classification"
            return NPZClassificationDataset(med_path, split=split, max_samples=max_samples), "classification"
        if name == "bloodmnist":
            return NPZClassificationDataset(root / "Classification/bloodmnist_64.npz?download=1", split=split, max_samples=max_samples), "classification"
    if name == "midog25-atypical":
        return CSVImageClassificationDataset(
            root / "segmentation/MIDOG25_Atypical_Classification_Train_Set.csv",
            root / "segmentation/MIDOG25_Binary_Classification_Train_Set",
            image_col="image_id",
            label_col="majority",
            label_map={"NMF": 0, "AMF": 1},
            max_samples=max_samples,
        ), "classification"
    if name == "nct-crc-he":
        pattern = _NCT_CRC_TRAIN_GLOB if split == "train" else _NCT_CRC_TEST_GLOB
        files = sorted(glob.glob(str(root / pattern)))
        if not files:
            raise FileNotFoundError(f"No NCT-CRC-HE parquet shards for split={split} under {root / pattern}")
        return ParquetClassificationDataset(
            files, max_samples=max_samples, max_per_class=max_per_class
        ), "classification"
    if name == "pcam":
        pattern = _PCAM_TRAIN_GLOB if split == "train" else _PCAM_TEST_GLOB
        files = sorted(glob.glob(str(root / pattern)))
        if not files:
            raise FileNotFoundError(f"No PatchCamelyon parquet shards for split={split} under {root / pattern}")
        return ParquetClassificationDataset(
            files, max_samples=max_samples, max_per_class=max_per_class
        ), "classification"
    if name == "lc25000":
        base = root / "Retrieval_Clustering/LC25000/images/lung_colon_image_set"
        class_dirs = {
            "colon_aca": base / "colon_image_sets/colon_aca",
            "colon_n": base / "colon_image_sets/colon_n",
            "lung_aca": base / "lung_image_sets/lung_aca",
            "lung_n": base / "lung_image_sets/lung_n",
            "lung_scc": base / "lung_image_sets/lung_scc",
        }
        return MappedImageFolderDataset(class_dirs, max_per_class=max_per_class), "classification"
    if name in _CHAMMI_SPECS:
        segment, task_split = _CHAMMI_SPECS[name]
        split_name = "Train" if split == "train" else task_split
        return CHAMMIClassificationDataset(
            root / "Classification/CHAMMI",
            segment=segment,
            split_name=split_name,
            max_samples=max_samples,
            max_per_class=max_per_class,
        ), "classification"
    if name == "bbbc013":
        return BBBC013RegressionDataset(root / "Regression/BBBC013", max_samples=max_samples), "regression"
    if name == "bbbc005":
        return BBBC005RegressionDataset(root / "Regression/BBBC005/extracted/BBBC005_v1_images", max_samples=max_samples), "regression"
    raise KeyError(f"Unknown dataset {name}")
