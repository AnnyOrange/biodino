# Dataset registry: name -> (Dataset, task).
#
# Vendored from `build_dataset` in `benchmark_model/run_benchmark.py` so the
# classification/regression/multilabel datasets resolve to exactly the same
# files/labels/tasks as the run that produced `benchmark_results_*.md`. The only
# change vs the reference is that the benchmark root is a parameter instead of a
# module-level constant.
from __future__ import annotations

from pathlib import Path

from .datasets import (
    BBBC005RegressionDataset,
    BBBC013RegressionDataset,
    CSVImageClassificationDataset,
    ImageFolderDataset,
    NPZClassificationDataset,
    NPZMultiLabelClassificationDataset,
)

DEFAULT_BENCHMARK_ROOT = Path("/mnt/huawei_deepcad/benchmark")

MEDMNIST_NAMES = {
    "bloodmnist", "pathmnist", "tissuemnist", "breastmnist",
    "organamnist", "organcmnist", "organsmnist", "dermamnist",
    "octmnist", "pneumoniamnist", "retinamnist", "chestmnist",
}

# Every dataset name understood by build_dataset, grouped by task. Mirrors the
# choices in the reference run_benchmark.py and is used for argparse validation.
CLASSIFICATION_DATASETS = sorted({"cyclops-protein-loc", "bbbc048-cellcycle", "midog25-atypical"} | (MEDMNIST_NAMES - {"chestmnist"}))
MULTILABEL_DATASETS = ["chestmnist"]
REGRESSION_DATASETS = ["bbbc013", "bbbc005"]
ALL_DATASETS = sorted(set(CLASSIFICATION_DATASETS) | set(MULTILABEL_DATASETS) | set(REGRESSION_DATASETS))


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
    if name == "bbbc013":
        return BBBC013RegressionDataset(root / "Regression/BBBC013", max_samples=max_samples), "regression"
    if name == "bbbc005":
        return BBBC005RegressionDataset(root / "Regression/BBBC005/extracted/BBBC005_v1_images", max_samples=max_samples), "regression"
    raise KeyError(f"Unknown dataset {name}")
