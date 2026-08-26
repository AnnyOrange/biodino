from __future__ import annotations

import csv
import hashlib
import logging
import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from dinov3.utils.bio_io import read_bio_image_as_numpy

logger = logging.getLogger(__name__)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class ArrayClassificationDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: Optional[Callable] = None):
        self.images = images
        self.labels = labels.reshape(-1).astype(np.int64)
        self.transform = transform
        self.NUM_CLASSES = int(self.labels.max() + 1) if len(self.labels) else 0

    def __len__(self) -> int:
        return int(len(self.labels))

    def get_targets(self) -> np.ndarray:
        return self.labels

    def get_target(self, index: int) -> int:
        return int(self.labels[index])

    def __getitem__(self, index: int):
        image = self.images[index]
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0 if np.issubdtype(image.dtype, np.integer) else image.astype(np.float32)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return image, int(self.labels[index])


class ImagePathClassificationDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, int]], class_names: Sequence[str], transform: Optional[Callable] = None):
        self.samples = list(samples)
        self.class_names = list(class_names)
        self.transform = transform
        self.labels = np.asarray([y for _, y in self.samples], dtype=np.int64)
        self.NUM_CLASSES = len(self.class_names)

    def __len__(self) -> int:
        return len(self.samples)

    def get_targets(self) -> np.ndarray:
        return self.labels

    def get_target(self, index: int) -> int:
        return int(self.labels[index])

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = read_bio_image_as_numpy(path, target_channels=3, normalize=True)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return image, int(label)


class CappedClassificationDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)
        self.NUM_CLASSES = getattr(dataset, "NUM_CLASSES", None)
        self.class_names = getattr(dataset, "class_names", None)

    def __len__(self) -> int:
        return len(self.indices)

    def get_targets(self) -> np.ndarray:
        if hasattr(self.dataset, "get_targets"):
            return np.asarray(self.dataset.get_targets())[self.indices]
        return np.asarray([self.dataset[i][1] for i in self.indices], dtype=np.int64)

    def get_target(self, index: int) -> int:
        return int(self.get_targets()[index])

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def _stable_split_key(path: str, seed: int) -> float:
    h = hashlib.sha1((str(seed) + "|" + path).encode("utf-8")).hexdigest()[:12]
    return int(h, 16) / float(16**12 - 1)


def _split_samples(samples: Sequence[Tuple[str, int]], split: str, seed: int = 0) -> List[Tuple[str, int]]:
    split = "val" if split == "valid" else split
    buckets = {"train": [], "val": [], "test": []}
    for sample in samples:
        r = _stable_split_key(sample[0], seed)
        if r < 0.70:
            buckets["train"].append(sample)
        elif r < 0.85:
            buckets["val"].append(sample)
        else:
            buckets["test"].append(sample)
    return buckets[split]


def _cap_per_split(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dataset), size=max_samples, replace=False))
    return CappedClassificationDataset(dataset, idx.tolist())


def _collect_class_folder(root: Path, *, only_merged: bool = False) -> Tuple[List[Tuple[str, int]], List[str]]:
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    class_names = [p.name for p in class_dirs]
    samples: List[Tuple[str, int]] = []
    for label, class_dir in enumerate(class_dirs):
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            if only_merged and "_merged" not in path.name.lower():
                continue
            samples.append((str(path), label))
    if not samples:
        raise FileNotFoundError(f"No image samples found under {root}")
    return samples, class_names


def _build_bloodmnist(benchmark_root: Path, split: str, transform: Optional[Callable]) -> Dataset:
    path = benchmark_root / "Classification" / "bloodmnist_64.npz?download=1"
    if not path.is_file():
        # NTFS-backed benchmark mounts reject '?' in filenames.
        path = benchmark_root / "Classification" / "bloodmnist_64.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    split_key = "val" if split == "valid" else split
    data = np.load(path)
    return ArrayClassificationDataset(data[f"{split_key}_images"], data[f"{split_key}_labels"], transform=transform)


def _build_bbbc048(benchmark_root: Path, split: str, transform: Optional[Callable], seed: int) -> Dataset:
    root = benchmark_root / "Classification" / "BBBC048v1" / "CellCycle"
    samples, class_names = _collect_class_folder(root, only_merged=True)
    return ImagePathClassificationDataset(_split_samples(samples, split, seed), class_names, transform=transform)


def _build_cyclops(benchmark_root: Path, split: str, transform: Optional[Callable], seed: int) -> Dataset:
    root = benchmark_root / "Classification" / "cyclops-protein-loc"
    samples, class_names = _collect_class_folder(root, only_merged=False)
    return ImagePathClassificationDataset(_split_samples(samples, split, seed), class_names, transform=transform)


def _build_midog25(benchmark_root: Path, split: str, transform: Optional[Callable], seed: int) -> Dataset:
    csv_path = benchmark_root / "segmentation" / "MIDOG25_Atypical_Classification_Train_Set.csv"
    image_root = benchmark_root / "segmentation" / "MIDOG25_Binary_Classification_Train_Set"
    class_names = ["NMF", "AMF"]
    samples: List[Tuple[str, int]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image_id") or row.get("patch") or row.get("filename")
            majority = (row.get("majority") or row.get("label") or "").strip().upper()
            if not image_id or majority not in {"NMF", "AMF"}:
                continue
            path = image_root / image_id
            if path.is_file():
                samples.append((str(path), 1 if majority == "AMF" else 0))
    if not samples:
        raise FileNotFoundError(f"No MIDOG25 samples from {csv_path}")
    return ImagePathClassificationDataset(_split_samples(samples, split, seed), class_names, transform=transform)


def build_bio_classification_dataset(
    dataset: str,
    benchmark_root: str,
    split: str,
    transform: Optional[Callable] = None,
    *,
    max_samples: int = 0,
    seed: int = 0,
) -> Dataset:
    key = dataset.lower().replace("-", "_")
    root = Path(benchmark_root)
    if key == "bloodmnist":
        ds = _build_bloodmnist(root, split, transform)
    elif key in {"bbbc048", "bbbc048v1", "cellcycle"}:
        ds = _build_bbbc048(root, split, transform, seed)
    elif key in {"cyclops", "cyclops_protein_loc"}:
        ds = _build_cyclops(root, split, transform, seed)
    elif key in {"midog25", "midog"}:
        ds = _build_midog25(root, split, transform, seed)
    else:
        raise ValueError("Unknown bio classification dataset %r" % dataset)
    ds = _cap_per_split(ds, max_samples=max_samples, seed=seed)
    logger.info("Built classification dataset=%s split=%s size=%d", dataset, split, len(ds))
    return ds


SUPPORTED_BIO_CLASSIFICATION_DATASETS = ("bloodmnist", "bbbc048", "cyclops", "midog25")
