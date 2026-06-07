# Frozen-feature benchmark datasets + split helpers.
#
# Vendored verbatim from the reference harness that produced the reported
# `benchmark_results_*.md` numbers (`benchmark_model/benchmark_eval/datasets.py`)
# so that `scripts/run_bio_benchmark_all.sh` reproduces those numbers exactly.
# Do NOT "improve" the split / preprocessing logic — it is the source of truth.
from __future__ import annotations

import re
import itertools
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        zero = np.zeros_like(arr[..., :1])
        arr = np.concatenate([arr, zero], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class ImageFolderDataset(Dataset):
    def __init__(self, root: str | Path, max_per_class: int | None = None, recursive: bool = False, class_names: list[str] | None = None):
        self.root = Path(root)
        classes = class_names if class_names is not None else sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        samples: list[tuple[Path, int]] = []
        for cls in classes:
            iterator = (self.root / cls).rglob("*") if recursive else (self.root / cls).iterdir()
            if max_per_class:
                files = list(itertools.islice(
                    (p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                    max_per_class,
                ))
            else:
                files = list(
                    p for p in iterator
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                )
            samples.extend((p, self.class_to_idx[cls]) for p in files)
        if not samples:
            raise ValueError(f"No image samples found under {self.root}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return load_image(path), int(label), str(path)


class NPZClassificationDataset(Dataset):
    def __init__(self, npz_path: str | Path, split: str = "train", max_samples: int | None = None):
        self.path = Path(npz_path)
        data = np.load(self.path, allow_pickle=True)
        x_key = f"{split}_images"
        y_key = f"{split}_labels"
        if x_key not in data.files or y_key not in data.files:
            raise ValueError(f"{self.path} does not contain {x_key}/{y_key}; found {data.files}")
        self.images = data[x_key]
        self.labels = data[y_key].reshape(-1)
        if max_samples:
            self.images = self.images[:max_samples]
            self.labels = self.labels[:max_samples]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        arr = _to_rgb_uint8(self.images[idx])
        return Image.fromarray(arr), int(self.labels[idx]), f"{self.path}:{idx}"


class NPZMultiLabelClassificationDataset(Dataset):
    def __init__(self, npz_path: str | Path, split: str = "train", max_samples: int | None = None):
        self.path = Path(npz_path)
        data = np.load(self.path, allow_pickle=True)
        x_key = f"{split}_images"
        y_key = f"{split}_labels"
        if x_key not in data.files or y_key not in data.files:
            raise ValueError(f"{self.path} does not contain {x_key}/{y_key}; found {data.files}")
        self.images = data[x_key]
        self.labels = data[y_key].astype(np.int64)
        if self.labels.ndim != 2:
            raise ValueError(f"{self.path} labels are not multi-label shaped: {self.labels.shape}")
        if max_samples:
            self.images = self.images[:max_samples]
            self.labels = self.labels[:max_samples]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        arr = _to_rgb_uint8(self.images[idx])
        return Image.fromarray(arr), self.labels[idx], f"{self.path}:{idx}"


class CSVImageClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        image_col: str,
        label_col: str,
        label_map: dict[str, int] | None = None,
        max_samples: int | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        samples: list[tuple[Path, int]] = []
        with self.csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                image_path = self.image_root / row[image_col]
                label_val = str(row[label_col])
                label = label_map[label_val] if label_map is not None else int(label_val)
                samples.append((image_path, int(label)))
                if max_samples and len(samples) >= max_samples:
                    break
        if not samples:
            raise ValueError(f"No image samples found from {self.csv_path}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return load_image(path), int(label), str(path)


@dataclass
class RegressionSample:
    image_path: Path
    target: float


class BBBC013RegressionDataset(Dataset):
    """BBBC013 dose regression using Channel1 BMP images and plate-map targets."""

    def __init__(
        self,
        root: str | Path,
        platemap: str = "BBBC013_v1_platemap_all.txt",
        max_samples: int | None = None,
    ):
        self.root = Path(root)
        targets = self._read_targets(self.root / platemap)
        img_dir = self.root / "BBBC013_v1_images_bmp"
        files = sorted(img_dir.glob("Channel1-*.BMP"), key=self._well_order)
        if len(files) < len(targets):
            raise ValueError(f"Found {len(files)} Channel1 BMP files but {len(targets)} targets")
        samples = [RegressionSample(p, float(t)) for p, t in zip(files, targets)]
        if max_samples:
            samples = samples[:max_samples]
        self.samples = samples

    @staticmethod
    def _read_targets(path: Path) -> list[float]:
        vals: list[float] = []
        for line in path.read_text(errors="ignore").splitlines()[1:]:
            line = line.strip()
            if line:
                vals.append(float(line))
        return vals

    @staticmethod
    def _well_order(path: Path) -> int:
        m = re.search(r"Channel1-(\d+)-", path.name)
        return int(m.group(1)) if m else 10**9

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return load_image(s.image_path), float(s.target), str(s.image_path)


class BBBC005RegressionDataset(Dataset):
    """BBBC005 synthetic cell-count regression. Cell count is encoded in the filename as
    `_C<n>_` (e.g. SIMCEPImages_A01_C10_F1_s01_w1.TIF -> 10). Uses the w1 (cell-body) channel."""

    def __init__(self, root: str | Path, channel: str = "w1", max_samples: int | None = None):
        self.root = Path(root)
        samples = []
        for p in sorted(self.root.glob(f"*_{channel}.TIF")):
            m = re.search(r"_C(\d+)_", p.name)
            if m:
                samples.append(RegressionSample(p, float(int(m.group(1)))))
        if max_samples:
            samples = samples[:max_samples]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return load_image(s.image_path), float(s.target), str(s.image_path)


def stratified_indices(labels: Iterable[int], train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(list(labels))
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for y in np.unique(labels):
        idx = np.flatnonzero(labels == y)
        rng.shuffle(idx)
        n_train = max(1, int(round(len(idx) * train_fraction)))
        if len(idx) > 1:
            n_train = min(n_train, len(idx) - 1)
        train_idx.extend(idx[:n_train].tolist())
        test_idx.extend(idx[n_train:].tolist())
    return np.asarray(train_idx), np.asarray(test_idx)


def random_indices(n: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = max(1, min(n - 1, int(round(n * train_fraction))))
    return idx[:n_train], idx[n_train:]
