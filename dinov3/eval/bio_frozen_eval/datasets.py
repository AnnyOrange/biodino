# Frozen-feature benchmark datasets + split helpers.
#
# Most loaders mirror the reference harness that produced the original
# `benchmark_results_*.md` numbers. Evaluation split policy now lives in
# `run_classification.py` / `make_group_splits.py`: official test sets where
# available, committed leakage-safe group splits where needed, and the historical
# deterministic 80/20 split only as a fallback.
from __future__ import annotations

import io
import re
import itertools
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
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


def _pad_to_square(image: Image.Image) -> Image.Image:
    """Letterbox an image with its mean color so full-image targets stay visible."""
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    fill = image.resize((1, 1), resample=Image.Resampling.BOX).getpixel((0, 0))
    square = Image.new("RGB", (side, side), color=fill)
    square.paste(image, ((side - width) // 2, (side - height) // 2))
    return square


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


class MappedImageFolderDataset(Dataset):
    """Image classification from an explicit ``{class_name: directory}`` map.

    For collections whose class folders are not direct children of a single root
    (e.g. LC25000, whose ``lung_*`` and ``colon_*`` classes live under two parent
    directories). Class index = position in the sorted class-name list.
    """

    def __init__(self, class_dirs: dict[str, str | Path], max_per_class: int | None = None):
        self.classes = sorted(class_dirs.keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        samples: list[tuple[Path, int]] = []
        for cls in self.classes:
            d = Path(class_dirs[cls])
            files = [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            if max_per_class:
                files = files[:max_per_class]
            samples.extend((p, self.class_to_idx[cls]) for p in files)
        if not samples:
            raise ValueError(f"No image samples found for class dirs {class_dirs}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return load_image(path), int(label), str(path)


def _robust_normalize_channel_stack(arr: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Percentile-normalize a ``C,H,W`` microscopy stack to float32 ``[0, 1]``."""
    arr = arr.astype(np.float32, copy=False)
    flat = arr.reshape(arr.shape[0], -1)
    lo = np.percentile(flat, p_low, axis=1).astype(np.float32)
    hi = np.percentile(flat, p_high, axis=1).astype(np.float32)
    lo = lo[:, None, None]
    hi = hi[:, None, None]
    denom = hi - lo
    out = np.zeros_like(arr, dtype=np.float32)
    valid = denom > 0
    clipped = np.clip(arr, lo, hi)
    np.divide(clipped - lo, denom + 1e-8, out=out, where=valid)
    return np.clip(out, 0.0, 1.0)


def load_flattened_multichannel_image(path: str | Path, channel_width: int, p_low: float = 1.0, p_high: float = 99.0) -> torch.Tensor:
    """Load CHAMMI-style flattened channels as a true ``C,H,W`` tensor.

    CHAMMI stores channels concatenated along image width. ``channel_width`` is
    the original per-channel width from metadata, so ``W_flat / channel_width``
    recovers the channel count.
    """
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr[..., :3].mean(axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D flattened channel image for {path}, got shape={arr.shape}")
    if channel_width <= 0:
        raise ValueError(f"Invalid channel_width={channel_width} for {path}")
    height, flat_width = arr.shape
    if flat_width % channel_width != 0:
        raise ValueError(
            f"Flattened width {flat_width} is not divisible by channel_width={channel_width} for {path}"
        )
    n_channels = flat_width // channel_width
    if n_channels <= 0:
        raise ValueError(f"No channels inferred from {path}")
    stack = arr[:, : n_channels * channel_width].reshape(height, n_channels, channel_width)
    stack = np.transpose(stack, (1, 0, 2))
    stack = _robust_normalize_channel_stack(stack, p_low=p_low, p_high=p_high)
    return torch.from_numpy(np.ascontiguousarray(stack))


class CHAMMIClassificationDataset(Dataset):
    """CHAMMI classification split with true flattened-channel decoding.

    Samples are returned as ``(tensor[C,H,W], label, path)`` rather than PIL RGB
    images so dual-route/channel-adaptive backbones can consume real channels.
    """

    def __init__(
        self,
        root: str | Path,
        segment: str,
        split_name: str,
        label_col: str = "Label",
        max_samples: int | None = None,
        max_per_class: int | None = None,
        p_low: float = 1.0,
        p_high: float = 99.0,
    ):
        self.root = Path(root)
        self.segment = segment
        self.split_name = split_name
        self.label_col = label_col
        self.p_low = float(p_low)
        self.p_high = float(p_high)
        self.meta_path = self.root / segment / "enriched_meta.csv"
        if not self.meta_path.exists():
            raise FileNotFoundError(f"CHAMMI metadata not found: {self.meta_path}")

        entries: list[tuple[str, str, str, int]] = []
        labels_all: set[str] = set()
        with self.meta_path.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                rel_path = row.get("file_path", "")
                label = row.get(label_col, "")
                row_split = row.get("train_test_split", "")
                if not rel_path or not label or not row_split:
                    continue
                try:
                    channel_width = int(float(row.get("channel_width", "0")))
                except ValueError:
                    channel_width = 0
                entries.append((rel_path, str(label), str(row_split), channel_width))
                labels_all.add(str(label))
        if not entries:
            raise ValueError(f"No CHAMMI rows found in {self.meta_path}")

        self.classes = sorted(labels_all)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        samples: list[tuple[Path, int, int]] = []
        per_class: dict[int, int] = {}
        for rel_path, label, row_split, channel_width in entries:
            if row_split != split_name:
                continue
            y = self.class_to_idx[label]
            if max_per_class is not None:
                if per_class.get(y, 0) >= max_per_class:
                    continue
                per_class[y] = per_class.get(y, 0) + 1
            samples.append((self.root / rel_path, y, channel_width))
            if max_samples is not None and len(samples) >= max_samples:
                break
        if not samples:
            raise ValueError(f"No CHAMMI samples for segment={segment} split={split_name}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, channel_width = self.samples[idx]
        image = load_flattened_multichannel_image(path, channel_width, p_low=self.p_low, p_high=self.p_high)
        return image, int(label), str(path)


class CHAMMIRegressionDataset(Dataset):
    """Continuous morphology targets on an official CHAMMI held-out split."""

    def __init__(
        self,
        root: str | Path,
        segment: str,
        split_name: str,
        target_col: str,
        max_samples: int | None = None,
        target_transform: str = "log1p",
        p_low: float = 1.0,
        p_high: float = 99.0,
    ):
        self.root = Path(root)
        self.segment = segment
        self.split_name = split_name
        self.target_col = target_col
        self.target_transform = target_transform
        self.p_low = float(p_low)
        self.p_high = float(p_high)
        self.meta_path = self.root / segment / "enriched_meta.csv"
        if not self.meta_path.exists():
            raise FileNotFoundError(f"CHAMMI metadata not found: {self.meta_path}")
        if target_transform not in {"none", "log1p"}:
            raise ValueError(f"Unsupported target_transform={target_transform!r}")

        samples: list[tuple[Path, float, int]] = []
        with self.meta_path.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                if row.get("train_test_split") != split_name or not row.get("file_path"):
                    continue
                try:
                    target = float(row.get(target_col, ""))
                    channel_width = int(float(row.get("channel_width", "0")))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(target) or target < 0:
                    continue
                if target_transform == "log1p":
                    target = float(np.log1p(target))
                samples.append((self.root / row["file_path"], target, channel_width))
                if max_samples is not None and len(samples) >= max_samples:
                    break
        if not samples:
            raise ValueError(
                f"No CHAMMI regression rows for segment={segment} split={split_name} target={target_col}"
            )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target, channel_width = self.samples[idx]
        image = load_flattened_multichannel_image(
            path,
            channel_width,
            p_low=self.p_low,
            p_high=self.p_high,
        )
        return image, float(target), str(path)


class ParquetClassificationDataset(Dataset):
    """HuggingFace-style parquet shards with an ``image`` struct ``{bytes, path}``
    column and an integer ``label`` column (e.g. NCT-CRC-HE, PatchCamelyon parquet).

    Image bytes are decoded lazily one shard at a time. The benchmark DataLoader
    iterates with ``shuffle=False`` (sequential sampler), so only the current shard
    stays resident in memory.
    """

    def __init__(
        self,
        parquet_files: Iterable[str | Path],
        image_col: str = "image",
        label_col: str = "label",
        max_samples: int | None = None,
        max_per_class: int | None = None,
    ):
        import pyarrow.parquet as pq

        self.files = [Path(p) for p in sorted(parquet_files)]
        if not self.files:
            raise ValueError("ParquetClassificationDataset: no parquet files given")
        self.image_col = image_col
        self.label_col = label_col
        index: list[tuple[int, int]] = []
        labels: list[int] = []
        per_class: dict[int, int] = {}
        stop = False
        for fi, f in enumerate(self.files):
            lab = pq.read_table(f, columns=[label_col]).column(label_col).to_numpy()
            for ri, y in enumerate(lab):
                y = int(y)
                if max_per_class is not None:
                    if per_class.get(y, 0) >= max_per_class:
                        continue
                    per_class[y] = per_class.get(y, 0) + 1
                index.append((fi, ri))
                labels.append(y)
                if max_samples is not None and len(index) >= max_samples:
                    stop = True
                    break
            if stop:
                break
        self.index = index
        self.labels = np.asarray(labels, dtype=np.int64)
        self._cache_fi: int | None = None
        self._cache_bytes: list | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard(self, fi: int) -> None:
        if self._cache_fi == fi:
            return
        import pyarrow.parquet as pq

        col = pq.read_table(self.files[fi], columns=[self.image_col]).column(self.image_col)
        self._cache_bytes = [d["bytes"] for d in col.to_pylist()]
        self._cache_fi = fi

    def __getitem__(self, idx: int):
        fi, ri = self.index[idx]
        self._load_shard(fi)
        with Image.open(io.BytesIO(self._cache_bytes[ri])) as img:
            img = img.convert("RGB")
        return img, int(self.labels[idx]), f"{self.files[fi].name}:{ri}"


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


class CoNICCellCountRegressionDataset(Dataset):
    """Official central-region nuclei counts on source-image-grouped CoNIC splits."""

    def __init__(self, root: str | Path, split: str, max_samples: int | None = None):
        self.root = Path(root)
        self.images_path = self.root / "data/images.npy"
        self.images = np.load(self.images_path, mmap_mode="r")
        samples: list[tuple[int, float]] = []
        with (self.root / "conic_cell_count.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"] != split:
                    continue
                samples.append((int(row["image_index"]), float(row["cell_count"])))
                if max_samples is not None and len(samples) >= max_samples:
                    break
        if not samples:
            raise ValueError(f"No CoNIC cell-count samples for split={split}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_index, target = self.samples[idx]
        image_array = _to_rgb_uint8(self.images[image_index])
        image = Image.fromarray(image_array[16:240, 16:240])
        return image, float(target), f"{self.images_path}:{image_index}"


class LIVECellCountRegressionDataset(Dataset):
    """Full-image cell counts from official LIVECell COCO splits."""

    def __init__(self, root: str | Path, split: str, max_samples: int | None = None):
        self.root = Path(root)
        samples: list[RegressionSample] = []
        with (self.root / "livecell_cell_count.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"] != split:
                    continue
                samples.append(
                    RegressionSample(
                        self.root / "data" / row["image_path"],
                        float(row["cell_count"]),
                    )
                )
                if max_samples is not None and len(samples) >= max_samples:
                    break
        if not samples:
            raise ValueError(f"No LIVECell count samples for split={split}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = _pad_to_square(load_image(sample.image_path))
        return image, float(sample.target), str(sample.image_path)


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
