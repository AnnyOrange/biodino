from __future__ import annotations

import csv
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Subset


Transform = Callable[[Image.Image], torch.Tensor]


def _as_float(value, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "null", "na"}:
            return default
        return float(text)
    except Exception:
        return default


def _decode_bytes(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _safe_uint8_from_array(
    arr: np.ndarray,
    percentiles: tuple[float, float],
    *,
    invert: bool = False,
) -> np.ndarray:
    """Convert microscopy arrays with arbitrary dynamic range to RGB uint8."""
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        x = np.zeros_like(x, dtype=np.float32)
        lo, hi = 0.0, 1.0
    else:
        lo, hi = np.percentile(x[finite], percentiles)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(x[finite]))
            hi = float(np.max(x[finite]))
        if hi <= lo:
            hi = lo + 1.0
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    if invert:
        x = 1.0 - x
    x = (x * 255.0 + 0.5).astype(np.uint8)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=-1)
    elif x.ndim == 3:
        if x.shape[0] in (1, 2, 3) and x.shape[-1] not in (1, 2, 3):
            x = np.moveaxis(x, 0, -1)
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        elif x.shape[-1] == 2:
            x = np.concatenate([x, x[..., -1:]], axis=-1)
        elif x.shape[-1] > 3:
            x = x[..., :3]
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape={x.shape}")
    return x


def _pil_from_uint8(arr: np.ndarray) -> Image.Image:
    arr = np.ascontiguousarray(arr)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB uint8 array, got {arr.shape}")
    return Image.fromarray(arr, mode="RGB")


def _even_slices(depth: int, count: int) -> list[int]:
    if depth <= 0:
        return []
    count = max(1, min(int(count), depth))
    if count == 1:
        return [depth // 2]
    lo = max(0, int(round(depth * 0.1)))
    hi = min(depth - 1, int(round(depth * 0.9)))
    if hi <= lo:
        lo, hi = 0, depth - 1
    return sorted(set(int(x) for x in np.linspace(lo, hi, count).round()))


def _read_xray_table(table_path: Path) -> dict[str, dict[str, str]]:
    if not table_path.exists():
        return {}
    with table_path.open(newline="") as f:
        return {str(row.get("tomoID", "")).strip(): row for row in csv.DictReader(f)}


@dataclass(frozen=True)
class XrayRecord:
    raw_path: Path
    json_path: Path
    volume_id: str
    tomo_id: str
    variant: str
    raw_shape_xyz: tuple[int, int, int]
    dose: float
    resolution: float
    sample_id: str
    resin_id: str
    epoxy_id: str
    z_index: int


class XrayTomogramSliceDataset(Dataset):
    """Downsampled WEBKNOSSOS tomograms as 2D or 2.5D slices."""

    def __init__(
        self,
        ood_root: str | Path,
        *,
        transform: Transform | None = None,
        raw_subdir: str = "raw_mag8",
        slices_per_volume: int = 8,
        input_mode: str = "three_slices",
        percentiles: tuple[float, float] = (0.5, 99.5),
        max_volumes: int | None = None,
    ):
        self.root = Path(ood_root) / "xray_brain_ultrastructure"
        self.transform = transform
        self.raw_dir = self.root / "webknossos" / raw_subdir
        self.input_mode = input_mode
        self.percentiles = percentiles
        if input_mode not in {"slice", "three_slices"}:
            raise ValueError("--xray-input-mode must be 'slice' or 'three_slices'")

        table = _read_xray_table(self.root / "metadata/github_1_dataset/tT_sorted_series_h_csv.csv")
        records: list[XrayRecord] = []
        json_paths = sorted(self.root.glob("webknossos/*.json"))
        if max_volumes is not None:
            json_paths = json_paths[: int(max_volumes)]
        for json_path in json_paths:
            with json_path.open() as f:
                meta = json.load(f)
            raw_path = self.raw_dir / f"{json_path.stem}.uint16.raw"
            if not raw_path.exists():
                continue
            shape_xyz_raw = meta.get("rawShapeXYZ") or []
            if len(shape_xyz_raw) != 3:
                continue
            shape_xyz = tuple(int(v) for v in shape_xyz_raw)
            expected = int(np.prod(shape_xyz)) * np.dtype(np.uint16).itemsize
            if raw_path.stat().st_size != expected:
                continue
            tomo_id = str(meta.get("tomoID", "")).strip()
            table_row = table.get(tomo_id, {})
            variant = str(meta.get("variant", "")).strip().lower()
            resolution_key = "resolution_nr" if variant == "nonrigid" else "resolution_r"
            depth = shape_xyz[2]
            for z_index in _even_slices(depth, slices_per_volume):
                records.append(
                    XrayRecord(
                        raw_path=raw_path,
                        json_path=json_path,
                        volume_id=json_path.stem,
                        tomo_id=tomo_id,
                        variant=variant,
                        raw_shape_xyz=shape_xyz,
                        dose=_as_float(table_row.get("dose")),
                        resolution=_as_float(table_row.get(resolution_key)),
                        sample_id=str(table_row.get("sampleID", "")),
                        resin_id=str(table_row.get("resinID", "")),
                        epoxy_id=str(table_row.get("epoxyID", "")),
                        z_index=int(z_index),
                    )
                )
        if not records:
            raise FileNotFoundError(f"No X-ray raw slices found under {self.raw_dir}")
        self.records = records
        self._memmaps: dict[Path, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _volume(self, rec: XrayRecord) -> np.memmap:
        mm = self._memmaps.get(rec.raw_path)
        if mm is None:
            x, y, z = rec.raw_shape_xyz
            mm = np.memmap(rec.raw_path, dtype=np.uint16, mode="r", shape=(z, y, x))
            self._memmaps[rec.raw_path] = mm
        return mm

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        vol = self._volume(rec)
        z = rec.z_index
        if self.input_mode == "slice":
            arr = vol[z]
        else:
            depth = vol.shape[0]
            z0 = max(0, z - 1)
            z2 = min(depth - 1, z + 1)
            arr = np.stack([vol[z0], vol[z], vol[z2]], axis=-1)
        image = _pil_from_uint8(_safe_uint8_from_array(arr, self.percentiles))
        if self.transform is not None:
            image = self.transform(image)
        meta = {
            "path": str(rec.raw_path),
            "volume_id": rec.volume_id,
            "tomo_id": rec.tomo_id,
            "variant": rec.variant,
            "dose": rec.dose,
            "resolution": rec.resolution,
            "sample_id": rec.sample_id,
            "resin_id": rec.resin_id,
            "epoxy_id": rec.epoxy_id,
            "z_index": rec.z_index,
        }
        return image, int(rec.tomo_id) if rec.tomo_id.isdigit() else idx, meta

    def meta_arrays(self) -> dict[str, np.ndarray]:
        return {
            "paths": np.asarray([str(r.raw_path) for r in self.records]),
            "volume_ids": np.asarray([r.volume_id for r in self.records]),
            "tomo_ids": np.asarray([r.tomo_id for r in self.records]),
            "variants": np.asarray([r.variant for r in self.records]),
            "doses": np.asarray([r.dose for r in self.records], dtype=np.float32),
            "resolutions": np.asarray([r.resolution for r in self.records], dtype=np.float32),
            "sample_ids": np.asarray([r.sample_id for r in self.records]),
            "resin_ids": np.asarray([r.resin_id for r in self.records]),
            "epoxy_ids": np.asarray([r.epoxy_id for r in self.records]),
            "z_indices": np.asarray([r.z_index for r in self.records], dtype=np.int32),
        }


class MRCStack:
    """Small MRC stack reader for cryoSPARC restacked particle files."""

    _DTYPES = {
        0: np.int8,
        1: np.int16,
        2: np.float32,
        6: np.uint16,
    }

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("rb") as f:
            header = f.read(1024)
        self.nx, self.ny, self.nz, self.mode = struct.unpack("<4i", header[:16])
        if self.mode not in self._DTYPES:
            raise ValueError(f"Unsupported MRC mode={self.mode} in {self.path}")
        nsymbt = struct.unpack("<i", header[92:96])[0]
        self.offset = 1024 + max(0, int(nsymbt))
        self.dtype = np.dtype(self._DTYPES[self.mode])
        self.data = np.memmap(
            self.path,
            dtype=self.dtype,
            mode="r",
            offset=self.offset,
            shape=(self.nz, self.ny, self.nx),
        )

    def read(self, index: int) -> np.ndarray:
        return np.asarray(self.data[int(index)], dtype=np.float32)


def _mrc_is_complete(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            header = f.read(1024)
        nx, ny, nz, mode = struct.unpack("<4i", header[:16])
        nsymbt = struct.unpack("<i", header[92:96])[0]
        dtype = MRCStack._DTYPES.get(mode)
        if dtype is None:
            return False
        expected = 1024 + max(0, int(nsymbt)) + int(nx) * int(ny) * int(nz) * np.dtype(dtype).itemsize
        return path.stat().st_size >= expected
    except Exception:
        return False


@dataclass(frozen=True)
class CryoRecord:
    project_id: str
    cs_path: Path
    mrc_path: Path
    particle_index: int
    class_id: int
    quality_score: float
    class_posterior: float
    ncc_score: float


def _load_scores(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    with path.open() as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}


def _sample_cryo_indices(
    labels: np.ndarray,
    *,
    max_particles: int | None,
    max_per_class: int | None,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(labels)
    if max_per_class and max_per_class > 0:
        picked: list[int] = []
        for cls in sorted(np.unique(labels).astype(int).tolist()):
            idx = np.flatnonzero(labels == cls)
            rng.shuffle(idx)
            picked.extend(idx[: int(max_per_class)].tolist())
        out = np.asarray(sorted(picked), dtype=np.int64)
        if max_particles and len(out) > max_particles:
            out = np.sort(rng.choice(out, size=int(max_particles), replace=False))
        return out
    if max_particles and n > max_particles:
        return np.sort(rng.choice(n, size=int(max_particles), replace=False))
    return np.arange(n, dtype=np.int64)


class CryoParticleDataset(Dataset):
    """CryoSPARC particle stacks from the Cryo-IEF genuine particle archives."""

    def __init__(
        self,
        ood_root: str | Path,
        *,
        transform: Transform | None = None,
        percentiles: tuple[float, float] = (0.5, 99.5),
        invert: bool = False,
        max_projects: int | None = None,
        max_particles_per_project: int | None = 20000,
        max_per_class: int | None = None,
        seed: int = 0,
    ):
        self.root = Path(ood_root) / "cryo_em_foundation_model" / "extracted"
        self.transform = transform
        self.percentiles = percentiles
        self.invert = invert
        projects = sorted(p for p in self.root.iterdir() if p.is_dir())
        if max_projects is not None:
            projects = projects[: int(max_projects)]
        records: list[CryoRecord] = []
        for project in projects:
            scores = _load_scores(project / "scores.json")
            cs_paths = sorted(project.glob("*_020_particles.cs"))
            if not cs_paths:
                continue
            complete_mrc: dict[Path, bool] = {}
            for cs_path in cs_paths:
                arr = np.load(cs_path, allow_pickle=False)
                if "alignments2D/class" in arr.dtype.names:
                    labels = arr["alignments2D/class"].astype(np.int64)
                else:
                    labels = np.zeros(len(arr), dtype=np.int64)
                chosen = _sample_cryo_indices(
                    labels,
                    max_particles=max_particles_per_project,
                    max_per_class=max_per_class,
                    seed=seed + sum(ord(c) for c in project.name),
                )
                blob_paths = arr["blob/path"] if "blob/path" in arr.dtype.names else None
                blob_indices = arr["blob/idx"] if "blob/idx" in arr.dtype.names else None
                class_post = (
                    arr["alignments2D/class_posterior"]
                    if "alignments2D/class_posterior" in arr.dtype.names
                    else np.full(len(arr), np.nan, dtype=np.float32)
                )
                ncc = (
                    arr["pick_stats/ncc_score"]
                    if "pick_stats/ncc_score" in arr.dtype.names
                    else np.full(len(arr), np.nan, dtype=np.float32)
                )
                for i in chosen:
                    raw_blob = _decode_bytes(blob_paths[i]) if blob_paths is not None else ""
                    mrc_name = Path(raw_blob).name
                    mrc_path = project / "restack" / mrc_name
                    if not mrc_path.exists():
                        alt = project / raw_blob
                        mrc_path = alt if alt.exists() else mrc_path
                    if not mrc_path.exists():
                        continue
                    if mrc_path not in complete_mrc:
                        complete_mrc[mrc_path] = _mrc_is_complete(mrc_path)
                    if not complete_mrc[mrc_path]:
                        continue
                    cls = int(labels[i])
                    records.append(
                        CryoRecord(
                            project_id=project.name,
                            cs_path=cs_path,
                            mrc_path=mrc_path,
                            particle_index=int(blob_indices[i]) if blob_indices is not None else int(i),
                            class_id=cls,
                            quality_score=float(scores.get(cls, float("nan"))),
                            class_posterior=float(class_post[i]),
                            ncc_score=float(ncc[i]),
                        )
                    )
        if not records:
            raise FileNotFoundError(f"No cryo-EM particles found under {self.root}")
        self.records = records
        self._stacks: dict[Path, MRCStack] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _stack(self, path: Path) -> MRCStack:
        stack = self._stacks.get(path)
        if stack is None:
            stack = MRCStack(path)
            self._stacks[path] = stack
        return stack

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        arr = self._stack(rec.mrc_path).read(rec.particle_index)
        image = _pil_from_uint8(
            _safe_uint8_from_array(arr, self.percentiles, invert=self.invert)
        )
        if self.transform is not None:
            image = self.transform(image)
        meta = {
            "path": str(rec.mrc_path),
            "project_id": rec.project_id,
            "cs_path": str(rec.cs_path),
            "particle_index": rec.particle_index,
            "class_id": rec.class_id,
            "quality_score": rec.quality_score,
            "class_posterior": rec.class_posterior,
            "ncc_score": rec.ncc_score,
        }
        return image, rec.class_id, meta

    def meta_arrays(self) -> dict[str, np.ndarray]:
        return {
            "paths": np.asarray([str(r.mrc_path) for r in self.records]),
            "project_ids": np.asarray([r.project_id for r in self.records]),
            "cs_paths": np.asarray([str(r.cs_path) for r in self.records]),
            "particle_indices": np.asarray([r.particle_index for r in self.records], dtype=np.int32),
            "class_ids": np.asarray([r.class_id for r in self.records], dtype=np.int32),
            "quality_scores": np.asarray([r.quality_score for r in self.records], dtype=np.float32),
            "class_posteriors": np.asarray([r.class_posterior for r in self.records], dtype=np.float32),
            "ncc_scores": np.asarray([r.ncc_score for r in self.records], dtype=np.float32),
        }


class _WrappedDataset(Dataset):
    def __init__(self, dataset: Dataset, source: str):
        self.dataset = dataset
        self.source = source

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        return image, int(label), {"path": f"{self.source}:{idx}", "source": self.source}


def _cap_dataset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dataset), size=max_samples, replace=False))
    return Subset(dataset, idx.tolist())


def build_id_reference_dataset(
    benchmark_root: str | Path,
    *,
    transform: Transform,
    dataset_names: Sequence[str] = ("bloodmnist", "bbbc048", "cyclops"),
    max_samples: int = 3000,
    seed: int = 0,
) -> Dataset:
    """Build a small in-distribution bio-image reference bank for OOD scoring."""
    from dinov3.eval.bio_classification.datasets.benchmark import build_bio_classification_dataset

    parts: list[Dataset] = []
    per_dataset = max(1, int(math.ceil(max_samples / max(1, len(dataset_names)))))
    for offset, name in enumerate(dataset_names):
        ds = build_bio_classification_dataset(
            name,
            str(benchmark_root),
            "train",
            transform=transform,
            max_samples=0,
            seed=seed,
        )
        ds = _cap_dataset(ds, per_dataset, seed + offset)
        parts.append(_WrappedDataset(ds, name))
    return ConcatDataset(parts)


def label_encode(values: Iterable[str]) -> np.ndarray:
    mapping: dict[str, int] = {}
    labels: list[int] = []
    for value in values:
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        labels.append(mapping[key])
    return np.asarray(labels, dtype=np.int64)
