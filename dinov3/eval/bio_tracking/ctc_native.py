"""Native 2-D/3-D Cell Tracking Challenge data adapter.

The CTC archives mix planar masks, complete multi-page volumes, and sparse
``man_seg_TIME_Z.tif`` annotations.  This module normalizes those layouts into
one deterministic slice-wise training contract while retaining native volumes
for inference and official scoring.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from dinov3.eval.bio_segmentation.constants import MICRO_RGB_MEAN, MICRO_RGB_STD
from dinov3.eval.bio_segmentation.instance_seg.targets import make_targets
from dinov3.eval.bio_tracking.ctc_2d import atomic_json, archive_signature, sha256


PROTOCOL_ID = "ctc-labelled-training-sequence-domain-heldout-v1"
CTC_DOMAINS = (
    "BF-C2DL-HSC",
    "BF-C2DL-MuSC",
    "DIC-C2DH-HeLa",
    "Fluo-C2DL-Huh7",
    "Fluo-C2DL-MSC",
    "Fluo-C3DH-A549",
    "Fluo-C3DH-A549-SIM",
    "Fluo-C3DH-H157",
    "Fluo-C3DL-MDA231",
    "Fluo-N2DH-GOWT1",
    "Fluo-N2DH-SIM+",
    "Fluo-N2DL-HeLa",
    "Fluo-N3DH-CE",
    "Fluo-N3DH-CHO",
    "Fluo-N3DH-SIM+",
    "Fluo-N3DL-DRO",
    "Fluo-N3DL-TRIC",
    "Fluo-N3DL-TRIF",
    "PhC-C2DH-U373",
    "PhC-C2DL-PSC",
)


def _raw_map(names: list[str], prefix: str) -> dict[int, str]:
    result: dict[int, str] = {}
    expression = re.compile(r"t(\d+)\.tif")
    for name in names:
        if not name.startswith(prefix):
            continue
        match = expression.fullmatch(Path(name).name)
        if match:
            result[int(match.group(1))] = name
    return result


def _volume_mask_map(names: list[str], prefix: str, stem: str) -> dict[int, str]:
    result: dict[int, str] = {}
    expression = re.compile(rf"{re.escape(stem)}(\d+)\.tif")
    for name in names:
        if not name.startswith(prefix):
            continue
        match = expression.fullmatch(Path(name).name)
        if match:
            result[int(match.group(1))] = name
    return result


def _seg_records(names: list[str], prefix: str) -> list[dict[str, Any]]:
    full = re.compile(r"man_seg(\d+)\.tif")
    sparse = re.compile(r"man_seg_(\d+)_(\d+)\.tif")
    records = []
    for name in names:
        if not name.startswith(prefix):
            continue
        base = Path(name).name
        match = full.fullmatch(base)
        if match:
            records.append({"frame": int(match.group(1)), "z": None, "member": name})
            continue
        match = sparse.fullmatch(base)
        if match:
            records.append({
                "frame": int(match.group(1)),
                "z": int(match.group(2)),
                "member": name,
            })
    return sorted(records, key=lambda item: (item["frame"], -1 if item["z"] is None else item["z"]))


def _extract_member(
    archive: zipfile.ZipFile,
    member: str,
    destination: Path,
    file_records: list[dict[str, Any]],
) -> None:
    info = archive.getinfo(member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest: str | None = None
    if not destination.is_file() or destination.stat().st_size != info.file_size:
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        hasher = hashlib.sha256()
        with archive.open(info) as source, temporary.open("wb") as target:
            while chunk := source.read(8 << 20):
                target.write(chunk)
                hasher.update(chunk)
        if temporary.stat().st_size != info.file_size:
            raise RuntimeError(f"incomplete extraction for {member}")
        digest = hasher.hexdigest()
        os.replace(temporary, destination)
    if digest is None:
        digest = sha256(destination)
    file_records.append({
        "path": str(destination),
        "source_member": member,
        "bytes": info.file_size,
        "crc32": f"{info.CRC:08x}",
        "sha256": digest,
    })


def _folds(split_manifest: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in split_manifest.read_text().splitlines() if line.strip()]
    folds = []
    held_out: list[str] = []
    for fold in range(5):
        current = [row for row in rows if int(row["fold"]) == fold]
        train = sorted({row["domain"] for row in current if row["role"] == "head_train"})
        test = sorted({row["domain"] for row in current if row["role"] == "test"})
        if set(train) & set(test) or set(train) | set(test) != set(CTC_DOMAINS):
            raise RuntimeError(f"invalid CTC fold {fold}")
        if len(train) != 16 or len(test) != 4:
            raise RuntimeError(f"CTC fold {fold} must be 16 train / 4 test domains")
        held_out.extend(test)
        folds.append({"fold": fold, "train_domains": train, "test_domains": test})
    if sorted(held_out) != sorted(CTC_DOMAINS):
        raise RuntimeError("every CTC domain must be held out exactly once")
    return folds


def _valid_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
        if payload.get("status") != "READY" or payload.get("protocol_id") != PROTOCOL_ID:
            return None
        for record in payload["files"]:
            candidate = Path(record["path"])
            if not candidate.is_file() or candidate.stat().st_size != int(record["bytes"]):
                return None
        return payload
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def prepare_cache(cache_root: Path, archives_root: Path, split_manifest: Path) -> Path:
    """Extract seq01 annotations/raw pairs and the labelled part of seq02."""
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "data_manifest.json"
    with (cache_root / ".prepare.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _valid_manifest(manifest_path) is not None:
            return manifest_path
        files: list[dict[str, Any]] = []
        domains: dict[str, Any] = {}
        for domain_index, domain in enumerate(CTC_DOMAINS, start=1):
            archive_path = archives_root / f"{domain}.zip"
            if not archive_path.is_file():
                raise FileNotFoundError(archive_path)
            print(f"[data {domain_index}/{len(CTC_DOMAINS)}] {domain}", flush=True)
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                raw01 = _raw_map(names, f"{domain}/01/")
                seg01 = _seg_records(names, f"{domain}/01_GT/SEG/")
                raw02 = _raw_map(names, f"{domain}/02/")
                tra02 = _volume_mask_map(names, f"{domain}/02_GT/TRA/", "man_track")
                seg02 = _seg_records(names, f"{domain}/02_GT/SEG/")
                if not seg01 or not raw02 or not tra02 or not seg02:
                    raise RuntimeError(f"incomplete CTC inventory for {domain}")
                if any(item["frame"] not in raw01 for item in seg01):
                    raise RuntimeError(f"{domain} seq01 SEG annotation lacks raw frame")
                labelled_frames = sorted(tra02)
                if labelled_frames != list(range(len(labelled_frames))):
                    raise RuntimeError(f"{domain} seq02 TRA frames are not contiguous from zero")
                if any(frame not in raw02 for frame in labelled_frames):
                    raise RuntimeError(f"{domain} seq02 TRA frame lacks raw data")

                raw01_paths: dict[int, Path] = {}
                train_samples = []
                for item in seg01:
                    frame = item["frame"]
                    if frame not in raw01_paths:
                        raw_member = raw01[frame]
                        raw_path = cache_root / domain / "01" / Path(raw_member).name
                        _extract_member(archive, raw_member, raw_path, files)
                        raw01_paths[frame] = raw_path
                    mask_path = cache_root / domain / "01_GT" / "SEG" / Path(item["member"]).name
                    _extract_member(archive, item["member"], mask_path, files)
                    train_samples.append({
                        "frame": frame,
                        "z": item["z"],
                        "image": str(raw01_paths[frame]),
                        "mask": str(mask_path),
                    })

                test_frames = []
                for frame in labelled_frames:
                    raw_member = raw02[frame]
                    raw_path = cache_root / domain / "02" / Path(raw_member).name
                    tra_member = tra02[frame]
                    tra_path = cache_root / domain / "02_GT" / "TRA" / Path(tra_member).name
                    _extract_member(archive, raw_member, raw_path, files)
                    _extract_member(archive, tra_member, tra_path, files)
                    test_frames.append({
                        "frame": frame,
                        "token": re.fullmatch(r"t(\d+)\.tif", Path(raw_member).name).group(1),
                        "image": str(raw_path),
                    })

                test_segmentation = []
                for item in seg02:
                    if item["frame"] not in tra02:
                        continue
                    destination = cache_root / domain / "02_GT" / "SEG" / Path(item["member"]).name
                    _extract_member(archive, item["member"], destination, files)
                    test_segmentation.append({
                        "frame": item["frame"], "z": item["z"], "mask": str(destination)
                    })
                track_member = f"{domain}/02_GT/TRA/man_track.txt"
                if track_member not in names:
                    raise RuntimeError(f"missing track table for {domain}")
                track_path = cache_root / domain / "02_GT" / "TRA" / "man_track.txt"
                _extract_member(archive, track_member, track_path, files)

            domains[domain] = {
                "ndim": 3 if "3D" in domain else 2,
                "archive": {
                    "path": str(archive_path.resolve()),
                    "bytes": archive_path.stat().st_size,
                    "central_directory_sha256": archive_signature(archive_path),
                },
                "train_samples": train_samples,
                "test_frames": test_frames,
                "test_segmentation": test_segmentation,
                "test_gt_dir": str(cache_root / domain / "02_GT"),
            }

        payload = {
            "status": "READY",
            "protocol_id": PROTOCOL_ID,
            "source_split_manifest": str(split_manifest.resolve()),
            "source_split_manifest_sha256": sha256(split_manifest),
            "folds": _folds(split_manifest),
            "domains": domains,
            "files": sorted(files, key=lambda item: item["path"]),
        }
        atomic_json(manifest_path, payload)
        return manifest_path


def load_cache_manifest(path: Path) -> dict[str, Any]:
    payload = _valid_manifest(path)
    if payload is None:
        raise RuntimeError(f"native CTC cache is absent or incomplete: {path}")
    return payload


def read_tiff(path: str | Path) -> np.ndarray:
    return np.squeeze(np.asarray(tifffile.imread(path)))


def read_slice(path: str | Path, z: int | None) -> np.ndarray:
    if z is None:
        array = read_tiff(path)
        if array.ndim != 2:
            raise ValueError(f"expected planar TIFF at {path}, got {array.shape}")
        return array
    array = np.asarray(tifffile.imread(path, key=int(z)))
    if array.ndim != 2:
        raise ValueError(f"expected 2-D TIFF page {z} at {path}, got {array.shape}")
    return array


def percentile_scale(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(array, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("CTC image contains non-finite values")
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)


def normalized_image_tensor(image: np.ndarray) -> torch.Tensor:
    scaled = percentile_scale(image)
    rgb = np.repeat(scaled[..., None], 3, axis=2)
    rgb = (rgb - np.asarray(MICRO_RGB_MEAN, dtype=np.float32)) / np.asarray(
        MICRO_RGB_STD, dtype=np.float32
    )
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()


class CTCNativeTrainDataset(Dataset):
    """One deterministic foreground-aware planar crop per SEG file and epoch.

    Complete 3-D SEG files contribute one foreground-containing z slice per
    epoch. Sparse ``TIME_Z`` files use their declared page. This keeps each
    official annotation file equally weighted and avoids treating empty volume
    pages as independent annotations.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        crop_size: int = 256,
        seed: int = 0,
        array_cache: dict[str, np.ndarray] | None = None,
    ) -> None:
        if not records:
            raise ValueError("CTC training records must not be empty")
        self.records = records
        self.crop_size = int(crop_size)
        self.seed = int(seed)
        self.epoch = 0
        self.array_cache = array_cache if array_cache is not None else {}

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _mask(self, path: str) -> np.ndarray:
        if path not in self.array_cache:
            self.array_cache[path] = read_tiff(path)
        return self.array_cache[path]

    def __getitem__(self, index: int):
        record = self.records[index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        mask = self._mask(record["mask"])
        declared_z = record.get("z")
        if mask.ndim == 3:
            foreground_pages = np.flatnonzero(np.any(mask > 0, axis=(1, 2)))
            choices = foreground_pages if len(foreground_pages) else np.arange(mask.shape[0])
            z = int(choices[int(rng.integers(len(choices)))])
            instance = mask[z].astype(np.int32, copy=False)
            image = percentile_scale(read_slice(record["image"], z))
        elif mask.ndim == 2:
            instance = mask.astype(np.int32, copy=False)
            image = percentile_scale(read_slice(record["image"], declared_z))
        else:
            raise ValueError(f"unsupported SEG shape {mask.shape} for {record}")
        if image.shape != instance.shape:
            raise ValueError(f"raw/SEG shape mismatch for {record}: {image.shape} != {instance.shape}")

        crop = self.crop_size
        pad_y, pad_x = max(0, crop - image.shape[0]), max(0, crop - image.shape[1])
        if pad_y or pad_x:
            mode = "reflect" if min(image.shape) > 1 else "edge"
            image = np.pad(image, ((0, pad_y), (0, pad_x)), mode=mode)
            instance = np.pad(instance, ((0, pad_y), (0, pad_x)), mode="constant")
        foreground_y, foreground_x = np.nonzero(instance)
        if len(foreground_y):
            selected = int(rng.integers(len(foreground_y)))
            cy, cx = int(foreground_y[selected]), int(foreground_x[selected])
            low_y, high_y = max(0, cy - crop + 1), min(cy, image.shape[0] - crop)
            low_x, high_x = max(0, cx - crop + 1), min(cx, image.shape[1] - crop)
            y = int(rng.integers(low_y, high_y + 1))
            x = int(rng.integers(low_x, high_x + 1))
        else:
            y = int(rng.integers(0, image.shape[0] - crop + 1))
            x = int(rng.integers(0, image.shape[1] - crop + 1))
        image, instance = image[y:y + crop, x:x + crop], instance[y:y + crop, x:x + crop]
        rotation, flip = int(rng.integers(0, 4)), bool(rng.integers(0, 2))
        if rotation:
            image, instance = np.rot90(image, rotation), np.rot90(instance, rotation)
        if flip:
            image, instance = np.fliplr(image), np.fliplr(instance)
        rgb = np.repeat(image[..., None], 3, axis=2)
        rgb = (rgb - np.asarray(MICRO_RGB_MEAN, dtype=np.float32)) / np.asarray(
            MICRO_RGB_STD, dtype=np.float32
        )
        target = make_targets(np.ascontiguousarray(instance))
        return (
            torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float(),
            target["np"], target["hv"], target["tp"],
        )
