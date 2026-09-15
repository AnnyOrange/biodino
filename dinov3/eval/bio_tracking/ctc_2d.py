"""Data preparation and deterministic crops for the 2-D CTC observation."""

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


PROTOCOL_ID = "ctc-native-2d-domain-heldout-observational-v1"
CTC_2D_DOMAINS = (
    "BF-C2DL-HSC",
    "BF-C2DL-MuSC",
    "DIC-C2DH-HeLa",
    "Fluo-C2DL-Huh7",
    "Fluo-C2DL-MSC",
    "Fluo-N2DH-GOWT1",
    "Fluo-N2DH-SIM+",
    "Fluo-N2DL-HeLa",
    "PhC-C2DH-U373",
    "PhC-C2DL-PSC",
)


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def archive_signature(path: Path) -> str:
    """Hash the ZIP central-directory metadata without rereading its payload."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            digest.update(f"{info.filename}\t{info.file_size}\t{info.CRC}\n".encode())
    return digest.hexdigest()


def _member_map(names: list[str], prefix: str, pattern: str) -> dict[int, str]:
    expression = re.compile(pattern)
    result: dict[int, str] = {}
    for name in names:
        if not name.startswith(prefix):
            continue
        match = expression.fullmatch(Path(name).name)
        if match:
            frame = int(match.group(1))
            if frame in result:
                raise RuntimeError(f"duplicate CTC frame {frame}: {result[frame]} and {name}")
            result[frame] = name
    return result


def _extract_member(
    archive: zipfile.ZipFile,
    member: str,
    destination: Path,
    file_records: list[dict[str, Any]],
) -> None:
    info = archive.getinfo(member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.stat().st_size != info.file_size:
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with archive.open(info) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 << 20)
        if temporary.stat().st_size != info.file_size:
            raise RuntimeError(f"incomplete extraction for {member}")
        os.replace(temporary, destination)
    file_records.append({
        "path": str(destination),
        "source_member": member,
        "bytes": info.file_size,
        "crc32": f"{info.CRC:08x}",
        "sha256": sha256(destination),
    })


def _folds_from_formal_manifest(split_manifest: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in split_manifest.read_text().splitlines() if line.strip()]
    folds = []
    observed_test_domains: list[str] = []
    for fold in range(5):
        fold_rows = [row for row in rows if int(row["fold"]) == fold]
        train_domains = sorted({
            row["domain"] for row in fold_rows
            if row["role"] == "head_train" and row["domain"] in CTC_2D_DOMAINS
        })
        test_domains = sorted({
            row["domain"] for row in fold_rows
            if row["role"] == "test" and row["domain"] in CTC_2D_DOMAINS
        })
        if set(train_domains) & set(test_domains):
            raise RuntimeError(f"CTC fold {fold} has train/test domain overlap")
        if set(train_domains) | set(test_domains) != set(CTC_2D_DOMAINS):
            raise RuntimeError(f"CTC fold {fold} does not cover all 2-D domains")
        observed_test_domains.extend(test_domains)
        folds.append({"fold": fold, "train_domains": train_domains, "test_domains": test_domains})
    if sorted(observed_test_domains) != sorted(CTC_2D_DOMAINS):
        raise RuntimeError("each 2-D CTC domain must be held out exactly once")
    return folds


def _validate_ready_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        if payload.get("status") != "READY" or payload.get("protocol_id") != PROTOCOL_ID:
            return None
        for record in payload["files"]:
            candidate = Path(record["path"])
            if (
                not record.get("sha256")
                or not candidate.is_file()
                or candidate.stat().st_size != int(record["bytes"])
            ):
                return None
        return payload
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def prepare_cache(cache_root: Path, archives_root: Path, split_manifest: Path) -> Path:
    """Extract only annotated seq01 samples and complete seq02 data for 2-D CTC."""
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "data_manifest.json"
    lock_path = cache_root / ".prepare.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _validate_ready_manifest(manifest_path) is not None:
            return manifest_path

        file_records: list[dict[str, Any]] = []
        domains: dict[str, Any] = {}
        for domain_index, domain in enumerate(CTC_2D_DOMAINS, start=1):
            archive_path = archives_root / f"{domain}.zip"
            if not archive_path.is_file():
                raise FileNotFoundError(archive_path)
            print(f"[data {domain_index}/{len(CTC_2D_DOMAINS)}] {domain}", flush=True)
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                train_raw = _member_map(names, f"{domain}/01/", r"t(\d+)\.tif")
                train_seg = _member_map(names, f"{domain}/01_GT/SEG/", r"man_seg(\d+)\.tif")
                test_raw = _member_map(names, f"{domain}/02/", r"t(\d+)\.tif")
                test_tra = _member_map(names, f"{domain}/02_GT/TRA/", r"man_track(\d+)\.tif")
                test_seg = _member_map(names, f"{domain}/02_GT/SEG/", r"man_seg(\d+)\.tif")
                if not train_seg or not test_raw or not test_tra or not test_seg:
                    raise RuntimeError(f"incomplete 2-D CTC archive inventory for {domain}")
                missing_raw = sorted(set(train_seg) - set(train_raw))
                if missing_raw:
                    raise RuntimeError(f"{domain} seq01 SEG frames lack raw images: {missing_raw}")
                expected_frames = list(range(len(test_raw)))
                if sorted(test_raw) != expected_frames or sorted(test_tra) != expected_frames:
                    raise RuntimeError(f"{domain} seq02 raw/TRA frames must be contiguous from zero")

                train_samples = []
                for frame in sorted(train_seg):
                    raw_member, seg_member = train_raw[frame], train_seg[frame]
                    raw_path = cache_root / domain / "01" / Path(raw_member).name
                    seg_path = cache_root / domain / "01_GT" / "SEG" / Path(seg_member).name
                    _extract_member(archive, raw_member, raw_path, file_records)
                    _extract_member(archive, seg_member, seg_path, file_records)
                    train_samples.append({
                        "frame": frame,
                        "image": str(raw_path),
                        "mask": str(seg_path),
                    })

                test_frames = []
                for frame in expected_frames:
                    raw_member = test_raw[frame]
                    raw_path = cache_root / domain / "02" / Path(raw_member).name
                    tra_member = test_tra[frame]
                    tra_path = cache_root / domain / "02_GT" / "TRA" / Path(tra_member).name
                    _extract_member(archive, raw_member, raw_path, file_records)
                    _extract_member(archive, tra_member, tra_path, file_records)
                    test_frames.append({
                        "frame": frame,
                        "token": re.fullmatch(r"t(\d+)\.tif", Path(raw_member).name).group(1),
                        "image": str(raw_path),
                    })
                test_segmentation = {}
                for frame, member in sorted(test_seg.items()):
                    destination = cache_root / domain / "02_GT" / "SEG" / Path(member).name
                    _extract_member(archive, member, destination, file_records)
                    test_segmentation[str(frame)] = str(destination)
                track_member = f"{domain}/02_GT/TRA/man_track.txt"
                if track_member not in names:
                    raise RuntimeError(f"missing track table: {track_member}")
                track_path = cache_root / domain / "02_GT" / "TRA" / "man_track.txt"
                _extract_member(archive, track_member, track_path, file_records)

            domains[domain] = {
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
            "folds": _folds_from_formal_manifest(split_manifest),
            "domains": domains,
            "files": sorted(file_records, key=lambda item: item["path"]),
        }
        atomic_json(manifest_path, payload)
        return manifest_path


def load_cache_manifest(path: Path) -> dict[str, Any]:
    payload = _validate_ready_manifest(path)
    if payload is None:
        raise RuntimeError(f"CTC 2-D cache is absent or incomplete: {path}")
    return payload


def verify_cache_content(payload: dict[str, Any]) -> None:
    """Fail closed if any extracted cache file differs from its pre-run digest."""
    for index, record in enumerate(payload["files"], start=1):
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"CTC cache hash drift: {path}")
        if index % 2000 == 0:
            print(f"[cache verify] {index}/{len(payload['files'])}", flush=True)


def read_2d_tiff(path: str | Path) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D TIFF at {path}, got {array.shape}")
    return array


def percentile_scale(image: np.ndarray) -> np.ndarray:
    """Map a grayscale image to [0, 1] using exact full-image p01/p99."""
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
    mean = np.asarray(MICRO_RGB_MEAN, dtype=np.float32)
    std = np.asarray(MICRO_RGB_STD, dtype=np.float32)
    normalized = (rgb - mean) / std
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float()


class CTC2DTrainDataset(Dataset):
    """One deterministic foreground-aware 256 crop per SEG annotation and epoch."""

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

    def _read(self, path: str) -> np.ndarray:
        if path not in self.array_cache:
            self.array_cache[path] = read_2d_tiff(path)
        return self.array_cache[path]

    def __getitem__(self, index: int):
        record = self.records[index]
        image = percentile_scale(self._read(record["image"]))
        instance = self._read(record["mask"]).astype(np.int32, copy=False)
        if image.shape != instance.shape:
            raise ValueError(f"raw/SEG shape mismatch for {record}")

        crop = self.crop_size
        pad_y, pad_x = max(0, crop - image.shape[0]), max(0, crop - image.shape[1])
        if pad_y or pad_x:
            pad_mode = "reflect" if min(image.shape) > 1 else "edge"
            image = np.pad(image, ((0, pad_y), (0, pad_x)), mode=pad_mode)
            instance = np.pad(instance, ((0, pad_y), (0, pad_x)), mode="constant")

        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        foreground_y, foreground_x = np.nonzero(instance)
        if len(foreground_y):
            selected = int(rng.integers(len(foreground_y)))
            center_y, center_x = int(foreground_y[selected]), int(foreground_x[selected])
            low_y, high_y = max(0, center_y - crop + 1), min(center_y, image.shape[0] - crop)
            low_x, high_x = max(0, center_x - crop + 1), min(center_x, image.shape[1] - crop)
            y = int(rng.integers(low_y, high_y + 1))
            x = int(rng.integers(low_x, high_x + 1))
        else:
            y = int(rng.integers(0, image.shape[0] - crop + 1))
            x = int(rng.integers(0, image.shape[1] - crop + 1))
        image = image[y : y + crop, x : x + crop]
        instance = instance[y : y + crop, x : x + crop]

        rotation = int(rng.integers(0, 4))
        horizontal_flip = bool(rng.integers(0, 2))
        if rotation:
            image = np.rot90(image, rotation)
            instance = np.rot90(instance, rotation)
        if horizontal_flip:
            image = np.fliplr(image)
            instance = np.fliplr(instance)

        rgb = np.repeat(image[..., None], 3, axis=2)
        mean = np.asarray(MICRO_RGB_MEAN, dtype=np.float32)
        std = np.asarray(MICRO_RGB_STD, dtype=np.float32)
        rgb = (rgb - mean) / std
        target = make_targets(np.ascontiguousarray(instance))
        return (
            torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float(),
            target["np"],
            target["hv"],
            target["tp"],
        )
