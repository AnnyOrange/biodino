#!/usr/bin/env python3
"""Delete only a verified old-complete H+ or L checkpoint copy on hxw.

Formal-v3 has NOT run: those later cells must re-stage this point from lyx-xr.
Never targets the lyx-xr originals or result artifacts.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import time


ROOT = Path("/data/hs6_hplus_5tb_eval_20260921")
OLD = ROOT / "old"
STATE = OLD / "_state"
AUDIT = ROOT / "logs" / "old_complete_checkpoint_deletions.jsonl"
IDS = (0, 487, 975, 1463, 1951, 2439, 2927, 3415, 3903, 4391, 4879, 5367, 5855, 6343, 6831)
CAMPAIGN = "hplus"
SIMPLE = {
    "classification_a": ("bio_classification", "bloodmnist pathmnist tissuemnist breastmnist organamnist organcmnist organsmnist"),
    "classification_b": ("bio_classification", "dermamnist octmnist pneumoniamnist retinamnist chestmnist bbbc048-cellcycle"),
    "classification_c": ("bio_classification", "cyclops-protein-loc midog25-atypical pcam nct-crc-he lc25000 chammi-allen-task1"),
    "classification_d": ("bio_classification", "chammi-allen-task2 chammi-cp-task1 chammi-cp-task2 chammi-cp-task3 chammi-hpa-task1 chammi-hpa-task2"),
    "regression": ("bio_regression", "bbbc013 bbbc005 conic-cell-count livecell-cell-count"),
    "retrieval": ("bio_retrieval", "lc25000 nct-crc-he-100 nct-crc-he-1k crc-val-he-7k hpa-subcellular rxrx1-cross"),
    "detection": ("bio_detection", "livecell bbbc038 conic"),
}
SEG = {
    "segmentation_a": ("bbbc038", "conic"),
    "segmentation_b": ("pannuke", "tissuenet"),
    "segmentation_c": ("livecell", "multimodal_cellseg"),
    "segmentation_d": ("monuseg", "cellpose"),
}
LANES = tuple(SIMPLE) + tuple(SEG)


def assert_safe(path: Path) -> None:
    if not path.is_relative_to(ROOT) or ROOT.is_symlink() or OLD.is_symlink():
        raise RuntimeError(f"Path outside the campaign: {path}")


def data_digest(path: Path, point_id: int) -> str:
    assert_safe(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or symlinked result {path}")
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("error"):
        raise ValueError(f"Invalid result {path}")
    ck = data.get("checkpoint")
    if ck is not None and str(ck) != str(point_id) and Path(str(ck)).parent.name != str(point_id):
        raise ValueError(f"Checkpoint mismatch in {path}: {ck}")

    def has_finite(obj):
        if isinstance(obj, dict):
            return any(has_finite(value) for value in obj.values())
        if isinstance(obj, list):
            return any(has_finite(value) for value in obj)
        return isinstance(obj, (int, float)) and not isinstance(obj, bool) and math.isfinite(obj)

    if not has_finite(data):
        raise ValueError(f"No finite numeric result in {path}")
    return hashlib.sha256(raw).hexdigest()


def validated_result_digest(point_id: int) -> tuple[str, int]:
    point = OLD / f"point_{point_id}"
    if point.is_symlink() or not point.is_dir():
        raise ValueError(f"Missing point_{point_id}")
    digest = hashlib.sha256()
    files = 0
    for lane in LANES:
        marker = STATE / "done" / f"ckpt_{point_id}__{lane}.json"
        assert_safe(marker)
        if marker.is_symlink() or not marker.is_file():
            raise ValueError(f"Incomplete lane {point_id}/{lane}")
        row = json.loads(marker.read_bytes())
        if row.get("checkpoint_id") != point_id or row.get("lane") != lane or row.get("returncode") != 0:
            raise ValueError(f"Invalid done marker {marker}")
        manifest = point / lane / "command_manifest.json"
        assert_safe(manifest)
        if manifest.is_symlink() or not manifest.is_file():
            raise ValueError(f"Missing command manifest {manifest}")
        json.loads(manifest.read_bytes())
        family, datasets = SIMPLE.get(lane, (None, None))
        if family:
            for dataset in datasets.split():
                suffix = "results_bio_detection.json" if lane == "detection" else "last_result.json"
                result = point / lane / family / dataset / str(point_id) / suffix
                digest.update(str(result.relative_to(point)).encode() + b"\0" + data_digest(result, point_id).encode())
                files += 1
            continue
        for dataset in SEG[lane]:
            matches = sorted((point / lane / "bio_segmentation").glob(f"*/{dataset}/{point_id}/results.json"))
            expected = 3 if dataset == "pannuke" else 1
            if len(matches) != expected:
                raise ValueError(f"Expected {expected} results, found {len(matches)} in {lane}/{dataset}")
            for result in matches:
                digest.update(str(result.relative_to(point)).encode() + b"\0" + data_digest(result, point_id).encode())
                files += 1
    if files != 48:
        raise ValueError(f"Expected 48 finite result artifacts, got {files}")
    for directory in (STATE / "claims", STATE / "terminal"):
        if list(directory.glob(f"ckpt_{point_id}__*")):
            raise ValueError(f"Claim or terminal marker exists for point_{point_id} in {directory}")
    return digest.hexdigest(), files


def active_process(point_id: int) -> bool:
    needles = (f"/point_{point_id}/", f"/adapters/{point_id}/", f"/source/eval/training_{point_id}/")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (PermissionError, ProcessLookupError, FileNotFoundError):
            continue
        if any(needle in command for needle in needles):
            return True
    return False


def copy_paths(point_id: int) -> tuple[Path, tuple[Path, ...], tuple[Path, ...]]:
    adapter_dir = ROOT / "adapters" / str(point_id)
    if point_id == 0:
        source = ROOT / "source" / "step0" / "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
        link_dir = ROOT / "source" / "step0_snapshot" / "training_487"
        links = (adapter_dir / "checkpoint.pth", link_dir / "teacher_checkpoint.pth")
        directories = (adapter_dir, link_dir, ROOT / "source" / "step0")
    else:
        source_dir = ROOT / "source" / "eval" / f"training_{point_id}"
        source = source_dir / "teacher_checkpoint.pth"
        if CAMPAIGN == "hplus":
            verified = ROOT / "source" / "verified_eval" / f"training_{point_id}"
            links = (adapter_dir / "checkpoint.pth", verified)
        else:
            links = (adapter_dir / "checkpoint.pth",)
        directories = (adapter_dir, source_dir)
    for path in (source, *links, *directories):
        assert_safe(path)
    return source, links, directories


def audit(row: dict) -> None:
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("a") as out:
        out.write(json.dumps(row, sort_keys=True) + "\n")
        out.flush()
        os.fsync(out.fileno())


def handle(point_id: int, *, dry_run: bool) -> bool:
    # Formal-v3 stages a temporary copy under a hold before any process starts.
    # Never delete that freshly staged copy between its transfer and launch.
    hold = ROOT / "v3" / "checkpoint_holds" / f"point_{point_id}.hold"
    assert_safe(hold)
    if hold.exists():
        raise ValueError(f"Formal-v3 restaging/assessment holds point_{point_id}")
    source, links, directories = copy_paths(point_id)
    if not source.is_file():
        return False
    result_sha, result_count = validated_result_digest(point_id)
    if active_process(point_id):
        raise ValueError(f"Active process references point_{point_id}")
    for link in links:
        expected = source.parent if link.name == f"training_{point_id}" else source
        if not link.is_symlink() or link.resolve(strict=True) != expected.resolve(strict=True):
            raise ValueError(f"Unexpected symlink at {link}")
    if source.is_symlink() or not source.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Unsafe checkpoint source {source}")
    size = source.stat().st_size
    if point_id == 0 and size != 3363232567:
        raise ValueError(f"Pretrained weight size mismatch at {source}")
    expected_teacher_size = 3556425127 if CAMPAIGN == "hplus" else 1401909871
    if point_id != 0 and size != expected_teacher_size:
        raise ValueError(f"Teacher checkpoint size mismatch at {source}")
    print(json.dumps({"point": point_id, "would_delete": str(source), "bytes": size,
                      "result_sha256": result_sha, "result_files": result_count,
                      "dry_run": dry_run}, sort_keys=True), flush=True)
    if dry_run:
        return True
    with source.open("rb") as checkpoint_file:
        checksum = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    audit({"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "event": "validated_before_deletion",
           "point": point_id, "checkpoint": str(source), "bytes": size,
           "checkpoint_sha256": checksum, "result_sha256": result_sha,
           "recovery": "Re-transfer from unchanged lyx-xr training export for formal-v3"})
    if active_process(point_id) or validated_result_digest(point_id)[0] != result_sha:
        raise RuntimeError(f"Point_{point_id} changed during validation")
    for link in links:
        link.unlink()
    source.unlink()
    for directory in directories:
        if directory.is_dir() and not directory.is_symlink():
            try:
                directory.rmdir()
            except OSError:
                pass
    audit({"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "event": "deleted_hxw_copy",
           "point": point_id, "checkpoint": str(source), "bytes": size,
           "checkpoint_sha256": checksum, "result_sha256": result_sha,
           "result_files_preserved": result_count})
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--campaign", choices=("hplus", "l"), default="hplus")
    args = parser.parse_args()
    if args.campaign == "l":
        global ROOT, OLD, STATE, AUDIT, IDS, CAMPAIGN
        CAMPAIGN = "l"
        ROOT = Path("/data/hs6_l_5tb_nogram_eval_20260921")
        OLD = ROOT / "results"
        STATE = OLD / "_state"
        AUDIT = ROOT / "logs" / "old_complete_checkpoint_deletions.jsonl"
        IDS = (29767, 30255, 30743, 31231, 31719, 32207, 32695, 33183)
    if args.interval < 10:
        parser.error("interval must be >= 10 seconds")
    while True:
        for point_id in IDS:
            try:
                handle(point_id, dry_run=args.dry_run)
            except ValueError as exc:
                if (OLD / f"point_{point_id}").is_dir():
                    print(f"point_{point_id}: not yet safe: {exc}", flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
