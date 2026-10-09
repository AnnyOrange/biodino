#!/usr/bin/env python3
"""Index the verified source archives before removing sibling checkouts."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
ARCHIVES = ROOT / "outputs/03_source_archives/dinov3_siblings_20261008"
REPORT = ROOT / "outputs/00_reports/source_cleanup_20261008/archive_manifest.json"
ACTIVE = {
    "dinov3_20tb_online_snapshot_20260918": "3090-qi evaluation process",
    "dinov3_method_full_v4_snapshot_20260928": "local and deepcad evaluation processes",
    "dinov3_retest_snapshot_20260918_fm_bound": "local and 3090-qi evaluation workers",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    archives = sorted(ARCHIVES.glob("*.tar.zst"))
    if len(archives) != 28:
        raise RuntimeError(f"Expected 28 archived sibling directories, found {len(archives)}")
    rows = []
    for archive in archives:
        source = PARENT / archive.name.removesuffix(".tar.zst")
        if source.exists() and not source.is_dir():
            raise RuntimeError(f"Archived source path changed type: {source}")
        if source.exists() and source.name not in ACTIVE:
            raise RuntimeError(f"Unexpected sibling still present: {source}")
        if not source.exists() and source.name in ACTIVE:
            raise RuntimeError(f"Active sibling was removed: {source}")
        rows.append({
            "source": str(source),
            "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256(archive),
            "state": "ACTIVE_RETAIN" if source.exists() else "ARCHIVED_REMOVED",
            "active_reason": ACTIVE.get(source.name),
        })
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "main_repo": str(ROOT),
        "entries": rows,
        "excluded_from_archives": [
            "*/dinov3/dataset_webdataset/test/dataset_webdataset/*",
            "*/__pycache__/*", "*/pymp-*", "*.pyc",
        ],
        "excluded_test_shards": "All seven large copies were byte-identical to the two files retained in the main repository.",
        "note": "Historical manifests retain original paths. Restore a named archive before replaying an old run.",
    }, indent=2) + "\n")
    print(f"indexed {len(rows)} source archives; active={len(ACTIVE)}; report={REPORT}")


if __name__ == "__main__":
    main()
