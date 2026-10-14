#!/usr/bin/env python3
"""Export provenance metadata and build a resumable inventory for rolling 1PB extraction."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (
    "/home/inspur/huaweimount_sam2p",
    "/home/inspur/as13000_sam2p",
    "/mnt/deepcad_nfs",
    "/mnt/mount_100t_zzf_2",
    "/mnt/mount_100t_zzf_3",
    "/slfm",
)


def load_script(name: str, module_name: str):
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--queue-root", type=Path,
                        default=Path("/mnt/huawei_blm/rolling_1pb_wds_v1"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--skip-metadata-export", action="store_true")
    args = parser.parse_args()
    roots = [Path(path).resolve() for path in (args.root or DEFAULT_ROOTS)]
    unavailable = [str(path) for path in roots if not path.is_dir()]
    if unavailable:
        raise FileNotFoundError(f"1PB source mounts are unavailable: {unavailable}")
    inventory_root = args.queue_root / "inventory"
    inventory_root.mkdir(parents=True, exist_ok=True)
    metadata_csv = inventory_root / "raw_1pb_source_metadata.csv"
    index_path = inventory_root / "raw_1pb_inventory.sqlite"
    report_path = inventory_root / "preparation.json"
    report = {"state": "STARTED", "roots": [str(path) for path in roots],
              "index": str(index_path), "started_unix": time.time()}
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        if not args.skip_metadata_export and not metadata_csv.is_file():
            export = subprocess.run([
                sys.executable, str(REPO / "scripts/export_raw_1pb_metadata.py"),
                "--output", str(metadata_csv),
            ], cwd=REPO, text=True, capture_output=True)
            report["metadata_export_returncode"] = export.returncode
            report["metadata_export_stdout_tail"] = export.stdout[-3000:]
            report["metadata_export_stderr_tail"] = export.stderr[-3000:]
            if export.returncode:
                metadata_csv.unlink(missing_ok=True)
                report["metadata_export"] = "UNAVAILABLE_FILESYSTEM_ONLY_MODE"
            else:
                report["metadata_export"] = "PASS"
            report_path.write_text(json.dumps(report, indent=2) + "\n")

        inventory = load_script("build_raw_1pb_inventory.py", "rolling_1pb_inventory")
        result = inventory.build(roots, index_path, world_size=1, workers=args.workers,
                                 metadata_csv=(metadata_csv if metadata_csv.is_file() else None),
                                 batch_size=args.batch_size)
        report.update(state=result["status"], inventory=result, completed_unix=time.time())
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if result["status"] != "PASS":
            raise SystemExit(f"1PB inventory is not ready: {report_path}")
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        report.update(state="ERROR", error=f"{type(exc).__name__}: {str(exc)[:500]}",
                      completed_unix=time.time())
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
