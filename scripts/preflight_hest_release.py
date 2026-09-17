#!/usr/bin/env python3
"""Check HEST's published file identities and every patch/target payload."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.eval.bio_frozen_eval.hest import inspect_hest


def release_files():
    errors = []
    for host in ("huggingface.co", "hf-mirror.com"):
        try:
            response = requests.get(f"https://{host}/api/datasets/MahmoodLab/hest-bench/tree/main",
                                    params={"recursive": "true", "limit": 1000}, timeout=(10, 40))
            response.raise_for_status()
            return response.json(), response.url
        except (requests.RequestException, ValueError) as error:
            errors.append(str(error))
    raise RuntimeError("Cannot retrieve authoritative release identities: " + "; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--release-manifest", help="Previously saved HF tree JSON for offline replay")
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    if args.release_manifest:
        published, transport = json.loads(Path(args.release_manifest).read_text()), "saved HF tree"
    else:
        published, transport = release_files()
    (output / "published_tree.json").write_text(json.dumps(published, indent=2) + "\n")
    files, errors = [], []
    for row in published:
        relative = row["path"]
        parts = Path(relative).parts
        if row["type"] != "file" or not parts or not (root / parts[0] / "splits").is_dir():
            continue
        if not (relative.endswith(".h5") or relative.endswith(".h5ad") or "/splits/" in relative
                or relative.endswith("var_50genes.json")):
            continue
        path = root / relative
        report = {"path": relative, "published_bytes": row["size"]}
        try:
            if path.stat().st_size != row["size"]:
                raise ValueError("Published file length mismatch")
            checksum = hashlib.sha256()
            git_checksum = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                    checksum.update(block)
                    git_checksum.update(block)
            report["sha256"] = checksum.hexdigest()
            identity = row.get("lfs", {}).get("oid")
            if identity and report["sha256"] != identity:
                raise ValueError("Published LFS SHA256 mismatch")
            if not identity and git_checksum.hexdigest() != row["oid"]:
                raise ValueError("Published Git blob identity mismatch")
            report["status"] = "PASS"
        except (OSError, ValueError) as error:
            errors.append({"path": relative, "error": str(error)})
            report.update(status="FAIL", error=str(error))
        files.append(report)
        print(relative, report["status"], flush=True)
    if not files:
        errors.append({"error": "No benchmark payloads in release manifest"})
    (output / "release_verification.json").write_text(json.dumps({"status": "FAIL" if errors else "PASS",
        "source": "https://huggingface.co/datasets/MahmoodLab/hest-bench", "transport": transport,
        "files": files, "errors": errors}, indent=2) + "\n")
    report = inspect_hest(root, verify_payload=True)
    report.update(release_status="FAIL" if errors else "PASS", release_files=len(files),
                  release_verification_sha256=hashlib.sha256((output / "release_verification.json").read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic()-started)
    if errors:
        report["status"] = "FAIL"
    (output / "preflight.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "tasks": len(report["tasks"]),
                      "release_files": len(files), "elapsed_seconds": report["elapsed_seconds"]}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
