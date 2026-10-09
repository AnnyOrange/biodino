"""Bind the user-authorized FM tail to the current authoritative local copy."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3")
MANIFEST = Path("/mnt/huawei_deepcad/benchmark_model/benchmark_runs/fm_v3_completion/campaign_manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    snapshot = manifest["source_snapshot"]
    rebound = {}
    missing = []
    for relative in snapshot["files"]:
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
        else:
            rebound[relative] = sha256(path)
    # The historical copied snapshot listed an optional evaluation_external
    # mirror which is no longer retained locally.  The live absolute external
    # sources are independently pinned by manifest['external_source_hashes'];
    # omit only missing mirror entries and record the omission explicitly.
    non_external_missing = [
        relative for relative in missing
        if not relative.startswith("evaluation_external/benchmark_model/")
    ]
    if non_external_missing:
        raise RuntimeError(f"Required snapshot files missing: {non_external_missing}")
    snapshot["files"] = rebound
    snapshot["sha256"] = hashlib.sha256(
        json.dumps(rebound, sort_keys=True).encode()
    ).hexdigest()
    snapshot["binding_amendment"] = (
        "USER_AUTHORIZED_LOCAL_COPY_20260923: current local code is authoritative; "
        "FM v3 completion starts only after this exact snapshot is bound"
    )
    snapshot["omitted_external_mirror_files"] = missing
    snapshot["rebound_unix"] = time.time()
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(snapshot["sha256"], len(rebound))


if __name__ == "__main__":
    main()
