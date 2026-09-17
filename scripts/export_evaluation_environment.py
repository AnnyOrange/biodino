#!/usr/bin/env python3
"""Export the installed pinned environment closure for offline venv overlays.

The archive contains distribution files, not benchmark code or mutable training
state. It is portable only to Linux x86_64 with the same Python minor version.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import tarfile

from packaging.requirements import Requirement

from evaluation_environment import fingerprint, requirements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-inventory", help="Remote installed-version JSON; export only differing distributions")
    args = parser.parse_args()
    base = json.loads(Path(args.base_inventory).read_text()) if args.base_inventory else {}
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    pending, distributions = list(requirements()), {}
    while pending:
        name = pending.pop()
        dist = metadata.distribution(name)
        key = dist.metadata["Name"].lower().replace("_", "-")
        if key in distributions:
            continue
        distributions[key] = dist
        for requirement in dist.requires or []:
            value = Requirement(requirement)
            if value.marker is None or value.marker.evaluate({"extra": ""}):
                pending.append(value.name)
    files = {}
    for dist in distributions.values():
        key = dist.metadata["Name"].lower().replace("_", "-")
        if base.get(key) == dist.version and key != "opencv-python-headless":
            continue
        for relative in dist.files or []:
            relative = Path(relative)
            if relative.is_absolute() or ".." in relative.parts or "__pycache__" in relative.parts:
                continue
            source = Path(dist.locate_file(relative))
            if source.is_file():
                files.setdefault(str(relative), source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz", dereference=True, compresslevel=1) as archive:
        for index, (relative, source) in enumerate(sorted(files.items())):
            archive.add(source, arcname=relative, recursive=False)
            if index % 5000 == 0:
                print(f"exported {index}/{len(files)} files", flush=True)
    checksum = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            checksum.update(block)
    manifest = {"sha256": checksum.hexdigest(), "bytes": target.stat().st_size, "compression": "gzip",
                "platform": platform.platform(), "machine": platform.machine(),
                "python_minor": platform.python_version_tuple()[:2], "environment": fingerprint(),
                "distributions": {name: dist.version for name, dist in sorted(distributions.items())},
                "files": len(files), "scope": "installed distribution closure; no source-code changes"}
    target.with_suffix(target.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"archive": str(target), "sha256": checksum.hexdigest(), "bytes": target.stat().st_size}))


if __name__ == "__main__":
    main()
