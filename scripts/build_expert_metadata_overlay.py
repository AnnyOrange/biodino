#!/usr/bin/env python3
"""Recover expert-bank source metadata from exported 100T tile records."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from scripts.build_bio_source_domain_inventory import source_domain
    from scripts.build_expert_feature_bank import load_catalog
except ModuleNotFoundError:  # Direct execution puts scripts/ first on sys.path.
    from build_bio_source_domain_inventory import source_domain
    from build_expert_feature_bank import load_catalog


OID_PATTERN = re.compile(r"_oid(?P<source_id>\d+)")
SOURCE_ID_PATTERN = re.compile(br'"source_id"\s*:\s*(\d+)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--domain-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--progress-gib", type=float, default=2.0)
    return parser.parse_args()


def source_id_from_key(key: str) -> int:
    match = OID_PATTERN.search(str(key))
    if match is None:
        raise ValueError(f"Sample key has no _oid source identifier: {key}")
    return int(match.group("source_id"))


def record_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.glob("*.records.jsonl")))
        else:
            raise FileNotFoundError(path)
    if not files:
        raise ValueError("No *.records.jsonl files were found")
    return files


def recover_source_paths(
    wanted_ids: set[int],
    files: list[Path],
    *,
    progress_gib: float,
) -> tuple[dict[int, str], dict[str, int]]:
    found: dict[int, str] = {}
    scanned_bytes = 0
    scanned_lines = 0
    next_progress = max(int(progress_gib * 1024**3), 1)

    for path in files:
        with path.open("rb") as handle:
            for line in handle:
                scanned_bytes += len(line)
                scanned_lines += 1
                match = SOURCE_ID_PATTERN.search(line)
                if match is None:
                    continue
                source_id = int(match.group(1))
                if source_id not in wanted_ids or source_id in found:
                    continue
                payload = json.loads(line)
                source_path = str(payload.get("source_path", "")).strip()
                if source_path:
                    found[source_id] = source_path
                    if len(found) == len(wanted_ids):
                        return found, {
                            "scanned_bytes": scanned_bytes,
                            "scanned_lines": scanned_lines,
                            "record_files_scanned": files.index(path) + 1,
                        }
                if scanned_bytes >= next_progress:
                    print(
                        f"[metadata-overlay] scanned={scanned_bytes / 1024**3:.1f} GiB "
                        f"recovered={len(found)}/{len(wanted_ids)}",
                        flush=True,
                    )
                    next_progress += max(int(progress_gib * 1024**3), 1)

    return found, {
        "scanned_bytes": scanned_bytes,
        "scanned_lines": scanned_lines,
        "record_files_scanned": len(files),
    }


def catalog_metadata(domain: str, catalog: dict[str, dict[str, str]]) -> dict[str, str]:
    row = catalog.get(domain.lower(), {})
    return {
        "domain": str(row.get("domain", domain)),
        "organism": str(row.get("organism", "")),
        "acquisition_family": str(row.get("acquisition_family", "unresolved")),
        "sample_type": str(row.get("sample_type", "")),
    }


def main() -> None:
    args = parse_args()
    if args.progress_gib <= 0:
        raise ValueError("--progress-gib must be positive")

    with np.load(args.bank, allow_pickle=False) as bank:
        keys = np.asarray(bank["keys"]).astype(str)
    source_ids = np.asarray([source_id_from_key(key) for key in keys], dtype=np.int64)
    wanted_ids = set(source_ids.tolist())
    files = record_files(args.records)
    recovered, scan_stats = recover_source_paths(
        wanted_ids,
        files,
        progress_gib=args.progress_gib,
    )
    catalog = load_catalog(args.domain_catalog)

    domains: list[str] = []
    organisms: list[str] = []
    acquisitions: list[str] = []
    sample_types: list[str] = []
    source_paths: list[str] = []
    recovered_mask: list[bool] = []
    for source_id in source_ids:
        path = recovered.get(int(source_id), "")
        if path:
            domain, _family = source_domain(path)
            metadata = catalog_metadata(domain, catalog)
            recovered_mask.append(True)
        else:
            metadata = {
                "domain": "unresolved",
                "organism": "",
                "acquisition_family": "unresolved",
                "sample_type": "",
            }
            recovered_mask.append(False)
        source_paths.append(path)
        domains.append(metadata["domain"])
        organisms.append(metadata["organism"])
        acquisitions.append(metadata["acquisition_family"])
        sample_types.append(metadata["sample_type"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        keys=keys,
        source_id=source_ids,
        source_path=np.asarray(source_paths),
        recovered=np.asarray(recovered_mask, dtype=np.bool_),
        domain=np.asarray(domains),
        organism=np.asarray(organisms),
        acquisition_family=np.asarray(acquisitions),
        sample_type=np.asarray(sample_types),
    )

    recovered_array = np.asarray(recovered_mask, dtype=np.bool_)
    known_organism = np.asarray([bool(value.strip()) for value in organisms])
    known_acquisition = np.asarray(
        [value.strip().lower() not in {"", "unresolved", "unknown"} for value in acquisitions]
    )
    report = {
        "bank": str(args.bank),
        "output": str(args.output),
        "samples": int(len(keys)),
        "unique_source_ids": int(len(wanted_ids)),
        "recovered_unique_source_ids": int(len(recovered)),
        "recovered_unique_fraction": len(recovered) / len(wanted_ids) if wanted_ids else 0.0,
        "recovered_samples": int(recovered_array.sum()),
        "recovered_sample_fraction": float(recovered_array.mean()) if len(keys) else 0.0,
        "known_organism_samples": int(known_organism.sum()),
        "known_acquisition_samples": int(known_acquisition.sum()),
        "domains_by_sample": dict(Counter(domains).most_common()),
        "organisms_by_sample": dict(Counter(value for value in organisms if value).most_common()),
        "acquisitions_by_sample": dict(
            Counter(value for value in acquisitions if value).most_common()
        ),
        **scan_stats,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
