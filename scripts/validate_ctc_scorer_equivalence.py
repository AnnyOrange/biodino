#!/usr/bin/env python3
"""Compare production and independent requested-only CTC scorer outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METRIC_KEYS = (
    "Valid",
    "DET",
    "SEG",
    "TRA",
    "AOGM",
    "AOGM_0",
    "AOGM_NS",
    "AOGM_FN",
    "AOGM_FP",
    "AOGM_ED",
    "AOGM_EA",
    "AOGM_EC",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    production_payload = json.loads(args.production.read_text())
    independent_payload = json.loads(args.independent.read_text())
    production = production_payload.get("metrics", production_payload)
    independent = independent_payload.get("metrics", independent_payload)
    comparisons = {
        key: {
            "production": production.get(key),
            "independent": independent.get(key),
            "exact_equal": production.get(key) == independent.get(key),
        }
        for key in METRIC_KEYS
    }
    valid = all(item["exact_equal"] for item in comparisons.values())
    report = {
        "status": "VALID_EXACT_EQUAL" if valid else "INVALID_MISMATCH",
        "production": {"path": str(args.production.resolve()), "sha256": sha256(args.production)},
        "independent": {
            "path": str(args.independent.resolve()),
            "sha256": sha256(args.independent),
        },
        "comparisons": comparisons,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
