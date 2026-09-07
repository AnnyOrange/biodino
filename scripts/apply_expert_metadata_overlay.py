#!/usr/bin/env python3
"""Create a labeled expert bank without modifying its frozen features."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from scripts.build_expert_feature_bank import role_reliability
except ModuleNotFoundError:  # Direct execution puts scripts/ first on sys.path.
    from build_expert_feature_bank import role_reliability


METADATA_FIELDS = ("domain", "organism", "acquisition_family", "sample_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-role", choices=("general", "organism_cell", "cell", "tissue"))
    return parser.parse_args()


def build_labeled_payload(
    bank: dict[str, np.ndarray],
    overlay: dict[str, np.ndarray],
    *,
    expert_role: str,
) -> dict[str, np.ndarray]:
    keys = np.asarray(bank["keys"]).astype(str)
    overlay_keys = np.asarray(overlay["keys"]).astype(str)
    if not np.array_equal(keys, overlay_keys):
        raise ValueError("Expert bank and metadata overlay keys differ")
    missing = set(METADATA_FIELDS).difference(overlay)
    if missing:
        raise ValueError(f"Metadata overlay is missing arrays: {sorted(missing)}")

    payload = {name: np.asarray(value) for name, value in bank.items()}
    metadata = {
        name: np.asarray(overlay[name]).astype(str) for name in METADATA_FIELDS
    }
    for name, values in metadata.items():
        if values.shape != keys.shape:
            raise ValueError(f"Overlay {name} has shape {values.shape}, expected {keys.shape}")
        payload[name] = values
    payload["reliability"] = np.asarray(
        [
            role_reliability(
                expert_role,
                {
                    "domain": metadata["domain"][row],
                    "organism": metadata["organism"][row],
                    "acquisition_family": metadata["acquisition_family"][row],
                    "sample_type": metadata["sample_type"][row],
                },
            )
            for row in range(len(keys))
        ],
        dtype=np.float32,
    )
    payload["expert_role"] = np.asarray(expert_role)
    return payload


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    with np.load(args.bank, allow_pickle=False) as source:
        bank = {name: np.asarray(source[name]) for name in source.files}
    with np.load(args.overlay, allow_pickle=False) as source:
        overlay = {name: np.asarray(source[name]) for name in source.files}
    if args.expert_role:
        expert_role = args.expert_role
    elif "expert_role" in bank:
        expert_role = str(np.asarray(bank["expert_role"]).item())
    else:
        raise ValueError("--expert-role is required when the bank has no expert_role scalar")
    payload = build_labeled_payload(bank, overlay, expert_role=expert_role)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    reliability = np.asarray(payload["reliability"], dtype=np.float32)
    report = {
        "bank": str(args.bank),
        "overlay": str(args.overlay),
        "output": str(args.output),
        "expert_role": expert_role,
        "samples": int(len(payload["keys"])),
        "feature_shape": list(np.asarray(payload["features"]).shape),
        "known_organism_fraction": float(np.mean(np.asarray(payload["organism"]) != "")),
        "known_acquisition_fraction": float(
            np.mean(np.asarray(payload["acquisition_family"]) != "")
        ),
        "organism_counts": dict(
            Counter(value for value in np.asarray(payload["organism"]).astype(str) if value)
        ),
        "acquisition_counts": dict(
            Counter(
                value
                for value in np.asarray(payload["acquisition_family"]).astype(str)
                if value
            )
        ),
        "reliability": {
            "min": float(reliability.min()),
            "mean": float(reliability.mean()),
            "max": float(reliability.max()),
        },
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
