#!/usr/bin/env python3
"""Merge disjoint expert-bank shards with strict metadata validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROW_ARRAYS = (
    "keys",
    "features",
    "reliability",
    "domain",
    "organism",
    "acquisition_family",
    "sample_type",
)
OPTIONAL_CONSTANT_ARRAYS = (
    "feature_protocol",
    "n_last_blocks",
    "use_avgpool",
    "normalization_protocol",
    "transform_resize_crop",
    "transform_mean",
    "transform_std",
    "checkpoint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [np.load(path, allow_pickle=False) for path in args.inputs]
    for path, payload in zip(args.inputs, payloads):
        missing = set(ROW_ARRAYS).difference(payload.files)
        if missing:
            raise ValueError(f"{path} is missing arrays {sorted(missing)}")

    models = {str(payload["model"].item()) for payload in payloads}
    roles = {str(payload["expert_role"].item()) for payload in payloads}
    if len(models) != 1 or len(roles) != 1:
        raise ValueError(f"Incompatible model/role values: models={models}, roles={roles}")
    normalizations = [np.asarray(payload["normalization_percentiles"]) for payload in payloads]
    if any(not np.array_equal(normalizations[0], value) for value in normalizations[1:]):
        raise ValueError("Input banks use different normalization percentiles")

    constant_metadata = {}
    for name in OPTIONAL_CONSTANT_ARRAYS:
        present = [name in payload.files for payload in payloads]
        if any(present) and not all(present):
            raise ValueError(f"Optional metadata {name!r} is missing from some input banks")
        if all(present):
            values = [np.asarray(payload[name]) for payload in payloads]
            if any(not np.array_equal(values[0], value) for value in values[1:]):
                raise ValueError(f"Input banks use different {name!r} metadata")
            constant_metadata[name] = values[0]

    merged = {name: np.concatenate([np.asarray(payload[name]) for payload in payloads]) for name in ROW_ARRAYS}
    if merged["features"].ndim != 2 or merged["features"].shape[0] != merged["keys"].shape[0]:
        raise ValueError("Merged feature matrix and keys are inconsistent")
    keys = merged["keys"].astype(str)
    if len(set(keys.tolist())) != len(keys):
        raise ValueError("Input banks contain duplicate sample keys")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **merged,
        **constant_metadata,
        model=np.asarray(next(iter(models))),
        expert_role=np.asarray(next(iter(roles))),
        source_shards=np.concatenate(
            [np.asarray(payload["source_shards"]) for payload in payloads]
        ),
        normalization_percentiles=normalizations[0],
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model": next(iter(models)),
                "expert_role": next(iter(roles)),
                "samples": int(merged["keys"].shape[0]),
                "feature_dim": int(merged["features"].shape[1]),
                "feature_protocol": str(constant_metadata["feature_protocol"].item())
                if "feature_protocol" in constant_metadata
                else None,
                "parts": len(payloads),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
