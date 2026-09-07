#!/usr/bin/env python3
"""Build one expert's full-bank cross-domain nearest-neighbor cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


UNKNOWN = frozenset({"", "unknown", "unresolved", "none", "nan", "n/a", "na"})
SCOPES = (
    "cross_organism",
    "cross_acquisition",
    "cross_organism_or_acquisition",
    "cross_organism_and_acquisition",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, default="cross_organism_or_acquisition")
    parser.add_argument("--topk-max", type=int, default=100)
    parser.add_argument("--query-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--shuffled-control", action="store_true")
    parser.add_argument("--expert-index", type=int, default=0)
    return parser.parse_args()


def encode_metadata(values: np.ndarray) -> np.ndarray:
    normalized = np.asarray([str(value).strip().lower() for value in values])
    known_values = sorted(set(normalized).difference(UNKNOWN))
    mapping = {value: index + 1 for index, value in enumerate(known_values)}
    return np.asarray([mapping.get(value, 0) for value in normalized], dtype=np.int64)


def candidate_mask(
    query_organism: torch.Tensor,
    query_acquisition: torch.Tensor,
    candidate_organism: torch.Tensor,
    candidate_acquisition: torch.Tensor,
    *,
    scope: str,
) -> torch.Tensor:
    cross_organism = (
        (query_organism[:, None] != 0)
        & (candidate_organism[None, :] != 0)
        & (query_organism[:, None] != candidate_organism[None, :])
    )
    cross_acquisition = (
        (query_acquisition[:, None] != 0)
        & (candidate_acquisition[None, :] != 0)
        & (query_acquisition[:, None] != candidate_acquisition[None, :])
    )
    if scope == "cross_organism":
        return cross_organism
    if scope == "cross_acquisition":
        return cross_acquisition
    if scope == "cross_organism_or_acquisition":
        return cross_organism | cross_acquisition
    if scope == "cross_organism_and_acquisition":
        return cross_organism & cross_acquisition
    raise ValueError(f"Unsupported scope {scope!r}")


def key_digest(keys: np.ndarray) -> str:
    digest = hashlib.sha256()
    for key in keys.astype(str):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "nonzero_fraction": float(np.mean(values > 0)),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.topk_max <= 0 or args.query_batch_size <= 0:
        raise ValueError("topk-max and query-batch-size must be positive")
    if args.expert_index < 0:
        raise ValueError("expert-index must be non-negative")

    with np.load(args.bank, allow_pickle=False) as payload:
        required = {"keys", "features", "reliability", "organism", "acquisition_family"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Bank is missing arrays: {sorted(missing)}")
        keys = np.asarray(payload["keys"]).astype(str)
        features = np.asarray(payload["features"])
        reliability = np.asarray(payload["reliability"], dtype=np.float32)
        organism = np.asarray(payload["organism"]).astype(str)
        acquisition = np.asarray(payload["acquisition_family"]).astype(str)

    sample_count = len(keys)
    if features.ndim != 2 or features.shape[0] != sample_count:
        raise ValueError("features must have shape [samples, dimensions]")
    for name, values in {
        "reliability": reliability,
        "organism": organism,
        "acquisition_family": acquisition,
    }.items():
        if values.shape != (sample_count,):
            raise ValueError(f"{name} has shape {values.shape}, expected {(sample_count,)}")
    if not np.isfinite(features).all() or not np.isfinite(reliability).all():
        raise ValueError("Bank contains non-finite features or reliability")

    organism_codes = encode_metadata(organism)
    acquisition_codes = encode_metadata(acquisition)
    metadata_known = (organism_codes != 0) | (acquisition_codes != 0)
    active_indices = np.flatnonzero(metadata_known & (reliability > 0)).astype(np.int64)
    if len(active_indices) < 2:
        raise ValueError("Fewer than two metadata-known reliable samples")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.compute_dtype]
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    shuffle_offset = 0
    if args.shuffled_control:
        shuffle_offset = 1 + (args.expert_index % (sample_count - 1))
    feature_rows = (active_indices - shuffle_offset) % sample_count
    candidate_features = torch.from_numpy(np.asarray(features[feature_rows], dtype=np.float32)).to(device)
    candidate_features = F.normalize(candidate_features, dim=-1).to(dtype=dtype)
    candidate_organism = torch.from_numpy(organism_codes[active_indices]).to(device)
    candidate_acquisition = torch.from_numpy(acquisition_codes[active_indices]).to(device)
    candidate_original = torch.from_numpy(active_indices).to(device)
    candidate_reliability = torch.from_numpy(reliability[active_indices]).to(device)

    effective_topk = min(args.topk_max, len(active_indices) - 1)
    neighbors = np.full((len(active_indices), args.topk_max), -1, dtype=np.int32)
    similarities = np.full((len(active_indices), args.topk_max), -np.inf, dtype=np.float16)
    pair_weights = np.zeros((len(active_indices), args.topk_max), dtype=np.float16)
    candidate_counts = np.zeros(len(active_indices), dtype=np.int32)

    for start in range(0, len(active_indices), args.query_batch_size):
        stop = min(start + args.query_batch_size, len(active_indices))
        query_original_np = active_indices[start:stop]
        query_original = torch.from_numpy(query_original_np).to(device)
        query_rows = (query_original_np - shuffle_offset) % sample_count
        query_features = torch.from_numpy(np.asarray(features[query_rows], dtype=np.float32)).to(device)
        query_features = F.normalize(query_features, dim=-1).to(dtype=dtype)
        mask = candidate_mask(
            torch.from_numpy(organism_codes[query_original_np]).to(device),
            torch.from_numpy(acquisition_codes[query_original_np]).to(device),
            candidate_organism,
            candidate_acquisition,
            scope=args.scope,
        )
        mask &= query_original[:, None] != candidate_original[None, :]
        counts = mask.sum(dim=1)
        similarity = query_features @ candidate_features.T
        similarity.masked_fill_(~mask, -torch.inf)
        values, positions = torch.topk(
            similarity,
            k=effective_topk,
            dim=1,
            largest=True,
            sorted=True,
        )
        valid = torch.isfinite(values)
        original_neighbors = candidate_original[positions]
        original_neighbors = original_neighbors.masked_fill(~valid, -1)
        weights = (
            torch.from_numpy(reliability[query_original_np]).to(device)[:, None]
            * candidate_reliability[positions]
        ).masked_fill(~valid, 0)

        width = effective_topk
        neighbors[start:stop, :width] = original_neighbors.cpu().numpy().astype(np.int32)
        similarities[start:stop, :width] = values.float().cpu().numpy().astype(np.float16)
        pair_weights[start:stop, :width] = weights.float().cpu().numpy().astype(np.float16)
        candidate_counts[start:stop] = counts.cpu().numpy().astype(np.int32)
        print(
            f"[global-knn] {stop}/{len(active_indices)} bank={args.bank.name} "
            f"shuffled={args.shuffled_control}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        keys=keys,
        key_digest=np.asarray(key_digest(keys)),
        bank_path=np.asarray(str(args.bank)),
        scope=np.asarray(args.scope),
        shuffled_control=np.asarray(args.shuffled_control),
        shuffle_offset=np.asarray(shuffle_offset, dtype=np.int64),
        query_indices=active_indices.astype(np.int32),
        neighbor_indices=neighbors,
        similarities=similarities,
        pair_weights=pair_weights,
        reliability=reliability,
        organism=organism,
        acquisition_family=acquisition,
        candidate_counts=candidate_counts,
    )
    valid_counts = np.sum(neighbors >= 0, axis=1)
    report = {
        "bank": str(args.bank),
        "output": str(args.output),
        "samples": sample_count,
        "active_samples": int(len(active_indices)),
        "scope": args.scope,
        "topk_max": args.topk_max,
        "effective_topk": effective_topk,
        "shuffled_control": args.shuffled_control,
        "shuffle_offset": shuffle_offset,
        "device": str(device),
        "compute_dtype": str(dtype),
        "candidate_counts": summary(candidate_counts),
        "valid_neighbor_counts": summary(valid_counts),
        "key_digest": key_digest(keys),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
