#!/usr/bin/env python3
"""Encode finite packed-WebDataset shards with an audited frozen expert."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dinov3.data.wds_decoder import decode_packed_sample_robust


MEMBER_PATTERN = re.compile(r"^(?P<key>.+)\.(?:ch\d+\.tiff?|meta\.json)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("virchow2", "gigapath", "conch", "bioclip", "jump_cp"),
        required=True,
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--encoder-module-root",
        type=Path,
        default=Path("/mnt/huawei_deepcad/benchmark_model"),
    )
    parser.add_argument("--domain-catalog", type=Path)
    parser.add_argument(
        "--expert-role",
        choices=("general", "organism_cell", "cell", "tissue"),
        default="general",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--p-low", type=float, default=1.0)
    parser.add_argument("--p-high", type=float, default=99.0)
    return parser.parse_args()


def load_catalog(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    catalog = {}
    for row in rows:
        domain = row.get("domain", "").lower()
        if not domain:
            continue
        catalog[domain] = row
        catalog[domain.split(":", 1)[-1]] = row
    return catalog


def sample_metadata(meta: dict, catalog: dict[str, dict[str, str]]) -> dict[str, str]:
    dataset_name = str(meta.get("dataset_name", "")).lower()
    original_path = str(meta.get("original_path", "")).lower()
    row = catalog.get(dataset_name)
    if row is None:
        for domain, candidate in catalog.items():
            if domain.startswith("idr") and domain in original_path:
                row = candidate
                break
    row = row or {}
    return {
        "domain": str(row.get("domain", dataset_name or "unresolved")),
        "organism": str(row.get("organism", "")),
        "acquisition_family": str(row.get("acquisition_family", "unresolved")),
        "sample_type": str(row.get("sample_type", "")),
    }


def role_reliability(role: str, metadata: dict[str, str]) -> float:
    if role == "general":
        return 1.0
    sample_type = metadata["sample_type"].lower()
    acquisition = metadata["acquisition_family"].lower()
    organism = metadata["organism"].lower()
    is_known_organism = organism not in {"", "unknown", "unresolved", "none", "nan"}
    is_human = organism in {"human", "homo sapiens", "h. sapiens"}
    if role == "tissue":
        if sample_type == "tissue" or acquisition == "histopathology":
            score = 1.0
        elif acquisition == "electron_microscopy":
            score = 0.65
        else:
            score = 0.20
        # Virchow-like pathology experts are most reliable on human tissue,
        # but retain a weak vote so cross-species consensus remains testable.
        return score if not is_known_organism or is_human else 0.75 * score
    cell_like = sample_type == "cell" or acquisition in {
        "fluorescence_microscopy",
        "imaging_mass_cytometry",
        "label_free_microscopy",
    }
    if role == "cell":
        return 1.0 if cell_like else 0.20
    # BioCLIP-like experts contribute organism semantics whenever species is
    # known and receive extra confidence on cellular microscopy.
    if cell_like:
        return 1.0
    if is_known_organism:
        return 0.85
    return 0.35


def iter_raw_samples(shard: Path):
    current_key = None
    current_sample: dict[str, bytes] = {}
    with tarfile.open(shard, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            match = MEMBER_PATTERN.match(member.name)
            if match is None:
                continue
            key = match.group("key")
            if current_key is not None and key != current_key:
                yield current_key, current_sample
                current_sample = {}
            current_key = key
            handle = archive.extractfile(member)
            if handle is None:
                continue
            suffix = member.name[len(key) + 1 :]
            current_sample[suffix] = handle.read()
        if current_key is not None:
            yield current_key, current_sample


def decode_sample(raw: dict[str, bytes], p_low: float, p_high: float) -> tuple[Image.Image, dict] | None:
    tensor = decode_packed_sample_robust(
        raw,
        target_channels=3,
        p_low=p_low,
        p_high=p_high,
    )
    if tensor is None:
        return None
    array = (
        tensor[:3]
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    meta_bytes = raw.get("meta.json", b"{}")
    try:
        meta = json.loads(meta_bytes)
    except (TypeError, json.JSONDecodeError):
        meta = {}
    return Image.fromarray(array, mode="RGB"), meta


def build_encoder(args: argparse.Namespace):
    sys.path.insert(0, str(args.encoder_module_root))
    module = importlib.import_module("benchmark_eval.encoders")
    module.MODEL_ROOT = args.encoder_module_root
    if args.model in {"virchow2", "gigapath"}:
        cls = module.TimmEncoder
    elif args.model == "conch":
        cls = module.CONCHEncoder
    elif args.model == "bioclip":
        cls = module.BioCLIPEncoder
    else:
        cls = module.ChannelViTJumpCPEncoder
    return cls(args.model_path, args.device, args.batch_size)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.torch_num_threads < 0:
        raise ValueError("--torch-num-threads must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if not 0 <= args.p_low < args.p_high <= 100:
        raise ValueError("Expected 0 <= p-low < p-high <= 100")
    for shard in args.shards:
        if not shard.is_file():
            raise FileNotFoundError(shard)

    if args.torch_num_threads:
        torch.set_num_threads(args.torch_num_threads)
        torch.set_num_interop_threads(min(4, args.torch_num_threads))
    encoder = build_encoder(args)
    catalog = load_catalog(args.domain_catalog)
    keys: list[str] = []
    features: list[np.ndarray] = []
    reliability: list[float] = []
    domains: list[str] = []
    organisms: list[str] = []
    acquisitions: list[str] = []
    sample_types: list[str] = []
    image_batch: list[Image.Image] = []
    pending: list[tuple[str, dict[str, str]]] = []

    def flush() -> None:
        if not image_batch:
            return
        device_type = torch.device(args.device).type
        amp_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[args.amp_dtype]
        with torch.autocast(
            device_type=device_type,
            dtype=amp_dtype,
            enabled=device_type == "cuda" and args.amp_dtype != "fp32",
        ):
            encoded = np.asarray(encoder.encode_pil(image_batch), dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] != len(pending):
            raise RuntimeError(
                f"Encoder returned {encoded.shape} for a batch of {len(pending)} samples"
            )
        features.append(encoded.astype(np.float16))
        for (key, metadata), _feature in zip(pending, encoded):
            keys.append(key)
            reliability.append(role_reliability(args.expert_role, metadata))
            domains.append(metadata["domain"])
            organisms.append(metadata["organism"])
            acquisitions.append(metadata["acquisition_family"])
            sample_types.append(metadata["sample_type"])
        image_batch.clear()
        pending.clear()
        if len(keys) % (20 * args.batch_size) == 0:
            print(f"[expert-bank] {args.model}: {len(keys)} samples", flush=True)

    stop = False
    for shard in args.shards:
        for key, raw in iter_raw_samples(shard):
            decoded = decode_sample(raw, args.p_low, args.p_high)
            if decoded is None:
                continue
            image, meta = decoded
            metadata = sample_metadata(meta, catalog)
            image_batch.append(image)
            pending.append((f"{shard.name}::{key}", metadata))
            if len(image_batch) >= args.batch_size:
                flush()
            if args.max_samples is not None and len(keys) + len(pending) >= args.max_samples:
                stop = True
                break
        if stop:
            break
    flush()
    if not keys:
        raise RuntimeError("No packed samples were encoded")
    feature_array = np.concatenate(features, axis=0)
    if feature_array.shape[0] != len(keys):
        raise RuntimeError("Feature and key counts diverged")
    if len(set(keys)) != len(keys):
        raise RuntimeError("Duplicate shard-qualified sample keys were produced")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        keys=np.asarray(keys),
        features=feature_array,
        reliability=np.asarray(reliability, dtype=np.float32),
        domain=np.asarray(domains),
        organism=np.asarray(organisms),
        acquisition_family=np.asarray(acquisitions),
        sample_type=np.asarray(sample_types),
        model=np.asarray(args.model),
        expert_role=np.asarray(args.expert_role),
        source_shards=np.asarray([str(path) for path in args.shards]),
        normalization_percentiles=np.asarray([args.p_low, args.p_high], dtype=np.float32),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model": args.model,
                "expert_role": args.expert_role,
                "samples": len(keys),
                "feature_dim": int(feature_array.shape[1]),
                "active_reliability": int(np.count_nonzero(reliability)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
