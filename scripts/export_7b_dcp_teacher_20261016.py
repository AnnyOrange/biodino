#!/usr/bin/env python3
"""Convert a completed DCP teacher into the existing evaluation file layout."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    metadata = FileSystemReader(args.source).read_metadata()
    expected = {k.removeprefix("model."): v for k, v in metadata.state_dict_metadata.items()
                if k.startswith("model.")}
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    converted = args.destination.with_suffix(".dcp_intermediate.pth")
    partial = args.destination.with_suffix(".partial")
    print("Converting", args.source, "tensors", len(expected), flush=True)
    dcp_to_torch_save(args.source, converted)
    checkpoint = torch.load(converted, map_location="cpu", mmap=True, weights_only=False)
    teacher = checkpoint["model"]
    assert set(teacher) == set(expected), "Missing or extra teacher parameters"
    for key, tensor in teacher.items():
        assert tensor.shape == expected[key].size and tensor.dtype == expected[key].properties.dtype, key
        assert torch.isfinite(tensor).all(), key
    # Preserve every tensor and its original dtype; no quantization or key filtering.
    torch.save({"teacher": teacher}, partial)
    reloaded = torch.load(partial, map_location="cpu", mmap=True, weights_only=False)["teacher"]
    for key, tensor in teacher.items():
        assert torch.equal(tensor, reloaded[key]), f"Round-trip mismatch: {key}"
    digest = hashlib.sha256()
    with partial.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024**2), b""):
            digest.update(chunk)
    partial.replace(args.destination)
    record = dict(source=str(args.source), destination=str(args.destination),
                  source_iteration=checkpoint.get("iteration"), tensors=len(teacher),
                  bytes=args.destination.stat().st_size, sha256=digest.hexdigest(),
                  torch=torch.__version__, exported_unix=time.time(), tensor_roundtrip_exact=True,
                  optimizer_included=False)
    args.destination.with_suffix(".export.json").write_text(json.dumps(record, indent=2) + "\n")
    converted.unlink()
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
