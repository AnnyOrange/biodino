#!/usr/bin/env python3
"""Extract the EMA teacher from a committed 7B DCP optimizer checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)

    source_metadata = FileSystemReader(args.source).read_metadata().state_dict_metadata
    reference_metadata = FileSystemReader(args.reference).read_metadata().state_dict_metadata
    prefix = 'model.teacher.'
    teacher_metadata = {key.removeprefix(prefix): value for key, value in source_metadata.items()
                        if key.startswith(prefix)}
    expected = {key.removeprefix('model.'): value for key, value in reference_metadata.items()
                if key.startswith('model.')}
    if not teacher_metadata or set(teacher_metadata) != set(expected):
        raise RuntimeError('Paused teacher keys differ from the verified eval teacher layout')
    for key, info in teacher_metadata.items():
        original = expected[key]
        if info.size != original.size or info.properties.dtype != original.properties.dtype:
            raise RuntimeError(f'Paused teacher shape or dtype differs: {key}')

    tensors = {prefix + key: torch.empty(tuple(info.size), dtype=info.properties.dtype)
               for key, info in teacher_metadata.items()}
    dcp.load(tensors, storage_reader=FileSystemReader(args.source))
    teacher = {key.removeprefix(prefix): tensor for key, tensor in tensors.items()}
    for key, tensor in teacher.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f'Non-finite teacher tensor: {key}')

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    partial = args.destination.with_suffix('.partial')
    torch.save({'teacher': teacher}, partial)
    reloaded = torch.load(partial, map_location='cpu', mmap=True, weights_only=False)['teacher']
    if set(reloaded) != set(teacher):
        raise RuntimeError('Exported teacher keys changed on reload')
    for key, tensor in teacher.items():
        if not torch.equal(tensor, reloaded[key]):
            raise RuntimeError(f'Exported teacher differs on reload: {key}')
    digest = sha256(partial)
    partial.replace(args.destination)
    record = dict(source=str(args.source), reference=str(args.reference),
                  destination=str(args.destination), step=int(args.source.name),
                  tensors=len(teacher), bytes=args.destination.stat().st_size,
                  sha256=digest, exported_unix=time.time(),
                  tensor_roundtrip_exact=True, optimizer_included=False)
    args.destination.with_suffix('.export.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
