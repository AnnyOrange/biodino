#!/usr/bin/env python3
"""Lock the authoritative MoNuSeg 30/7/14 sample assignment for v4."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/mnt/huawei_deepcad/benchmark/segmentation/monuseg/extracted')
SOURCE_MANIFEST = SOURCE / 'monuseg2018_train30_extra7val_test14_manifest.json'
SOURCE_SHA256 = '52dd206e30f91238666b80e5b99501bb3111f79129c8e6400d91a2895a3632ec'
OUT = ROOT / 'outputs/02_eval_inputs/monuseg37_v4_20261012/manifest.json'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    source_bytes = SOURCE_MANIFEST.read_bytes()
    if sha256(source_bytes) != SOURCE_SHA256:
        raise RuntimeError('Authoritative MoNuSeg 30/7/14 manifest changed')
    source = json.loads(source_bytes)
    if source.get('protocol') != 'monuseg2018-train30-extra7val-test14-v1':
        raise RuntimeError('Unexpected source protocol')
    samples = source['samples']
    counts = {split: sum(item['split'] == split for item in samples)
              for split in ('train', 'val', 'test')}
    if counts != {'train': 30, 'val': 7, 'test': 14}:
        raise RuntimeError(f'Unexpected split counts: {counts}')
    ids = {split: {item['id'] for item in samples if item['split'] == split}
           for split in counts}
    if any(ids[a] & ids[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise RuntimeError('MoNuSeg sample IDs overlap across splits')
    for item in samples:
        for field in ('image', 'xml'):
            path = SOURCE / item[field]
            if not path.is_file() or sha256(path.read_bytes()) != item[f'{field}_sha256']:
                raise RuntimeError(f"Source file hash mismatch: {item[field]}")

    manifest = {
        'protocol': source['protocol'],
        'source': source['official_source_url'],
        'pool': 'official challenge train30 + seven extra annotated training-archive samples + official test14',
        'selection': source['selection'],
        'source_manifest': str(SOURCE_MANIFEST),
        'source_manifest_sha256': SOURCE_SHA256,
        'archive_sha256': source['official_archive_sha256'],
        'inventory_sha256': source['official_inventory_sha256'],
        'counts': counts,
        'samples': samples,
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2) + '\n'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.read_text() != payload:
        raise RuntimeError('Existing v4 MoNuSeg lock differs; refusing to overwrite')
    OUT.write_text(payload)
    print(json.dumps({'manifest': str(OUT), 'sha256': sha256(OUT.read_bytes()),
                      'counts': counts}, indent=2))


if __name__ == '__main__':
    main()
