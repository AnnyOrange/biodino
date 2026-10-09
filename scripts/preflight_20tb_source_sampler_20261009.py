#!/usr/bin/env python3
"""Validate actual source probabilities, tar headers, decoding and cold I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tarfile
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.data.source_balanced import SourceBalancedDataset, SourceDraws, make_source_balanced_mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--samples-per-pool', type=int, default=512)
    args = ap.parse_args()
    torch.set_num_threads(1)
    info = json.loads(args.manifest.read_text())
    root = Path(info['index_dir'])
    db = sqlite3.connect(f'file:{info["database"]}?mode=ro&immutable=1', uri=True)
    reports = []
    for stream in info['streams']:
        if stream['kind'] != 'indexed':
            import glob
            assert glob.glob(stream['spec']), stream
            continue
        name = stream['name']
        with np.load(root / f'{name}_sources.npz') as data:
            a = {k: data[k] for k in data.files}
        rows = np.load(root / f'{name}_rows.npy', mmap_mode='r')
        probabilities = np.diff(a['cdf'], prepend=0.)
        n = 300000
        draw = SourceDraws(a['cdf'], a['starts'], a['counts'], np.random.SeedSequence([7741, 0, 0]))
        sources, positions = draw.draw(n)
        top_n = max(1, len(probabilities) // 100)
        top = np.argpartition(probabilities, -top_n)[-top_n:]
        expected = float(probabilities[top].sum())
        observed = float(np.isin(sources, top).mean())
        assert abs(observed - expected) < 6 * (expected*(1-expected)/n)**.5 + 1/n
        # Verify payload offsets against independent tar headers, not merely
        # against the same SQLite metadata used by the reader.
        for rowid in rows[positions[:16]]:
            aid, key, members = db.execute('SELECT archive_id,key,members FROM samples WHERE id=?', (int(rowid),)).fetchone()
            archive = db.execute('SELECT path FROM archives WHERE id=?', (aid,)).fetchone()[0]
            with open(archive, 'rb', buffering=0) as f:
                for suffix, offset, size in json.loads(members):
                    f.seek(offset - 512)
                    header = tarfile.TarInfo.frombuf(f.read(512), encoding='utf8', errors='strict')
                    assert header.name == key + '.' + suffix and header.size == size
        started = time.monotonic()
        dataset = SourceBalancedDataset(str(args.manifest), name, seed=9913, prefetch=32)
        iterator = iter(dataset)
        keys, oids = set(), set()
        for _ in range(args.samples_per_pool):
            sample, _ = next(iterator)
            assert sample['image'].shape[0] == 3
            assert sample['image'].min() >= 0 and sample['image'].max() <= 1
            keys.add(sample['__key__'])
            oids.add(sample['source_oid'])
        iterator.close()
        elapsed = time.monotonic() - started
        report = dict(pool=name, decoded=args.samples_per_pool, unique_crops=len(keys), unique_oids=len(oids),
                      seconds=elapsed, samples_per_second=args.samples_per_pool/elapsed,
                      draws=n, top1pct_expected=expected, top1pct_observed=observed,
                      independently_checked_tar_samples=16, failures=0)
        reports.append(report)
        print(json.dumps(report), flush=True)
        if report['samples_per_second'] < 4:
            raise RuntimeError('Cold I/O below required throughput; adjust loading before training')
    db.close()
    # Exercise the production six-pool router, including old streaming pools.
    # This private descriptor is never used by a training launcher.
    provisional = root / '.preflight_only_manifest.json'
    provisional.write_text(json.dumps(dict(info, status='PASS')))
    mixed_counts = Counter()
    try:
        mixed = make_source_balanced_mix(str(provisional), lambda image: {'image': image},
                                         3, 32, 779, True)
        iterator = iter(mixed)
        for _ in range(1024):
            sample, _ = next(iterator)
            assert torch.isfinite(sample['image']).all()
            mixed_counts[sample['mixture_domain_name']] += 1
        iterator.close()
        for stream in info['streams']:
            p = stream['weight']
            assert abs(mixed_counts[stream['name']] / 1024 - p) < 6 * (p*(1-p)/1024)**.5
    finally:
        provisional.unlink(missing_ok=True)
    record = dict(status='PASS', checked_at=time.time(), pools=reports,
                  actual_six_pool_decoded_counts=dict(mixed_counts),
                  note='Sampled decoder/header audit, not exhaustive decoding of every image. Runtime fails on invalid samples.',
                  crop_scope='Rotation is per worker; independent workers may revisit the same source/crop.')
    (root / 'PREFLIGHT.json').write_text(json.dumps(record, indent=2) + '\n')
    info['status'] = 'PASS'
    info['preflight'] = str(root / 'PREFLIGHT.json')
    hashes = {}
    for path in sorted(root.iterdir()):
        if path.suffix not in {'.npy', '.npz', '.parquet', '.csv', '.sqlite'}:
            continue
        digest = hashlib.sha256()
        with path.open('rb') as f:
            while block := f.read(16 << 20):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    info['index_sha256'] = hashes
    temporary = args.manifest.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(info, indent=2) + '\n')
    os.replace(temporary, args.manifest)
    print('PASS', flush=True)


if __name__ == '__main__':
    main()
