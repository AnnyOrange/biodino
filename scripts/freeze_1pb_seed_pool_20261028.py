#!/usr/bin/env python3
"""Freeze all verified rolling 1PB batches into a two-rank WDS inventory."""

import hashlib
import json
from pathlib import Path


SOURCES = (
    Path('/mnt/huawei_blm/rolling_1pb_wds_v1/queue'),
    Path('/mnt/liyixin22/rolling_1pb_wds_v1/queue'),
)
ROOT = Path('/mnt/liyixin22/seed_repeats_1pb_100tb_20261028')
POOL = ROOT / '1pb_pool'
TARGET_UPDATES = 80000
GLOBAL_BATCH = 1024


def main():
    candidates = {}
    for source in SOURCES:
        for state in ('completed', 'ready', 'active'):
            for record in (source / state).glob('batch-*/READY.json'):
                data = json.loads(record.read_text())
                if data.get('status') != 'PASS' or data.get('batch_id') != record.parent.name:
                    continue
                if len(data.get('shards', [])) != data.get('shard_count') or (
                        sum(int(item['samples']) for item in data['shards']) != data['samples']):
                    raise ValueError(f'Invalid READY inventory: {record}')
                current = candidates.get(record.parent.name)
                if current:
                    prior = current[1]
                    if (prior['samples'], prior['shard_count']) != (
                            data['samples'], data['shard_count']):
                        raise ValueError(f'Conflicting batch inventory: {record.parent.name}')
                # Prefer physical shards over symlinks and the new writable disk.
                priority = (not (record.parent / 'shards').is_symlink(),
                            source == SOURCES[1])
                if current is None or priority > current[2]:
                    candidates[record.parent.name] = (record, data, priority)

    if POOL.exists():
        raise FileExistsError(f'Frozen pool already exists: {POOL}')
    if len(candidates) < 160:
        raise ValueError(f'Only {len(candidates)} verified batches; expected at least 160')
    ranks = [[], []]
    counts = [0, 0]
    batches = []
    links = []
    for batch, (record, data, _) in sorted(candidates.items()):
        source = record.parent.resolve()
        shards_dir = source / 'shards'
        if not shards_dir.is_dir():
            raise FileNotFoundError(shards_dir)
        links.append((batch, source))
        for item in data['shards']:
            source_tar = shards_dir / item['path']
            if not source_tar.is_file() or source_tar.stat().st_size != item['bytes']:
                raise ValueError(f'Missing or changed verified shard: {source_tar}')
            rank = 0 if counts[0] <= counts[1] else 1
            ranks[rank].append(str(POOL / batch / 'shards' / item['path']))
            counts[rank] += int(item['samples'])
        batches.append(dict(batch=batch, source=str(source), samples=data['samples'],
                            shards=data['shard_count'],
                            ready_sha256=hashlib.sha256(record.read_bytes()).hexdigest()))
    target_per_rank = TARGET_UPDATES * GLOBAL_BATCH // 2
    if min(counts) < target_per_rank:
        raise ValueError(f'Insufficient verified samples per rank: {counts}, need {target_per_rank}')
    ROOT.mkdir(parents=True, exist_ok=True)
    POOL.mkdir()
    for batch, source in links:
        (POOL / batch).symlink_to(source, target_is_directory=True)
    assignment = dict(status='PASS', world_size=2, rank_shards=ranks,
                      rank_sample_counts=counts, target_per_rank=target_per_rank,
                      source='verified rolling 1PB READY batches, deduplicated by batch ID')
    (ROOT / '1pb_two_rank_assignment.json').write_text(
        json.dumps(assignment, separators=(',', ':')) + '\n')
    manifest = dict(status='PASS', batches=batches, unique_batches=len(batches),
                    sample_count=sum(row['samples'] for row in batches),
                    shard_count=sum(row['shards'] for row in batches),
                    rank_sample_counts=counts, target_updates=TARGET_UPDATES,
                    effective_global_batch=GLOBAL_BATCH,
                    dataset_path=f'packwds_once_robust:{POOL}/batch-*/shards/*.tar::pct=1,99',
                    assignment=str(ROOT / '1pb_two_rank_assignment.json'))
    (ROOT / '1pb_pool_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: manifest[key] for key in
                      ('unique_batches', 'sample_count', 'shard_count', 'rank_sample_counts')},
                     indent=2))


if __name__ == '__main__':
    main()
