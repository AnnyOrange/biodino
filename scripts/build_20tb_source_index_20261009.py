#!/usr/bin/env python3
"""Build a resumable, host-local payload index; never copy image archives.

Reuse read-only shuffle inventories where archive sizes match. Missing archives
are scanned by seeking over payloads. A manifest is published only after exact
pool membership, source metadata, and concentration constraints are validated.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import re
import sqlite3
import tarfile
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = Path('/mnt/deepcad_nfs/deepcad_100t/final-data/20TB_takeover_20260922_2350/route2_strict_pathology_20260923')
SOURCE_RE = re.compile(r'_s(\d+)_i(\d+)_')
POOL_NAMES = ['new15_r0', 'boundary_replay', 'boundary_novel']


def log(**kw):
    print(json.dumps(dict(time=time.time(), **kw)), flush=True)


def compact_members(key, members):
    result = []
    for m in members:
        prefix = key + '.'
        if not m['name'].startswith(prefix):
            raise ValueError((key, m['name']))
        result.append([m['name'][len(prefix):], int(m['offset']), int(m['size'])])
    if 'meta.json' not in [m[0] for m in result] or not any(m[0].startswith('ch') for m in result):
        raise ValueError(f'Incomplete sample {key}')
    return json.dumps(result, separators=(',', ':'))


def scan_archive_generic(path):
    records = []
    with tarfile.open(path, 'r:') as tf:
        last_key, members = None, []
        for m in tf:
            if not m.isfile():
                continue
            key, _ = m.name.split('.', 1)
            if last_key is not None and key != last_key:
                records.append((last_key, compact_members(last_key, members)))
                members = []
            last_key = key
            members.append(dict(name=m.name, offset=m.offset_data, size=m.size))
            # tarfile otherwise retains every TarInfo, unnecessary for a scan.
            tf.members.clear()
        if last_key is not None:
            records.append((last_key, compact_members(last_key, members)))
    return str(path), records


def scan_archive(path):
    """One small read per header; avoid tarfile's extra end-of-payload reads.

    FADV_RANDOM disables large, wasted NFS read-ahead between sparse headers.
    PAX/GNU extension records fall back to Python's complete tar parser.
    """
    records, members, last_key = [], [], None
    fd = os.open(path, os.O_RDONLY)
    try:
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        offset, total = 0, os.fstat(fd).st_size
        while offset + 512 <= total:
            buf = os.pread(fd, 512, offset)
            if buf == b'\0' * 512:
                break
            header = tarfile.TarInfo.frombuf(buf, 'utf8', 'strict')
            if not header.isfile():
                return scan_archive_generic(path)
            key, _ = header.name.split('.', 1)
            if last_key is not None and key != last_key:
                records.append((last_key, compact_members(last_key, members)))
                members = []
            last_key = key
            if offset + 512 + header.size > total:
                raise ValueError('Truncated tar: ' + str(path))
            members.append(dict(name=header.name, offset=offset + 512, size=header.size))
            offset += 512 + ((header.size + 511) // 512) * 512
        if last_key is not None:
            records.append((last_key, compact_members(last_key, members)))
    finally:
        os.close(fd)
    return str(path), records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--reuse-cache', type=Path)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Index already complete: ' + str(out))
    import fcntl
    lock = (out / 'build.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frame = pq.read_table(DATA / 'final_selection/selected_100tb_sources.parquet',
                          columns=['source_key', 'source_dataset', 'imaging_family']).to_pandas()
    if frame.source_key.duplicated().any() or frame.isna().any().any():
        raise ValueError('Source metadata is ambiguous or incomplete')
    source_ids = dict(zip(frame.source_key, range(len(frame))))
    old_oids = set(pq.read_table(DATA / 'original_5tb_oid_index_20260923/all_5tb_oids.parquet',
                                 columns=['oid']).column(0).to_pylist())
    r0 = sorted((DATA / 'route2_15tb_no_old5_micro_tars').glob('*-r0*.tar'))
    boundary = sorted((DATA / 'route2_5tb_boundary_tars').glob('*.tar'))
    if not r0 or not boundary:
        raise ValueError('Missing formal archive pools')
    paths = r0 + boundary
    inventory = {str(p): (i, 0 if i < len(r0) else 1, p.stat().st_size) for i, p in enumerate(paths)}
    db = sqlite3.connect(out / 'samples.sqlite')
    db.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-262144;'
                     'CREATE TABLE IF NOT EXISTS archives(id INTEGER PRIMARY KEY,path TEXT UNIQUE,bytes INTEGER,complete INTEGER);'
                     'CREATE TABLE IF NOT EXISTS samples(id INTEGER PRIMARY KEY,source_id INTEGER,pool INTEGER,archive_id INTEGER,key TEXT UNIQUE,members TEXT);')
    for path, (aid, _, size) in inventory.items():
        old = db.execute('SELECT path,bytes FROM archives WHERE id=?', (aid,)).fetchone()
        if old and old != (path, size):
            raise ValueError('Archive inventory changed during resumed indexing')
        db.execute('INSERT OR IGNORE INTO archives VALUES(?,?,?,0)', (aid, path, size))
    db.commit()
    done = {r[0] for r in db.execute('SELECT id FROM archives WHERE complete=1')}

    def rows_for(path, records):
        aid, coarse_pool, _ = inventory[path]
        for key, members in records:
            match = SOURCE_RE.search(key)
            if match is None:
                raise ValueError('No OID in sample key: ' + key)
            run, oid = match.groups()
            source = f'{run}:{oid}'
            sid = source_ids[source]
            pool = (1 if int(oid) in old_oids else 2) if coarse_pool else 0
            if not coarse_pool and int(oid) in old_oids:
                raise ValueError('Old OID unexpectedly occurs in novel r0: ' + source)
            yield (sid, pool, aid, key, members)

    insert = 'INSERT INTO samples(source_id,pool,archive_id,key,members) VALUES(?,?,?,?,?)'
    # Each historical worker database is processed transactionally. A failed
    # import leaves no partially indexed archive marked complete.
    candidates = sorted((DATA / 'route2_full_random_shuffle_candidate_20260927').glob('*/work/worker_*/*.sqlite'))
    for candidate in candidates:
        if args.reuse_cache:
            relative = candidate.relative_to(DATA / 'route2_full_random_shuffle_candidate_20260927')
            cached = args.reuse_cache / relative
            if cached.with_suffix('.ready').exists():
                candidate = cached
        src = sqlite3.connect(f'file:{candidate}?mode=ro', uri=True)
        archives = {aid: path for aid, path, size in src.execute('SELECT archive_id,path,bytes FROM archives')
                    if path in inventory and inventory[path][0] not in done and inventory[path][2] == size}
        if not archives:
            src.close()
            continue
        log(stage='reuse_index', database=str(candidate), archives=len(archives))
        imported = set()
        with db:
            batch = []
            for key, aid, members in src.execute('SELECT sample_id,archive_id,members FROM samples ORDER BY seq'):
                if aid not in archives:
                    continue
                path = archives[aid]
                batch.extend(rows_for(path, [(key, compact_members(key, json.loads(members)))]))
                imported.add(inventory[path][0])
                if len(batch) >= 10000:
                    db.executemany(insert, batch)
                    batch.clear()
            db.executemany(insert, batch)
            db.executemany('UPDATE archives SET complete=1 WHERE id=?', [(i,) for i in imported])
        src.close()
        done.update(imported)
        log(stage='reused', archives_done=len(done), total_archives=len(paths))
    missing = [p for p in paths if inventory[str(p)][0] not in done]
    log(stage='scan_missing_headers', archives=len(missing), workers=args.workers)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        for start in range(0, len(missing), args.workers * 4):
            futures = [executor.submit(scan_archive, p) for p in missing[start:start + args.workers * 4]]
            for future in as_completed(futures):
                path, records = future.result()
                aid = inventory[path][0]
                with db:
                    db.executemany(insert, rows_for(path, records))
                    db.execute('UPDATE archives SET complete=1 WHERE id=?', (aid,))
                done.add(aid)
                if len(done) % 20 == 0:
                    log(stage='scanned', archives_done=len(done), total_archives=len(paths))
    assert len(done) == len(paths)
    log(stage='compute_actual_policy', samples=db.execute('SELECT count(*) FROM samples').fetchone()[0])
    module_spec = importlib.util.spec_from_file_location('policy', Path(__file__).with_name('plan_20tb_resampling_adaptive_v2_20261009.py'))
    policy = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(policy)
    policy.OUT = out
    summaries, seen_sources = [], set()
    for pool, name in enumerate(POOL_NAMES):
        count = db.execute('SELECT count(*) FROM samples WHERE pool=?', (pool,)).fetchone()[0]
        sid = np.empty(count, dtype=np.int32)
        row = np.empty(count, dtype=np.int64)
        cursor = db.execute('SELECT id,source_id FROM samples WHERE pool=?', (pool,))
        n = 0
        while batch := cursor.fetchmany(100000):
            a = np.asarray(batch, dtype=np.int64)
            row[n:n+len(a)], sid[n:n+len(a)] = a[:, 0], a[:, 1]
            n += len(a)
        assert n == count
        order = np.argsort(sid, kind='stable')
        ids, starts, counts = np.unique(sid[order], return_index=True, return_counts=True)
        if seen_sources.intersection(ids.tolist()):
            raise ValueError('Source occurs in multiple indexed pools')
        seen_sources.update(ids.tolist())
        np.save(out / f'{name}_rows.npy', row[order])
        actual = frame.iloc[ids].copy()
        actual['qualified_patches'] = counts
        groups, summary = policy.policy_for_pool(actual, name)
        groups.to_csv(out / f'{name}_projects.csv', index=False)
        probability = pd.read_parquet(out / f'{name}_source_probabilities.parquet')
        assert probability.source_key.tolist() == actual.source_key.tolist()
        split = actual.source_key.str.split(':', expand=True).astype('int64')
        cdf = np.cumsum(probability.target_probability_source.to_numpy())
        assert abs(cdf[-1] - 1) < 1e-8
        cdf[-1] = 1.
        np.savez(out / f'{name}_sources.npz', cdf=cdf, starts=starts, counts=counts,
                 source_run=split.iloc[:, 0].to_numpy(), oid=split.iloc[:, 1].to_numpy())
        summaries.append(summary)
        log(stage='pool_complete', **summary)
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.execute('PRAGMA journal_mode=DELETE')
    db.close()
    streams = [dict(name='legacy_1tb', weight=.09, kind='stream', spec='/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*.tar'),
               dict(name='legacy_4tb', weight=.21, kind='stream', spec='/mnt/huawei_blm/deepcad_5t_v1/wds_patched_shuffle/filtered_mixed_train*.tar'),
               dict(name='new15_r0', weight=.4581, kind='indexed'),
               dict(name='new15_r9_mixed', weight=.1419, kind='stream', spec=str(DATA / 'route2_15tb_no_old5_micro_tars/*-r9*.tar')),
               dict(name='boundary_replay', weight=.04, kind='indexed'),
               dict(name='boundary_novel', weight=.06, kind='indexed')]
    manifest = dict(version=1, status='INDEX_COMPLETE_REQUIRES_DECODE_PREFLIGHT', index_dir=str(out),
                    database=str(out / 'samples.sqlite'), streams=streams, concentration=summaries,
                    archive_counts=dict(new15_r0=len(r0), boundary=len(boundary)),
                    io='Indexed sample payload reads, batch sorted by archive/offset; no image copies.',
                    crop_rotation='Without replacement per source within each worker; workers use independent seeds.',
                    validity='Unique sample keys and complete member structure; decoder preflight required before launch.')
    (out / 'manifest.json.tmp').write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(out / 'manifest.json.tmp', out / 'manifest.json')
    log(stage='COMPLETE', output=str(out))


if __name__ == '__main__':
    main()
