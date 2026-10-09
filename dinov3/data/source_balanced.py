"""Exact source-level sampling from a validated local tar payload inventory.

Only metadata is indexed locally. Image payloads stay in their original tars.
Selections are independent of read order: bounded batches are sorted for I/O,
then yielded in their original random order. Decode errors abort instead of
silently changing the intended source distribution.
"""
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import threading

import numpy as np
import torch


class SourceDraws:
    """Sample source probabilities, rotating that source's tiles per worker."""

    def __init__(self, cdf, starts, counts, seed):
        self.cdf, self.starts, self.counts = cdf, starts, counts
        if len(cdf) == 0 or not np.all(np.isfinite(cdf)) or not np.all(np.diff(cdf) > 0):
            raise ValueError('Source probabilities must be finite and positive')
        if abs(float(cdf[-1]) - 1.) > 1e-10 or np.any(counts <= 0):
            raise ValueError('Invalid source CDF/counts')
        self.rng = np.random.default_rng(seed)
        self.rotation = {}

    def draw(self, batch_size):
        sources = np.searchsorted(self.cdf, self.rng.random(batch_size), side='right')
        positions = np.empty(batch_size, dtype=np.int64)
        for j, source in enumerate(sources):
            source = int(source)
            n = int(self.counts[source])
            if source not in self.rotation or self.rotation[source][2] == n:
                start = int(self.rng.integers(n))
                stride = int(self.rng.integers(1, n + 1))
                while math.gcd(stride, n) != 1:
                    stride = int(self.rng.integers(1, n + 1))
                self.rotation[source] = [start, stride, 0]
            state = self.rotation[source]
            positions[j] = self.starts[source] + (state[0] + state[1] * state[2]) % n
            state[2] += 1
        return sources, positions


class TarPayloadReader:
    def __init__(self, database, max_open=64, read_workers=4):
        self.db = sqlite3.connect(f'file:{database}?mode=ro&immutable=1', uri=True)
        self.db.execute('PRAGMA cache_size=-8192')
        self.archives = {i: (path, size) for i, path, size in self.db.execute('SELECT id,path,bytes FROM archives')}
        read_workers = min(max_open, read_workers)
        self.file_caches = {}
        self.cache_lock = threading.Lock()
        self.max_open = max(1, max_open // read_workers)
        self.executor = ThreadPoolExecutor(max_workers=read_workers)
        self.bytes_read = 0

    def read_batch(self, row_ids):
        records = []
        for i, rowid in enumerate(row_ids):
            record = self.db.execute('SELECT archive_id,key,members FROM samples WHERE id=?', (int(rowid),)).fetchone()
            if record is None:
                raise KeyError(f'Missing sample row {rowid}')
            aid, key, encoded = record
            members = json.loads(encoded)
            lo = min(m[1] for m in members)
            hi = max(m[1] + m[2] for m in members)
            records.append((aid, lo, hi, i, key, members))
        result = [None] * len(records)
        for i, sample, size in self.executor.map(self._read_payload, sorted(records)):
            self.bytes_read += size
            result[i] = sample
        return result

    def _read_payload(self, record):
        aid, lo, hi, i, key, members = record
        path, size = self.archives[aid]
        # Each thread owns its handles: eviction can never close a descriptor
        # used by another outstanding pread. At most 64 handles in total.
        with self.cache_lock:
            files = self.file_caches.setdefault(threading.get_ident(), OrderedDict())
        if aid in files:
            fd = files.pop(aid)
        else:
            fd = os.open(path, os.O_RDONLY)
            if os.fstat(fd).st_size != size:
                os.close(fd)
                raise RuntimeError('Indexed tar changed size: ' + path)
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        files[aid] = fd
        if len(files) > self.max_open:
            _, old_fd = files.popitem(last=False)
            os.close(old_fd)
        if lo < 0 or hi > size or hi <= lo or hi - lo > 128 * 1024 * 1024:
            raise ValueError(f'Invalid payload range {key}: {lo}..{hi}')
        payload = os.pread(fd, hi - lo, lo)
        if len(payload) != hi - lo:
            raise IOError('Short tar payload read: ' + path)
        sample = {'__key__': key, '__url__': path}
        for suffix, offset, length in members:
            sample[suffix] = payload[offset-lo:offset-lo+length]
        return i, sample, len(payload)

    def close(self):
        self.executor.shutdown(wait=True)
        for files in self.file_caches.values():
            for fd in files.values():
                os.close(fd)
        self.file_caches.clear()
        self.db.close()


class SourceBalancedDataset(torch.utils.data.IterableDataset):
    def __init__(self, manifest, pool, transform=None, target_channels=3, seed=0,
                 prefetch=32, rank=None):
        super().__init__()
        self.manifest, self.pool = manifest, pool
        self.transform, self.target_channels = transform, target_channels
        self.seed, self.prefetch = int(seed), int(prefetch)
        # Capture rank in the parent process (also works with spawn workers).
        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.environ.get('RANK', 0))
        self.rank = rank

    def __iter__(self):
        from .wds_decoder import decode_packed_sample_robust

        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        info = json.loads(Path(self.manifest).read_text())
        root = Path(info['index_dir'])
        with np.load(root / f'{self.pool}_sources.npz') as data:
            arrays = {key: data[key] for key in data.files}
        import pyarrow.parquet as pq

        metadata_table = pq.read_table(root / f'{self.pool}_source_probabilities.parquet',
                                       columns=['source_dataset', 'imaging_family'])
        encoded_project = metadata_table['source_dataset'].combine_chunks().dictionary_encode()
        encoded_modality = metadata_table['imaging_family'].combine_chunks().dictionary_encode()
        project_codes = encoded_project.indices.to_numpy()
        modality_codes = encoded_modality.indices.to_numpy()
        project_names = encoded_project.dictionary.to_pylist()
        modality_names = encoded_modality.dictionary.to_pylist()
        if len(project_codes) != len(arrays['cdf']):
            raise ValueError('Source metadata and probability arrays differ in length')
        rows = np.load(root / f'{self.pool}_rows.npy', mmap_mode='r')
        stream_seed = np.random.SeedSequence([self.seed, self.rank, worker_id])
        draws = SourceDraws(arrays['cdf'], arrays['starts'], arrays['counts'], stream_seed)
        reader = TarPayloadReader(info['database'])
        accepted, unique_oids = 0, set()
        project_hits, modality_hits = Counter(), Counter()
        started = time.monotonic()
        audit_dir = os.environ.get('SOURCE_SAMPLER_AUDIT_DIR')
        try:
            while True:
                sources, positions = draws.draw(self.prefetch)
                samples = reader.read_batch(rows[positions])
                for source, sample in zip(sources, samples):
                    oid = int(arrays['oid'][source])
                    source_key = f"{int(arrays['source_run'][source])}:{oid}"
                    metadata = json.loads(sample['meta.json'])
                    if metadata.get('source_key') != source_key:
                        raise ValueError(f'Source metadata mismatch: {sample["__key__"]} vs {source_key}')
                    project = project_names[project_codes[source]]
                    modality = modality_names[modality_codes[source]]
                    if metadata.get('source_dataset') != project:
                        raise ValueError('Project metadata mismatch: ' + sample['__key__'])
                    tensor = decode_packed_sample_robust(sample, target_channels=self.target_channels, p_low=1., p_high=99.)
                    if tensor is None or not torch.isfinite(tensor).all():
                        raise RuntimeError('Invalid decoded sample: ' + sample['__key__'])
                    result = self.transform(tensor) if self.transform else {'image': tensor}
                    result.update(__key__=sample['__key__'], __url__=sample['__url__'],
                                  source_key=source_key, source_oid=oid, source_pool=self.pool,
                                  source_project=project, source_modality=modality)
                    accepted += 1
                    unique_oids.add(oid)
                    project_hits[project] += 1
                    modality_hits[modality] += 1
                    if audit_dir and (accepted == 1 or accepted % 1024 == 0):
                        directory = Path(audit_dir)
                        directory.mkdir(parents=True, exist_ok=True)
                        record = dict(pool=self.pool, rank=self.rank, worker=worker_id,
                                      seed=self.seed, accepted=accepted, unique_oids=len(unique_oids),
                                      project_hits=dict(project_hits), modality_hits=dict(modality_hits),
                                      bytes_read=reader.bytes_read, elapsed=time.monotonic()-started,
                                      decode_rejections=0, time=time.time())
                        with (directory / f'{self.pool}_r{self.rank}_w{worker_id}.jsonl').open('a') as f:
                            f.write(json.dumps(record) + '\n')
                    yield result, ()
        finally:
            reader.close()


def make_source_balanced_mix(manifest, transform, target_channels, shuffle_buffer,
                             resample_seed, deterministic_resampling):
    from .loaders import _make_packed_robust_webdataset
    from .wds_pipeline import WeightedIterableDataset

    info = json.loads(Path(manifest).read_text())
    if info['status'] != 'PASS':
        raise ValueError('Source index has not passed decoder preflight: ' + manifest)
    datasets, names, weights = [], [], []
    for i, stream in enumerate(info['streams']):
        seed = int(resample_seed) + i * 1000003
        if stream['kind'] == 'indexed':
            ds = SourceBalancedDataset(manifest, stream['name'], transform,
                                       target_channels or 3, seed=seed)
        else:
            ds = _make_packed_robust_webdataset(stream['spec'] + '::pct=1,99', transform,
                                               target_channels, shuffle_buffer, seed,
                                               deterministic_resampling)
        datasets.append(ds)
        names.append(stream['name'])
        weights.append(stream['weight'])
    if abs(sum(weights) - 1.) > 1e-10:
        raise ValueError('Pool weights do not sum to one')
    return WeightedIterableDataset(datasets, weights, seed=resample_seed, names=names, domain_block_size=1)
