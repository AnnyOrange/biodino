"""Distribution, crop-rotation, rank independence, and real tar offset checks."""
import io
import json
from pathlib import Path
import sqlite3
import tarfile

import numpy as np
import pytest
import tifffile
import torch
import pandas as pd

from dinov3.data.source_balanced import SourceDraws, TarPayloadReader, SourceBalancedDataset
from dinov3.data.wds_decoder import decode_packed_sample_robust


def test_weighted_sources_and_crop_rotation():
    draw = SourceDraws(np.array([.1, .4, 1.]), np.array([0, 3, 8]), np.array([3, 5, 7]), 32)
    sources, positions = draw.draw(100000)
    np.testing.assert_allclose(np.bincount(sources) / len(sources), [.1, .3, .6], atol=.006)
    for source, n in enumerate([3, 5, 7]):
        selected = positions[sources == source]
        for start in range(0, len(selected) - n, n):
            assert len(set(selected[start:start + n])) == n
    same = SourceDraws(np.array([.1, .4, 1.]), np.array([0, 3, 8]), np.array([3, 5, 7]), 32)
    np.testing.assert_array_equal(same.draw(100000)[1], positions)
    other = SourceDraws(np.array([.1, .4, 1.]), np.array([0, 3, 8]), np.array([3, 5, 7]),
                        np.random.SeedSequence([32, 1, 0]))
    assert not np.array_equal(other.draw(100000)[1], positions)


def make_fixture(tmp_path):
    archive = tmp_path / 'images.tar'
    records, originals = [], []
    with tarfile.open(archive, 'w') as tf:
        for i in range(5):
            key = f'sample_p{i:04d}_sid{i}_s1_i42_r0_c{i}'
            content = io.BytesIO()
            tifffile.imwrite(content, np.arange(256, dtype=np.uint16).reshape(16, 16) + i)
            sample = {'ch1.tif': content.getvalue(), 'meta.json': json.dumps({'source_key': '1:42', 'source_dataset': 'project'}).encode()}
            for ext, payload in sample.items():
                member = tarfile.TarInfo(key + '.' + ext)
                member.size = len(payload)
                tf.addfile(member, io.BytesIO(payload))
            originals.append(sample)
    with tarfile.open(archive) as tf:
        members = list(tf)
        for i in range(0, len(members), 2):
            key = members[i].name.split('.', 1)[0]
            records.append((key, json.dumps([[m.name[len(key)+1:], m.offset_data, m.size] for m in members[i:i+2]])))
    db = sqlite3.connect(tmp_path / 'samples.sqlite')
    db.executescript('CREATE TABLE archives(id INTEGER PRIMARY KEY,path TEXT,bytes INTEGER);'
                     'CREATE TABLE samples(id INTEGER PRIMARY KEY,archive_id INTEGER,key TEXT,members TEXT);')
    db.execute('INSERT INTO archives VALUES(0,?,?)', (str(archive), archive.stat().st_size))
    for i, (key, members) in enumerate(records):
        db.execute('INSERT INTO samples VALUES(?,0,?,?)', (i+1, key, members))
    db.commit()
    db.close()
    np.save(tmp_path / 'test_rows.npy', np.arange(1, 6))
    np.savez(tmp_path / 'test_sources.npz', cdf=np.array([1.]), counts=np.array([5]), starts=np.array([0]),
             oid=np.array([42]), source_run=np.array([1]))
    pd.DataFrame([{'source_dataset': 'project', 'imaging_family': 'fluorescence'}]).to_parquet(
        tmp_path / 'test_source_probabilities.parquet')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(index_dir=str(tmp_path), database=str(tmp_path / 'samples.sqlite'))))
    return archive, originals, manifest


@pytest.mark.parametrize('max_open', [1, 8])
def test_read_order_decoder_and_corruption(tmp_path, max_open):
    archive, originals, manifest = make_fixture(tmp_path)
    reader = TarPayloadReader(tmp_path / 'samples.sqlite', max_open=max_open)
    samples = reader.read_batch([5, 1, 3])
    for sample, original in zip(samples, [originals[4], originals[0], originals[2]]):
        assert sample['ch1.tif'] == original['ch1.tif']
        assert torch.equal(decode_packed_sample_robust(sample, 3), decode_packed_sample_robust(original, 3))
    reader.close()
    dataset = SourceBalancedDataset(str(manifest), 'test', seed=8, prefetch=3)
    iterator = iter(dataset)
    accepted = [next(iterator)[0] for _ in range(5)]
    assert len({x['__key__'] for x in accepted}) == 5
    assert {x['source_oid'] for x in accepted} == {42}
    iterator.close()
    with archive.open('ab') as f:
        f.write(b'changed')
    reader = TarPayloadReader(tmp_path / 'samples.sqlite')
    try:
        with pytest.raises(RuntimeError, match='changed size'):
            reader.read_batch([1])
    finally:
        reader.close()


def test_invalid_probability_rejected():
    with pytest.raises(ValueError):
        SourceDraws(np.array([.4, .3, 1.]), np.arange(3), np.ones(3), 1)
