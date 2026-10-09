import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'eval/bio_segmentation/datasets/monuseg_official.py'
spec = importlib.util.spec_from_file_location('monuseg_official', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unverified_37_pool_is_blocked(tmp_path):
    with pytest.raises(RuntimeError, match='NOT TESTED'):
        module.official_paths(tmp_path, 'train', 'PENDING')


def test_identity_hash_and_roles(tmp_path):
    rows = []
    for i in range(44):
        role = 'train' if i < 24 else 'val' if i < 30 else 'test'
        row = dict(id=str(i), split=role)
        for kind in ('image', 'xml'):
            p = tmp_path / f'{i}.{kind}'
            p.write_bytes(str(i).encode())
            row[kind] = p.name
            row[kind + '_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append(row)
    manifest = tmp_path / 'monuseg2018_official_manifest.json'
    manifest.write_text(json.dumps(dict(samples=rows, official_source_url='synthetic-test',
                                        official_inventory_sha256='test-evidence')))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert len(module.official_paths(tmp_path, 'val', digest)[0]) == 6
    (tmp_path / '0.image').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        module.official_paths(tmp_path, 'train', digest)
