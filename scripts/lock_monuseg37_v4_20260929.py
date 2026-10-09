#!/usr/bin/env python3
"""Lock the 37-pool MoNuSeg split used for matched cross-model comparison."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/huawei_deepcad/benchmark/segmentation')
EXTRACTED = DATA / 'monuseg/extracted'
ARCHIVES = DATA / 'MoNuSeg'
OUT = ROOT / 'outputs/02_eval_inputs/monuseg37_v4_20260929/manifest.json'
VAL_SHA = '932a09d0e936bd2ee83438145f6e6955dac224b747f2536ae06d31c764ff5c91'
TRAIN_SHA = '25d3d3185bb2970b397cafa72eb664c9b4d24294aee382e7e3df9885affce742'
TEST_SHA = '13e522387ae8b1bcc0530e13ff9c7b4d91ec74959ef6f6e57747368d7ee6f88a'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def members(archive):
    result = {}
    for name in archive.namelist():
        path = Path(name)
        if '__MACOSX' in path.parts or path.name.startswith('.'):
            continue
        if path.suffix.lower() not in ('.tif', '.tiff', '.xml'):
            continue
        result.setdefault(path.stem, {})[path.suffix.lower()] = name
    return result


def main():
    val_path = EXTRACTED / 'monuseg_val_indices.npy'
    assert sha(val_path.read_bytes()) == VAL_SHA
    indices = set(map(int, np.load(val_path).tolist()))
    assert len(indices) == 7 and max(indices) < 37
    records = []
    for split, filename, digest, expected in (
        ('train_pool', 'MoNuSeg 2018 Training Data.zip', TRAIN_SHA, 37),
        ('test', 'MoNuSegTestData.zip', TEST_SHA, 14),
    ):
        path = ARCHIVES / filename
        assert sha(path.read_bytes()) == digest
        with ZipFile(path) as archive:
            assert archive.testzip() is None
            entries = members(archive)
            assert len(entries) == expected
            for i, identity in enumerate(sorted(entries)):
                entry = entries[identity]
                image_name = entry.get('.tif', entry.get('.tiff'))
                xml_name = entry['.xml']
                assert image_name is not None
                image = EXTRACTED / image_name
                xml = EXTRACTED / xml_name
                image_sha, xml_sha = sha(archive.read(image_name)), sha(archive.read(xml_name))
                assert image.is_file() and xml.is_file()
                assert sha(image.read_bytes()) == image_sha and sha(xml.read_bytes()) == xml_sha
                role = 'test' if split == 'test' else 'val' if i in indices else 'train'
                records.append(dict(id=identity, split=role, image=str(image.relative_to(EXTRACTED)),
                                    image_sha256=image_sha, xml=str(xml.relative_to(EXTRACTED)),
                                    xml_sha256=xml_sha))
    ids = {role: {r['id'] for r in records if r['split'] == role} for role in ('train', 'val', 'test')}
    assert {k: len(v) for k, v in ids.items()} == {'train': 30, 'val': 7, 'test': 14}
    assert not (ids['train'] & ids['val'] or ids['train'] & ids['test'] or ids['val'] & ids['test'])
    manifest = dict(protocol='monuseg37-locked-val7-test14-v4-amendment-20260929',
                    source='https://monuseg.grand-challenge.org/Data/',
                    pool='all 37 image/XML identities in downloaded training archive',
                    validation_index_sha256=VAL_SHA,
                    archive_sha256={'train': TRAIN_SHA, 'test': TEST_SHA},
                    counts={'train': 30, 'val': 7, 'test': 14}, samples=records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, sort_keys=True, indent=2) + '\n'
    if OUT.exists() and OUT.read_text() != payload:
        raise RuntimeError('Existing MoNuSeg37 manifest differs; refusing to overwrite')
    OUT.write_text(payload)
    print(json.dumps({'manifest': str(OUT), 'sha256': sha(OUT.read_bytes()),
                      'counts': manifest['counts']}, indent=2))


if __name__ == '__main__':
    main()
