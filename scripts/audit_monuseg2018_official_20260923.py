#!/usr/bin/env python3
"""Validate official MoNuSeg blobs and lock the classic 30/14 identities."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


BASE = Path('/mnt/huawei_deepcad/benchmark/segmentation')
ARCHIVE_ROOT = BASE / 'MoNuSeg'
ROOT = BASE / 'monuseg/extracted'
OUTPUT = ROOT / 'monuseg2018_official_manifest.json'
OFFICIAL_PAGE = 'https://monuseg.grand-challenge.org/Data/'
URLS = {
    'train': 'https://drive.google.com/uc?export=download&id=1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA',
    'test': 'https://drive.google.com/uc?export=download&id=1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw',
}
ARCHIVES = {
    'train': (ARCHIVE_ROOT / 'MoNuSeg 2018 Training Data.zip',
              '25d3d3185bb2970b397cafa72eb664c9b4d24294aee382e7e3df9885affce742'),
    'test': (ARCHIVE_ROOT / 'MoNuSegTestData.zip',
             '13e522387ae8b1bcc0530e13ff9c7b4d91ec74959ef6f6e57747368d7ee6f88a'),
}

# The 30 identities in the original Kumar/MoNuSeg 2018 training inventory,
# grouped by the challenge organ-information inventory (flattened here).
CLASSIC_TRAIN30 = tuple('''
TCGA-A7-A13E-01Z-00-DX1 TCGA-A7-A13F-01Z-00-DX1
TCGA-AR-A1AK-01Z-00-DX1 TCGA-AR-A1AS-01Z-00-DX1
TCGA-E2-A1B5-01Z-00-DX1 TCGA-E2-A14V-01Z-00-DX1
TCGA-B0-5711-01Z-00-DX1 TCGA-HE-7128-01Z-00-DX1
TCGA-HE-7129-01Z-00-DX1 TCGA-HE-7130-01Z-00-DX1
TCGA-B0-5710-01Z-00-DX1 TCGA-B0-5698-01Z-00-DX1
TCGA-18-5592-01Z-00-DX1 TCGA-38-6178-01Z-00-DX1
TCGA-49-4488-01Z-00-DX1 TCGA-50-5931-01Z-00-DX1
TCGA-21-5784-01Z-00-DX1 TCGA-21-5786-01Z-00-DX1
TCGA-G9-6336-01Z-00-DX1 TCGA-G9-6348-01Z-00-DX1
TCGA-G9-6356-01Z-00-DX1 TCGA-G9-6363-01Z-00-DX1
TCGA-CH-5767-01Z-00-DX1 TCGA-G9-6362-01Z-00-DX1
TCGA-DK-A2I6-01A-01-TS1 TCGA-G2-A2EK-01A-02-TSB
TCGA-AY-A8YK-01A-01-TS1 TCGA-NH-A8F7-01A-01-TS1
TCGA-KB-A93J-01A-01-TS1 TCGA-RD-A8N9-01A-01-TS1
'''.split())


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def zip_pairs(path: Path, split: str) -> dict[str, tuple[str, str]]:
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist()
                 if not name.startswith('__MACOSX/') and not name.endswith('/')]
        images = {Path(name).stem: name for name in names if Path(name).suffix.lower() in ('.tif', '.tiff')}
        xmls = {Path(name).stem: name for name in names if Path(name).suffix.lower() == '.xml'}
        if set(images) != set(xmls):
            raise ValueError(f'{split}: image/XML identities differ')
        for identity in images:
            image = archive.read(images[identity])
            xml = archive.read(xmls[identity])
            if Image.open(io.BytesIO(image)).size != (1000, 1000):
                raise ValueError(f'{split}/{identity}: expected 1000x1000 image')
            if not list(ET.fromstring(xml).iter('Region')):
                raise ValueError(f'{split}/{identity}: annotation has no Region')
        return {identity: (images[identity], xmls[identity]) for identity in sorted(images)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-existing', action='store_true',
                        help='recompute the identity lock and compare it byte-for-byte with the existing manifest')
    args = parser.parse_args()
    if OUTPUT.exists() and not args.verify_existing:
        raise FileExistsError(f'Refusing to overwrite identity lock: {OUTPUT}')
    if args.verify_existing and not OUTPUT.exists():
        raise FileNotFoundError(f'Identity lock does not exist: {OUTPUT}')
    archive_pairs = {}
    for split, (path, expected) in ARCHIVES.items():
        actual = digest_file(path)
        if actual != expected:
            raise ValueError(f'{split} archive SHA256 mismatch: {actual} != {expected}')
        archive_pairs[split] = zip_pairs(path, split)
    classic = set(CLASSIC_TRAIN30)
    current_train = set(archive_pairs['train'])
    if len(classic) != 30 or not classic <= current_train:
        raise ValueError('Classic official train30 is not a distinct subset of the official current blob')
    extras = sorted(current_train - classic)
    if len(extras) != 7 or len(archive_pairs['test']) != 14:
        raise ValueError(f'Expected current blob train30+7 extras/test14, got extras={len(extras)} test={len(archive_pairs["test"])}')

    ordered_train = sorted(classic)
    rng = np.random.default_rng(42)
    val_indices = set(map(int, rng.permutation(len(ordered_train))[:6]))
    rows = []
    for split, identities in (('official_train_pool', ordered_train),
                              ('test', sorted(archive_pairs['test']))):
        archive_key = 'train' if split == 'official_train_pool' else 'test'
        archive_path = ARCHIVES[archive_key][0]
        with ZipFile(archive_path) as archive:
            for index, identity in enumerate(identities):
                image_member, xml_member = archive_pairs[archive_key][identity]
                if split == 'test':
                    role = 'test'
                    image_rel = Path('MoNuSegTestData') / f'{identity}.tif'
                    xml_rel = Path('MoNuSegTestData') / f'{identity}.xml'
                else:
                    role = 'val' if index in val_indices else 'train'
                    image_rel = Path('MoNuSeg 2018 Training Data/Tissue Images') / f'{identity}.tif'
                    xml_rel = Path('MoNuSeg 2018 Training Data/Annotations') / f'{identity}.xml'
                image_path, xml_path = ROOT / image_rel, ROOT / xml_rel
                if not image_path.is_file() or not xml_path.is_file():
                    raise FileNotFoundError(f'Extracted original missing for {identity}')
                image_sha = digest_bytes(archive.read(image_member))
                xml_sha = digest_bytes(archive.read(xml_member))
                if digest_file(image_path) != image_sha or digest_file(xml_path) != xml_sha:
                    raise ValueError(f'Extracted bytes differ from locked archive for {identity}')
                rows.append({'id': identity, 'split': role,
                             'image': image_rel.as_posix(), 'image_sha256': image_sha,
                             'xml': xml_rel.as_posix(), 'xml_sha256': xml_sha})
    if {role: sum(row['split'] == role for row in rows) for role in ('train', 'val', 'test')} != {
            'train': 24, 'val': 6, 'test': 14}:
        raise AssertionError('Wrong formal split counts')
    inventory = json.dumps([(row['id'], row['split'], row['image_sha256'], row['xml_sha256'])
                            for row in rows], separators=(',', ':'))
    manifest = {
        'protocol': 'monuseg2018-official30-seed42-val6-test14-v1',
        'official_source_url': OFFICIAL_PAGE,
        'official_download_urls': URLS,
        'official_archive_sha256': {key: value[1] for key, value in ARCHIVES.items()},
        'official_inventory_sha256': digest_bytes(inventory.encode()),
        'selection': 'classic challenge train30 identities; sorted IDs; numpy default_rng(42) permutation first 6 as val',
        'excluded_current_training_blob_extras': extras,
        'counts': {'official_train_pool': 30, 'train': 24, 'val': 6, 'test': 14},
        'samples': rows,
    }
    if args.verify_existing:
        existing = json.loads(OUTPUT.read_text())
        if existing != manifest:
            raise ValueError(f'Existing identity lock differs from recomputed manifest: {OUTPUT}')
        print(json.dumps({'verified': True, 'manifest': str(OUTPUT),
                          'manifest_sha256': digest_file(OUTPUT),
                          'inventory_sha256': manifest['official_inventory_sha256'],
                          'counts': manifest['counts'], 'excluded': extras}, indent=2))
        return
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'manifest': str(OUTPUT), 'manifest_sha256': digest_file(OUTPUT),
                      'inventory_sha256': manifest['official_inventory_sha256'],
                      'counts': manifest['counts'], 'excluded': extras}, indent=2))


if __name__ == '__main__':
    main()
