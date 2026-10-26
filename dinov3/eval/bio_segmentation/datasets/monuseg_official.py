"""Identity-locked MoNuSeg2018 admission; never infer the official pool by count."""
import hashlib
import json
from pathlib import Path


def official_paths(root, split, approved_sha256, manifest_file='monuseg2018_official_manifest.json', counts=None):
    root = Path(root)
    path = root / manifest_file
    counts = counts or {'train': 24, 'val': 6, 'test': 14}
    if split not in ('train', 'val', 'test'):
        raise ValueError('MoNuSeg split must be train/val/test')
    if not path.is_file() or len(approved_sha256) != 64:
        raise RuntimeError('MoNuSeg NOT TESTED: official train30/test14 identity manifest not admitted')
    if hashlib.sha256(path.read_bytes()).hexdigest() != approved_sha256:
        raise RuntimeError('MoNuSeg official identity manifest hash mismatch')
    manifest = json.loads(path.read_text())
    rows = manifest['samples']
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or len(rows) != sum(counts.values()):
        raise ValueError('MoNuSeg manifest requires distinct sample IDs matching the registered counts')
    roles = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
    if [len(roles[s]) for s in roles] != [counts['train'], counts['val'], counts['test']]:
        raise ValueError('MoNuSeg split counts differ from the registered protocol')
    if counts == {'train': 30, 'val': 7, 'test': 14}:
        if manifest.get('protocol') != 'monuseg2018-train30-extra7val-test14-v1' or \
                not (manifest.get('inventory_sha256') or manifest.get('official_inventory_sha256')) or \
                not (manifest.get('archive_sha256') or manifest.get('official_archive_sha256')):
            raise ValueError('The locked v4 MoNuSeg manifest must include inventory and archive hashes')
    elif not manifest.get('official_source_url') or not manifest.get('official_inventory_sha256'):
        raise ValueError('Official inventory/source evidence required')
    for row in rows:
        for kind in ('image', 'xml'):
            file = (root / row[kind]).resolve()
            if not file.is_relative_to(root.resolve()) or not file.is_file():
                raise ValueError('MoNuSeg original file missing/outside data root')
            if hashlib.sha256(file.read_bytes()).hexdigest() != row[kind + '_sha256']:
                raise ValueError('MoNuSeg original image/XML changed: ' + row['id'])
    return ([str(root / r['image']) for r in roles[split]],
            [str(root / r['xml']) for r in roles[split]])
