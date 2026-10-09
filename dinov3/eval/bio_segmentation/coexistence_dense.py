#!/usr/bin/env python3
"""Create matched Cellpose spatial fusion caches from three v4 frozen teachers.

No evaluation labels are used to fit PCA. The source Cellpose split is a fixed,
ordered, deterministic loader; segmentation caches do not themselves store
filenames, so we record its ordered image/mask identity in the provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .datasets.cellpose import get_cellpose_paths


ROLES = {'E': ('early', 12687), 'M': ('middle', 20007), 'L': ('late', 29279)}
SPLITS = ('train', 'val', 'test')
META = ('orig_H', 'orig_W', 'patch_size', 'embed_dim', 'n_layers', 'autocast_dtype', 'feature_batch_size')


def _digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _input_path(root: Path, role: str, split: str) -> Path:
    label, step = ROLES[role]
    tag = 'last1__pad__s512_spformal_static_v1'
    run = f'hs6_l5_{label}_ampbf16_b32__best__{tag}'
    return root / run / 'cellpose' / str(step) / f'cellpose_{split}_config_last1_pad_s512_spformal_static_v1_ampbf16_b32.npz'


def _read_verified(root: Path, split: str, data_root: Path):
    files = {role: _input_path(root, role, split) for role in ROLES}
    loaded = {}
    for role, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as z:
            if int(z['chunked']) != 0:
                raise ValueError(f'Chunked source cache requires per-chunk fusion: {path}')
            loaded[role] = {key: np.asarray(z[key]) for key in ('features', 'sem_masks', 'inst_maps', *META)}
    ref = loaded['E']
    for role, source in loaded.items():
        if source['features'].ndim != 4 or source['features'].shape != ref['features'].shape:
            raise ValueError(f'{role} feature geometry differs: {source["features"].shape}')
        for key in (*META, 'sem_masks', 'inst_maps'):
            if not np.array_equal(source[key], ref[key]):
                raise ValueError(f'{split}/{role}: {key} mismatch across teacher banks')
    image_paths, mask_paths = get_cellpose_paths(str(data_root), split)
    if len(image_paths) != len(ref['features']) or len(set(image_paths)) != len(image_paths):
        raise ValueError(f'{split}: ordered source-path inventory count/uniqueness mismatch')
    ids = json.dumps(list(zip(image_paths, mask_paths)), separators=(',', ':'))
    provenance = {
        'split': split, 'ordered_image_mask_paths_sha256': hashlib.sha256(ids.encode()).hexdigest(),
        'ordered_images': len(image_paths),
        'source_banks': {role: {'path': str(path), 'sha256': _digest_file(path)} for role, path in files.items()},
        'identity_contract': 'Same deterministic source split/loader and matching ordered semantic+instance masks; '
                             'source feature caches do not embed image paths.',
    }
    return loaded, provenance


def _concat(left: np.ndarray, right: np.ndarray, block: int = 8) -> np.ndarray:
    n, d, h, w = left.shape
    result = np.empty((n, 2 * d, h, w), dtype=np.float16)
    for start in range(0, n, block):
        end = min(start + block, n)
        for array, column in ((left, 0), (right, d)):
            x = array[start:end].astype(np.float32)
            norm = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
            if np.any(norm <= 1e-12):
                raise ValueError(f'Zero-length patch vector at images {start}:{end}')
            result[start:end, column:column + d] = (x / norm / np.sqrt(2.0)).astype(np.float16)
    return result


def _save(path: Path, feat: np.ndarray, ref: dict, provenance: dict) -> None:
    if path.exists():
        raise FileExistsError(f'Refusing to replace dense cache: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    if temp.exists():
        raise FileExistsError(f'Unclaimed incomplete dense cache: {temp}')
    np.savez(temp, features=feat, sem_masks=ref['sem_masks'], inst_maps=ref['inst_maps'],
             chunked=np.int8(0), orig_H=ref['orig_H'], orig_W=ref['orig_W'],
             patch_size=ref['patch_size'], embed_dim=np.int32(feat.shape[1]),
             n_layers=np.int32(1), autocast_dtype=np.asarray('bf16'),
             feature_batch_size=np.int32(32))
    temp.replace(path)
    path.with_suffix('.identity.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'[dense] {path}: {feat.shape}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('/mnt/huawei_deepcad/benchmark/segmentation/Cellpose'))
    args = p.parse_args()
    identities = {}
    split_paths = {split: set(get_cellpose_paths(str(args.data_root), split)[0]) for split in SPLITS}
    if any(split_paths[a] & split_paths[b] for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]):
        raise ValueError('Cellpose train/val/test image source identities overlap')
    for split in SPLITS:
        sources, provenance = _read_verified(args.input_root, split, args.data_root)
        identities[split] = provenance['ordered_image_mask_paths_sha256']
        for name, left in (('E+L', 'E'), ('M+L', 'M')):
            result = _concat(sources[left]['features'], sources['L']['features'])
            _save(args.output_root / name / f'{split}.npz', result, sources[left], provenance)
            del result
    if len(set(identities.values())) != len(SPLITS):
        raise ValueError('Cellpose train/val/test ordered split identities unexpectedly equal')


if __name__ == '__main__':
    main()
