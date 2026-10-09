#!/usr/bin/env python3
"""Train-only PCA of v4 Cellpose [E;L] patch vectors back to native d=1024.

The calibration is a fixed-seed, uniform 4096-patch subset of the *training*
spatial bank only. Validation/test patches never influence the PCA basis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def project(split: str, source: Path, dest: Path, pca: PCA, evidence: dict) -> None:
    if dest.exists():
        raise FileExistsError(dest)
    with np.load(source, allow_pickle=False) as z:
        if int(z['chunked']) != 0:
            raise ValueError(f'Unexpected chunked source: {source}')
        x = z['features']
        n, two_d, h, w = x.shape
        if two_d != 2048 or pca.n_components_ != 1024:
            raise ValueError(f'Unexpected dense PCA feature width: {two_d}/{pca.n_components_}')
        output = np.empty((n, 1024, h, w), dtype=np.float16)
        for start in range(n):
            patches = x[start].reshape(two_d, -1).T.astype(np.float32)
            projected = pca.transform(patches)
            projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)
            output[start] = projected.T.reshape(1024, h, w).astype(np.float16)
            if (start + 1) % 50 == 0 or start + 1 == n:
                print(f'[pca] {split}: {start + 1}/{n}', flush=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix('.tmp.npz')
        if tmp.exists():
            raise FileExistsError(tmp)
        np.savez(tmp, features=output, sem_masks=z['sem_masks'], inst_maps=z['inst_maps'],
                 chunked=np.int8(0), orig_H=z['orig_H'], orig_W=z['orig_W'],
                 patch_size=z['patch_size'], embed_dim=np.int32(1024), n_layers=np.int32(1),
                 autocast_dtype=np.asarray('bf16'), feature_batch_size=np.int32(32))
    tmp.replace(dest)
    identity = json.loads(source.with_suffix('.identity.json').read_text())
    identity.update({'train_only_pca': evidence, 'projection_split': split})
    dest.with_suffix('.identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(f'[saved] {dest}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    training = args.input_dir / 'train.npz'
    with np.load(training, allow_pickle=False) as z:
        features = z['features']
        n, twice_dim, height, width = features.shape
        if twice_dim != 2048 or n * height * width < 4096:
            raise ValueError('PCA_UNAVAILABLE_INSUFFICIENT_TRAIN_RANK: need >=4096 train patches')
        rng = np.random.default_rng(0)
        choices = np.sort(rng.choice(n * height * width, size=4096, replace=False))
        samples, patch = np.divmod(choices, height * width)
        y, x = np.divmod(patch, width)
        calibration = features[samples, :, y, x].astype(np.float32)
    if calibration.shape != (4096, 2048):
        raise ValueError(f'Unexpected PCA train sample shape: {calibration.shape}')
    pca = PCA(n_components=1024, svd_solver='randomized', random_state=0)
    print('[pca] fitting 4096 uniformly drawn training patches to 1024 dims', flush=True)
    pca.fit(calibration)
    evidence = {'input': 'E+L', 'fit_split': 'train', 'seed': 0, 'calibration_patches': 4096,
                'native_d': 1024, 'fit_patch_indices_sha256': hashlib.sha256(choices.tobytes()).hexdigest(),
                'explained_variance_ratio_sum': float(np.sum(pca.explained_variance_ratio_))}
    for split in ('train', 'val', 'test'):
        project(split, args.input_dir / f'{split}.npz', args.output_dir / f'{split}.npz', pca, evidence)


if __name__ == '__main__':
    main()
