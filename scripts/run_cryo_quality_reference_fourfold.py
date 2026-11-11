#!/usr/bin/env python3
"""Four project-held-out Cryo quality 1-NN label-demand matrix."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from dinov3.eval.ood_transfer.common import (
    code_identity, process_policy, read_json, require_inputs, save_json, sha256,
)
from run_cryo_quality_full_finetune import source_records, class_budget, metrics
from run_cryo_quality_reference_scale import features, neighbors
from run_ood_frozen_transfer import Encoder


FOLDS = [
    dict(test='10535', val='11387', train=['11043', '11388']),
    dict(test='11387', val='10535', train=['11043', '11388']),
    dict(test='11043', val='11388', train=['10535', '11387']),
    dict(test='11388', val='11043', train=['10535', '11387']),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--input-receipt', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-config', required=True)
    parser.add_argument('--source-cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    process_policy()
    require_inputs(args.manifest, args.data_root, args.input_receipt)
    code = code_identity()
    if code['dirty']:
        raise ValueError('Clean committed code required')
    out, source = Path(args.output), Path(args.source_cache)
    out.mkdir(parents=True, exist_ok=True)
    earlier = read_json(source/'feature_identity.json')
    if earlier['source_manifest_sha256'] != sha256(args.manifest) or earlier['checkpoint_sha256'] != sha256(args.checkpoint):
        raise ValueError('Source feature cache has different model or population')
    if sha256(source/'train_features.npy') != earlier['train_feature_sha256'] or sha256(source/'test_features.npy') != earlier['test_feature_sha256']:
        raise ValueError('Source feature cache changed')
    projects = source_records(read_json(args.manifest), args.data_root)
    train_cache = np.load(source/'train_features.npy', mmap_mode='r')
    test_cache = np.load(source/'test_features.npy', mmap_mode='r')
    if train_cache.shape[0] != 160000 or test_cache.shape[0] != 80000:
        raise ValueError('Feature cache population differs')
    project_features = {'11043': train_cache[:80000], '11388': train_cache[80000:], '10535': test_cache}
    encoder = Encoder(args)
    project_features['11387'] = features(projects['11387'], args, 'project11387', encoder, out)
    del encoder
    identity = dict(protocol='cryo-quality-frozen-reference-fourfold-v1', code=code,
                    manifest_sha256=sha256(args.manifest), checkpoint_sha256=sha256(args.checkpoint),
                    config_sha256=sha256(args.train_config), source_cache=earlier,
                    heldout_folds=FOLDS, class_fractions=[.06, .2, .5, 1.0],
                    per_class_particles=[8, 32, 128], seeds=list(range(5)),
                    project11387_feature_sha256=sha256(out/'project11387_features.npy'))
    if (out/'identity.json').exists() and read_json(out/'identity.json') != identity:
        raise ValueError('Identity changed')
    save_json(out/'identity.json', identity)
    for fold in FOLDS:
        name = fold['test']
        fold_out = out/f'test_{name}'
        fold_out.mkdir(exist_ok=True)
        train = [r for p in fold['train'] for r in projects[p]]
        test = projects[name]
        xtrain = np.concatenate([project_features[p] for p in fold['train']])
        xtest = project_features[name]
        ytrain = np.asarray([r['tier'] for r in train])
        ytest = np.asarray([r['tier'] for r in test])
        index_by_id = {r['sample_id']: i for i, r in enumerate(train)}
        save_json(fold_out/'test_sample_ids.json', [r['sample_id'] for r in test])
        rows = []
        for fraction in identity['class_fractions']:
            for seed in identity['seeds']:
                selected, classes = class_budget(train, fraction, seed)
                by_class = {}
                for row in selected:
                    by_class.setdefault((row['project'], row['class_id']), []).append(index_by_id[row['sample_id']])
                for per_class in identity['per_class_particles']:
                    rng = np.random.default_rng(seed+1000003*per_class)
                    indices = []
                    for key in sorted(by_class):
                        candidates = np.asarray(by_class[key])
                        indices.extend(rng.choice(candidates, size=min(per_class, len(candidates)), replace=False).tolist())
                    indices = np.asarray(indices)
                    prediction = ytrain[indices[neighbors(xtest, xtrain[indices])]]
                    score = metrics(ytest, prediction)
                    prediction_path = fold_out/f'predictions_class{sum(map(len,classes.values()))}_seed{seed}_k{per_class}.npz'
                    np.savez_compressed(prediction_path, truth=ytest.astype(np.uint8), predicted=prediction.astype(np.uint8))
                    row = dict(test_project=name, val_project=fold['val'], train_projects=fold['train'],
                               class_fraction=fraction, seed=seed,
                               reference_particles_per_class=per_class,
                               labeled_quality_classes=sum(map(len, classes.values())),
                               reference_particles=len(indices), metrics=score,
                               reference_sample_ids_sha256=hashlib.sha256(
                                   '\n'.join(train[i]['sample_id'] for i in indices).encode()).hexdigest(),
                               predictions=str(prediction_path), predictions_sha256=sha256(prediction_path))
                    rows.append(row)
                    save_json(fold_out/'results.json', dict(identity=identity, rows=rows,
                                                           optimization_steps=0, label_unit='author 2-D quality class'))
                    print('FOLD', name, fraction, seed, per_class, score['macro_f1'], flush=True)


if __name__ == '__main__':
    main()
