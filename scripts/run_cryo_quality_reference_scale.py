#!/usr/bin/env python3
"""Cryo quality label demand for frozen 1-NN, with project-disjoint queries.

Counts both labeled 2-D classes and reference particles. No optimizer runs.
"""
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
import torch
import torch.nn.functional as F
from dinov3.eval.eval_ood.datasets import MRCStack
from dinov3.eval.ood_transfer.common import (
    code_identity, process_policy, read_json, require_inputs, resolve_input, save_json, sha256,
)
from run_ood_frozen_transfer import Encoder, minmax
from run_cryo_quality_full_finetune import source_records, class_budget, metrics


def features(rows, args, name, encoder, out):
    target = out / f'{name}_features.npy'
    if target.exists():
        values = np.load(target, mmap_mode='r')
        if len(values) != len(rows) or values.dtype != np.float16:
            raise ValueError('Existing feature population mismatch')
        return values
    staging = target.with_suffix('.partial.npy')
    if staging.exists():
        raise FileExistsError('Partial feature cache requires inspection')
    stacks = {}
    data = None
    for start in range(0, len(rows), args.batch):
        planes = []
        for row in rows[start:start+args.batch]:
            path = str(resolve_input(args.data_root, row['path']))
            if path not in stacks:
                stacks[path] = MRCStack(Path(path))
            planes.append(minmax(stacks[path].read(row['index'])))
        value = encoder(planes).astype(np.float16)
        if data is None:
            data = np.lib.format.open_memmap(staging, mode='w+', dtype=np.float16,
                                              shape=(len(rows), value.shape[1]))
        data[start:start+len(value)] = value
        if start % (args.batch*100) == 0:
            print('FEATURES', name, start, len(rows), flush=True)
    data.flush()
    del data
    staging.replace(target)
    return np.load(target, mmap_mode='r')


def neighbors(query, reference, chunk=256):
    bank = F.normalize(torch.as_tensor(np.asarray(reference), device='cuda').float(), dim=1)
    answers = []
    for start in range(0, len(query), chunk):
        q = F.normalize(torch.as_tensor(np.asarray(query[start:start+chunk]), device='cuda').float(), dim=1)
        answers.extend((q@bank.T).argmax(1).cpu().tolist())
    return np.asarray(answers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--input-receipt', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-projects', nargs=2, default=['11043', '11388'])
    parser.add_argument('--test-project', default='10535')
    parser.add_argument('--excluded-val-project', default='11387')
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    process_policy()
    if len(set(args.train_projects+[args.test_project, args.excluded_val_project])) != 4:
        raise ValueError('Project splits must be disjoint')
    require_inputs(args.manifest, args.data_root, args.input_receipt)
    code = code_identity()
    if code['dirty']:
        raise ValueError('Clean committed code required')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol='cryo-quality-frozen-reference-scale-v1', code=code,
                    manifest_sha256=sha256(args.manifest), checkpoint_sha256=sha256(args.checkpoint),
                    config_sha256=sha256(args.train_config), train_projects=args.train_projects,
                    test_project=args.test_project, excluded_val_project=args.excluded_val_project,
                    class_fractions=[.06, .2, .5, 1.0], per_class_particles=[8, 32, 128],
                    sampling_seeds=list(range(5)))
    if (out/'identity.json').exists() and read_json(out/'identity.json') != identity:
        raise ValueError('Run identity changed')
    save_json(out/'identity.json', identity)
    projects = source_records(read_json(args.manifest), args.data_root)
    train = [r for p in args.train_projects for r in projects[p]]
    test = projects[args.test_project]
    encoder = Encoder(args)
    xtrain = features(train, args, 'train', encoder, out)
    xtest = features(test, args, 'test', encoder, out)
    del encoder
    ytrain = np.asarray([r['tier'] for r in train])
    ytest = np.asarray([r['tier'] for r in test])
    index_by_id = {r['sample_id']:i for i,r in enumerate(train)}
    rows = []
    for fraction in identity['class_fractions']:
        for seed in identity['sampling_seeds']:
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
                row = dict(class_fraction=fraction, seed=seed, reference_particles_per_class=per_class,
                           labeled_quality_classes=sum(map(len, classes.values())),
                           reference_particles=len(indices), test_project=args.test_project,
                           metrics=score, reference_sample_ids_sha256=hashlib.sha256(
                               '\n'.join(train[i]['sample_id'] for i in indices).encode()).hexdigest())
                rows.append(row)
                save_json(out/'results.json', dict(identity=identity, rows=rows,
                                                   unit='labeled author 2-D quality class; reference particle count',
                                                   optimization_steps=0))
                print('REFERENCE', fraction, seed, per_class, len(indices), score, flush=True)


if __name__ == '__main__':
    main()
