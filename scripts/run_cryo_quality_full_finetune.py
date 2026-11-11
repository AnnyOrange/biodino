#!/usr/bin/env python3
"""Project-disjoint Cryo quality full fine-tuning and label-budget study.

The released quality score is assigned to a 2-D class, so the unit of label
sampling is a class, not a particle. Test projects are never read during fit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from dinov3.eval.eval_ood.datasets import MRCStack
from dinov3.eval.ood_transfer.common import (
    code_identity, process_policy, read_json, require_inputs, resolve_input,
    save_json, save_torch, seed_all, sha256,
)
from dinov3.eval.ood_transfer.models import load_backbone


class Particles(Dataset):
    def __init__(self, records, data_root, mean, std, augment=False):
        self.records, self.data_root = records, data_root
        self.mean = torch.tensor(mean)[:, None, None]
        self.std = torch.tensor(std)[:, None, None]
        self.augment, self.stacks = augment, {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        path = str(resolve_input(self.data_root, r['path']))
        if path not in self.stacks:
            self.stacks[path] = MRCStack(Path(path))
        image = np.asarray(self.stacks[path].read(r['index']), dtype=np.float32)
        low, high = float(image.min()), float(image.max())
        image = torch.from_numpy(((image-low) / max(high-low, 1e-12)).copy())[None, None]
        image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)[0].expand(3, -1, -1).clone()
        if self.augment:
            if random.random() < .5:
                image = image.flip(-1)
            if random.random() < .5:
                image = image.flip(-2)
            image = image.rot90(random.randrange(4), (-2, -1))
        return (image-self.mean)/self.std, int(r['tier']), r['sample_id']


class QualityModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = backbone
        self.head = nn.Sequential(nn.LayerNorm(backbone.embed_dim*2), nn.Linear(backbone.embed_dim*2, 3))

    def forward(self, images):
        features = self.encoder.forward_features(images)
        x = torch.cat((features['x_norm_clstoken'], features['x_norm_patchtokens'].mean(1)), dim=1)
        return self.head(x)


def source_records(manifest, root):
    projects = {}
    for project in manifest['projects']:
        name = str(project['project'])
        cs = np.load(resolve_input(root, project['particles']), allow_pickle=False)
        scores = read_json(resolve_input(root, project['quality_scores']))
        rows = []
        for row in cs:
            class_id = int(row['alignments2D/class'])
            score = float(scores[str(class_id)])
            tier = 0 if score < .3 else 2 if score > .7 else 1
            blob = row['blob/path']
            blob = blob.decode() if isinstance(blob, bytes) else str(blob)
            rows.append(dict(sample_id=f"{name}:{int(row['uid'])}", project=name,
                             class_id=class_id, tier=tier,
                             micrograph_uid=int(row['location/micrograph_uid']),
                             path=str(Path(project['stack_root'])/Path(blob).name),
                             index=int(row['blob/idx'])))
        if len(rows) != 80000 or len({r['sample_id'] for r in rows}) != len(rows):
            raise ValueError('Unexpected project population')
        projects[name] = rows
    if len(projects) != 4:
        raise ValueError('Expected four author projects')
    return projects


def class_budget(rows, fraction, seed):
    """Sample labeled 2-D classes; all particles of chosen classes are eligible.

    Class labels are the independent annotations in the author release. Using a
    fraction of particles from every class would falsely imply more labels.
    """
    rng = np.random.default_rng(seed)
    chosen = {}
    for project in sorted({r['project'] for r in rows}):
        candidates = [r for r in rows if r['project'] == project]
        by_class = {}
        for r in candidates:
            by_class.setdefault(r['class_id'], []).append(r)
        classes = np.asarray(sorted(by_class))
        count = max(3, math.ceil(len(classes)*fraction))
        # At least one class per quality tier if the budget permits.
        groups = {tier: [c for c in classes if by_class[int(c)][0]['tier'] == tier] for tier in range(3)}
        picked = set()
        for tier in range(3):
            if groups[tier]:
                picked.add(int(rng.choice(groups[tier])))
        rest = [int(c) for c in rng.permutation(classes) if int(c) not in picked]
        picked.update(rest[:max(0, count-len(picked))])
        chosen[project] = sorted(picked)
    selected = [r for r in rows if r['class_id'] in chosen[r['project']]]
    if len({r['tier'] for r in selected}) != 3:
        raise ValueError('Missing training quality tier')
    return selected, chosen


def metrics(y_true, y_pred):
    return dict(accuracy=float(np.mean(np.asarray(y_true)==np.asarray(y_pred))),
                balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro', zero_division=0)),
                confusion=confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(), n=len(y_true))


@torch.inference_mode()
def evaluate(model, dataset, batch):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    true, predicted, identifiers = [], [], []
    for images, labels, ids in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(images.cuda(non_blocking=True))
        true.extend(labels.tolist())
        predicted.extend(logits.float().argmax(1).cpu().tolist())
        identifiers.extend(ids)
    return metrics(true, predicted), [dict(sample_id=i, truth=int(y), prediction=int(p))
                                      for i, y, p in zip(identifiers, true, predicted)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'test'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--input-receipt', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-projects', nargs=2, default=['11043', '11388'])
    parser.add_argument('--val-project', default='11387')
    parser.add_argument('--test-project', default='10535')
    parser.add_argument('--class-fraction', type=float, default=.02)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    args = parser.parse_args()
    process_policy()
    if len(set(args.train_projects + [args.val_project, args.test_project])) != 4:
        raise ValueError('Train, validation and test projects must be disjoint')
    if args.class_fraction <= 0 or args.class_fraction > 1:
        raise ValueError('Invalid class budget')
    require_inputs(args.manifest, args.data_root, args.input_receipt)
    code = code_identity()
    if code['dirty']:
        raise ValueError('Committed, clean code required')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol='cryo-quality-project-disjoint-full-finetune-v1',
                    code=code, manifest_sha256=sha256(args.manifest),
                    checkpoint_sha256=sha256(args.checkpoint), config_sha256=sha256(args.train_config),
                    train_projects=args.train_projects, val_project=args.val_project,
                    test_project=args.test_project, class_fraction=args.class_fraction,
                    seed=args.seed, epochs=args.epochs, batch=args.batch,
                    encoder_lr=args.encoder_lr, head_lr=args.head_lr)
    if (out/'identity.json').exists() and read_json(out/'identity.json') != identity:
        raise ValueError('Run identity changed')
    save_json(out/'identity.json', identity)
    projects = source_records(read_json(args.manifest), args.data_root)
    seed_all(args.seed)
    train_rows, classes = class_budget([r for p in args.train_projects for r in projects[p]],
                                       args.class_fraction, args.seed)
    val_rows = projects[args.val_project]
    test_rows = projects[args.test_project]
    save_json(out/'split.json', dict(train_projects=args.train_projects, val_project=args.val_project,
                                    test_project=args.test_project, selected_classes=classes,
                                    n_train_particles=len(train_rows), n_val_particles=len(val_rows),
                                    n_test_particles=len(test_rows),
                                    labeled_quality_classes=sum(map(len, classes.values())),
                                    unit_of_label='author 2-D class quality score',
                                    train_sample_ids_sha256=__import__('hashlib').sha256(
                                        '\n'.join(r['sample_id'] for r in train_rows).encode()).hexdigest()))
    backbone, mean, std = load_backbone(args.checkpoint, args.train_config, True)
    model = QualityModel(backbone).cuda()
    if args.command == 'train':
        if (out/'train_complete.json').exists():
            return
        train = Particles(train_rows, args.data_root, mean, std, augment=True)
        val = Particles(val_rows, args.data_root, mean, std)
        groups = [dict(params=model.encoder.parameters(), lr=args.encoder_lr, weight_decay=.05),
                  dict(params=model.head.parameters(), lr=args.head_lr, weight_decay=.05)]
        optimizer = torch.optim.AdamW(groups)
        counts = np.bincount([r['tier'] for r in train_rows], minlength=3)
        weights = torch.tensor(len(train_rows)/(3*np.maximum(counts, 1)), dtype=torch.float32, device='cuda')
        best = -1.0
        for epoch in range(args.epochs):
            model.train()
            loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=2,
                                pin_memory=True, generator=torch.Generator().manual_seed(args.seed+epoch))
            total_loss = 0.0
            for step, (images, labels, _) in enumerate(loader, 1):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(images.cuda(non_blocking=True))
                    loss = F.cross_entropy(logits.float(), labels.cuda(non_blocking=True), weight=weights)
                loss.backward()
                if epoch == 0 and step == 1:
                    count = sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in model.encoder.parameters())
                    if not count:
                        raise ValueError('No backbone gradients: not full fine-tuning')
                    save_json(out/'first_step.json', dict(backbone_gradient_tensors=count,
                                                         loss=float(loss), peak_gpu_bytes=torch.cuda.max_memory_allocated()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach())
                if step % 20 == 0:
                    save_json(out/'progress.json', dict(epoch=epoch+1, step=step,
                                                       steps_per_epoch=len(loader), loss=total_loss/step,
                                                       time=time.time()))
                    print('TRAIN', epoch+1, step, len(loader), total_loss/step, flush=True)
            validation, _ = evaluate(model, val, args.batch)
            row = dict(epoch=epoch+1, training_loss=total_loss/len(loader), validation=validation)
            print('EPOCH', json.dumps(row), flush=True)
            if validation['macro_f1'] > best:
                best = validation['macro_f1']
                save_torch(out/'best.pth', dict(model=model.state_dict(), identity=identity,
                                                selected_epoch=epoch+1, validation=validation))
                save_json(out/'best_validation.json', row)
        save_json(out/'train_complete.json', dict(identity=identity, best_validation=read_json(out/'best_validation.json'),
                                                  best_checkpoint_sha256=sha256(out/'best.pth'), test_used=False))
    else:
        complete = read_json(out/'train_complete.json')
        if complete['identity'] != identity or complete['best_checkpoint_sha256'] != sha256(out/'best.pth'):
            raise ValueError('Training identity or checkpoint changed')
        state = torch.load(out/'best.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        result, predictions = evaluate(model, Particles(test_rows, args.data_root, mean, std), args.batch)
        save_json(out/'test_predictions.json', predictions)
        save_json(out/'test_result.json', dict(identity=identity, metrics=result,
                                               selected_epoch=state['selected_epoch'],
                                               prediction_sha256=sha256(out/'test_predictions.json'),
                                               test_tuning=False))


if __name__ == '__main__':
    main()
