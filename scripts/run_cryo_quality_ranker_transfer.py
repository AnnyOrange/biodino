#!/usr/bin/env python3
"""Project-disjoint Cryo quality ranking with full encoder fine-tuning.

This follows CryoRanker's two-logit cross-entropy plus margin-ranking objective
on the four released evaluation projects. It is a new comparison protocol, not
a reproduction of the author's 42-million-particle training run.
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
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

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
        return (image-self.mean)/self.std, float(r['score']), r['sample_id']


class QualityModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = backbone
        self.head = nn.Sequential(nn.LayerNorm(backbone.embed_dim*2), nn.Linear(backbone.embed_dim*2, 2))

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
                             class_id=class_id, tier=tier, score=score,
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
    """Sample 2-D class IDs without inspecting their quality labels."""
    rng = np.random.default_rng(seed)
    chosen = {}
    for project in sorted({r['project'] for r in rows}):
        candidates = [r for r in rows if r['project'] == project]
        by_class = {}
        for r in candidates:
            by_class.setdefault(r['class_id'], []).append(r)
        classes = np.asarray(sorted(by_class))
        count = max(1, math.ceil(len(classes)*fraction))
        picked = set(map(int, rng.choice(classes, size=count, replace=False)))
        chosen[project] = sorted(picked)
    selected = [r for r in rows if r['class_id'] in chosen[r['project']]]
    return selected, chosen


def metrics(y_true, y_pred):
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    extremes = (true <= .3) | (true > .7)
    positive = true[extremes] > .7
    ranked = np.argsort(-pred, kind='stable')
    correlation = float(spearmanr(true, pred).statistic)
    out = dict(n=len(true), n_extremes=int(extremes.sum()),
               spearman=correlation if np.isfinite(correlation) else None,
               auc_extremes=float(roc_auc_score(positive, pred[extremes])) if len(set(positive)) == 2 else None,
               ap_extremes=float(average_precision_score(positive, pred[extremes])) if positive.any() else None)
    for fraction in (.1, .25, .5):
        count = max(1, int(len(true) * fraction))
        out[f'top_{int(fraction*100)}_mean_quality'] = float(true[ranked[:count]].mean())
        out[f'top_{int(fraction*100)}_high_fraction'] = float((true[ranked[:count]] > .7).mean())
    return out


def cryoranker_loss(logits, quality):
    """Author-stated two-logit CE on extremes plus pairwise margin 0.2.

    Both terms are means so the objective does not scale with batch size.
    Pairing is within a batch; the author's unpublished sampler is unknown.
    """
    probability = logits.float().softmax(1)[:, 1]
    extreme = (quality <= .3) | (quality > .7)
    ce = F.cross_entropy(logits.float()[extreme], (quality[extreme] > .7).long()) if extreme.any() else probability.sum()*0
    higher = quality[:, None] > quality[None, :]
    margin = (0.2 - (probability[:, None] - probability[None, :]))[higher]
    ranking = margin.clamp_min(0).mean() if margin.numel() else probability.sum()*0
    return ce + ranking, ce, ranking


@torch.inference_mode()
def evaluate(model, dataset, batch):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    true, predicted, identifiers = [], [], []
    for images, scores, ids in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(images.cuda(non_blocking=True))
        true.extend(scores.tolist())
        predicted.extend(logits.float().softmax(1)[:, 1].cpu().tolist())
        identifiers.extend(ids)
    return metrics(true, predicted), [dict(sample_id=i, quality=float(y), predicted_score=float(p))
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
    parser.add_argument('--class-fraction', type=float, default=1.0)
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
    identity = dict(protocol='cryo-ranker-style-project-disjoint-v1',
                    code=code, manifest_sha256=sha256(args.manifest),
                    checkpoint_sha256=sha256(args.checkpoint), config_sha256=sha256(args.train_config),
                    train_projects=args.train_projects, val_project=args.val_project,
                    test_project=args.test_project, class_fraction=args.class_fraction,
                    class_selection='uniform_random_class_ids_without_quality_labels',
                    seed=args.seed, epochs=args.epochs, batch=args.batch,
                    encoder_lr=args.encoder_lr, head_lr=args.head_lr,
                    loss='mean_cross_entropy_on_quality_extremes_plus_mean_pairwise_margin_0.2',
                    validation_selection='mean_quality_of_top_10_percent',
                    author_training_reproduction=False)
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
                                    selected_particle_tiers=np.bincount([r['tier'] for r in train_rows], minlength=3).tolist(),
                                    class_selection='uniform_random_class_ids_without_quality_labels',
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
        best = -1.0
        for epoch in range(args.epochs):
            model.train()
            loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=2,
                                pin_memory=True, generator=torch.Generator().manual_seed(args.seed+epoch))
            total_loss = 0.0
            for step, (images, scores, _) in enumerate(loader, 1):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(images.cuda(non_blocking=True))
                    loss, ce, ranking = cryoranker_loss(logits, scores.cuda(non_blocking=True))
                loss.backward()
                if epoch == 0 and step == 1:
                    count = sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in model.encoder.parameters())
                    if not count:
                        raise ValueError('No backbone gradients: not full fine-tuning')
                    save_json(out/'first_step.json', dict(backbone_gradient_tensors=count,
                                                         loss=float(loss.detach()), ce=float(ce.detach()),
                                                         ranking=float(ranking.detach()),
                                                         peak_gpu_bytes=torch.cuda.max_memory_allocated()))
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
            if validation['top_10_mean_quality'] > best:
                best = validation['top_10_mean_quality']
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
