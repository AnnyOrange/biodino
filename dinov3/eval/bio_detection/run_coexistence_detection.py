#!/usr/bin/env python3
"""Matched BBBC038 v4 center-to-patch proxy with two frozen EMA teachers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dinov3.eval.bio_classification.common import load_backbone
from .center_probe import PatchFeatureModel, _estimate_pos_weight, _eval, build_center_dataset


class PairedPatchModel(nn.Module):
    def __init__(self, first: nn.Module, second: nn.Module):
        super().__init__()
        self.first = first
        self.second = second

    @torch.inference_mode()
    def forward(self, images):
        a = self.first(images)
        b = self.second(images)
        if a.shape != b.shape or a.ndim != 3:
            raise ValueError(f'Incompatible teacher patch geometries: {a.shape}/{b.shape}')
        return torch.cat((F.normalize(a, dim=-1), F.normalize(b, dim=-1)), dim=-1) / np.sqrt(2.0)


class UnitPatchModel(nn.Module):
    """Single checkpoint with the same per-patch normalization as paired arms."""

    def __init__(self, feature_model: nn.Module):
        super().__init__()
        self.feature_model = feature_model

    @torch.inference_mode()
    def forward(self, images):
        return F.normalize(self.feature_model(images), dim=-1)


def identity(ds):
    base = ds.dataset
    if not hasattr(base, 'img_paths') or not hasattr(base, 'mask_paths'):
        raise ValueError('BBBC038 must have explicit image/mask ordered identities')
    names = [(str(base.img_paths[i]), str(base.mask_paths[i])) for i in ds.indices]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate BBBC038 source image/mask identities')
    payload = json.dumps(names, separators=(',', ':'))
    return {'count': len(names), 'sha256': hashlib.sha256(payload.encode()).hexdigest(),
            'ordered_first': names[0], 'ordered_last': names[-1]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True, choices=('E+L', 'M+L', 'unit_E', 'unit_M', 'unit_L'))
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--benchmark-root', default='/mnt/huawei_deepcad/benchmark')
    args = p.parse_args()
    out = args.output_dir / 'results_bio_detection.json'
    if out.exists():
        raise FileExistsError(out)
    specs = {'E': 12687, 'M': 20007, 'L': 29279}
    source_roles = [args.arm.removeprefix('unit_')] if args.arm.startswith('unit_') else args.arm.split('+')
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    models = []
    for role in source_roles:
        backbone = load_backbone(
            repo_dir='.', arch='dinov3_vitb16', weights='',
            checkpoint=str(args.run_root / f'eval/training_{specs[role]}/teacher_checkpoint.pth'),
            train_config=str(args.run_root / 'config.yaml'),
        )
        models.append(PatchFeatureModel(backbone, autocast_dtype=torch.bfloat16, channel_policy='auto').cuda().eval())
    model = (UnitPatchModel(models[0]) if len(models) == 1 else PairedPatchModel(*models)).cuda().eval()
    datasets = {split: build_center_dataset('bbbc038', args.benchmark_root, split, 224, 0, seed)
                for split in ('train', 'val', 'test')}
    ids = {split: identity(ds) for split, ds in datasets.items()}
    sets = {split: set(str(datasets[split].dataset.img_paths[i]) for i in datasets[split].indices)
            for split in datasets}
    if any(sets[a] & sets[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('BBBC038 train/val/test source image identities overlap')
    train_loader = DataLoader(datasets['train'], batch_size=8, shuffle=True, num_workers=2,
                              pin_memory=True, drop_last=False, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(datasets['val'], batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(datasets['test'], batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    sample_images, sample_labels = next(iter(train_loader))
    sample_feats = model(sample_images.cuda(non_blocking=True)).clone()
    expected_width = 1024 if len(models) == 1 else 2048
    if sample_feats.shape[1] != sample_labels.shape[1] or sample_feats.shape[-1] != expected_width:
        raise ValueError(f'Incompatible frozen patch map: {sample_feats.shape}, labels={sample_labels.shape}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = nn.Linear(int(sample_feats.shape[-1]), 1).cuda()
    pos_weight = torch.tensor([_estimate_pos_weight(train_loader)], device='cuda')
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    validation_history = []
    for epoch in range(5):
        head.train()
        for images, labels in train_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            logits = head(model(images).clone()).squeeze(-1)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        validation_history.append(_eval(model, head, val_loader))
        print(f'[epoch] {epoch + 1}/5 val_patch_f1={validation_history[-1]["patch_f1"]:.4f}', flush=True)
    val_metrics = _eval(model, head, val_loader)
    test_metrics = _eval(model, head, test_loader)
    results = {
        'dataset': 'bbbc038', 'task': 'detection_proxy', 'arm': args.arm,
        'feature': 'per-patch unit single' if len(models) == 1 else 'per-patch unit branches, equally weighted concat',
        'image_size': 224, 'batch_size': 8, 'num_workers': 2, 'epochs': 5,
        'lr': 1e-3, 'weight_decay': 1e-4, 'seed': seed,
        'ordered_split_identities': ids, 'pos_weight': float(pos_weight.item()),
        'validation_history': validation_history,
        **{f'val_{k}': v for k, v in val_metrics.items()},
        **{f'test_{k}': v for k, v in test_metrics.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + '\n')
    print(f'[done] {out}', flush=True)


if __name__ == '__main__':
    main()
