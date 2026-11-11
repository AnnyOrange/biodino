#!/usr/bin/env python3
"""Evaluate the released Cryo-IEF ViT-B on the frozen BioDINO Cryo population.

This uses the identical source particles and 1-NN scorer as
run_ood_frozen_transfer.py; model-native single-channel input and normalization
are explicit adaptations. It is not a claim of byte-identical author inference.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.eval.ood_transfer.common import (
    code_identity, process_policy, read_json, require_inputs, save_json, sha256,
)
from run_ood_frozen_transfer import cryo


class CryoIEFEncoder:
    def __init__(self, source: Path, checkpoint: Path):
        sys.path.insert(0, str(source))
        from Cryo_IEF import vits

        model = vits.vit_base(
            num_classes=2, dynamic_img_size=True, stop_grad_conv1=False,
            use_bn=True, patch_size=14,
        )
        raw = load_file(str(checkpoint), device='cpu')
        state = {
            key.removeprefix('base_encoder.'): value
            for key, value in raw.items()
            if key.startswith('base_encoder.') and not key.startswith('base_encoder.head.')
        }
        if not state:
            raise ValueError('Cryo-IEF checkpoint lacks base_encoder')
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith('head.') for key in missing):
            raise ValueError(f'Cryo-IEF load mismatch: missing={missing}, unexpected={unexpected}')
        self.model = model.cuda().eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, planes):
        images = torch.cat([
            F.interpolate(torch.as_tensor(np.array(plane, dtype=np.float32, copy=True))[None, None],
                          size=(224, 224), mode='bilinear', align_corners=False)
            for plane in planes
        ]).cuda()
        images = (images - 0.53786290141223) / 0.11803331075935841
        with torch.autocast('cuda', dtype=torch.bfloat16):
            features = self.model.forward_features(images)[:, 0]
        return F.normalize(features.float(), dim=1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'input-receipt', 'data-root', 'checkpoint', 'author-source', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    process_policy()
    if args.batch < 1:
        raise ValueError('batch must be positive')
    require_inputs(args.manifest, args.data_root, args.input_receipt)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(
        protocol='ood-lw5-adaptation-v1-cryo-ief-same-population',
        source_samples='four author real projects; 320000 particles',
        evaluator='shared cosine leave-self-out 1-NN from run_ood_frozen_transfer.py',
        preprocessing='per-particle minmax; Cryo-IEF native 224 single-channel mean/std',
        checkpoint_sha256=sha256(args.checkpoint),
        manifest_sha256=sha256(args.manifest),
        code=code_identity(),
    )
    save_json(output / 'identity.json', identity)
    encoder = CryoIEFEncoder(Path(args.author_source), Path(args.checkpoint))
    metrics = cryo(args, read_json(args.manifest), encoder, output)
    save_json(output / 'complete.json', dict(identity=identity, metrics=metrics))
    print('CRYO_IEF_COMPLETE', metrics, flush=True)


if __name__ == '__main__':
    main()
