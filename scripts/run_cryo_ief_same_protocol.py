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
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.eval.ood_transfer.common import (
    code_identity, process_policy, read_json, require_inputs, save_json, sha256,
)
from run_ood_frozen_transfer import cryo


class CryoIEFEncoder:
    def __init__(self, source: Path, checkpoint: Path, preprocessing: str):
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
        self.preprocessing = preprocessing

    @staticmethod
    def author_native_plane(plane):
        """CryoData mrcs_resize and to_int8, without its temporary LMDB cache."""
        plane = np.asarray(plane, dtype=np.float32)
        height, width = plane.shape
        if height == width and width > 224:
            transformed = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(plane)))
            hartley = transformed.real - transformed.imag
            start = int(width / 2 - 112)
            stop = int(width / 2 + 112)
            crop = hartley[start:stop, start:stop]
            transformed = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(crop)))
            resized = (transformed.real - transformed.imag) / (224 * 224)
        elif height != 224 or width != 224:
            resized = zoom(plane, (224 / height, 224 / width), order=3)
        else:
            resized = plane
        low = float(np.min(resized))
        span = float(np.max(resized) - low)
        if span:
            image = ((resized - low) / span * 255).astype(np.uint8)
        else:
            image = np.asarray(resized).astype(np.uint8)
        return torch.from_numpy(image.astype(np.float32) / 255)[None, None]

    @torch.inference_mode()
    def __call__(self, planes):
        if self.preprocessing == 'author_native':
            images = torch.cat([self.author_native_plane(plane) for plane in planes]).cuda()
        else:
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
    parser.add_argument('--preprocessing', choices=('shared', 'author_native'), default='shared')
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
        preprocessing=(
            'CryoData FFT/cubic resize to224, per-particle uint8 minmax, ToTensor, model mean/std'
            if args.preprocessing == 'author_native' else
            'per-particle minmax, bilinear resize224, model mean/std'
        ),
        checkpoint_sha256=sha256(args.checkpoint),
        manifest_sha256=sha256(args.manifest),
        code=code_identity(),
    )
    save_json(output / 'identity.json', identity)
    encoder = CryoIEFEncoder(Path(args.author_source), Path(args.checkpoint), args.preprocessing)
    image_transform = (lambda image: np.asarray(image, dtype=np.float32)) if args.preprocessing == 'author_native' else None
    if image_transform is None:
        metrics = cryo(args, read_json(args.manifest), encoder, output)
    else:
        metrics = cryo(args, read_json(args.manifest), encoder, output, image_transform=image_transform)
    save_json(output / 'complete.json', dict(identity=identity, metrics=metrics))
    print('CRYO_IEF_COMPLETE', metrics, flush=True)


if __name__ == '__main__':
    main()
