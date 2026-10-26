#!/usr/bin/env python3
"""Screening E50/B16/seed0 with the corrected author Multimodal split.

Uses the canonical extractor and probe, without changing the 44 running tasks.
The old pipeline CLI does not accept this newly certified split identifier.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'neurips22-author-public52-v2'
COUNTS = {'train': 1000, 'val': 101, 'test': 52}
MANIFEST_SHA256 = {
    'train': '6bcc726b8bd3dc3acf761886c53e0cac821b2e6c0bd625ccee7a27a636ac7d39',
    'val': '833423bce65f916942a0183b5cbffd5c1d715d25858f91cd904fe2a728547128',
    'test': 'a7f77a24beb1107c06e014dc207cbfe8e79d804989777141fbee0740714e112a',
}


def certify(data_root):
    ready_file = ROOT / 'outputs/02_eval_runs/unified_paper/multimodal_official_readiness.json'
    ready = json.loads(ready_file.read_text())
    assert ready['protocol'] == PROTOCOL and ready['counts'] == COUNTS
    image_ids, mask_ids, records = set(), set(), {}
    for split, expected in COUNTS.items():
        manifest = Path(ready['manifests'][split])
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        assert digest == ready['manifest_sha256'][split] == MANIFEST_SHA256[split]
        rows = list(csv.DictReader(manifest.open()))
        assert len(rows) == expected
        for row in rows:
            image, mask = (data_root / row[k] for k in ('image_path', 'mask_path'))
            assert image.is_file() and mask.is_file(), (image, mask)
            assert image.parent.parent == mask.parent.parent
            assert mask.stem == image.stem + '_label'
            assert str(image.resolve()) not in image_ids
            assert str(mask.resolve()) not in mask_ids
            image_ids.add(str(image.resolve()))
            mask_ids.add(str(mask.resolve()))
        records[split] = dict(manifest=str(manifest), sha256=digest, count=expected)
    return dict(protocol=PROTOCOL, counts=COUNTS, manifests=records,
                readiness_sha256=hashlib.sha256(ready_file.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--checkpoint-id', type=int, required=True)
    p.add_argument('--train-config', required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--cell', type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    proof = certify(args.data_root)
    assert Path(args.checkpoint).is_file() and Path(args.train_config).is_file()
    if args.preflight_only:
        print(json.dumps(proof))
        return
    import numpy as np
    args.cell.mkdir(parents=True, exist_ok=True)
    cache = args.cell / 'cache'
    cache.mkdir(exist_ok=True)
    commands, caches = [], {}
    for split, count in COUNTS.items():
        command = [sys.executable, '-m', 'dinov3.eval.bio_segmentation.feature_extractor',
                   '--dataset', 'multimodal_cellseg', '--data-root', str(args.data_root),
                   '--checkpoint', args.checkpoint, '--train-config', args.train_config,
                   '--output-dir', str(cache), '--split', split,
                   '--dataset-split-protocol', PROTOCOL, '--img-size', '512',
                   '--resize-mode', 'pad', '--batch-size', '4', '--num-workers', '2',
                   '--autocast-dtype', 'bf16', '--channel-policy', 'auto',
                   '--channel-tta-samples', '8', '--channel-policy-seed', '0',
                   '--chunked-cache', '--no-compress-cache']
        # This private cache is never reused across a split/protocol change.
        subprocess.run(command, cwd=ROOT, check=True)
        commands.append(command)
        files = list(cache.glob(f'multimodal_cellseg_{split}_*.npz'))
        assert len(files) == 1, files
        with np.load(files[0]) as z:
            assert int(z['num_samples']) == count
            assert int(z['n_layers']) == 1
            assert int(z['feature_batch_size']) == 4
        caches[split] = str(files[0])
    output = args.cell / 'results' / PROTOCOL / 'multimodal_cellseg' / str(args.checkpoint_id)
    command = [sys.executable, '-m', 'dinov3.eval.bio_segmentation.linear_probe',
               '--dataset', 'multimodal_cellseg', '--use-cached-features',
               '--train-cache', caches['train'], '--val-cache', caches['val'],
               '--test-cache', caches['test'], '--output-dir', str(output),
               '--epochs', '50', '--batch-size', '16', '--lr', '0.001',
               '--weight-decay', '0.0001', '--num-workers', '2',
               '--eval-every', '50', '--seed', '0', '--class-weight-mode', 'none']
    commands.append(command)
    proof.update(checkpoint=args.checkpoint, checkpoint_id=args.checkpoint_id,
                 commands=commands, scope='screening45 with corrected Multimodal membership; not complete V4-ID')
    (args.cell / 'split_proof.json').write_text(json.dumps(proof, indent=2))
    subprocess.run(command, cwd=ROOT, check=True)
    result_file = output / 'results.json'
    result = json.loads(result_file.read_text())
    meta = result['_meta']
    assert (meta['probe_epochs'], meta['probe_batch_size'], meta['seed'],
            meta['probe_eval_every'], meta['test_evaluations']) == (50, 16, 0, 50, 1)
    result['screening_split_proof'] = proof
    temp = result_file.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(result_file)
    print('VALIDATED_CORRECTED_MULTIMODAL', args.checkpoint_id, result['test']['mDice'], flush=True)


if __name__ == '__main__':
    main()
