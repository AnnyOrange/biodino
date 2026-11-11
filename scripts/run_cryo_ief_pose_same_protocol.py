#!/usr/bin/env python3
"""Cryo-IEF author-native features on the shared 12-dataset pose 1-NN scorer."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.eval.eval_ood.datasets import MRCStack
from dinov3.eval.ood_transfer.common import code_identity, process_policy, read_json, save_json, sha256
from run_cryo_ief_same_protocol import CryoIEFEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--author-source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    process_policy()
    code = code_identity()
    if code['dirty']:
        raise ValueError('Clean committed code required')
    source = Path(args.manifest)
    manifest = read_json(source)
    if manifest['protocol'] != 'cryo-ief-simulated-pose-knn-v1' or len(manifest['records']) != 288000:
        raise ValueError('Pose source population differs')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol='cryo-ief-same-pose-population-v1', code=code,
                    manifest_sha256=sha256(source), checkpoint_sha256=sha256(args.checkpoint),
                    preprocessing='author CryoData native FFT/cubic resize224, uint8 minmax',
                    scorer='per-dataset cosine 1NN excluding self', n=288000)
    if (out/'complete.json').exists():
        if read_json(out/'complete.json')['identity'] != identity:
            raise ValueError('Completed result identity changed')
        return
    save_json(out/'identity.json', identity)
    # Same resource gates used by the main OOD runner on the shared host.
    sys.path.insert(0, '/mnt/huawei_deepcad/dinov3/scripts')
    from unified_paper_host_admission import reserve_for_current_command as reserve_host
    from unified_paper_gpu_admission import reserve_for_current_command as reserve_gpu
    reserve_host(dict(key='ood_author:cryo_ief:cryo_pose', entrypoint=Path(__file__).name))
    reserve_gpu()
    encoder = CryoIEFEncoder(Path(args.author_source), Path(args.checkpoint), 'author_native')
    records = manifest['records']
    by_dataset = {}
    predictions = []
    stacks = {}
    for dataset in sorted({r['dataset'] for r in records}):
        rows = [r for r in records if r['dataset'] == dataset]
        if len(rows) != 24000 or sorted(np.bincount([r['pose_class'] for r in rows], minlength=12)) != [2000]*12:
            raise ValueError('Pose dataset labels/population differ: ' + dataset)
        bank = None
        for start in range(0, len(rows), args.batch):
            images = []
            for row in rows[start:start+args.batch]:
                path = row['path']
                if path not in stacks:
                    stacks[path] = MRCStack(Path(path))
                images.append(stacks[path].read(int(row['index'])))
            value = encoder(images).astype(np.float32)
            if bank is None:
                bank = np.empty((len(rows), value.shape[1]), dtype=np.float32)
            bank[start:start+len(value)] = value
            if start % (args.batch*100) == 0:
                print('FEATURES', dataset, start, len(rows), flush=True)
        labels = np.asarray([r['pose_class'] for r in rows])
        x = F.normalize(torch.from_numpy(bank).cuda(), dim=1)
        neighbors = []
        for start in range(0, len(rows), 256):
            end = min(start+256, len(rows))
            similarity = x[start:end] @ x.T
            similarity[torch.arange(end-start, device='cuda'), torch.arange(start,end,device='cuda')] = -torch.inf
            neighbors.extend(similarity.argmax(1).cpu().tolist())
        predicted = labels[np.asarray(neighbors)]
        score = float(np.mean(predicted == labels))
        by_dataset[dataset] = score
        predictions.extend(dict(sample_id=r['sample_id'], dataset=dataset,
                                truth=int(labels[i]), prediction=int(predicted[i]),
                                neighbor_id=rows[int(neighbors[i])]['sample_id']) for i,r in enumerate(rows))
        print('DATASET', dataset, score, flush=True)
        del x, bank
        torch.cuda.empty_cache()
    save_json(out/'predictions.json', predictions)
    save_json(out/'complete.json', dict(identity=identity, metrics=dict(
        pose_top1_1nn_accuracy_by_dataset=by_dataset,
        pose_top1_1nn_accuracy_macro=float(np.mean(list(by_dataset.values()))),
        n_datasets=12, n_particles=288000, k=1),
        predictions_sha256=sha256(out/'predictions.json')))


if __name__ == '__main__':
    main()
