#!/usr/bin/env python3
"""Prepare exact-byte Cryo/PXCT inputs and author-generated FILM pair arrays."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dinov3.eval.ood_transfer.common import read_json, save_json, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['cryo','pxct','film'],required=True)
    parser.add_argument('--canonical-inputs',required=True)
    parser.add_argument('--benchmark-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--supplement-root',required=True)
    args=parser.parse_args()
    canonical=Path(args.canonical_inputs); benchmark=Path(args.benchmark_root).resolve()
    output=Path(args.output);supplement=Path(args.supplement_root)
    output.mkdir(parents=True,exist_ok=True);supplement.mkdir(parents=True,exist_ok=True)
    files=[]
    def add(path,relative=None):
        path=Path(path).resolve()
        if relative is None:relative=str(path.relative_to(benchmark))
        files.append(dict(sample_id=relative,path=relative,input_sha256=sha256(path),bytes=path.stat().st_size))
        print('HASHED',args.task,relative,flush=True)
        return relative
    if args.task=='cryo':
        source=canonical/'cryo_genuine.json';d=read_json(source);projects=[]
        for p in d['projects']:
            particles=add(p['particles']);quality=add(p['quality_scores'])
            for stack in p['stacks']:add(stack['path'])
            projects.append(dict(project=p['project'],particles=particles,quality_scores=quality,
                                 stack_root=str(Path(particles).parent/'restack')))
        result=dict(task='cryo',projects=projects)
    elif args.task=='pxct':
        source=canonical/'pxct_synapse.json';d=read_json(source);records=[]
        for row in d['records']:
            relative='pxct_rois/'+Path(row['path']).name
            target=supplement/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.symlink_to(Path(row['path']).resolve())
            records.append(dict(row,path=add(row['path'],relative)))
        result=dict(task='pxct',records=records,independent_specimens=2)
    else:
        source=canonical/'film_spend.json';d=read_json(source)
        from dinov3.eval.ood_transfer import film_pairing
        film_pairing.DATA=benchmark/'ood/ood_classification/datasets/FILM/extracted'
        film_pairing.AUTHOR=film_pairing.DATA/'SPEND/SPEND-main'
        film_pairing.RAW=film_pairing.DATA/'Fig2 SPEND example/hyper raw.tif'
        x,y,meta=film_pairing.training_pairs(seed=0)
        import hashlib
        if (hashlib.sha256(x.tobytes()).hexdigest()!=d['pairs_sha256']['input'] or
            hashlib.sha256(y.tobytes()).hexdigest()!=d['pairs_sha256']['target']):
            raise ValueError('Author pairing does not match locked manifest')
        target=supplement/'film/pairs.npz';target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            previous=np.load(target,allow_pickle=False)
            if not np.array_equal(previous['x'],x) or not np.array_equal(previous['y'],y):
                raise ValueError('Existing pair archive differs')
        else:
            np.savez_compressed(target,x=x,y=y)
        pairs=add(target,'film/pairs.npz')
        raw_path=supplement/'film/raw.tif'
        if not raw_path.exists():raw_path.symlink_to(Path(d['raw']).resolve())
        raw=add(d['raw'],'film/raw.tif')
        result=dict(task='film',pairs=pairs,raw=raw,pairing=meta,pairs_sha256=d['pairs_sha256'])
    result.update(schema='ood-transfer-frozen-inputs-v1',files=files,source_manifest_sha256=sha256(source))
    save_json(output/(args.task+'_frozen.json'),result)
    (output/(args.task+'_frozen_files.txt')).write_text('\n'.join(f['path'] for f in files)+'\n')


if __name__=='__main__':
    os.nice(15)
    main()
