#!/usr/bin/env python3
"""H100 LC25000 classification with the exact unified original-tile split/probe."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

PROTOCOL = 'lc25000-clean-original-tile-80-20-seed0-v1'
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO/'outputs/02_eval_runs/l_seed_repeats_h100_20261028/lc25000_correction'
PINS = {'probes_snapshot.py':'3d997535046493ee0049382df1b2cb6934cd21bd21f35999a785d8eb3d3f6763',
        'groups.csv':'2ff9e992f759f265dfc9aa2363de0dfcc8e6559884d8bb02a4c925214fdf7e23',
        'images.json':'61dc924b3799fd23bee8a63fe293b1750f124648cc10699b79f8fefc53dffb9b',
        'split.json':'d6af8fbe6dc872ad01261d294c2f1d6864943a5e2cbcfb7d3647b19729e41662'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def membership(dataset):
    import numpy as np
    for name,expected in PINS.items():
        if sha(INPUT/name)!=expected:raise ValueError(f'Pinned input changed: {name}')
    with (INPUT/'groups.csv').open() as f:
        mapping={(row['stem'],row['label']):row['group_id'] for row in csv.DictReader(f)}
    assert len(mapping)==25000 and len(set(mapping.values()))==1246
    split=json.loads((INPUT/'split.json').read_text())
    records=json.loads((INPUT/'images.json').read_text())['records']
    samples=[(Path(p),int(y)) for p,y in dataset.samples]
    expected=[(Path(r['sample_id']).stem,Path(r['sample_id']).parent.name,r['label']) for r in records]
    assert [(p.stem,p.parent.name,y) for p,y in samples]==expected
    assert set((p.stem,p.parent.name) for p,_ in samples)==set(mapping)
    groups=np.asarray([mapping[p.stem,p.parent.name] for p,_ in samples])
    test=np.flatnonzero(np.isin(groups,split['test_groups']));train=np.flatnonzero(~np.isin(groups,split['test_groups']))
    assert (len(train),len(test))==(19949,5051)
    assert not set(groups[train]) & set(groups[test])
    return train,test,groups,records,split


def audit(dataset,records,verify_images=False):
    stats=[(str(p),Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p,_ in dataset.samples]
    stat_digest=hashlib.sha256(json.dumps(stats).encode()).hexdigest()
    path=INPUT/'h100_image_audit.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    if verify_images or previous.get('file_stats_sha256')!=stat_digest or previous.get('manifest_sha256')!=PINS['images.json']:
        for (p,_),record in zip(dataset.samples,records):
            if sha(p)!=record['image_sha256']:raise ValueError(f'Image differs from unified inventory: {p}')
        save(path,dict(status='PASS',count=25000,manifest_sha256=PINS['images.json'],file_stats_sha256=stat_digest))
    return stat_digest


def validate(output):
    row=json.loads((output/'component_result.json').read_text())
    assert row['protocol']==PROTOCOL and row['annotation_sha256']==PINS['groups.csv']
    assert row['committed_split_sha256']==PINS['split.json'] and row['probe_source_sha256']==PINS['probes_snapshot.py']
    assert (row['n_train'],row['n_test'])==(19949,5051)
    assert row['image_size']==224 and row['resize_size']==256
    assert row['feature_readout']=='final CLS + patch mean'
    identity=json.loads((Path(row['checkpoint']).parent/'identity.json').read_text())
    assert row['checkpoint_sha256']==identity['sha256'] and row['config_sha256']==identity['config_sha256']
    assert row['prediction_sha256']==sha(output/'test_predictions.npz')
    assert row['sample_manifest_sha256']==sha(output/'sample_manifest.json')
    import math
    assert all(math.isfinite(float(row[k])) for k in ('accuracy','balanced_accuracy','macro_f1'))
    with (output/'summary.csv').open() as f:
        summaries=list(csv.DictReader(f))
    assert len(summaries)==1 and summaries[0]['protocol']==PROTOCOL
    assert all(float(summaries[0][k])==float(row[k]) for k in ('accuracy','balanced_accuracy','macro_f1'))
    return {k:row[k] for k in ('accuracy','balanced_accuracy','macro_f1')} | {'protocol':PROTOCOL,'dataset':'lc25000'}


def install(module):
    original_command=module.command
    original_validate=module.validate
    def command(task,iteration,output):
        if task['family']=='classification' and task['dataset']=='lc25000':
            checkpoint=module.ROOT/f'checkpoints/{iteration}/checkpoint.pth'
            return [module.PYTHON,'-u',str(Path(__file__).resolve()),'--checkpoint',str(checkpoint),
                    '--train-config',str(checkpoint.parent/'config.yaml'),'--output',str(output),
                    '--benchmark-root',str(module.BENCHMARK),'--model',f'{module.MODEL_TAG}_ck{iteration}']
        return original_command(task,iteration,output)
    def check(task,output):
        if task['family']=='classification' and task['dataset']=='lc25000':return validate(output)
        return original_validate(task,output)
    module.command=command
    module.validate=check


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path);p.add_argument('--train-config',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--model')
    p.add_argument('--benchmark-root',default='/data_2/suxin/test-dataset')
    p.add_argument('--audit-only',action='store_true')
    a=p.parse_args()
    import numpy as np
    import torch
    import sklearn
    from threadpoolctl import threadpool_limits
    from dinov3.eval.bio_frozen_eval.registry import build_dataset
    assert sklearn.__version__=='1.8.0'
    torch.set_num_threads(2);np.random.seed(0);torch.manual_seed(0)
    dataset,task=build_dataset('lc25000','train',None,None,benchmark_root=a.benchmark_root)
    assert task=='classification'
    train,test,groups,records,split=membership(dataset)
    audit_digest=audit(dataset,records,verify_images=a.audit_only)
    if a.audit_only:
        print(json.dumps(dict(status='PASS',train=len(train),test=len(test),groups=len(set(groups)),group_overlap=0,image_inventory='25000 exact image SHA256 matches')),flush=True)
        return
    assert a.checkpoint and a.train_config and a.output and a.model
    a.output.mkdir(parents=True,exist_ok=True)
    from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder,extract_features
    encoder=Dinov3CkptEncoder(checkpoint=a.checkpoint,train_config=a.train_config,device='cuda:0',
        n_last_blocks=1,use_avgpool=True,autocast_dtype=torch.bfloat16,image_size=224,resize_size=256,channel_policy='auto')
    # Match unified extraction's 64-item loader and 16-image encoder microbatches.
    original=encoder.encode_images
    encoder.encode_images=lambda images:np.concatenate([original(images[i:i+16]) for i in range(0,len(images),16)])
    features,labels=extract_features(dataset,encoder,a.output/'unused_features.npz',64,2,True,a.model,save_features=False,save_paths=False)
    features=features.astype(np.float32)
    assert features.shape[0]==25000 and np.isfinite(features).all()
    assert np.array_equal(labels,np.asarray([r['label'] for r in records]))
    del encoder,original
    import gc
    gc.collect();torch.cuda.empty_cache()
    name='dinov3.eval.bio_frozen_eval._lc25000_pinned_probes'
    spec=importlib.util.spec_from_file_location(name,INPUT/'probes_snapshot.py')
    probe=importlib.util.module_from_spec(spec);sys.modules[name]=probe;spec.loader.exec_module(probe)
    with threadpool_limits(limits=2):
        result=probe.run_classification_probe_split(features[train],labels[train],features[test],labels[test])
    ids=np.asarray([r['sample_id'] for r in records])
    prediction=a.output/'test_predictions.npz'
    temp=a.output/'test_predictions.pending.npz'
    np.savez_compressed(temp,**result.predictions,sample_id=ids[test],group_id=groups[test]);temp.replace(prediction)
    manifest=a.output/'sample_manifest.json'
    save(manifest,dict(protocol=PROTOCOL,train=[(ids[i],int(labels[i])) for i in train],
         test=[(ids[i],int(labels[i])) for i in test],test_groups=groups[test].tolist(),
         annotation_sha256=PINS['groups.csv'],split_seed=0,independent_unit=split['independent_unit']))
    identity=json.loads((a.checkpoint.parent/'identity.json').read_text())
    row={**result.to_dict(),'protocol':PROTOCOL,'split':PROTOCOL,'model':a.model,'dataset':'lc25000',
         'checkpoint':str(a.checkpoint),'checkpoint_sha256':identity['sha256'],
         'config_sha256':sha(a.train_config),'annotation_sha256':PINS['groups.csv'],
         'committed_split_sha256':PINS['split.json'],'probe_source_sha256':PINS['probes_snapshot.py'],
         'image_manifest_sha256':PINS['images.json'],'image_audit_stats_sha256':audit_digest,
         'image_size':224,'resize_size':256,'feature_readout':'final CLS + patch mean',
         'sklearn_version':sklearn.__version__,'solver_seed':0,'training_seed':'not_applicable',
         'group_note':'Published original-tile groups; patient/slide identities unavailable',
         'prediction_sha256':sha(prediction),'sample_manifest_sha256':sha(manifest)}
    save(a.output/'component_result.json',row)
    summary_tmp=a.output/'summary.csv.pending'
    with summary_tmp.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    summary_tmp.replace(a.output/'summary.csv')
    validate(a.output)
    print('COMPLETE',PROTOCOL,result.metrics,flush=True)

if __name__=='__main__':main()
