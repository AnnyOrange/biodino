#!/usr/bin/env python3
"""Matched formal v4 OOD, shared native data and scorer, all frozen encoders."""
from __future__ import annotations
import argparse, csv, fcntl, hashlib, json, logging, os, socket, sys, time
from pathlib import Path
CANONICAL_PYTHON = os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable != CANONICAL_PYTHON and os.environ.get('UNIFIED_PAPER_LOCAL_PILOT') != '1':
    os.execv(CANONICAL_PYTHON, [CANONICAL_PYTHON, '-u', __file__, *sys.argv[1:]])
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), '/mnt/huawei_deepcad/benchmark_model']
from dinov3.eval.eval_ood.datasets import build_id_reference_dataset, XrayTomogramSliceDataset, CryoParticleDataset
from dinov3.eval.eval_ood.metrics import id_vs_ood_knn
OUT = ROOT/'outputs/02_eval_runs/unified_paper'
LOCKED = ROOT/'outputs/02_eval_inputs/formal_v4/fm18_ood_selection_20261021.json'

def _collate(batch):
    images, labels, metas = zip(*batch)
    return list(images), np.asarray(labels), list(metas)

def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def inputs(task):
    locked = json.loads(LOCKED.read_text())
    source_path = Path(os.environ.get('UNIFIED_PAPER_OOD_SOURCE_RECORDS',locked['source_records']))
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == locked['source_records_sha256']
    source = json.loads(source_path.read_text())
    bench = Path(os.environ.get('UNIFIED_PAPER_BENCHMARK_ROOT','/mnt/huawei_deepcad/benchmark'))
    packed=os.environ.get('UNIFIED_PAPER_OOD_PACKED_ID')
    if packed:
        from unified_paper_ood_packed import PackedIDDataset
        ids=PackedIDDataset(packed);assert ids.meta['source_records_sha256']==locked['source_records_sha256']
        selected=ids.meta['selected_sources']
    else:
        ids = build_id_reference_dataset(bench, transform=None, max_samples=3000, seed=0)
        selected = [{'source': d.source, 'source_size': len(d.dataset.dataset),
                     'indices': list(map(int,d.dataset.indices))} for d in ids.datasets]
    if selected != source['id']:
        raise RuntimeError('ID selection differs from admitted v4 manifest')
    identities = list(ids.meta['identities']) if packed else []
    for part, records in ([] if packed else zip(ids.datasets, selected)):
        base = part.dataset.dataset
        for index in records['indices']:
            sample = f"{part.source}:{index}"
            if hasattr(base, 'samples'):
                stem = Path(base.samples[index][0]).stem
                group = stem.split('_',1)[0] if part.source == 'bbbc048' else stem.rsplit('_',1)[0]
                group = part.source + ':' + group
            else:
                group = sample
            identities.append({'sample_id':sample, 'group_id':group, 'source':part.source})
    packed_ood=os.environ.get('UNIFIED_PAPER_OOD_PACKED_'+task.upper())
    if packed_ood:
        from unified_paper_ood_packed import PackedIDDataset
        ds=PackedIDDataset(packed_ood);assert ds.meta['source_records_sha256']==locked['source_records_sha256']
        observed=ds.meta['selected_sources'];other=ds.meta['identities']
    elif task == 'xray':
        ds = XrayTomogramSliceDataset(bench/'ood', slices_per_volume=8, input_mode='three_slices')
        observed = [[r.volume_id,r.tomo_id,r.variant,r.z_index,list(r.raw_shape_xyz),r.raw_path.stat().st_size] for r in ds.records]
        other = [{'sample_id':f'{r.volume_id}:{r.z_index}', 'group_id':'xray:'+r.sample_id,
                  'source':'xray', 'volume_id':r.volume_id, 'tomo_id':r.tomo_id} for r in ds.records]
    else:
        ds = CryoParticleDataset(bench/'ood', max_particles_per_project=20000, seed=0)
        observed = [[r.project_id,r.cs_path.name,r.mrc_path.name,r.particle_index,r.class_id] for r in ds.records]
        other = [{'sample_id':f'{r.project_id}:{r.cs_path.name}:{r.mrc_path.name}:{r.particle_index}',
                  'group_id':'cryo:'+r.project_id, 'source':'cryo'} for r in ds.records]
    if observed != source[task]:
        raise RuntimeError(f'{task} population/order differs from admitted v4 manifest')
    assert len(set(x['sample_id'] for x in other)) == len(other)
    return locked, ids, ds, identities, other

def encode(ds, encoder, target, identity_hash, batch_size, workers):
    meta = target.with_suffix('.json')
    fingerprint_root=os.environ.get('UNIFIED_PAPER_OOD_INPUT_FINGERPRINTS')
    certified=None
    if fingerprint_root:
        from verify_unified_ood_inputs import fingerprint
        part=target.stem.removesuffix('_features')
        certified=json.loads((Path(fingerprint_root)/f'{part}.json').read_text())
        assert certified['identity_sha256']==identity_hash and certified['n_samples']==len(ds)
    staged=getattr(ds,'meta',{}).get('transport_stage')=='dinov3_spatial_uint8_v1'
    if staged:
        assert certified and ds.meta['all_normalized_inputs_elementwise_equal']
        fp_path=Path(fingerprint_root)/f'{part}.json'
        assert hashlib.sha256(fp_path.read_bytes()).hexdigest()==ds.meta['input_fingerprints_sha256']
        if encoder is not None:
            from PIL import Image
            from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder
            assert isinstance(encoder,Dinov3CkptEncoder)
            with np.load(ds.root/ds.meta['raw_validation_examples']) as examples:
                raw_images=[Image.fromarray(examples[f'image_{i:06d}']) for i in range(2)]
            raw_features=encoder.encode_images(raw_images);prepared_features=encoder.encode_images([ds[i][0] for i in range(2)])
            assert np.array_equal(raw_features,prepared_features),'Real encoder outputs differ after spatial transport'
            write_json(target.with_name('spatial_transport_validation.json'),dict(n_validation_examples=2,real_encoder_features_bitwise_equal=True,all80000_normalized_inputs_elementwise_equal=True,canonical_input_fingerprints=str(fp_path),input_pack=str(ds.root/'manifest.json')))
    if target.exists() and meta.exists():
        check = json.loads(meta.read_text())
        if check['identity_sha256'] != identity_hash:
            raise RuntimeError('Cache identity mismatch')
        a = np.load(target)
        if len(a) != len(ds) or not np.isfinite(a).all():
            raise RuntimeError('Invalid feature cache')
        return a.astype(np.float32)
    chunks = []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=_collate)
    for i,(images,_,_) in enumerate(loader):
        if certified:
            expected=(ds.meta['transport_fingerprints'] if staged else certified['fingerprints'])[i*batch_size:i*batch_size+len(images)]
            if [fingerprint(image) for image in images]!=expected:raise ValueError(f'Resident input pixels differ from shared canonical input at batch {i}: {target.name}')
        chunks.append(encoder.encode_images(images).astype(np.float16))
        if i % 25 == 0:
            print(f'{target.name} {min((i+1)*batch_size,len(ds))}/{len(ds)}',flush=True)
    a = np.concatenate(chunks)
    if not np.isfinite(a).all():
        raise RuntimeError('Nonfinite encoder features')
    with target.with_suffix('.tmp').open('wb') as f:
        np.save(f,a)
    target.with_suffix('.tmp').replace(target)
    write_json(meta, {'identity_sha256':identity_hash,'shape':list(a.shape),'all_input_pixels_verified':bool(certified)})
    return a.astype(np.float32)

def main():
    from unified_paper_gpu_admission import reserve_for_current_command
    reserve_for_current_command()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--task', choices=['xray','cryo'],required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--train-config',type=Path)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--input-fix-v2',action='store_true',help='Correct raw floating CyclOps normalization; retain exact selected IDs')
    p.add_argument('--group-fix-v3',action='store_true',help='Score directly on the shared source/FOV-disjoint bank and test members')
    a = p.parse_args()
    torch.set_num_threads(2)
    key=f'{a.model}__ood__{a.task}'+('__inputfix_v2' if a.input_fix_v2 else '')
    legacy=OUT/'results'/a.model/'ood'
    if a.group_fix_v3 and not a.input_fix_v2:p.error('--group-fix-v3 requires --input-fix-v2')
    root=OUT/'results'/a.model/('ood_groupfix_v3' if a.group_fix_v3 else 'ood_inputfix_v2' if a.input_fix_v2 else 'ood')
    root.mkdir(parents=True,exist_ok=True)
    with (OUT/'locks'/f'{a.model}__ood.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result=root/f'{a.task}_result.json'
        if result.exists():
            print('COMPLETE',result,flush=True);return
        started=time.time()
        if not a.preflight:
            write_json(OUT/'states'/f'{key}.json', {'state':'running','host':socket.gethostname(),'pid':os.getpid(),
                       'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'started':started,'result':str(result)})
        try:
            locked,ids,ds,identity,other = inputs(a.task)
            selection_path=LOCKED
            if a.group_fix_v3:
                selection_path=LOCKED.with_name('fm18_ood_group_selection_v3.json')
                grouped=json.loads(selection_path.read_text());assert grouped['source_records_sha256']==locked['source_records_sha256']
                assert grouped['id_group_overlap']==0
                locked=grouped
            print(f'PREFLIGHT PASS id={len(ids)} {a.task}={len(ds)}',flush=True)
            if a.preflight:return
            ready_id=(root/'id_features.npy').exists() and (root/'id_features.json').exists()
            if a.input_fix_v2 and ready_id and (root/f'{a.task}_features.npy').exists():
                # This second modality only scores existing features; loading a giant
                # GPU encoder here wastes memory and can fail amid unrelated jobs.
                prior_path=legacy/f'{a.task}_result.json'
                prior=json.loads(prior_path.read_text()) if prior_path.exists() else {}
                encoder=None;checkpoint=prior.get('checkpoint',str(a.checkpoint));adaptation=prior.get('adaptation','Frozen features from fixed checkpoint')+'; corrected raw-float ID input v2'
            else:
                if a.checkpoint:
                    from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder
                    encoder=Dinov3CkptEncoder(a.checkpoint,a.train_config,'cuda',1,True,torch.bfloat16)
                    checkpoint=str(a.checkpoint)
                    adaptation='shared DINOv3 final CLS + patch mean; config normalization; auto channels'
                elif a.model in ['distillpath','morphem','genbio_pathfm','openpath']:
                    from run_new_fm_v4_linear_20261016 import MODELS
                    from unified_paper_encoders import NewFormalGlobalEncoder
                    encoder=NewFormalGlobalEncoder(a.model,'cuda',a.batch_size)
                    checkpoint=str(MODELS[a.model])
                    adaptation='shared final CLS + patch mean; existing published NewEncoder normalization/channel adaptation; resize256/crop224'
                else:
                    from benchmark_eval.encoders import MODEL_REGISTRY
                    from run_fm_rules import StandardExternalEncoder
                    encoder=StandardExternalEncoder(a.model,224,256)
                    checkpoint=str(MODEL_REGISTRY[a.model].path)
                    adaptation='audited RuleFMFeatures final CLS + patch mean; published normalization and documented no-CLS architecture fallback'
            if a.input_fix_v2:
                from ood_input_fix import CorrectedIDDataset,REVISION
                corrected=ids if getattr(ids,'input_revision',None)==REVISION else CorrectedIDDataset(ids)
                # Retain unaffected BloodMNIST/BBBC048 features byte-for-byte.
                if not ready_id and (legacy/'id_features.npy').exists():
                    positions=[i for i,x in enumerate(identity) if x['source']=='cyclops']
                    subset=torch.utils.data.Subset(corrected,positions)
                    patch=encode(subset,encoder,root/'cyclops_corrected_features.npy',sha({'ids':[identity[i] for i in positions],'input_revision':REVISION}),a.batch_size,a.workers)
                    merged=np.load(legacy/'id_features.npy').copy();merged[positions]=patch
                    with (root/'id_features.tmp').open('wb') as f:np.save(f,merged)
                    (root/'id_features.tmp').replace(root/'id_features.npy')
                    write_json(root/'id_features.json',{'identity_sha256':sha(identity),'input_revision':REVISION,'shape':list(merged.shape),'reencoded':len(positions),'unchanged':len(identity)-len(positions)})
                # OOD image path is unaffected; share its existing cache, not predictions.
                for ext in ['.npy','.json']:
                    target=root/f'{a.task}_features{ext}'
                    if not target.exists() and (legacy/target.name).exists():os.link(legacy/target.name,target)
                ids=corrected
            idf=encode(ids,encoder,root/'id_features.npy',sha(identity),a.batch_size,a.workers)
            if a.input_fix_v2:
                metadata=json.loads((root/'id_features.json').read_text());metadata['input_revision']=REVISION
                write_json(root/'id_features.json',metadata)
            of=encode(ds,encoder,root/f'{a.task}_features.npy',sha(other),a.batch_size,a.workers)
            split_kwargs={'bank_indices':np.asarray([r['position'] for r in locked['id_bank']]),'test_indices':np.asarray([r['position'] for r in locked['id_test']])} if a.group_fix_v3 else {}
            metrics,pred=id_vs_ood_knn(idf,of,k=10,train_fraction=.7,seed=0,return_predictions=True,**split_kwargs)
            expected=[r['position'] for r in locked['id_test']]
            assert expected==pred['id_test_indices'].tolist()
            rows=[identity[i] for i in expected]+other
            fields=['model','checkpoint','dataset','task','fold','seed','sample_id','group_id','source','true_label','score']
            prediction=root/f'{a.task}_predictions.csv'
            with prediction.with_suffix('.tmp').open('w') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                for row,y,s in zip(rows,pred['labels'],pred['scores']):
                    w.writerow({**{k:row[k] for k in ('sample_id','group_id','source')}, 'model':a.model,
                        'checkpoint':checkpoint,'dataset':a.task,'task':'ood','fold':'group_fixed' if a.group_fix_v3 else 'fixed','seed':'not_applicable',
                        'true_label':int(y),'score':float(s)})
            prediction.with_suffix('.tmp').replace(prediction)
            write_json(result, {'protocol':'bio-eval-union-v4/formal-ood','model':a.model,'checkpoint':checkpoint,
                'dataset':a.task,'metrics':metrics,'seed_applicable':False,'test_ids_sha256':sha(rows),
                'selection':str(selection_path),'prediction_path':str(prediction),'adaptation':adaptation,
                'split_revision':'source_stratified_fov_disjoint_v3' if a.group_fix_v3 else 'legacy_image_random_v2',
                'input_revision':REVISION if a.input_fix_v2 else 'legacy_float_clip','statistics_status':'pending_group_bootstrap','seconds':time.time()-started})
            write_json(OUT/'states'/f'{key}.json',{'state':'completed_metrics','host':socket.gethostname(),
                'pid':os.getpid(),'result':str(result),'ended':time.time()})
            print('COMPLETE',metrics,flush=True)
        except Exception as e:
            write_json(OUT/'states'/f'{key}.json',{'state':'failed','host':socket.gethostname(),'pid':os.getpid(),
                'error':repr(e),'ended':time.time()})
            raise
if __name__=='__main__':main()
