#!/usr/bin/env python3
"""Grouped uncertainty of mean single-seed semantic Dice, preserving every seed."""
import argparse,fcntl,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
from unified_paper_segmentation_artifacts import attach_source_manifest
from dinov3.eval.bio_segmentation.linear_probe import DATASET_CONFIGS

def dump(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def metric(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    denominator=cm.sum(-1)+cm.sum(-2)
    dice=np.divide(2*tp,denominator+1e-8,out=np.full(tp.shape,np.nan),where=denominator>0)
    return np.nanmean(dice,axis=-1).mean(axis=-1)

def ci_path(model,dataset,split,budget):
    return OUT/'statistics'/f'{model}__segmentation__{dataset}__{split}__E{budget}__ci.json'

def count_path(cell,budget):
    store=OUT/'statistics/segmentation_counts';store.mkdir(exist_ok=True)
    tag=cell.get('result_family','segmentation')
    return store/f"{cell['model']}__{cell['dataset']}__{cell['split']}__E{budget}.npz" if tag=='segmentation' else store/f"{cell['model']}__{tag}__{cell['dataset']}__{cell['split']}__E{budget}.npz"

def process(cell,budget,settings):
    paths=[OUT/'results'/cell['model']/cell.get('result_family','segmentation')/cell['dataset']/cell['split']/f'E{budget}'/f'seed{s}'/'component_result.json' for s in range(5)]
    available=[p for p in paths if p.exists()]
    for p in available:
        result=json.loads(p.read_text())
        if cell['dataset']=='livecell' and result.get('sample_manifest_status')=='verified':
            old_path=Path(result['sample_manifest']);old=json.loads(old_path.read_text())
            if old['samples'] and any('#image_id=' not in r['sample_id'] for r in old['samples']):
                archive=OUT/'corrections/livecell_record_ids'/sha_name(p)
                archive.mkdir(parents=True,exist_ok=True)
                for original in [p,old_path]:
                    saved=archive/original.name
                    if not saved.exists():saved.write_bytes(original.read_bytes())
                fixed=attach_source_manifest(cell,p.parent);new=json.loads(Path(fixed['sample_manifest']).read_text())
                assert len(old['samples'])==len(new['samples'])
                assert all(a['sample_id'].split('#image_id=')[0]==b['sample_id'].split('#image_id=')[0] and a['group_id']==b['group_id'] and a['label_sha256']==b['label_sha256'] for a,b in zip(old['samples'],new['samples']))
                result.update(fixed);dump(p,result)
                dump(archive/'correction.json',dict(reason='Add official COCO record ID to distinguish repeated records of one image. Original-image groups, every true-label hash, predictions and all metrics unchanged.',corrected_result=str(p)))
        if result.get('sample_manifest_status')!='verified':
            result.update(attach_source_manifest(cell,p.parent));dump(p,result)
    if len(available)!=5:return
    target=ci_path(cell['model'],cell['dataset'],cell['split'],budget)
    if cell.get('requires_fresh_training'):target=target.with_name(target.name.replace('__segmentation__','__segmentation_refit__'))
    counts_file=count_path(cell,budget)
    grouping_version=None
    if cell['dataset']=='livecell':
        from unified_paper_livecell_groups import VERSION
        grouping_version=VERSION
    if target.exists() and counts_file.exists():
        with np.load(counts_file) as cached:
            previous=cached['source_result_sha256'].tolist()
        current=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        if previous==current and json.loads(target.read_text()).get('grouping_version')==grouping_version:return
        revision=hashlib.sha256(json.dumps(previous).encode()).hexdigest()
        archive=OUT/'statistics/source_revisions'/revision;archive.mkdir(parents=True,exist_ok=True)
        for file in [target,counts_file]:
            archived=archive/file.name
            if archived.exists():archived=archive/(hashlib.sha256(file.read_bytes()).hexdigest()+'.'+file.name)
            file.replace(archived)
    all_counts=[];identity=None;source=[];scores=[]
    for p in paths:
        result=json.loads(p.read_text());manifest=json.loads(Path(result['sample_manifest']).read_text())
        rows=manifest['samples']
        if cell['dataset']=='livecell':
            from unified_paper_livecell_groups import regroup
            rows,metadata=regroup(rows);manifest.update(metadata,group_unit=metadata['grouping'])
        ids=[(r['sample_id'],r['group_id'],r['label_sha256']) for r in rows]
        if identity is None:identity=ids
        if ids!=identity:raise ValueError('Seed memberships or labels differ')
        groups=sorted({r['group_id'] for r in rows});gpos={v:i for i,v in enumerate(groups)}
        num_classes=DATASET_CONFIGS[cell['dataset']]['num_classes']
        counts=np.zeros((len(groups),num_classes,num_classes),dtype=np.float64)
        with np.load(result['prediction_path']) as z:
            pred=z['prediction'];truth=z['true_label']
            for row,yhat,y in zip(rows,pred,truth):
                valid=y!=255
                counts[gpos[row['group_id']]]+=np.bincount((y[valid].astype(int)*num_classes+yhat[valid]).ravel(),minlength=num_classes**2).reshape(num_classes,num_classes)
        score=float(metric(counts.sum(0)[None]))
        if abs(score-result['test']['mDice'])>1e-6:raise ValueError('Predictions do not reproduce mDice')
        all_counts.append(counts);scores.append(result['test']['mDice'])
        source.append({'result':str(p),'prediction':result['prediction_path'],'manifest':result['sample_manifest'],
            'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'prediction_metric':score})
    a=np.stack(all_counts);rng=np.random.default_rng(settings['random_seed']);draw=[]
    strata={r['group_id']:r.get('stratum','all') for r in rows}
    positions=[np.array([gpos[g] for g in groups if strata[g]==st]) for st in sorted(set(strata.values()))]
    temp=counts_file.with_suffix('.pending.npz')
    np.savez_compressed(temp,counts=a,groups=np.asarray(groups),strata=np.asarray([strata[g] for g in groups]),sample_identity=np.asarray(identity),scores=np.asarray(scores),source_results=np.asarray([x['result'] for x in source]),source_result_sha256=np.asarray([x['result_sha256'] for x in source]),grouping_version=np.asarray(grouping_version or 'source-manifest-v1'))
    temp.replace(counts_file)
    if target.exists():return
    for start in range(0,settings['bootstrap_replicates'],128):
        w=np.zeros((min(128,settings['bootstrap_replicates']-start),len(groups)))
        for pos in positions:w[:,pos]=rng.multinomial(len(pos),np.full(len(pos),1/len(pos)),size=len(w))
        draw.extend(metric(np.einsum('bn,snij->bsij',w,a)).tolist())
    low,high=np.quantile(draw,[.025,.975]).tolist()
    dump(target,{'model':cell['model'],'task':'segmentation','dataset':cell['dataset'],'split':cell['split'],'budget':budget,
        'n_test':len(identity),'n_groups':len(groups),'group_unit':manifest['group_unit'],'group_limitation':manifest['group_limitation'],'bootstrap_strata':{st:sum(v==st for v in strata.values()) for st in set(strata.values())},
        'training_seeds':list(range(5)),'per_seed_values':scores,'seed_sd':float(np.std(scores,ddof=1)),
        **{k:manifest[k]for k in ['grouping_version','grouping_proof','grouping_proof_sha256','primary_grouping_verified']if k in manifest},
        'bootstrap_replicates':settings['bootstrap_replicates'],'random_seed':settings['random_seed'],
        'uncertainty':'Group bootstrap of pooled-pixel class-macro Dice, then mean across five separate fits; same group draws across all seeds; conditional on these fits',
        'source':source,'metrics':{'mDice':{'value':float(np.mean(scores)),'ci_low':low,'ci_high':high}}})
    print('SEG CI',cell['model'],cell['dataset'],budget,low,high,flush=True)

def sha_name(path):return hashlib.sha256(str(path).encode()).hexdigest()

def run():
    settings=json.loads((OUT/'config.json').read_text())['statistics']
    errors=[]
    cells=json.loads((OUT/'biodino_segmentation_inventory.json').read_text())['cells']
    fm=OUT/'fm_segmentation_inventory.json'
    if fm.exists():cells+=json.loads(fm.read_text())['cells']
    for cell in cells:
        if cell.get('superseded_protocol'):continue
        if cell['dataset']=='multimodal_cellseg' and cell['split']!='neurips22-author-public52-v2':continue
        for budget in [20,50]:
            try:process(cell,budget,settings)
            except Exception as error:
                errors.append({'model':cell['model'],'dataset':cell['dataset'],'budget':budget,'error':repr(error)})
                traceback.print_exc()
    dump(OUT/'states/segmentation_statistics.json',{'updated':time.time(),'errors':errors})
    from unified_paper_segmentation_pairs import run_pairs
    run_pairs(cells,settings)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    with (OUT/'locks/segmentation_statistics.local.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(0)
        while True:
            run()
            if not args.watch:break
            time.sleep(60)
