"""Paired released-image inference, keeping all five decoder fits together."""
import argparse,fcntl,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dinov3.eval.bio_segmentation.datasets.bbbc038_official import dsb_mean_precision
import unified_paper_global_statistics as common
OUT=common.OUT;STORE=OUT/'statistics';CACHE=STORE/'bbbc038_index';CACHE.mkdir(exist_ok=True)
PENDING=[]
def local_artifact(value):
    remote='/data/xuzijing/l20tb_selected29_dense_20261021/unified_paper'
    return str(ROOT)+value[len(remote):]if value.startswith(remote+'/')else value
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(spec,stage):
    model=spec['model'];found=[]
    manifest=ROOT/'outputs/02_eval_inputs/formal_v4/bbbc038_official_manifest.json';data=json.loads(manifest.read_text());expected=data['splits'][stage];manifest_sha=sha(manifest)
    for seed in range(5):
        path=OUT/'results'/model/'bbbc038_official'/f'seed{seed}'/'component_result.json'
        if not path.exists():continue
        raw=json.loads(path.read_text());assert raw['seed']==seed and raw['model']==model and raw['checkpoint']==spec['checkpoint']and raw['manifest_sha256']==manifest_sha
        prediction=Path(raw['prediction_path']);digest=sha(prediction);assert digest==raw['prediction_sha256'];records=[r for r in json.loads(prediction.read_text())if r['fold']==stage]
        records=[{**r,**{key:local_artifact(r[key])for key in ['truth_path','prediction_path']}}for r in records]
        missing=[r[key]for r in records for key in ['truth_path','prediction_path']if not Path(r[key]).exists()]
        if missing:
            PENDING.append(dict(model=model,stage=stage,seed=seed,missing_artifacts=len(missing),example=missing[0],reason='Resident predictions complete; artifact mirror not yet complete'))
            continue
        assert [r['sample_id']for r in records]==[r['sample_id']for r in expected]
        assert [r['group_id']for r in records]==[r['group_id']for r in expected]
        signature=dict(component_sha256=sha(path),prediction_sha256=digest,manifest_sha256=manifest_sha,
            artifacts=[(str(p),p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns)for r in records for p in [Path(r['truth_path']),Path(r['prediction_path'])]])
        signature=json.loads(json.dumps(signature));cache=CACHE/f'{model}__{stage}__seed{seed}.json'
        previous=json.loads(cache.read_text())if cache.exists()else{}
        if previous.get('signature')==signature:values=np.asarray(previous['values'])
        else:
            values=[]
            for r,e in zip(records,expected):
                assert r['truth_sha256']==e['instances_sha256']and sha(r['truth_path'])==r['truth_sha256']and sha(r['prediction_path'])==r['prediction_sha256']
                value=dsb_mean_precision(np.load(r['prediction_path']),np.load(r['truth_path']));assert abs(value-r['metrics']['dsb_mean_precision'])<1e-12;values.append(value)
            values=np.asarray(values);common.dump(cache,dict(signature=signature,values=values.tolist()))
        assert abs(values.mean()-raw['stages'][stage]['metrics']['dsb_mean_precision'])<1e-12
        found.append((seed,values,digest,signature['component_sha256']))
    if len(found)!=5:return None,len(found)
    return dict(model=model,stage=stage,values=np.stack([v for _,v,_,_ in found]),prediction_sha256=[h for _,_,h,_ in found],component_sha256=[h for _,_,_,h in found],manifest_sha256=manifest_sha,ids=[r['sample_id']for r in expected],groups=[r['group_id']for r in expected]),5
def run():
    PENDING.clear()
    cfg=json.loads((OUT/'config.json').read_text());settings=cfg['statistics'];models={m['model']:m for m in cfg['models']};state=dict(pid=os.getpid(),updated=time.time(),errors=[],indexed_seed_counts={},ci_completed=0,pairs_completed=0)
    for stage in ['stage1','stage2']:
        frames={}
        for model,spec in models.items():
            try:
                data,count=load(spec,stage);state['indexed_seed_counts'][model+'::'+stage]=count
                if data is not None:frames[model]=data
            except Exception as e:state['errors'].append(dict(model=model,stage=stage,error=repr(e)))
        if not frames:continue
        reference=next(iter(frames.values()));n=len(reference['ids']);assert n==len(set(reference['groups']))
        rng=np.random.default_rng(settings['random_seed']);weights=rng.multinomial(n,np.full(n,1/n),size=settings['bootstrap_replicates'])/n
        signs=2*np.random.default_rng(settings['random_seed']).integers(0,2,size=(settings['permutation_replicates'],n))-1
        for model,a in frames.items():
            assert a['ids']==reference['ids']and a['groups']==reference['groups']and a['manifest_sha256']==reference['manifest_sha256']
            a['mean_image_score']=a['values'].mean(0);mean=float(a['mean_image_score'].mean());ci=np.quantile(weights@a['mean_image_score'],[.025,.975]);seed_scores=a['values'].mean(1)
            prov=dict(implementation='bbbc038-official-paired-image-fivefits-v1',prediction_sha256=a['prediction_sha256'],component_sha256=a['component_sha256'],manifest_sha256=a['manifest_sha256'],settings=settings,stage=stage)
            target=STORE/f'{model}__bbbc038-official__{stage}__ci.json'
            if not common.valid_existing(target,prov):common.save_stat(target,dict(model=model,dataset='bbbc038-official',task='instance_segmentation',fold=stage,n_test=n,n_groups=n,training_seeds=list(range(5)),seed_values=seed_scores.tolist(),seed_sd=float(seed_scores.std(ddof=1)),prediction_sha256=a['prediction_sha256'],provenance=prov,primary_grouping_verified=False,uncertainty='Released image-ID paired bootstrap retaining all five fits; parent biological/source-image mapping unavailable, conditional image inference only',metrics={'dsb_mean_precision':dict(value=mean,ci_low=float(ci[0]),ci_high=float(ci[1]))}))
            state['ci_completed']+=1
        for model,a in frames.items():
            if models[model]['kind']!='biodino':continue
            for fm,b in frames.items():
                if models[fm]['kind']!='fm':continue
                delta=a['mean_image_score']-b['mean_image_score'];point=float(delta.mean());ci=np.quantile(weights@delta,[.025,.975]);null=(signs@delta)/n;p=(1+int((np.abs(null)>=abs(point)-1e-12).sum()))/(len(null)+1)
                prov=dict(implementation='bbbc038-official-paired-image-fivefits-v1',prediction_sha256=[a['prediction_sha256'],b['prediction_sha256']],component_sha256=[a['component_sha256'],b['component_sha256']],manifest_sha256=a['manifest_sha256'],settings=settings,stage=stage);target=STORE/f'{model}__vs__{fm}__bbbc038-official__{stage}.json'
                if not common.valid_existing(target,prov):common.save_stat(target,dict(candidate=model,reference=fm,dataset='bbbc038-official',task='instance_segmentation',fold=stage,n_test=n,n_groups=n,training_seeds=list(range(5)),primary_grouping_verified=False,provenance=prov,bootstrap_replicates=settings['bootstrap_replicates'],permutation_replicates=settings['permutation_replicates'],test_definition='Two-sided paired released-image swaps of all five native predictions together; each image precision is recomputed, then average five individual fit metrics. Conditional image inference; parent source groups unavailable.',metrics={'dsb_mean_precision':dict(difference=point,ci_low=float(ci[0]),ci_high=float(ci[1]),p_two_sided=float(p))}))
                state['pairs_completed']+=1
    state.update(updated=time.time(),status='watching',pending_artifact_transfers=list(PENDING));common.dump(OUT/'states/bbbc038_official_statistics.json',state)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    with(OUT/'locks/bbbc038_official_statistics.lock').open('a')as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(0)
        while True:
            try:run()
            except Exception:traceback.print_exc()
            if not args.watch:break
            time.sleep(60)
