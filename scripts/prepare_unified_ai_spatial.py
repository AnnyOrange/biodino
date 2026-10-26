"""Admit resident frozen spatial evaluations using byte-verified source files."""
import argparse, csv, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from manage_unified_ai_branches import ROOT, OUT, HOST, REMOTE_ROOT, REMOTE_OUT, MODELS

def write(path, obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['cellpose','conic','monuseg','tissuenet','pannuke','livecell','multimodal_cellseg'],default='cellpose');parser.add_argument('--split');parser.add_argument('--remaining',action='store_true');args=parser.parse_args();ds=args.dataset
    if ds=='multimodal_cellseg' and args.split is None:args.split='neurips22-author-public52-v2'
    if args.remaining:
        with (OUT/'locks/ai_spatial_prepare_remaining.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            templates=[c for c in json.loads((OUT/'biodino_segmentation_inventory.json').read_text())['cells'] if c['model']=='biodino_1tb_l' and c['dataset'] in ['monuseg','tissuenet','pannuke','livecell']]
            results=[]
            for c in sorted(templates,key=lambda c:(['monuseg','tissuenet','pannuke','livecell'].index(c['dataset']),c['split'])):
                known={(x['model'],x['dataset'],x['split']) for x in json.loads((OUT/'biodino_segmentation_inventory.json').read_text())['cells']}
                if all((m,c['dataset'],c['split']) in known for m in MODELS):continue
                state=OUT/'states/ai_spatial_remaining_preparation.json';write(state,dict(pid=os.getpid(),updated=time.time(),current=dict(dataset=c['dataset'],split=c['split']),finished=results,status='preparing_and_verifying_resident_inputs'))
                r=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--dataset',c['dataset'],'--split',c['split']],cwd=ROOT)
                results.append(dict(dataset=c['dataset'],split=c['split'],exit_code=r.returncode))
                write(state,dict(pid=os.getpid(),updated=time.time(),finished=results,status='processing'))
            write(OUT/'states/ai_spatial_remaining_preparation.json',dict(pid=os.getpid(),updated=time.time(),finished=results,status='finished_admissions; inspect individual failures'))
        return
    inventory=json.loads((OUT/'biodino_segmentation_inventory.json').read_text())
    template=next(c for c in inventory['cells'] if c['model']=='biodino_1tb_l' and c['dataset']==ds and (args.split is None or c['split']==args.split))
    invocation=json.loads(Path(template['invocation']).read_text());cmd=invocation.get('command',[])
    input_spec=template.get('input_spec',{})
    aliases={'--data-root-base':'data_root_base','--feature-img-size':'image_size','--resize-mode':'resize_mode'}
    def arg(name):
        if name in cmd:return cmd[cmd.index(name)+1]
        if name=='--probe-class-weight-mode':return template['class_weight_mode']
        return input_spec[aliases[name]]
    validation=json.loads(Path(template['validation']).read_text());counts=validation.get('expected_counts',validation.get('counts'))
    sys.path.insert(0,str(ROOT))
    from dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline import _resolve_data_root
    source=_resolve_data_root(Path(arg('--data-root-base')),ds)
    files={}
    if ds=='cellpose':selected=sorted(source.rglob('*.png'))
    elif ds=='conic':selected=[source/name for name in ['images.npy','labels.npy','patch_info.csv']]
    elif ds=='tissuenet':selected=[source/f'tissuenet_v1.1_{part}.npz' for part in ['train','val','test']]
    elif ds=='pannuke':selected=sorted(source.rglob('*.npy'))
    elif ds=='livecell':selected=sorted(f for f in source.rglob('*') if f.suffix.lower() in ['.tif','.tiff','.png','.json'] and f.is_file())
    elif ds=='multimodal_cellseg':
        assert template['split']=='neurips22-author-public52-v2'
        selected=[]
        for part,path in validation['manifests'].items():
            manifest=Path(path);assert hashlib.sha256(manifest.read_bytes()).hexdigest()==validation['manifest_sha256'][part]
            with manifest.open() as stream:rows=list(csv.DictReader(stream))
            assert len(rows)==counts[part]
            selected.extend(source/r[k] for r in rows for k in ['image_path','mask_path'])
        selected=sorted(set(selected))
        resident=json.loads(json.dumps(validation).replace(str(ROOT),REMOTE_ROOT).replace('/mnt/huawei_deepcad/benchmark','/data/xuzijing/benchmark'))
        prepared=OUT/'validation/ai_multimodal_readiness.json';write(prepared,resident)
        subprocess.run(['rsync','-aR',*[str(Path(p).relative_to(ROOT)) for p in validation['manifests'].values()],HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True,timeout=180)
        subprocess.run(['rsync','-a',str(prepared),HOST+':'+REMOTE_OUT+'/multimodal_official_readiness.json'],check=True,timeout=180)
    else:
        manifest=source/'monuseg2018_train30_extra7val_test14_manifest.json';rows=json.loads(manifest.read_text())['samples']
        selected=[manifest,*sorted({source/r[k] for r in rows for k in ['image','xml']})]
        default=source/'monuseg2018_official_manifest.json'
        if default.exists():selected.append(default)
        # Only source manifests, never weights. Preserve their exact bytes.
        subprocess.run(['rsync','-a',*[str(f) for f in selected if f.suffix=='.json'],HOST+':'+str(source).replace('/mnt/huawei_deepcad/benchmark','/data/xuzijing/benchmark')+'/'],check=True,timeout=180)
    for f in selected:
        with f.open('rb') as stream:files[str(f.relative_to(source))]=hashlib.file_digest(stream,'sha256').hexdigest()
    assert files
    dataset_metadata=invocation.get('dataset',{})
    source_hashes=dataset_metadata.get('source_hashes',{}) if isinstance(dataset_metadata,dict) else {}
    proof=dict(dataset=ds,canonical_root=str(source),files=files,template_invocation=template['invocation'],template_invocation_sha256=hashlib.sha256(Path(template['invocation']).read_bytes()).hexdigest(),expected_counts=counts,template_source_hashes={k.replace('/mnt/huawei_deepcad/benchmark','/data/xuzijing/benchmark'):v for k,v in source_hashes.items()})
    proof_name=f'ai_{ds}_{template["split"]}_source_files.json';write(OUT/'validation'/proof_name,proof)
    subprocess.run(['rsync','-aR','Evaluation Rules/protocol_v3.json','scripts/run_unified_paper_biodino_segmentation_cache.py','scripts/run_unified_paper_segmentation.py','scripts/unified_paper_segmentation_artifacts.py','dinov3/eval/bio_frozen_eval/external_fm_protocol.py',f'outputs/02_eval_runs/unified_paper/validation/{proof_name}',HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True,timeout=180)
    subprocess.run(['rsync','-a','--include=*/','--include=*.py','--exclude=*',str(ROOT/'dinov3/eval/bio_segmentation')+'/',HOST+':'+REMOTE_ROOT+'/dinov3/eval/bio_segmentation/'],check=True,timeout=180)
    payload=dict(dataset=ds,split=template['split'],proof_name=proof_name,source=str(source).replace('/mnt/huawei_deepcad/benchmark','/data/xuzijing/benchmark'),models=MODELS,counts=counts,input_spec=dict(data_root_base='/data/xuzijing/benchmark/segmentation',image_size=int(arg('--feature-img-size')),resize_mode=arg('--resize-mode')),class_weight_mode=arg('--probe-class-weight-mode'))
    code='''import json,pathlib,hashlib,fcntl,time
root=pathlib.Path(__ROOT__);out=root/'outputs/02_eval_runs/unified_paper';params=__PARAMS__
ds=params['dataset'];split=params['split'];proof_path=out/'validation'/params['proof_name'];proof=json.loads(proof_path.read_text());source=pathlib.Path(params['source']);fingerprints={}
if ds=='cellpose':
 actual={str(f.relative_to(source)) for f in source.rglob('*.png')}
 assert actual==set(proof['files']),'Resident Cellpose file membership differs'
for name,digest in proof['files'].items():
 path=source/name
 sha=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):sha.update(block)
 assert sha.hexdigest()==digest,name
 fingerprints[str(path)]=digest
with (out/'locks/model_admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);ip=out/'biodino_segmentation_inventory.json';inv=json.loads(ip.read_text()) if ip.exists() else {'cells':[]};qp=out/'segmentation_queue.json';q=json.loads(qp.read_text()) if qp.exists() else {'tasks':[]};cfg=json.loads((out/'config.json').read_text());added=[]
 for model in params['models']:
  if any(c['model']==model and c['dataset']==ds and c['split']==split for c in inv['cells']):continue
  spec=next(m for m in cfg['models'] if m['model']==model);folder=out/'segmentation_cache_recovery'/model/ds/split;folder.mkdir(parents=True,exist_ok=True);recipe=folder/'invocation_manifest.json';recipe.write_text(json.dumps(dict(model=spec,expected_counts=params['counts'],template_invocation=str(proof_path),policy='Same canonical files, frozen final spatial layer, shared B32 E20/E50 five-seed head recipe'))+'\\n')
  index=len(inv['cells']);done=folder/'cache_complete.json';cell=dict(model=model,dataset=ds,split=split,cell=str(folder),invocation=str(recipe),validation=str(proof_path),existing=[],caches={part:str(folder/'cache'/f'{part}.npz') for part in ['train','val','test']},cache_ready=False,cache_completion=str(done),input_spec=params['input_spec'],class_weight_mode=params['class_weight_mode'],source_file_fingerprints=fingerprints,execution_host='5090-lyx-xr');inv['cells'].append(cell);added.append(cell)
  key=model+'__spatial_cache__'+ds+'__'+split;q['tasks'].append(dict(key=key,model=model,dataset=ds,task='segmentation_cache_recovery',entrypoint='run_unified_paper_biodino_segmentation_cache.py',args=['--cell-index',str(index)],claim_key=key,min_free_mib=10000,expected_peak_mib=5500,result=str(done),log=str(out/'logs'/f'{key}.log')))
  for budget in [20,50]:
   for seed in range(5):
    key=model+f'__segmentation__{ds}__{split}__E{budget}_seed{seed}';q['tasks'].append(dict(key=key,model=model,dataset=ds,task='segmentation',split=split,fold='fixed',budget=budget,seed=seed,entrypoint='run_unified_paper_segmentation.py',args=['--cell-index',str(index),'--budget',str(budget),'--seed',str(seed)],requires=[str(done)],claim_key=model+'__'+ds+'_head',min_free_mib=3500,expected_peak_mib=1500,result=str(out/'results'/model/'segmentation'/ds/split/f'E{budget}'/f'seed{seed}'/'component_result.json'),log=str(out/'logs'/f'{key}.log')))
 for path,obj in [(ip,inv),(qp,q)]:
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\\n');tmp.replace(path)
 print(json.dumps({'cells':[c for c in inv['cells'] if c['dataset']==ds],'verified_files':len(fingerprints),'added':len(added)}))
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__PARAMS__',repr(payload))
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,'python3 -u -'],input=code,capture_output=True,text=True,timeout=3600)
    if r.returncode:raise RuntimeError(r.stderr)
    result=json.loads(r.stdout)
    with (OUT/'locks/model_admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);path=OUT/'biodino_segmentation_inventory.json';inv=json.loads(path.read_text());known={(c['model'],c['dataset'],c['split']) for c in inv['cells']}
        for c in result['cells']:
            if (c['model'],c['dataset'],c['split']) not in known:inv['cells'].append(c)
        write(path,inv)
    write(OUT/'states'/f'ai_spatial_{ds}_preparation.json',dict(updated=time.time(),host=HOST,verified_files=result['verified_files'],resident_queue=REMOTE_OUT+'/segmentation_queue.json',models=MODELS,task=ds+': frozen cache plus 10 head fits per model',status='queue_prepared'))
    print(json.dumps(result)[:500],flush=True)

if __name__=='__main__':main()
