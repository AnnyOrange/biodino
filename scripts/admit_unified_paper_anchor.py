"""Append a verified, previously selected checkpoint to existing shared queues."""
import argparse, fcntl, hashlib, json, subprocess, sys, time
from pathlib import Path
from admit_unified_paper_20tb_best import OUT, ROOT, REPORT, read, write, admit_segmentation

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);args=parser.parse_args()
    spec=read(args.spec);assert spec['kind']=='biodino';model=spec['model']
    for name in ['checkpoint','train_config']:assert Path(spec[name]).is_file(),spec[name]
    for name in ['checkpoint','train_config']:
        with Path(spec[name]).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if spec.get(name+'_sha256'):assert spec[name+'_sha256']==actual
        spec[name+'_sha256']=actual
    with (OUT/'locks/model_admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cfg=read(OUT/'config.json');base=next(m for m in cfg['models'] if m['model']=='biodino_20tb_l')
        previous=next((m for m in cfg['models'] if m['model']==model),None)
        if previous:
            assert (previous['checkpoint'],previous['step'])==(spec['checkpoint'],spec['step'])
        else:cfg['models'].append(spec)
        own=sum(m['kind']=='biodino' for m in cfg['models']);fm=sum(m['kind']=='fm' for m in cfg['models'])
        cfg['statistics']['planned_comparisons_per_family']=own*fm
        cfg['statistics']['primary_comparisons']='Each fixed configured BioDINO anchor versus each configured FM within matched task/dataset/budget/fold/metric; retrospective checkpoint selection provenance retained'
        write(OUT/'config.json',cfg)
        counts={}
        for name in ['queue','global_queue','detection_queue','native_detection_queue','nucls_queue','retrieval_queue']:
            path=OUT/(name+'.json');queue=read(path);known={j['key'] for j in queue['tasks']};added=[]
            for job in queue['tasks']:
                if job['model']!=base['model']:continue
                if name=='queue' and '--input-fix-v2' not in job['args']:continue
                raw=json.dumps(job).replace(base['checkpoint'],spec['checkpoint']).replace(base['train_config'],spec['train_config']).replace(base['model'],model)
                new=json.loads(raw)
                if new['key'] not in known:added.append(new);known.add(new['key'])
            queue['tasks'].extend(added)
            if 'models' in queue and model not in {m['model'] for m in queue['models']}:queue['models'].append(spec)
            write(path,queue);counts[name]=len(added)
        counts.update(admit_segmentation(spec))
        stat=read(OUT/'nucls_statistics_config.json')
        stat.update(primary_family=f'{own} fixed BioDINO anchors x{fm} FM; {own*fm} planned comparisons; four-fold equal-weight mean AP',
            secondary_families=f'Each secondary metric has a separate {own*fm}-comparison exploratory family',
            paired_test='Two-sided paired patient swap across all five fitted seeds within each official evaluation fold; equal fold mean; Holm over all configured BioDINO-vs-FM pairs')
        write(OUT/'nucls_statistics_config.json',stat)
        write(OUT/'corrections'/f'{model}_admission.json',{'spec':spec,'source_spec':str(args.spec),'time':time.time(),'new_jobs':counts,'models':len(cfg['models']),'family_size':own*fm})
    subprocess.run([sys.executable,str(ROOT/'scripts/admit_unified_biodino_spatial_gaps.py'),'--model',model],check=True)
    print('ADMITTED',model,spec['step'],counts,flush=True)

if __name__=='__main__':main()
