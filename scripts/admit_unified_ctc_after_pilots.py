"""Expand each validated encoder-family CTC pilot to fixed folds and fit seeds."""
import fcntl,time
from pathlib import Path
from admit_unified_paper_20tb_best import OUT,read,write

def run():
    pilots={'biodino':'biodino_1tb_splus','fm14':'dinov2','newfm':'distillpath'};ready={}
    for family,model in pilots.items():
        path=OUT/'results'/model/'ctc/fold0/seed0/component_result.json'
        if not path.exists():continue
        result=read(path)
        assert result['n_domains']==4 and all(r['ctc_metrics']['Valid']==1 for r in result['domain_rows'])
        ready[family]=str(path)
    with (OUT/'locks/model_admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);cfg=read(OUT/'config.json');path=OUT/'ctc_queue.json';queue=read(path);known={j['key'] for j in queue['tasks']};added=[]
        for spec in cfg['models']:
            family='biodino' if spec['kind']=='biodino' else 'newfm' if spec['model'] in ['distillpath','morphem','genbio_pathfm','openpath'] else 'fm14'
            if family not in ready or spec.get('execution_host'):continue
            for fold in range(5):
                for seed in range(5):
                    model=spec['model'];key=f'{model}__ctc__fold{fold}_seed{seed}'
                    if key in known:continue
                    queue['tasks'].append(dict(key=key,model=model,task='cell_tracking',dataset='ctc',fold=fold,seed=seed,entrypoint='run_unified_paper_ctc.py',args=['--model',model,'--fold',str(fold),'--seed',str(seed)],claim_key=key,min_free_mib=max(6000,spec['min_free_mib']),result=str(OUT/'results'/model/'ctc'/f'fold{fold}'/f'seed{seed}'/'component_result.json'),log=str(OUT/'logs'/f'{key}.log'),pilot_evidence=ready[family]));known.add(key);added.append(key)
        order={spec['model']:i for i,spec in enumerate(cfg['models'])}
        for job in queue['tasks']:
            if job.get('claim_key')==job['key']:continue
            claim=OUT/'locks'/f"{job.get('claim_key',job['model'])}.claimdir"
            owner=claim/'owner.json'
            if claim.exists() and (not owner.exists() or read(owner).get('task')==job['key']):continue
            job['claim_key']=job['key']
        queue['tasks'].sort(key=lambda j:(j.get('seed',0),j.get('fold',0),order.get(j['model'],999)))
        queue['stage']='Family-specific real pilots validated; five prescribed folds x five downstream seeds; no score-selected fit'
        queue['dispatch_order']='Across all fixed models for each fold/seed before the next repetition; running claims unchanged'
        write(path,queue)
        write(OUT/'states/ctc_expansion.json',dict(updated=time.time(),ready_families=ready,new_jobs=added,total_jobs=len(queue['tasks']),remaining='Other adapter families await their running pilots; AI-resident anchors await resident data adapter'))
    return len(added)

if __name__=='__main__':print('CTC JOBS ADDED',run())
