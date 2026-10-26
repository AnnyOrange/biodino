"""Pilot-gated completion of the two existing within-set v4 cohorts."""
import argparse,fcntl,hashlib,json,os,time
from pathlib import Path
from unified_paper_retrieval_scheduling import requirements
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def dump(path,value):
    tmp=path.with_suffix('.additional.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def job(spec,dataset):
    m=spec['model'];key=f'{m}__retrieval_clustering__{dataset}'
    result=dict(key=key,model=m,task='retrieval_clustering',dataset=dataset,fold='fixed',seed='bundled0to4',entrypoint='run_unified_paper_retrieval.py',args=['--model',m,'--dataset',dataset],claim_key='retrieval_'+m,min_free_mib=16000 if spec['min_free_mib']>=16000 else 6000,result=str(OUT/'results'/m/'retrieval_clustering'/dataset/'component_result.json'),log=str(OUT/'logs'/f'{key}.log'))
    result.update(requirements(spec,dataset))
    if spec.get('execution_host'):result.update(blocked_reason='Complete cohort and shared adapter require resident-host preparation; no cross-building weights transfer',allowed_hosts=[spec['execution_host']])
    return result
def run():
    with (OUT/'locks/model_admission.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);cfg=json.loads((OUT/'config.json').read_text());models=cfg['models'];pilot=next(m for m in models if m['model']=='biodino_1tb_splus')
        path=OUT/'retrieval_queue.json';queue=json.loads(path.read_text());jobs={j['key']:j for j in queue['tasks']};state=[];pilot_jobs=[]
        for dataset in ['nct-crc-he-100','lc25000']:
            manifest=ROOT/'outputs/02_eval_inputs/formal_v4'/f'{dataset}_retrieval_manifest.json'
            if not manifest.exists():state.append(dict(dataset=dataset,status='manifest_not_prepared'));continue
            j=job(pilot,dataset);p=Path(j['result']);passed=False
            if p.exists():
                raw=json.loads(p.read_text());assert raw['checkpoint']==pilot['checkpoint'] and raw['completed_clustering_seeds']==5
                assert raw['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
                assert raw['n_samples']=={'nct-crc-he-100':99,'lc25000':25000}[dataset]
                for artifact in [raw['retrieval_result'],*raw['clustering_results']]:
                    r=json.loads(Path(artifact).read_text());assert hashlib.sha256(Path(r['prediction_path']).read_bytes()).hexdigest()==r['prediction_sha256']
                passed=True
                for spec in models:
                    j=job(spec,dataset);jobs.setdefault(j['key'],j)
            if not passed:
                j=job(pilot,dataset)
                if dataset=='lc25000':
                    j['allowed_hosts']=['deepcad'];pilot_jobs.append(j);jobs.pop(j['key'],None)
                else:jobs.setdefault(j['key'],j)
            state.append(dict(dataset=dataset,status='all_models_admitted_after_real_pilot'if passed else'one_real_pilot_queued',pilot_result=str(p)))
        if list(jobs.values())!=queue['tasks']:queue['tasks']=list(jobs.values());dump(path,queue)
        pilot_path=OUT/'additional_retrieval_pilot_queue.json'
        previous=json.loads(pilot_path.read_text())if pilot_path.exists()else{}
        if previous.get('tasks')!=pilot_jobs:dump(pilot_path,dict(tasks=pilot_jobs,reason='One full LC25000 real pilot on a dedicated worker; do not let old general workers claim a new cohort onto a full GPU'))
        dump(OUT/'states/additional_retrieval_admission.json',dict(pid=os.getpid(),updated=time.time(),cohorts=state))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    with (OUT/'locks/additional_retrieval_admission.lock').open('a')as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(0)
        while True:
            run()
            if not args.watch:break
            time.sleep(45)
