"""Verify resident data and publish the remaining common deterministic probes."""
import argparse,fcntl,hashlib,json,os,shlex,subprocess,sys,time,traceback
from pathlib import Path
from manage_unified_ai_branches import ROOT,OUT,HOST,REMOTE_ROOT,REMOTE_OUT,MODELS,PY
LOCAL_PY='/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'
def write(path,value):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def wait_existing_transfer(proof):
    needle=str(proof.relative_to(ROOT)).encode()
    while True:
        active=[]
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:args=path.read_bytes().split(b'\0')
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            if args and Path(args[0].decode()).name=='rsync' and needle in args and any(HOST.encode() in arg for arg in args):active.append(int(path.parent.name))
        if not active:return
        print('WAIT LIVE PROOF TRANSFER',proof.name,active,flush=True);time.sleep(15)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset');parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    with (OUT/'locks'/('ai_global_preparation'+('_'+args.dataset if args.dataset else '')+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=['scripts/verify_unified_global_inputs.py','scripts/run_unified_paper_global.py','scripts/run_shared_frozen_stage1_20260917.py','dinov3/eval/bio_frozen_eval/datasets.py','dinov3/eval/bio_frozen_eval/registry.py','dinov3/eval/bio_frozen_eval/group_keys.py','dinov3/eval/bio_frozen_eval/make_group_splits.py','dinov3/eval/bio_frozen_eval/splits/']
        subprocess.run(['rsync','-aR',*files,HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True,timeout=180)
        jobs=json.loads((OUT/'global_queue.json').read_text())['tasks'];datasets=sorted({j['dataset'] for j in jobs},key=lambda d:({'bbbc013':0,'conic-cell-count':1,'bbbc005':2,'cyclops-protein-loc':3}.get(d,4),d));finished=[]
        for ds in datasets:
            if args.dataset and ds!=args.dataset:continue
            remote_queue=json.loads((OUT/'remote_queue.json').read_text())['tasks'];known={(j['model'],j['dataset'],j['task']) for j in remote_queue}
            task=next(j['task'] for j in jobs if j['dataset']==ds)
            if all((m,ds,task) in known for m in MODELS):continue
            dataset_lock=(OUT/'locks'/f'ai_global_input_admission_{ds}.lock').open('a')
            try:fcntl.flock(dataset_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:dataset_lock.close();continue
            state=OUT/'states'/('ai_global_preparation'+('_'+args.dataset if args.dataset else '')+'.json');write(state,dict(pid=os.getpid(),updated=time.time(),dataset=ds,status='verifying_inputs',finished=finished))
            try:
                proof=OUT/'validation'/f'ai_global_{ds}_canonical.json'
                wait_existing_transfer(proof)
                subprocess.run([LOCAL_PY,'-u',str(ROOT/'scripts/verify_unified_global_inputs.py'),'--dataset',ds,'--output',str(proof)],cwd=ROOT,check=True)
                subprocess.run(['rsync','-azR','--partial','--partial-dir=.rsync-partial',str(proof.relative_to(ROOT)),HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
                remote_proof=REMOTE_OUT+'/validation/'+proof.name;verified=REMOTE_OUT+'/validation/'+f'ai_global_{ds}_resident.json'
                command=shlex.join([PY,'-u',REMOTE_ROOT+'/scripts/verify_unified_global_inputs.py','--dataset',ds,'--benchmark-root','/data/xuzijing/benchmark','--expected',remote_proof,'--output',verified])
                r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,command],capture_output=True,text=True,timeout=3600)
                if r.returncode:raise RuntimeError(r.stderr[-2500:])
                print(r.stdout,flush=True)
                code='''import pathlib,json,fcntl,hashlib
out=pathlib.Path(__OUT__);ds=__DS__;task=__TASK__;proof=pathlib.Path(__PROOF__)
with (out/'locks/model_admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);cfg_path=out/'config.json';cfg=json.loads(cfg_path.read_text());cfg.setdefault('global_input_proofs',{})[ds]={'path':str(proof),'sha256':hashlib.sha256(proof.read_bytes()).hexdigest()};qp=out/'global_queue.json';q=json.loads(qp.read_text()) if qp.exists() else {'tasks':[]};known={j['key'] for j in q['tasks']}
 for model in __MODELS__:
  key=model+'__global__'+ds
  if key in known:continue
  q['tasks'].append(dict(key=key,model=model,dataset=ds,task=task,entrypoint='run_unified_paper_global.py',args=['--model',model,'--dataset',ds],claim_key=key,min_free_mib=10000,expected_peak_mib=5500,result=str(out/'results'/model/'global'/ds/'component_result.json'),log=str(out/'logs'/f'{key}.log'),requires=[str(proof)]))
 for path,obj in [(cfg_path,cfg),(qp,q)]:
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\\n');tmp.replace(path)
 print('RESIDENT GLOBAL QUEUED',ds,len(q['tasks']))
'''.replace('__OUT__',repr(REMOTE_OUT)).replace('__DS__',repr(ds)).replace('__TASK__',repr(task)).replace('__PROOF__',repr(verified)).replace('__MODELS__',repr(MODELS))
                subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,'python3 -u -'],input=code,text=True,check=True,timeout=60)
                finished.append(dict(dataset=ds,status='resident_queue_admitted',proof=str(proof),remote_proof=verified))
            except Exception as e:
                finished.append(dict(dataset=ds,status='input_verification_failed',error=repr(e)));print('DATASET BLOCKED',ds,repr(e),flush=True)
            write(state,dict(pid=os.getpid(),updated=time.time(),dataset=ds,status='processing',finished=finished))
            dataset_lock.close()
        write(OUT/'states'/('ai_global_preparation'+('_'+args.dataset if args.dataset else '')+'.json'),dict(pid=os.getpid(),updated=time.time(),status='finished_admissions',finished=finished))
if __name__=='__main__':
    if '--watch' not in sys.argv:main()
    else:
        while True:
            try:main()
            except BlockingIOError:pass
            except Exception:traceback.print_exc()
            time.sleep(180)
