"""Publish byte-identical FILM inputs and fixed five-head resident evaluations."""
import argparse,fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
from manage_unified_ai_branches import ROOT,OUT,HOST,REMOTE_ROOT,REMOTE_OUT,MODELS,PY

def write(path,value):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def run():
    state=OUT/'states/ai_film_preparation.json'
    if state.exists() and json.loads(state.read_text()).get('status')=='resident_queue_admitted':return
    with (OUT/'locks/ai_film_preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=ROOT/'outputs/02_eval_inputs/formal_v4/film_fixed_manifest.json'
        cfg=json.loads(manifest.read_text());base=Path('/mnt/huawei_deepcad/benchmark/ood/ood_classification/datasets/FILM')
        for row in cfg['records']:
            with (base/row['path']).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256']
        files=OUT/'validation/ai_film_files.txt';files.write_text(''.join(row['path']+'\n' for row in cfg['records']))
        write(state,dict(pid=os.getpid(),updated=time.time(),status='transferring_all28_verified_stacks',files=28,checkpoint_transfer=False))
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,'mkdir -p /data/xuzijing/benchmark/ood/ood_classification/datasets/FILM'],check=True)
        subprocess.run(['rsync','-a','--partial','--partial-dir=.rsync-partial','--files-from',str(files),str(base)+'/',HOST+':/data/xuzijing/benchmark/ood/ood_classification/datasets/FILM/'],check=True)
        inputs=['scripts/run_unified_paper_film.py','dinov3/eval/bio_frozen_eval/film.py',str(manifest.relative_to(ROOT)),str(Path(cfg['source_manifest']).relative_to(ROOT)),'outputs/02_eval_runs/unified_paper/film_config.json']
        subprocess.run(['rsync','-azR',*inputs,HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
        templates=json.loads((OUT/'film_queue.json').read_text())['tasks'];template=next(j for j in templates if j['model']=='biodino_1tb_l');jobs=[]
        for model in MODELS:
            job=json.loads(json.dumps(template).replace('biodino_1tb_l',model).replace(str(ROOT),REMOTE_ROOT));job.pop('blocked_reason',None);job.pop('allowed_hosts',None);job['expected_peak_mib']=5500;jobs.append(job)
        code='''import pathlib,json,hashlib,fcntl
root=pathlib.Path(__ROOT__);out=root/'outputs/02_eval_runs/unified_paper';manifest=root/'outputs/02_eval_inputs/formal_v4/film_fixed_manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==__SHA__
cfg=json.loads(manifest.read_text());base=pathlib.Path('/data/xuzijing/benchmark/ood/ood_classification/datasets/FILM')
for row in cfg['records']:
 with (base/row['path']).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256']
with (out/'locks/model_admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);path=out/'film_queue.json';q=json.loads(path.read_text()) if path.exists() else {'tasks':[]};known={j['key'] for j in q['tasks']};q['tasks'].extend(j for j in __JOBS__ if j['key'] not in known);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(path)
print('FILM28SOURCEFILESVERIFIED',len(q['tasks']))
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__SHA__',repr(hashlib.sha256(manifest.read_bytes()).hexdigest())).replace('__JOBS__',repr(jobs))
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,PY+' -u -'],input=code,text=True,check=True)
        write(state,dict(pid=os.getpid(),updated=time.time(),status='resident_queue_admitted',jobs=2,files=28,queue=REMOTE_OUT+'/film_queue.json',manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        try:run()
        except BlockingIOError:pass
        except Exception:
            traceback.print_exc()
            if not args.watch:raise
        if not args.watch:break
        time.sleep(60)
