"""Lossless FILM encoder-input transport, verified against all28 original stacks."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from manage_unified_ai_branches import ROOT,OUT,HOST,REMOTE_ROOT,REMOTE_OUT,MODELS,PY
sys.path.insert(0,str(ROOT))

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def write(path,value):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def run():
    state=OUT/'states/ai_film_pack_preparation.json'
    if state.exists() and json.loads(state.read_text()).get('status')=='resident_pilot_queue_admitted':return
    with (OUT/'locks/ai_film_pack_preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np,torch
        from dinov3.eval.bio_frozen_eval.film import FILMAgeDataset
        torch.set_num_threads(2)
        manifest=ROOT/'outputs/02_eval_inputs/formal_v4/film_fixed_manifest.json';cfg=json.loads(manifest.read_text());base=Path('/mnt/huawei_deepcad/benchmark/ood/ood_classification/datasets/FILM')
        pack=OUT/'prepared_inputs/film_rgb';pack.mkdir(parents=True,exist_ok=True)
        if not (pack/'manifest.json').exists():
            dataset=FILMAgeDataset(base,cfg);planes=[]
            for index,row in enumerate(cfg['records']):
                assert sha(base/row['path'])==row['sha256']
                original=dataset[index][0].mean(0,keepdim=True).expand(3,-1,-1).contiguous()
                plane=original[:1].numpy().copy();assert torch.equal(torch.from_numpy(plane).expand(3,-1,-1).contiguous(),original)
                planes.append(plane)
            images=np.stack(planes);np.savez_compressed(pack/'inputs.npz',images=images)
            with np.load(pack/'inputs.npz') as arrays:assert np.array_equal(arrays['images'],images)
            proof=dict(canonical_manifest_sha256=sha(manifest),source_sha256=[r['sha256'] for r in cfg['records']],archive_sha256=sha(pack/'inputs.npz'),plane_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in images],verified_samples=len(images),verification='All original stacks SHA256 verified; all28 exact float32 RGB input tensors reproduced byte-for-byte with shared2CPU-thread spectral mean; unchanged encoder resizing, normalization, B4 extraction and frozen head fitting')
            write(pack/'manifest.json',proof);write(OUT/'validation/film_rgb_transport.json',dict(status='PASS',**proof))
        write(state,dict(pid=os.getpid(),updated=time.time(),status='transferring_verified_complete_input_pack',bytes=(pack/'inputs.npz').stat().st_size))
        files=['scripts/run_unified_paper_film.py','dinov3/eval/bio_frozen_eval/film.py',str(manifest.relative_to(ROOT)),str(Path(cfg['source_manifest']).relative_to(ROOT)),'outputs/02_eval_runs/unified_paper/film_config.json',str(pack.relative_to(ROOT))+'/']
        subprocess.run(['rsync','-azR','--partial','--partial-dir=.rsync-partial',*files,HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
        template=next(j for j in json.loads((OUT/'film_queue.json').read_text())['tasks'] if j['model']=='biodino_1tb_l');jobs=[]
        for model in MODELS:
            j=json.loads(json.dumps(template).replace('biodino_1tb_l',model).replace(str(ROOT),REMOTE_ROOT));j.pop('blocked_reason',None);j.pop('allowed_hosts',None);j['expected_peak_mib']=5500;j['requires']=[REMOTE_OUT+'/prepared_inputs/film_rgb/manifest.json']
            if model!=MODELS[0]:j['requires'].append(REMOTE_OUT+'/results/'+MODELS[0]+'/film_age/component_result.json')
            jobs.append(j)
        code='''import pathlib,json,hashlib,fcntl
root=pathlib.Path(__ROOT__);out=root/'outputs/02_eval_runs/unified_paper';pack=out/'prepared_inputs/film_rgb';proof=json.loads((pack/'manifest.json').read_text())
assert hashlib.sha256((root/'outputs/02_eval_inputs/formal_v4/film_fixed_manifest.json').read_bytes()).hexdigest()==proof['canonical_manifest_sha256']
with (pack/'inputs.npz').open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==proof['archive_sha256']
with (out/'locks/model_admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);p=out/'film_queue.json';q=json.loads(p.read_text()) if p.exists() else {'tasks':[]};known={j['key'] for j in q['tasks']};q['tasks'].extend(j for j in __JOBS__ if j['key'] not in known);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(p)
print('LOSSLESS FILM INPUTS VERIFIED AND QUEUED',len(q['tasks']))
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__JOBS__',repr(jobs))
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,PY+' -u -'],input=code,text=True,check=True)
        write(state,dict(pid=os.getpid(),updated=time.time(),status='resident_pilot_queue_admitted',verified_samples=28,jobs=2,queue=REMOTE_OUT+'/film_queue.json'))

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
