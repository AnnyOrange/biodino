"""Transfer complete native inputs, verify canonical bytes, admit resident fits."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from manage_unified_ai_branches import ROOT,OUT,HOST,REMOTE_ROOT,REMOTE_OUT,MODELS,PY
BENCH=Path('/mnt/huawei_deepcad/benchmark')

def write(path,obj):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def admit_prepared_dataset(ds):
    """Shared full-input verification and queue admission; cached SHA checks require exact file stats."""
    state=OUT/'states'/f'ai_native_{ds}_preparation.json'
    if state.exists() and json.loads(state.read_text()).get('status')=='resident_queue_admitted':return
    if ds=='rsna_pneumonia':
        manifest=ROOT/'outputs/02_eval_inputs/formal_v4/rsna_pneumonia_manifest.json';config=manifest
    else:
        config=OUT/('nucls_native_config.json' if ds=='nucls_qc' else 'native_detection_config.json');cfg=json.loads(config.read_text())
        manifest=Path(cfg['manifest'] if ds=='nucls_qc' else cfg['manifests'][ds]['path'])
    proof=OUT/'validation'/f'ai_native_{ds}_canonical.json';canonical=json.loads(proof.read_text())
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==canonical['manifest_sha256']
    assert hashlib.sha256(config.read_bytes()).hexdigest()==canonical['config_sha256']
    subprocess.run(['rsync','-azR',*[str(p.relative_to(ROOT)) for p in [manifest,config,proof]],HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
    queue=OUT/('rsna_queue.json' if ds=='rsna_pneumonia' else 'nucls_queue.json' if ds=='nucls_qc' else 'native_detection_queue.json')
    job_dataset='xray_chest' if ds=='rsna_pneumonia' else ds
    template=[j for j in json.loads(queue.read_text())['tasks'] if j['model']=='biodino_1tb_l' and j['dataset']==job_dataset]
    assert len(template)==(20 if ds=='nucls_qc' else 5),(ds,len(template))
    jobs=[]
    for model in MODELS:
        for job in template:
            j=json.loads(json.dumps(job).replace('biodino_1tb_l',model).replace(str(ROOT),REMOTE_ROOT));j.pop('blocked_reason',None);j.pop('allowed_hosts',None)
            j['min_free_mib']=12000 if ds=='nucls_qc' else 10000;j['expected_peak_mib']=11500 if ds=='nucls_qc' else 5500
            j.setdefault('requires',[]).append(REMOTE_OUT+'/validation/'+f'ai_native_{ds}_resident.json');jobs.append(j)
    code='''import fcntl,pathlib,json,hashlib,time,os
root=pathlib.Path(__ROOT__);out=root/'outputs/02_eval_runs/unified_paper';ds=__DS__;proof=json.loads((out/'validation'/f'ai_native_{ds}_canonical.json').read_text());manifest=root/__MANIFEST__;config=root/__CONFIG__
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==proof['manifest_sha256']
assert hashlib.sha256(config.read_bytes()).hexdigest()==proof['config_sha256']
data=json.loads(manifest.read_text());assert len(data['records'])==proof['n_records'];fingerprints={};stats={}
proof_sha=hashlib.sha256((out/'validation'/f'ai_native_{ds}_canonical.json').read_bytes()).hexdigest();certified={};certificates=[]
for certificate in (out/'validation/native_transfer_chunks'/ds).glob('*.json'):
 record=json.loads(certificate.read_text())
 if record.get('canonical_proof_sha256')==proof_sha and record.get('status')=='PASS':certified.update(record['files']);certificates.append(str(certificate))
def signature(path):
 s=path.stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino)
for name,expected in proof['canonical_files'].items():
 path=pathlib.Path('/data/xuzijing/benchmark')/name;before=signature(path);cached=certified.get(name,{})
 if cached.get('sha256')!=expected or cached.get('stat')!=before:
  with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected,name
  assert signature(path)==before,'Input changed during verification: '+name
 fingerprints[str(path)]=expected;stats[str(path)]=before
verified=out/'validation'/f'ai_native_{ds}_resident.json';tmp=verified.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(dict(dataset=ds,n_records=proof['n_records'],manifest_sha256=proof['manifest_sha256'],source_fingerprints=fingerprints,source_stats=stats,verified_batch_certificates=certificates,canonical_proof_sha256=hashlib.sha256((out/'validation'/f'ai_native_{ds}_canonical.json').read_bytes()).hexdigest()),indent=2));tmp.replace(verified)
with (out/'locks/model_admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);path=out/__QUEUE__;q=json.loads(path.read_text()) if path.exists() else {'tasks':[]};known={j['key'] for j in q['tasks']};q['tasks'].extend(j for j in __JOBS__ if j['key'] not in known);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\\n');tmp.replace(path)
 print('RESIDENT NATIVE VERIFIED AND QUEUED',ds,len(fingerprints),len(q['tasks']))
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__DS__',repr(ds)).replace('__MANIFEST__',repr(str(manifest.relative_to(ROOT)))).replace('__CONFIG__',repr(str(config.relative_to(ROOT)))).replace('__QUEUE__',repr(queue.name)).replace('__JOBS__',repr(jobs))
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,PY+' -u -'],input=code,text=True,check=True)
    write(state,dict(pid=os.getpid(),updated=time.time(),status='resident_queue_admitted',files=len(canonical['canonical_files']),jobs=len(jobs),queue=REMOTE_OUT+'/'+queue.name,proof=str(proof)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['bccd','nucls_qc','bbbc041','rsna_pneumonia']);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    datasets=[args.dataset] if args.dataset else ['bccd','nucls_qc','bbbc041']
    with (OUT/'locks'/f'ai_native_prepare_{args.dataset or "all"}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=['scripts/run_unified_paper_native_detection.py','scripts/run_unified_paper_nucls.py','scripts/run_unified_paper_rsna.py','scripts/unified_paper_resident_native_inputs.py','dinov3/eval/bio_detection/','outputs/02_eval_runtime/pycocotools_2_0_10/']
        subprocess.run(['rsync','-azR',*files,HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
        for ds in datasets:
            state=OUT/'states'/f'ai_native_{ds}_preparation.json'
            if state.exists() and json.loads(state.read_text()).get('status')=='resident_queue_admitted':continue
            if ds=='rsna_pneumonia':
                manifest=ROOT/'outputs/02_eval_inputs/formal_v4/rsna_pneumonia_manifest.json';config=manifest
            else:
                config=OUT/('nucls_native_config.json' if ds=='nucls_qc' else 'native_detection_config.json');cfg=json.loads(config.read_text())
                manifest=Path(cfg['manifest'] if ds=='nucls_qc' else cfg['manifests'][ds]['path'])
            raw=manifest.read_bytes();data=json.loads(raw)
            paths=sorted({Path(r['path']) for r in data['records']});assert all(p.is_relative_to(BENCH) for p in paths)
            write(state,dict(pid=os.getpid(),updated=time.time(),status='verifying_complete_canonical_sources',files=len(paths),manifest=str(manifest)))
            fingerprints={}
            for path in paths:
                with path.open('rb') as stream:fingerprints[str(path.relative_to(BENCH))]=hashlib.file_digest(stream,'sha256').hexdigest()
            proof=OUT/'validation'/f'ai_native_{ds}_canonical.json';write(proof,dict(dataset=ds,manifest_sha256=hashlib.sha256(raw).hexdigest(),n_records=len(data['records']),canonical_files=fingerprints,config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),data_bytes=sum(p.stat().st_size for p in paths)))
            filelist=OUT/'validation'/f'ai_native_{ds}_files.txt';filelist.write_text(''.join(name+'\n' for name in fingerprints))
            state=OUT/'states'/f'ai_native_{ds}_preparation.json';write(state,dict(pid=os.getpid(),updated=time.time(),status='transferring_complete_source_images',files=len(paths),proof=str(proof)))
            subprocess.run(['rsync','-az' if ds=='rsna_pneumonia' else '-a','--relative','--partial','--partial-dir=.rsync-partial','--files-from',str(filelist),str(BENCH)+'/',HOST+':/data/xuzijing/benchmark/'],check=True)
            admit_prepared_dataset(ds)

if __name__=='__main__':
    if '--watch' not in sys.argv:main()
    else:
        while True:
            datasets=[sys.argv[sys.argv.index('--dataset')+1]] if '--dataset' in sys.argv else ['bccd','nucls_qc','bbbc041']
            states=[OUT/'states'/f'ai_native_{ds}_preparation.json' for ds in datasets]
            if not all(p.exists() and json.loads(p.read_text()).get('status')=='resident_queue_admitted' for p in states):
                try:main()
                except BlockingIOError:pass
                except Exception:traceback.print_exc()
            time.sleep(60)
