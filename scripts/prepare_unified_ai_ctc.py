"""Stage the shared native CTC runner and persistently admit a resident pilot."""
import argparse,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from manage_unified_ai_branches import ROOT,OUT,HOST,REMOTE_ROOT,REMOTE_OUT,MODELS,PY

def write(path,value):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def admit_after_complete_sequence_pilots():
    """Independent native sequence pilots cover both 2D and 3D end to end.

    The long fourth sequence does not block the other frozen-head fits. A fold
    component remains incomplete until every prescribed domain is complete.
    """
    code='''import pathlib,json,hashlib,fcntl,os,time
root=pathlib.Path(__ROOT__);out=root/'outputs/02_eval_runs/unified_paper';pilot=out/'results'/__MODEL__/'ctc/fold0/seed0';proof=out/'validation/ai_ctc_sequence_pilots.json'
names=['BF-C2DL-HSC','Fluo-C3DH-A549-SIM','Fluo-N2DH-SIM+']
if not all((pilot/'domains'/name/'result.json').is_file()for name in names):print(json.dumps(dict(status='waiting_for_complete_native_sequences')));raise SystemExit(0)
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
data=json.loads((root/'outputs/02_eval_inputs/formal_v3/ctc_native_full/data_manifest.json').read_text());head=json.loads((pilot/'head_metadata.json').read_text());invocation=json.loads((pilot/'invocation_manifest.json').read_text());inputs=json.loads((out/'validation/ai_ctc_inputs_complete.json').read_text())
assert inputs['status']=='PASS'and inputs['domains']==20
assert head['status']=='VALID_EPOCH_50'and head['epochs']==50 and head['fold']==0
assert invocation['recipe']['backbone']=='frozen'and invocation['recipe']['batch_size']==8 and invocation['recipe']['epochs']==50
assert invocation['dataset']['split_sha256']=='7a0f5bc2579f6ae22a2b8f2b16103530cd678466a4b2bbd1f617692b8e3cac27'
assert head['campaign_manifest_sha256']==sha(pilot/'invocation_manifest.json')and head['checkpoint_sha256']==invocation['checkpoint']['sha256']
evidence=[];ndims=set()
for name in names:
 directory=pilot/'domains'/name;path=directory/'result.json';r=json.loads(path.read_text());frames=data['domains'][name]['test_frames'];pred=directory/'02_RES'
 assert r['status']=='VALID_COMPLETE'and r['ctc_metrics']['Valid']==1 and r['test_frames']==len(frames)
 assert r['head_sha256']==head['head_sha256']and r['campaign_manifest_sha256']==head['campaign_manifest_sha256']and r['fold']==0
 expected={'mask'+frame['token']+'.tif'for frame in frames};assert {p.name for p in pred.glob('mask*.tif')}==expected
 assert (pred/'res_track.txt').is_file();ndims.add(r['ndim'])
 evidence.append(dict(domain=name,ndim=r['ndim'],test_frames=len(frames),result=str(path),result_sha256=sha(path),native_metrics=r['ctc_metrics'],tracking_sha256=sha(pred/'res_track.txt')))
assert ndims=={2,3}
payload=dict(status='PASS',updated=time.time(),head_sha256=head['head_sha256'],training_epochs=50,training_batch_size=8,complete_sequences=evidence,scope='Operational admission after complete 2D and3D native sequence pilots; fold results still require all4domains; all5folds and5seeds remain mandatory')
tmp=proof.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(proof)
with(out/'locks/model_admission.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);path=out/'ctc_queue.json';q=json.loads(path.read_text());changed=False
 for job in q['tasks']:
  if job['model']not in __MODELS__:continue
  requires=[str(out/'validation/ai_ctc_inputs_complete.json')]
  if not(job['model']==__MODEL__and job['fold']==0 and job['seed']==0):requires.append(str(proof))
  if job.get('requires')!=requires:job['requires']=requires;changed=True
  # Both resident branches use the same L architecture and fixed B8 decoder.
  # Real50epoch+2D/3D pilot retained about3.6GiB NVML memory; reserve5GiB per
  # subsequent fit, plus the unchanged20percent card margin and observed-peak
  # adjustment in child admission. Existing reservations are not reduced.
  if job.get('expected_peak_mib')!=5000 or job.get('min_free_mib')!=7000:
   job.update(expected_peak_mib=5000,min_free_mib=7000);changed=True
 if changed:
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(path)
print(json.dumps(dict(status='resident_native_sequences_verified_all_fits_admitted',proof=str(proof),jobs=len(q['tasks']),pilot_fold_complete=(pilot/'component_result.json').exists())))
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__MODEL__',repr(MODELS[0])).replace('__MODELS__',repr(MODELS))
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,PY+' -u -'],input=code,text=True,capture_output=True,timeout=45)
    result.check_returncode();record=json.loads(result.stdout)
    write(OUT/'states/ai_ctc_sequence_pilot_admission.json',dict(pid=os.getpid(),updated=time.time(),**record))

def run():
    state=OUT/'states/ai_ctc_runner_preparation.json'
    if state.exists() and json.loads(state.read_text()).get('status')=='resident_pilot_queue_admitted':
        admit_after_complete_sequence_pilots();return
    with (OUT/'locks/ai_ctc_runner_preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=['scripts/run_unified_paper_ctc.py','scripts/unified_paper_ctc_runner.py','scripts/run_ctc_native_full_hs6.py','scripts/run_ctc_native_2d_observation.py','scripts/prepare_unified_ctc_resident_data.py','scripts/unified_paper_queue.py','scripts/unified_paper_gpu_admission.py','dinov3/eval/bio_tracking/','dinov3/eval/bio_segmentation/instance_seg/','outputs/02_eval_runtime/py-ctcmetrics/','outputs/02_eval_runtime/imagecodecs_py311_wheel_2026.3.6/','outputs/02_eval_runtime/imagecodecs_wheel/imagecodecs-2026.3.6-cp311-abi3-manylinux_2_28_x86_64.whl']
        write(state,dict(pid=os.getpid(),updated=time.time(),status='transferring_shared_runner_and_pinned_scorer',checkpoint_transfer=False))
        subprocess.run(['rsync','-azR','--partial','--partial-dir=.rsync-partial',*files,HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
        jobs=[];template=[j for j in json.loads((OUT/'ctc_queue.json').read_text())['tasks'] if j['model']=='biodino_1tb_l'];assert len(template)==25
        pilot=REMOTE_OUT+'/results/'+MODELS[0]+'/ctc/fold0/seed0/component_result.json'
        for model in MODELS:
            for job in template:
                j=json.loads(json.dumps(job).replace('biodino_1tb_l',model).replace(str(ROOT),REMOTE_ROOT));j.pop('blocked_reason',None);j.pop('allowed_hosts',None)
                j['expected_peak_mib']=9000;j['min_free_mib']=10000;j['requires']=[REMOTE_OUT+'/validation/ai_ctc_inputs_complete.json']
                if not (model==MODELS[0] and j['fold']==0 and j['seed']==0):j['requires'].append(pilot)
                jobs.append(j)
        code='''import pathlib,sys,json,fcntl
root=pathlib.Path(__ROOT__);sys.path[:0]=[str(root),str(root/'scripts')]
import unified_paper_ctc_runner as shared
shared.native.base.ctcmetrics_commit=shared.ctcmetrics_commit
shared.native_ctc_modules()
assert shared.ctcmetrics_commit()=='59481c48a62d4376fe34bed3e3606b4ec4d60972'
out=root/'outputs/02_eval_runs/unified_paper'
with (out/'locks/model_admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);p=out/'ctc_queue.json';q=json.loads(p.read_text()) if p.exists() else {'tasks':[]};known={j['key'] for j in q['tasks']};q['tasks'].extend(j for j in __JOBS__ if j['key'] not in known);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\\n');tmp.replace(p)
print('SHARED CTC IMPORT AND PINNED SCORER VERIFIED; PILOT-GATED QUEUE',len(q['tasks']))
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__JOBS__',repr(jobs))
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,PY+' -u -'],input=code,text=True,check=True)
        write(state,dict(pid=os.getpid(),updated=time.time(),status='resident_pilot_queue_admitted',jobs=50,pilot=pilot,data_dependency=REMOTE_OUT+'/validation/ai_ctc_inputs_complete.json',queue=REMOTE_OUT+'/ctc_queue.json'))

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
