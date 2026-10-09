#!/usr/bin/env python3
"""Native-host 5TB RxRx3 / OOD queue. Never transfers teacher weights."""
import argparse,datetime,hashlib,importlib.util,json,os,subprocess,time,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',choices=['hxw','lyx'],required=True);p.add_argument('--gpus',type=int,nargs='+',required=True);a=p.parse_args()
PREFIX=Path('/data' if a.site=='hxw' else '/data/xuzijing')
BASE=PREFIX/'v2_resident_v4_20261008';BASE.mkdir(exist_ok=True)
INPUT=PREFIX/'v2_native_v4_inputs_20261008'
E=PREFIX/'hs6_l5_v2_recovery_eval_20260930'
RESUME_RUNS={
 'lyx':('global_cls',Path('/data/xuzijing/biodino_v2_20260930/outputs/01_training_runs/hs6_l5_v2_recovery_fork29279_20260930/global_cls')),
 'hxw':('global_cls_w3',Path('/home/xzj/biodino_v2_20260930/outputs/01_training_runs/hs6_l5_v2_recovery_fork29279_20260930/global_cls_w3')),
}
SOURCE=Path('/data/hs6_l_5tb_nogram_eval_20260921/bin/v4_monuseg_source_snapshot_20260924') if a.site=='hxw' else INPUT/'source'
RX=Path('/data/hs6_5tb_v4_rxrx3_20260924') if a.site=='hxw' else INPUT/'rxrx3'
BENCH=Path('/data/benchmark') if a.site=='hxw' else PREFIX/'benchmark'
PY='/home/xzj/eval_envs/hs6_protocol_v2/bin/python' if a.site=='hxw' else '/data/xuzijing/eval_envs/hs6_protocol_v2/bin/python'
ARMS=['global_cls','global_cls_w3','global_cls_slow2','global_cls_slow','global_cls_slow2_w3','global_cls_slow2_w03','global_cls_w03','global','global_local','global_slow','global_w03','global_cls_early','gram_ext']
def load(name):
 s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
rx=load('run_hxw_v4_rxrx3_queue_20260924');ood=load('run_hxw_v4_ood_queue_20260924')
ood.BASE=BASE/'ood_inputs';ood.SOURCE=SOURCE;ood.BENCHMARK=BENCH;ood.PYTHON=PY
for name in ['claims','logs','failures','manifests']:(BASE/name).mkdir(exist_ok=True)
ready=set();active=[];last_status=0;last_preflight=0;hashes={}
def teacher_sha(path):
 st=path.stat();key=(str(path.resolve()),st.st_size,st.st_mtime_ns)
 if key not in hashes:hashes[key]=rx.sha(path)
 return hashes[key]

def assets():
 for arm in ARMS:
  root=E/arm
  candidates={int(f.parent.name.split('_')[-1]):f for f in (root/'eval').glob('training_*/teacher_checkpoint.pth')}
  candidates.update({int(f.parent.name):f for f in (root/'adapters').glob('*/checkpoint.pth') if f.parent.name.isdigit()})
  resume_arm,resume_run=RESUME_RUNS[a.site]
  if arm==resume_arm:
   candidates.update({int(f.parent.name.split('_')[-1]):f for f in (resume_run/'eval').glob('training_*/teacher_checkpoint.pth')})
  for ck,f in sorted(candidates.items()):
   ceiling=50263 if arm==resume_arm else 35135
   if not (29279<ck<=ceiling) or not f.is_file() or f.stat().st_size<10**9 or time.time()-f.stat().st_mtime<300:continue
   config=root/'source/config.yaml'
   if not config.is_file():config=Path('/data/hs6_l_5tb_nogram_eval_20260921/source/config.yaml')
   if not config.is_file():continue
   yield arm,ck,f,config

def preflight():
 if not (SOURCE/'dinov3/eval/bio_frozen_eval/encoder.py').is_file():
  print('SOURCE_WAIT',str(SOURCE),flush=True);return
 if 'rxrx3' not in ready:
  try:
   lock=json.loads((RX/'protocol_v4.json').read_text());cache=RX/'cache/rxrx3-core'
   assert rx.sha(cache/'split_manifest.jsonl')==lock['retrieval_splits']['rxrx3-core']['manifest_sha256']
   meta=json.loads((cache/'metadata.json').read_text());assert meta['protocol_id']==rx.SPLIT
   assert [sum(r['split']==v for r in meta['rows']) for v in ['query','gallery']]==[734,734]
   assert (cache/'images.npy').stat().st_size>10**8 and (cache/'labels.npy').is_file()
   alias=cache.parent/'rxrx3'
   if not alias.exists():alias.symlink_to(cache.name)
   assert alias.resolve()==cache.resolve()
   ready.add('rxrx3')
  except (OSError,ValueError,KeyError,AssertionError) as exc:print('RX_PREFLIGHT_WAIT',repr(exc),flush=True)
 if 'xray' not in ready:
  try:ood.preflight();ready.add('xray')
  except Exception as exc:print('XRAY_PREFLIGHT_WAIT',repr(exc),flush=True)
 if 'cryo' not in ready:
  try:
   for project in ['10535','11043','11387','11388']:ood.cryo_project_preflight(project)
   ready.add('cryo')
  except Exception as exc:print('CRYO_PREFLIGHT_WAIT',repr(exc),flush=True)

def valid(task,arm,ck,out):
 if task=='rxrx3':return rx.valid(out/'models'/f'{arm}_{ck}'/'results.json',arm,ck)
 f=out/f'{arm}_{ck}_{task}'/str(ck)/'last_result.json'
 return f.is_file() and bool(json.loads(f.read_text()))

def launch(gpu,asset,task):
 arm,ck,checkpoint,config=asset;key=f'{arm}__{ck}__{task}';claim=BASE/'claims'/key
 try:claim.mkdir()
 except FileExistsError:return False
 out=BASE/'results'/task/arm/f'point_{ck}';out.mkdir(parents=True,exist_ok=True)
 if task=='rxrx3':
  cmd=[PY,'-u',str(RX/'scripts/run_external4_fixedbudget_model.py'),'--model',f'{arm}_{ck}','--campaign',str(out),'--cache-root',str(RX/'cache'),'--datasets','rxrx3','--checkpoint',str(checkpoint),'--train-config',str(config),'--device','cuda:0','--batch-size','64','--logical-batch-size','64']
 else:
  adapter=BASE/'adapters'/arm/str(ck);adapter.mkdir(parents=True,exist_ok=True);link=adapter/'checkpoint.pth'
  if not link.exists():link.symlink_to(checkpoint.resolve())
  assert link.resolve()==checkpoint.resolve()
  cmd=[PY,'-u','-m','dinov3.eval.eval_ood.dinov3_runner','--model-name',f'{arm}_{ck}_{task}','--ckpt-root',str(adapter.parent),'--ckpt-iter',str(ck),'--train-config',str(config),'--output-dir',str(out),'--benchmark-root',str(BENCH),'--ood-root',str(BENCH/'ood'),'--tasks',task,'--device','cuda:0','--batch-size','64','--num-workers','2','--n-last-blocks','1','--autocast-dtype','bf16','--resize-size','256','--crop-size','224','--xray-input-mode','three_slices','--xray-slices-per-volume','8','--id-max-samples','3000','--id-datasets','bloodmnist','bbbc048','cyclops','--seed','0','--phase','all']
  if task=='cryo':cmd+=['--cryo-max-projects','4','--cryo-max-particles-per-project','20000']
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(SOURCE),DINOV3_ROOT=str(SOURCE),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 lib='/home/xzj/miniconda3/envs/dinov3/lib/python3.11/site-packages/nvidia' if a.site=='hxw' else '/home/server/miniconda3/envs/dinov3/lib/python3.11/site-packages/nvidia'
 env['LD_LIBRARY_PATH']=f'{lib}/cuda_runtime/lib:{lib}/cuda_cupti/lib:'+env.get('LD_LIBRARY_PATH','')
 manifest=dict(site=a.site,arm=arm,checkpoint_id=ck,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=teacher_sha(checkpoint),config=str(config),config_sha256=rx.sha(config),task=task,command=cmd,gpu=gpu,source=str(SOURCE),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),weights_transferred=False)
 (BASE/'manifests'/f'{key}.json').write_text(json.dumps(manifest,indent=2))
 log=(BASE/'logs'/f'{key}.log').open('a');proc=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT);log.close()
 active.append((proc,gpu,key,task,arm,ck,out,claim));print('START',key,'GPU',gpu,'PID',proc.pid,flush=True);return True

# Single scheduler per site; prevents duplicate resource reservations.
import fcntl
lock=(BASE/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
 for item in list(active):
  proc,gpu,key,task,arm,ck,out,claim=item
  if proc.poll() is None:continue
  okay=proc.returncode==0 and valid(task,arm,ck,out)
  (BASE/('manifests' if okay else 'failures')/(key+'.completion.json')).write_text(json.dumps(dict(returncode=proc.returncode,valid=okay,time=time.time())))
  claim.rmdir();active.remove(item);print('FINISH' if okay else 'FAILED',key,proc.returncode,flush=True)
 if time.time()-last_preflight>300:preflight();last_preflight=time.time()
 available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))/1024**2
 cards={int(v[0]):(int(v[1]),int(v[2])) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).splitlines() if (v:=line.replace(' ','').split(','))}
 pending=[]
 for asset in assets():
  arm,ck,_,_=asset
  for task in ['rxrx3','xray','cryo']:
   key=f'{arm}__{ck}__{task}';out=BASE/'results'/task/arm/f'point_{ck}'
   if task in ready and not valid(task,arm,ck,out) and not (BASE/'claims'/key).exists() and not (BASE/'failures'/(key+'.completion.json')).exists():pending.append((asset,task))
 status=dict(time=time.time(),site=a.site,ready=sorted(ready),pending=len(pending),running=[x[2] for x in active],gpus=cards,mem_available_gib=available,weights_transferred=False,ctc='BLOCKED: no resident CTC dataset',target_memory_fraction=.75,max_jobs_per_gpu=5)
 (BASE/'STATUS.json').write_text(json.dumps(status,indent=2))
 if pending and available>(180 if a.site=='lyx' else 80) and shutil.disk_usage(PREFIX).free>60*2**30:
  for gpu in a.gpus:
   used,total=cards[gpu]
   if used/total>=.75 or total-used<10000 or sum(x[1]==gpu for x in active)>=5:continue
   if launch(gpu,*pending[0]):break
 time.sleep(15)
