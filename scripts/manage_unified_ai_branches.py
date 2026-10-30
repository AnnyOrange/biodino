"""Monitor AI-resident branch pilots and import outputs, never checkpoints."""
import argparse, fcntl, hashlib, json, os, shlex, subprocess, time, threading
from pathlib import Path
from admit_unified_paper_20tb_best import ROOT, OUT, read, write

HOST='5090-lyx-xr'
REMOTE_ROOT='/data/xuzijing/l20tb_selected29_dense_20261021/unified_paper'
REMOTE_OUT=REMOTE_ROOT+'/outputs/02_eval_runs/unified_paper'
PY='/data/xuzijing/eval_envs/hs6_protocol_v2/bin/python'
MODELS=['biodino_20tb_rebalanced_best','biodino_20tb_residual_best']
SYNC_CHILDREN={}
SYNC_CHILDREN_LOCK=threading.Lock()

def reap_sync_children():
    # Poll every owned transfer before network/filesystem work in the next pass.
    # Keep its Popen for ensure_sync to report the previous exit status.
    with SYNC_CHILDREN_LOCK:
        children=list(SYNC_CHILDREN.values())
    for child in children:
        child.poll()

def sync_reaper():
    while True:
        reap_sync_children()
        time.sleep(5)

def ensure_sync(kind,mirror):
    """Adopt live transfers across controller restarts; no elapsed-time cutoff."""
    source=HOST+':'+REMOTE_OUT+'/'+kind+'/'
    state_path=OUT/'states'/f'ai_sync_{kind}.json'
    active=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=path.read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if args and Path(args[0].decode()).name=='rsync' and source.encode() in args:active.append(int(path.parent.name))
    if active:
        write(state_path,dict(updated=time.time(),status='transfer_running',pid=min(active),observed_rsync_pids=active,source=source,timeout_policy='No wall-clock timeout; reattach to live transfer'))
        return
    previous=SYNC_CHILDREN.get(kind);exit_code=previous.poll() if previous else None
    target=mirror/kind;target.mkdir(parents=True,exist_ok=True)
    argv=['rsync','-az','--partial','--partial-dir=.rsync-partial','--exclude=*.tmp*']
    if kind=='results':argv+=['--exclude=*features*','--exclude=best_head.pth','--exclude=head_epoch*.pth','--exclude=training_state.pth']
    argv += [source,str(target)+'/']
    log=OUT/'logs'/f'ai_sync_{kind}.log'
    with log.open('a') as stream:proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    with SYNC_CHILDREN_LOCK:SYNC_CHILDREN[kind]=proc
    write(state_path,dict(updated=time.time(),status='transfer_started',pid=proc.pid,previous_exit_code=exit_code,source=source,log=str(log)))

def remote_snapshot():
    code='''import json,pathlib,subprocess,os,time
root=pathlib.Path(__REMOTE_ROOT__);out=root/'outputs/02_eval_runs/unified_paper';states=[]
oodq=out/'ood_repair_queue.json'
if oodq.exists():
 import fcntl
 with (out/'locks/model_admission.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);q=json.loads(oodq.read_text());changed=False
  for j in q['tasks']:
   desired=j['model']+'__resident_ood';owner=out/'locks'/f"{j.get('claim_key',j['model'])}.claimdir/owner.json";s=json.loads(owner.read_text()) if owner.exists() else {}
   if j.get('claim_key')!=desired and s.get('task')!=j['key']:j['claim_key']=desired;changed=True
  if changed:
   tmp=oodq.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\\n');tmp.replace(oodq)
env=os.environ.copy();env.update(UNIFIED_PAPER_PYTHON=__REMOTE_PY__,UNIFIED_PAPER_BENCHMARK_ROOT='/data/xuzijing/benchmark',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',UNIFIED_PAPER_OOD_SOURCE_RECORDS=str(root/'outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927/provenance/ood_selected_records.json'),UNIFIED_PAPER_OOD_PACKED_ID=str(out/'prepared_inputs/ood_fixed_id'),UNIFIED_PAPER_OOD_INPUT_FINGERPRINTS=str(out/'validation/ood_input_fingerprints'))
env['UNIFIED_PAPER_OOD_PACKED_CRYO']=str(out/'prepared_inputs/ood_fixed_cryo_spatial')
# Older persistent workers retain the legacy cryo environment path. Preserve the
# abandoned partial raw pack and route that path to the exact certified pack.
cryo_pack=out/'prepared_inputs/ood_fixed_cryo_spatial';cryo_legacy=out/'prepared_inputs/ood_fixed_cryo'
if (cryo_pack/'manifest.json').is_file() and (cryo_pack/'cryo_inputs.npz').is_file():
 archived=out/'prepared_inputs/ood_fixed_cryo_incomplete_raw_archive'
 if cryo_legacy.is_dir() and not cryo_legacy.is_symlink() and not (cryo_legacy/'manifest.json').exists() and not archived.exists():cryo_legacy.rename(archived)
 if not cryo_legacy.exists():cryo_legacy.symlink_to(cryo_pack.name,target_is_directory=True)
 if cryo_legacy.resolve()==cryo_pack.resolve():
  repaired=[]
  for model in __MODELS__:
   key=model+'__ood__cryo__groupfix_v3';attempt=out/'states'/f'{key}_attempts.json';log=out/'logs'/f'{key}.log'
   if not attempt.is_file() or not log.is_file():continue
   record=json.loads(attempt.read_text());tail=log.read_text()[-6000:]
   if record.get('attempts',0)>=2 and 'FileNotFoundError' in tail and 'ood_fixed_cryo/manifest.json' in tail and not record.get('legacy_cryo_path_repaired'):
    archive=out/'recovered_attempts';archive.mkdir(exist_ok=True);(archive/(attempt.stem+'.missing_legacy_pack_path.json')).write_bytes(attempt.read_bytes())
    tmp=attempt.with_suffix('.tmp');tmp.write_text(json.dumps(dict(attempts=0,legacy_cryo_path_repaired=True,repaired_at=time.time(),repair='Legacy directory alias now points to complete spatial pack; runner still validates full SHA before encoding'))+'\\n');tmp.replace(attempt);repaired.append(key)
  if repaired:(out/'validation/cryo_legacy_worker_path_repair.json').write_text(json.dumps(dict(updated=time.time(),status='path_repaired_runner_SHA_still_required',legacy_path=str(cryo_legacy),canonical_path=str(cryo_pack),preserved_partial=str(archived),retried=repaired),indent=2))
env['UNIFIED_PAPER_BONE_AGE_PACK']=str(out/'prepared_inputs/bone_age_spatial')
env['DINOV3_BBBC038_SPLIT_FILE']=str(out/'manifests/bbbc038_splits.npz')
ctc_data_script=root/'scripts/prepare_unified_ctc_resident_data.py'
if ctc_data_script.is_file() and (out/'validation/ctc_canonical_manifest.json').is_file():
 active=[]
 for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
  try:argv=path.read_bytes().split(b'\\0')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if str(ctc_data_script).encode() in argv and b'--watch' in argv:active.append(int(path.parent.name))
 if not active:
  with (out/'logs/ai_ctc_data_preparation.log').open('a') as stream:
   child=subprocess.Popen([__REMOTE_PY__,'-u',str(ctc_data_script),'--watch'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  print('',end='')
ctc_acceleration_script=root/'scripts/accelerate_unified_ctc_download.py'
if ctc_acceleration_script.is_file() and (out/'validation/ctc_canonical_manifest.json').is_file():
 active=[]
 for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
  try:argv=path.read_bytes().split(b'\\0')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if str(ctc_acceleration_script).encode() in argv and b'--watch' in argv:active.append(int(path.parent.name))
 if not active:
  with (out/'logs/ctc_download_acceleration.log').open('a') as stream:
   child=subprocess.Popen([__REMOTE_PY__,'-u',str(ctc_acceleration_script),'--watch'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
boneq=out/'bone_age_queue.json'
if boneq.exists():
 import fcntl
 with (out/'locks/model_admission.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);q=json.loads(boneq.read_text());changed=False
  for j in q['tasks']:
   if j['key'].endswith('__bone_age_official_validation'):
    claim=out/'locks'/f"{j.get('claim_key',j['model'])}.claimdir"
    if not claim.exists():j['key']=j['model']+'__bone_age__official_validation';j['claim_key']=j['key'];changed=True
  if changed:
   tmp=boneq.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\\n');tmp.replace(boneq)
# Once validation-only LR selection is frozen, independent seeds can share cards.
nativeq=out/'native_detection_queue.json'
if nativeq.exists():
 import fcntl
 with (out/'locks/model_admission.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);q=json.loads(nativeq.read_text());changed=False
  for j in q['tasks']:
   if j.get('dataset')!='bbbc041' or j.get('seed',0)==0:continue
   recipe=out/'results'/j['model']/'native_box/bbbc041/selected_recipe.json'
   if not recipe.exists():continue
   r=json.loads(recipe.read_text());assert r['test_used'] is False and r['selected']['status']=='complete'
   owner=out/'locks'/f"{j.get('claim_key',j['model'])}.claimdir/owner.json";s=json.loads(owner.read_text()) if owner.exists() else {}
   if s.get('task')==j['key']:continue
   desired=j['key']
   if j.get('claim_key')!=desired:j['claim_key']=desired;changed=True
  if changed:
   tmp=nativeq.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(nativeq)
targets=[(4,'unified_branches','retrieval_queue.json'),(6,'unified_branches','retrieval_queue.json')]
if (out/'ood_repair_queue.json').exists():targets += [(2,'unified_ood','ood_repair_queue.json'),(3,'unified_ood','ood_repair_queue.json')]
if (out/'segmentation_queue.json').exists():targets += [(gpu,slot,'segmentation_queue.json') for gpu in [2,3] for slot in ['unified_spatial','unified_spatial_extra','unified_spatial_2','unified_spatial_3','unified_spatial_4','unified_spatial_5']]
if (out/'global_queue.json').exists():targets += [(gpu,slot,'global_queue.json') for gpu in [2,3] for slot in ['unified_global_extra','unified_global_0','unified_global_1','unified_global_2']]
if boneq.exists():targets += [(gpu,'unified_bone_age','bone_age_queue.json') for gpu in [2,3,4,6]]
if (out/'native_detection_queue.json').exists():targets += [(gpu,slot,'native_detection_queue.json') for gpu in [2,3,4,6] for slot in ['unified_native','unified_native_parallel']]
if (out/'nucls_queue.json').exists():targets += [(gpu,slot,'nucls_queue.json') for gpu in [2,3,4,6] for slot in ['unified_nucls','unified_nucls_1','unified_nucls_2','unified_nucls_3']]
if (out/'detection_queue.json').exists():targets += [(gpu,'unified_detection_proxy','detection_queue.json')for gpu in [2,3,4,6]]
if (out/'film_queue.json').exists():targets += [(2,'unified_film','film_queue.json'),(3,'unified_film','film_queue.json')]
if (out/'rsna_queue.json').exists():targets += [(gpu,'unified_rsna','rsna_queue.json') for gpu in [2,3,4,6]]
if (out/'bbbc038_official_queue.json').exists():targets += [(gpu,'unified_bbbc038_official','bbbc038_official_queue.json') for gpu in [2,3,4,6]]
if (out/'rxrx3_retrieval_queue.json').exists():targets += [(gpu,'unified_rxrx3','rxrx3_retrieval_queue.json')for gpu in [2,3,4,6]]
if (out/'rxrx1_retrieval_queue.json').exists():targets += [(gpu,'unified_rxrx1','rxrx1_retrieval_queue.json')for gpu in [2,3,4,6]]
if (out/'hpa_retrieval_queue.json').exists():targets += [(gpu,'unified_hpa','hpa_retrieval_queue.json')for gpu in [2,3,4,6]]
if (out/'ctc_queue.json').exists():
 targets += [(gpu,'unified_ctc','ctc_queue.json') for gpu in [2,3,4,6]]
 if (out/'validation/ai_ctc_sequence_pilots.json').exists():targets += [(gpu,'unified_ctc_'+str(slot),'ctc_queue.json')for gpu in [2,3,4,6]for slot in range(1,5)]
for gpu,slot,queue_name in targets:
 ident=f'5090-lyx-xr_gpu{gpu}_slot{slot}';p=out/'states'/f'worker_{ident}.json';s=json.loads(p.read_text()) if p.exists() else {}
 def alive(pid):
  try:return b'unified_paper' in pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
  except (FileNotFoundError,ProcessLookupError):return False
 if not alive(s.get('pid')) and not alive(s.get('child_pid')):
  log=out/'logs'/f'worker_{ident}.log'
  with log.open('a') as f:proc=subprocess.Popen([__REMOTE_PY__,'-u',str(root/'scripts/unified_paper_queue.py'),'--host-label','5090-lyx-xr','--gpu',str(gpu),'--slot',slot,'--queue-file',str(out/queue_name)],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  s.update(pid=proc.pid,status='restarted')
 s['child_verified_live']=alive(s.get('child_pid'));states.append(s)
reservations=[]
for p in (out/'gpu_reservations').glob('process_*.json'):reservations.append(json.loads(p.read_text()))
cards=[]
process_memory={}
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
 fields=[x.strip()for x in line.split(',')]
 if len(fields)==2 and all(x.isdigit()for x in fields):process_memory[fields[0]]=int(fields[1])
for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
 index,total,used,util=map(int,line.split(','));cards.append(dict(gpu=index,total_mib=total,used_mib=used,vram_fraction=used/total,gpu_utilization=util,target_met=used/total>=.6))
tasks=[]
for name in ['rxrx1_retrieval_queue.json','rxrx3_retrieval_queue.json','hpa_retrieval_queue.json','retrieval_queue.json','ood_repair_queue.json','segmentation_queue.json','global_queue.json','bone_age_queue.json','native_detection_queue.json','nucls_queue.json','detection_queue.json','film_queue.json','rsna_queue.json','bbbc038_official_queue.json','ctc_queue.json']:
 if (out/name).exists():tasks.extend(json.loads((out/name).read_text())['tasks'])
for job in tasks:
 result=pathlib.Path(job['result']);job['remote_result_verified']=False
 if result.is_file():
  raw=json.loads(result.read_text())
  job['remote_result_verified']=all(pathlib.Path(raw[key]).is_file() for key in ['prediction_path','sample_manifest'] if key in raw)
  if job['remote_result_verified']:
   import hashlib
   job['remote_result_sha256']=hashlib.sha256(result.read_bytes()).hexdigest();job['remote_result_path']=str(result)
cache_completions={j['result']:json.loads(pathlib.Path(j['result']).read_text()) for j in tasks if j['task']=='segmentation_cache_recovery' and pathlib.Path(j['result']).is_file()}
print(json.dumps(dict(states=states,reservations=reservations,gpus=cards,gpu_process_memory_mib=process_memory,queue={'tasks':tasks},cache_completions=cache_completions,complete=[str(p) for p in (out/'results').glob('*/retrieval_clustering/*/component_result.json')])))
'''.replace('__REMOTE_ROOT__',repr(REMOTE_ROOT)).replace('__REMOTE_PY__',repr(PY)).replace('__MODELS__',repr(MODELS))
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,'python3 -u -'],input=code,text=True,capture_output=True,timeout=45)
    if result.returncode:raise RuntimeError(result.stderr)
    return json.loads(result.stdout)

def run():
    snapshot=remote_snapshot();mirror=OUT/'remote_mirror'/HOST;mirror.mkdir(parents=True,exist_ok=True)
    for remote_path,data in snapshot.get('cache_completions',{}).items():
        local=Path(remote_path.replace(REMOTE_OUT,str(OUT)));local.parent.mkdir(parents=True,exist_ok=True)
        data.update(remote_host=HOST,remote_cache_completion=remote_path,feature_caches_retained_on_resident_host=True)
        if not local.exists() or read(local)!=data:write(local,data)
    # Transfers are persistent independent children, so slow network I/O cannot
    # stall GPU/queue monitoring. Each completed file arrives by atomic rename.
    for kind in ['results','logs','states']:ensure_sync(kind,mirror)
    sources=sorted((p for p in (mirror/'results').rglob('*') if p.is_file() and '.rsync-partial' not in p.parts and '.tmp' not in p.name),key=lambda p:(p.name=='component_result.json',p.suffix=='.json',str(p)))
    for src in sources:
        if not src.is_file():continue
        rel=src.relative_to(mirror/'results');dst=OUT/'results'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if src.suffix=='.json':
            raw=read(src)
            if not isinstance(raw,dict):
                # Epoch histories and metric arrays are JSON lists, not result
                # dictionaries. Preserve them verbatim without metadata fields.
                if not dst.exists() or read(dst)!=raw:write(dst,raw)
                continue
            normalized=dict(raw)
            if isinstance(raw.get('seeds'),list):
                normalized['seeds']=[dict(seed,**{key:seed[key].replace(REMOTE_ROOT,str(ROOT)) for key in ['prediction_path','manifest','sample_manifest','selection','invocation'] if isinstance(seed.get(key),str)}) if isinstance(seed,dict) else seed for seed in raw['seeds']]
            for key in ['prediction_path','retrieval_result','sample_manifest','invocation']:
                if key in raw:normalized[key]=raw[key].replace(REMOTE_OUT,str(OUT))
            if src.name=='component_result.json' and any(not Path(normalized[key]).is_file() for key in ['prediction_path','sample_manifest'] if key in normalized):continue
            if 'selection' in raw:normalized['selection']=raw['selection'].replace(REMOTE_ROOT,str(ROOT))
            if 'clustering_results' in raw:normalized['clustering_results']=[x.replace(REMOTE_OUT,str(OUT)) for x in raw['clustering_results']]
            if 'manifest' in raw:normalized['manifest']=raw['manifest'].replace(REMOTE_ROOT,str(ROOT))
            if 'sample_manifest_sha256' in raw and 'sample_manifest' in normalized:
                manifest_path=Path(normalized['sample_manifest'])
                if not manifest_path.is_file():continue
                normalized['remote_sample_manifest_sha256']=raw['sample_manifest_sha256']
                normalized['sample_manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            normalized.update(remote_host=HOST,remote_raw_result=REMOTE_OUT+'/results/'+str(rel),mirrored_raw_result=str(src),remote_raw_result_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
            if not dst.exists() or read(dst)!=normalized:write(dst,normalized)
        elif not dst.exists():os.link(src,dst)
    with (OUT/'locks/model_admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cfg=read(OUT/'config.json');known={s['model'] for s in cfg['models']}
        # Admit both remote representatives after one real shared-entry pilot passes.
        passed=any((OUT/'results'/m/'retrieval_clustering/nct-crc-he-1k/component_result.json').exists() for m in MODELS)
        if passed:
            for model in MODELS:
                if model not in known:cfg['models'].append(read(OUT/'anchor_specs'/f'{model}.json'))
            own=sum(m['kind']=='biodino' for m in cfg['models']);fm=sum(m['kind']=='fm' for m in cfg['models']);cfg['statistics']['planned_comparisons_per_family']=own*fm;write(OUT/'config.json',cfg)
            stat=read(OUT/'nucls_statistics_config.json');stat.update(primary_family=f'{own} fixed BioDINO anchors x{fm} FM; {own*fm} planned comparisons; four-fold equal-weight mean AP',secondary_families=f'Each secondary metric has a separate {own*fm}-comparison exploratory family');write(OUT/'nucls_statistics_config.json',stat)
        jobs=[]
        for job in snapshot['queue']['tasks']:
            j=dict(job);j['result']=j['result'].replace(REMOTE_OUT,str(OUT));j['remote_log']=j['log'];j['log']=str(mirror/'logs'/Path(j['log']).name);j['allowed_hosts']=[HOST];jobs.append(j)
        write(OUT/'remote_queue.json',{'tasks':jobs,'remote_host':HOST,'remote_queue':REMOTE_OUT+'/retrieval_queue.json','admitted_to_main_config':passed})
    live={};reservations={r['pid']:r for r in snapshot['reservations']}
    for s in snapshot['states']:
        if s.get('child_verified_live') and s.get('task'):
            s['execution_phase']=reservations.get(s['child_pid'],{}).get('status','active_or_loading');s['remote_log']=s.get('log');s['log']=str(mirror/'logs'/Path(s['log']).name);live[s['task']]=s
    write(OUT/'remote_resource_snapshot.json',{'observed_at_controller':time.time(),'host':HOST,'live_tasks':live,'states':snapshot['states'],'nodes':{HOST:{'gpus':snapshot['gpus'],'gpu_process_memory_mib':snapshot.get('gpu_process_memory_mib',{})}}})
    write(OUT/'states/ai_branch_manager.json',{'pid':os.getpid(),'updated':time.time(),'status':'live','completed_pilot_outputs':len(snapshot['complete']),'models_admitted':passed,'remote_queue':REMOTE_OUT+'/retrieval_queue.json'})
    print('AI BRANCHS',len(snapshot['complete']),'completed outputs;',len(live),'live;',passed,'admitted',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    with (OUT/'locks/ai_branch_manager.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        threading.Thread(target=sync_reaper,daemon=True,name='sync-reaper').start()
        while True:
            reap_sync_children()
            try:run()
            except Exception as e:
                write(OUT/'states/ai_branch_manager.json',{'pid':os.getpid(),'updated':time.time(),'status':'error','error':repr(e)});print('ERROR',repr(e),flush=True)
                if not args.watch:raise
            if not args.watch:break
            time.sleep(60)

if __name__=='__main__':main()
