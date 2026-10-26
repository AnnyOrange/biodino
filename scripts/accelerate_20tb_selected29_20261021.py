#!/usr/bin/env python3
"""Keep the authorized 20TB catch-up fleet populated and audit real progress."""
import argparse
import csv
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import manage_20tb_selected29_fill_20261021 as fill

BASE = fill.BASE
REGISTRY = BASE / 'capacity_workers.json'
REACHABLE_CPUS = [1, 2, 5, 7, 8, 9, 10, 11, 12, 15, 18, 19, 20]


def prepare():
    records = []
    specs = [('5090-lyx-xr', 'lyx:capacity', [4, 5, 6, 7], 32),
             ('deepcad', 'deepcad:capacity', [1, 5, 6], 32),
             ('3090-qi', 'qi:capacity', list(range(8)), 48)]
    specs += [(f'cpu{i}', f'cpu{i}:capacity', [0], 8) for i in REACHABLE_CPUS]
    specs += [('local', 'zxr:capacity', list(range(8)), 8)]
    for host, name, gpus, jobs in specs:
        lyx = host == fill.LYX
        python = '/data/xuzijing/eval_envs/hs6_protocol_v2/bin/python' if lyx else fill.PYTHON
        script = fill.REMOTE + '/worker_capacity.py' if lyx else str(fill.ROOT / 'scripts/run_20tb_selected29_fill_worker_20261021.py')
        output = fill.REMOTE_CAMPAIGN if lyx else str(BASE / 'lw5')
        snapshot = fill.REMOTE_SOURCE if lyx else str(fill.SNAPSHOT)
        command = [python, '-u', script, '--snapshot', snapshot, '--output', output,
                   '--host', name, '--family', 'mixed', '--gpus', *map(str, gpus),
                   '--max-jobs', str(jobs), '--target-per-gpu', '8', '--max-global-jobs', '256',
                   '--allow-dense-colocation', '--minimum-per-gpu', '5',
                   '--memory-target-fraction', '.68', '--memory-headroom-mib', '2048']
        if host == 'local':
            command[command.index('--target-per-gpu') + 1] = '1'
            command[command.index('--minimum-per-gpu') + 1] = '1'
            command += ['--allow-training-share']
        records.append(dict(host=host, name=name, command=command, output=output,
                            log=f'{output}/launcher_logs/{name.replace(":", "_")}.log'))
    fill.save(REGISTRY, records)
    print(json.dumps(dict(workers=len(records), hosts=[r['host'] for r in records])))


def inspect_worker(record):
    # Locks and running claims are shared with the original fleet. Never restart
    # a launcher while any of its evaluation children still exist.
    code = '''import fcntl,json,os,shutil,subprocess,time
from pathlib import Path
r=RECORD;root=Path(r['output']);state=root/'_state';name=r['name']
workers=state/'workers';workers.mkdir(parents=True,exist_ok=True)
lock=(workers/(name+'.lock')).open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 p=workers/(name+'.json');w=json.loads(p.read_text()) if p.exists() else {}
 pid=str(w.get('pid',0));children=Path('/proc',pid,'task',pid,'children');active=[]
 if children.exists():
  for c in children.read_text().split():
   try:
    if b'dinov3.eval.' in Path('/proc',c,'cmdline').read_bytes():active.append(int(c))
   except OSError:pass
 print(json.dumps(dict(host=r['host'],name=name,status='EVALUATING' if active else 'WAITING_OR_LOADING',
       active_evaluation_pids=active,worker=w)));raise SystemExit
pause=state/'PAUSED.json'
if pause.exists():print(json.dumps(dict(host=r['host'],name=name,status='PAUSED',reason=json.loads(pause.read_text()))));raise SystemExit
owned=[]
for p in (state/'running').glob('*.json'):
 try:
  if json.loads(p.read_text()).get('host')==name:owned.append(p)
 except FileNotFoundError:pass
commands=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:commands.append(p.read_bytes())
 except OSError:pass
live_owned=[p for p in owned if any(str(root/'cells'/p.stem).encode() in c for c in commands)]
owned=[p for p in owned if p not in live_owned]
archive=state/'recovery'/('capacity_'+str(int(time.time())));archive.mkdir(parents=True,exist_ok=True)
if owned:
 import sys,copy
 snapshot=r['command'][r['command'].index('--snapshot')+1]
 sys.path[:0]=[str(Path(snapshot)/'scripts'),snapshot]
 import run_l20tb_selected29_worker_20261019 as locked
 manifest=json.loads((root/'campaign_manifest.json').read_text())
 for p in owned:
  directory=root/'cells'/p.stem
  if not (state/'done'/p.name).exists():
   try:
    task=copy.deepcopy(next(t for t in manifest['tasks'] if t['key']==p.stem))
    invocation=json.loads((directory/'invocation_manifest.json').read_text())
    task['asset']['path']=invocation['checkpoint']['path']
    report=locked.validate(task,directory,invocation)
    locked.fleet.queue.save(state/'done'/p.name,report)
   except Exception:pass
for p in owned:
 claim=state/'claims'/p.stem
 if claim.exists():shutil.move(str(claim),str(archive/claim.name))
 p.rename(archive/p.name)
if live_owned:
 print(json.dumps(dict(host=r['host'],name=name,status='LIVE_CHILDREN',live_tasks=[p.stem for p in live_owned])));raise SystemExit
if r.get('retired'):
 print(json.dumps(dict(host=r['host'],name=name,status='RETIRED')));raise SystemExit
for command in commands:
 parts=command.split(bytes([0]))
 if b'--host' in parts and name.encode() in parts and any(p.endswith((b'/worker.py',b'/worker_capacity.py',b'/run_20tb_selected29_fill_worker_20261021.py')) for p in parts):
  print(json.dumps(dict(host=r['host'],name=name,status='STARTING_WAITING_FOR_IMPORTS')));raise SystemExit
log=Path(r['log']);log.parent.mkdir(parents=True,exist_ok=True)
with log.open('a') as f:
 child=subprocess.Popen(r['command'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(host=r['host'],name=name,status='STARTED',pid=child.pid)))
'''.replace('RECORD', repr(record))
    if record['host'] == 'local':
        out = subprocess.run(['python', '-'], input=code, text=True, capture_output=True, check=True, timeout=45).stdout
    else:
        out = fill.remote_python(record['host'], code, timeout=45)
    return json.loads(out)


def audit():
    from selected29_metrics_20261019 import collect_campaigns
    campaigns = [fill.OLD, BASE / 'lw5', BASE / 'hxw', BASE / 'lw5_hxw']
    points, _ = collect_campaigns(campaigns)
    arms = {'l20tb_domain', 'l20tb_complement_bio5tb', '20tb_from47823_lw5',
            '20tb_distill_lw5', '20tb_nogram_fill'}
    points = [p for p in points if p['arm'] in arms]
    previous_path = BASE / 'capacity_progress.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    now = time.time()
    done = sum(p['done_cells'] for p in points)
    start = previous.get('started_unix', now)
    initial = previous.get('initial_done_tasks', done)
    rate = (done - initial) / max((now - start) / 3600, .001)
    remaining = sum(29-p['done_cells'] for p in points)
    exported = {(p['arm'],p['checkpoint']) for p in points}
    for arm, spec in fill.ARMS.items():
        exported.update((arm,int(p.name.split('_')[-1])) for p in (spec['local']/'eval').glob('training_*')
                        if (p/'teacher_checkpoint.pth').is_file())
    training_path = fill.ROOT / 'outputs/00_reports/l20tb_domain_7b_baseline_20261018/overview_20261021/full_figure_training.json'
    if training_path.exists():
        training=json.loads(training_path.read_text())
        for key,arm in [('domain','l20tb_domain'),('complement','l20tb_complement_bio5tb'),('lw5','20tb_distill_lw5')]:
            exported.update((arm,ck) for ck in training['arms'][key]['exports'])
    admitted={(p['arm'],p['checkpoint']) for p in points}
    unadmitted=sorted(exported-admitted)
    deadline=dt.datetime(2026,10,21,16,tzinfo=dt.timezone.utc).timestamp()
    remaining_hours=max((deadline-now)/3600,0)
    total_remaining=remaining+29*len(unadmitted)
    result = dict(updated_unix=now, started_unix=start, initial_done_tasks=initial,
                  done_tasks=done, remaining_admitted_tasks=remaining,
                  completed_checkpoints=sum(p['complete'] for p in points),
                  admitted_checkpoints=len(points), validated_tasks_per_hour=rate,
                  exported_but_not_admitted=[dict(arm=a,checkpoint=c) for a,c in unadmitted],
                  remaining_exported_tasks=total_remaining,
                  deadline='2026-10-21 23:59 Asia/Shanghai',
                  hours_until_deadline=remaining_hours,
                  required_tasks_per_hour=total_remaining/max(remaining_hours,.001),
                  estimated_hours_for_admitted_tasks=remaining/rate if rate > 0 else None,
                  estimate_limitation='Observed throughput only; long segmentation and future exports can change ETA',
                  points=points)
    fill.save(previous_path, result)
    table=BASE/'capacity_missing_checkpoints.csv'
    with table.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['arm','checkpoint','valid_tasks','remaining_tasks','status'])
        writer.writeheader()
        for p in sorted(points,key=lambda p:(p['arm'],p['checkpoint'])):
            writer.writerow(dict(arm=p['arm'],checkpoint=p['checkpoint'],valid_tasks=p['done_cells'],
                                 remaining_tasks=29-p['done_cells'],status='COMPLETE' if p['complete'] else 'EVALUATING_OR_QUEUED'))
        for arm,ck in unadmitted:
            writer.writerow(dict(arm=arm,checkpoint=ck,valid_tasks=0,remaining_tasks=29,status='WAITING_ADMISSION'))
    with (BASE / 'capacity_progress_history.jsonl').open('a') as f:
        f.write(json.dumps({k:v for k,v in result.items() if k != 'points'})+'\n')
    return result


def retry_failures(root=None):
    """Retry runtime failures with bounded attempts; never bypass validation."""
    root = Path(root) if root is not None else BASE / 'lw5'
    state = root / '_state'
    history_path = state / 'capacity_retries.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    with (state / 'manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads((root / 'campaign_manifest.json').read_text())
        changed = False
        for task in manifest['tasks']:
            spec = task['dataset']
            profile = manifest.get('resource_profiles', {}).get(spec['dataset'])
            if spec['task'] == 'classification' and profile:
                reserve = max(int(profile), int(task['asset'].get('reserve_mib') or 0))
                if spec.get('reserve_mib') != reserve:
                    task['dataset'] = dict(spec, reserve_mib=reserve);changed = True
        for p in (state / 'failed_resource').glob('*.json'):
            key = p.stem
            if (state / 'done' / p.name).exists() or (state / 'running' / p.name).exists():
                continue
            record = history.get(key, dict(attempts=0, last_retry=0))
            if record['attempts'] >= 3 or time.time() - record['last_retry'] < 300:
                continue
            failure = json.loads(p.read_text())
            if failure.get('status') == 'INVALID_PROTOCOL':
                continue
            log_path = root / 'cells' / key / 'run.log'
            log = log_path.read_text(errors='replace')[-20000:] if log_path.exists() else ''
            if 'out of memory' in log.lower():
                for task in manifest['tasks']:
                    if task['key'] == key:
                        task['asset'] = dict(task['asset'], reserve_mib=max(8192, task['asset'].get('reserve_mib', 4096)))
                        if task['dataset'].get('reserve_mib'):
                            task['dataset'] = dict(task['dataset'], reserve_mib=max(8192, task['dataset']['reserve_mib']))
                        changed = True
            archive = state / 'recovery' / f'retry_{int(time.time())}' / key
            archive.mkdir(parents=True, exist_ok=True)
            p.rename(archive / 'failure.json')
            claim = state / 'claims' / key
            if claim.exists():claim.rename(archive / 'claim')
            record.update(attempts=record['attempts']+1, last_retry=time.time(), last_failure=failure)
            history[key] = record
        if changed:fill.save(root / 'campaign_manifest.json', manifest)
    fill.save(history_path, history)


def recover_identical_source_metadata():
    """Repair a restored file's stat cache only after rehashing locked bytes."""
    import hashlib
    import shutil
    root = BASE / 'lw5'
    state = root / '_state'
    pause = state / 'PAUSED.json'
    if not pause.exists():
        return
    failure = json.loads(pause.read_text())
    if failure.get('error') != 'Source changed after registration':
        fill.save(BASE / 'blocked_queue.json', dict(updated_unix=time.time(), reason=failure,
                                                   action='REQUIRES_DIAGNOSIS_NOT_AUTOMATICALLY_BYPASSED'))
        return
    manifest = json.loads((root / 'campaign_manifest.json').read_text())
    snapshot = Path(manifest['source_snapshot_path'])
    for relative, digest in manifest['source_snapshot']['files'].items():
        if hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Cannot repair metadata: frozen evaluator content changed: ' + relative)
    sources = {p: h for t in manifest['tasks'] for p, h in t['dataset']['source_hashes'].items()}
    verified = {}
    for path, digest in sources.items():
        p = Path(path); before = p.stat()
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Cannot repair metadata: split content changed: ' + path)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Cannot repair a changing source: ' + path)
        verified[path] = dict(path=path, bytes=after.st_size, mtime_ns=after.st_mtime_ns, sha256=digest)
    archive = state / 'recovery' / ('verified_metadata_' + str(time.time_ns()))
    archive.mkdir(parents=True)
    repaired = []
    for record in (state / 'sources').glob('*.json'):
        with record.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            old = json.loads(record.read_text()); new = verified.get(old['path'])
            if new is None or old == new:continue
            if old['sha256'] != new['sha256']:
                raise RuntimeError('Registered source hash changed; refusing repair')
            shutil.copy2(record, archive / record.name)
            fill.save(record, new);repaired.append(dict(old=old, new=new))
    key = failure['task']
    if (state / 'running' / (key + '.json')).exists():
        raise RuntimeError('Paused task still running; refusing claim recovery')
    claim = state / 'claims' / key
    if claim.exists():claim.rename(archive / ('claim_' + key))
    fill.save(archive / 'proof.json', dict(verified_source_count=len(sources), repairs=repaired, failure=failure))
    # Do not clear a different pause written while verification was running.
    if json.loads(pause.read_text()) == failure:pause.rename(archive / 'PAUSED.json')


def retry_remote_failures():
    # Run the same bounded queue repair beside the AI-building checkpoints.
    import inspect
    function = inspect.getsource(retry_failures).replace('fill.save(', 'save(')
    code = '''from pathlib import Path
import fcntl,json,time,os
def save(path,value):
 temporary=path.with_name(path.name+'.retry.'+str(os.getpid()))
 temporary.write_text(json.dumps(value,indent=2)+'\\n');temporary.replace(path)
'''+function+'\nretry_failures('+repr(fill.REMOTE_CAMPAIGN)+')\n'
    fill.remote_python(fill.LYX, code, timeout=30)


def prioritize_checkpoints():
    code = '''from pathlib import Path
import fcntl,json,os,collections
root=Path(ROOT);state=root/'_state'
with (state/'manifest.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 p=root/'campaign_manifest.json';m=json.loads(p.read_text())
 done={p.stem for p in (state/'done').glob('*.json')};counts=collections.Counter()
 for t in m['tasks']:
  if t['key'] in done:counts[(t['asset']['arm'],t['asset']['checkpoint_id'])]+=1
 def rank(t):
  a=t['asset'];key=(a['arm'],a['checkpoint_id'])
  return (-counts[key],int(a['checkpoint_id']),a['arm'],t['dataset']['task']!='segmentation',t['key'])
 m['tasks'].sort(key=rank);m['checkpoint_first']=True
 m['scheduling_policy']='Finish the most-complete checkpoint first; run its long segmentation tasks early'
 temporary=p.with_name(p.name+'.priority.'+str(os.getpid()))
 temporary.write_text(json.dumps(m,indent=2)+'\\n');temporary.replace(p)
'''
    subprocess.run(['python', '-'], input=code.replace('ROOT',repr(str(BASE/'lw5'))), text=True, check=True, timeout=30)
    fill.remote_python(fill.LYX, code.replace('ROOT',repr(fill.REMOTE_CAMPAIGN)), timeout=30)


def sweep():
    records = json.loads(REGISTRY.read_text())
    results = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        jobs = {pool.submit(inspect_worker, r): r for r in records}
        for future in as_completed(jobs):
            r = jobs[future]
            try:results.append(future.result())
            except Exception as exc:results.append(dict(host=r['host'],name=r['name'],status='ERROR',error=str(exc)))
    fill.save(BASE / 'capacity_supervision.json', dict(updated_unix=time.time(),workers=results))
    return results


def gpu_floor_audit():
    """Measure CUDA processes, not launcher counts, and retain floor breaches."""
    records = json.loads(REGISTRY.read_text())
    hosts = {}
    for record in records:
        command = record['command']
        start = command.index('--gpus') + 1
        end = next((i for i in range(start, len(command)) if command[i].startswith('--')), len(command))
        hosts.setdefault(record['host'], set()).update(map(int, command[start:end]))
    code = '''import csv,json,subprocess,time,collections
from pathlib import Path
root=Path(ROOT);state=root/'_state';assigned=collections.Counter()
for p in (state/'running').glob('*.json'):
 try:
  row=json.loads(p.read_text())
  if row.get('host','').split(':')[0]==PREFIX:assigned[row['gpu']]+=1
 except FileNotFoundError:pass
manifest=json.loads((root/'campaign_manifest.json').read_text());pending=[]
for t in manifest['tasks']:
 if not (state/'done'/(t['key']+'.json')).exists() and not (state/'claims'/t['key']).exists():
  d=t['dataset'];pending.append(int(d.get('reserve_mib') or max(18432 if d['image_size']>=384 else 4096,int(t['asset'].get('reserve_mib') or 0))))
cards=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()))
apps=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines()))
result=[]
for index,uid,used,total,util in cards:
 if int(index) not in GPUS:continue
 tests=[];training=[];other=[]
 for row in apps:
  if len(row)!=3 or row[0].strip()!=uid.strip():continue
  pid=row[1].strip()
  try:cmd=Path('/proc',pid,'cmdline').read_bytes().replace(b'\\0',b' ').decode()
  except OSError:continue
  evaluation=any(t in cmd for t in ('dinov3.eval.','/dinov3/eval/','/bio_segmentation/','linear_probe','--probe-epochs','/run_ctc_native_','/run_unified_paper_'))
  train=any(t in cmd for t in ('dinov3/train/','train_irflow','/train.py',' train ')) and not evaluation
  item=dict(pid=int(pid),memory_mib=int(row[2].strip()) if row[2].strip().isdigit() else None,
   selected29_20tb=('l20tb_selected29' in cmd and '/cells/' in cmd))
  (tests if evaluation else training if train else other).append(item)
 fraction=int(used)/int(total);meets=fraction>=.6 and len(tests)>=5;headroom_safe=fraction<=.8
 own_count=sum(t['selected29_20tb'] for t in tests)
 reason='OK'
 if not headroom_safe:reason='MEMORY_ABOVE_80_PERCENT_STOP_ADMISSION'
 elif training:reason='EXISTING_TRAINING_PRESERVED'
 elif not meets:
  if not pending:reason='ALL_REMAINING_SELECTED29_TASKS_ALREADY_ASSIGNED'
  elif int(total)*.8-int(used)<min(pending):reason='PENDING_TASKS_DO_NOT_FIT_20_PERCENT_HEADROOM'
  elif assigned[int(index)]>own_count:reason='ASSIGNED_TASKS_LOADING_OR_IN_CPU_PHASE'
  else:reason='REFILL_PENDING'
 result.append(dict(gpu=int(index),used_mib=int(used),total_mib=int(total),memory_fraction=fraction,
  utilization_percent=int(util),test_count=len(tests),selected29_20tb_test_count=own_count,test_processes=tests,training_processes=training,
  assigned_selected29_tasks=assigned[int(index)],unclaimed_selected29_tasks=len(pending),constraint=reason,
  other_processes=other,meets_floor=meets,headroom_safe=headroom_safe,
  free_mib=int(total)-int(used),
  action='DRAIN_EXCESS_NO_NEW_TASKS' if not headroom_safe and not training else 'OK' if meets and headroom_safe else 'PRESERVE_TRAINING_AND_FILL_AVAILABLE_CAPACITY' if training else 'AUTOMATIC_REFILL_ACTIVE'))
print(json.dumps(dict(host=HOST,updated_unix=time.time(),gpus=result)))
'''
    def inspect(host, gpus):
        prefix = next(r['name'].split(':')[0] for r in records if r['host']==host)
        root = fill.REMOTE_CAMPAIGN if host==fill.LYX else str(BASE/'lw5')
        script = code.replace('GPUS', repr(sorted(gpus))).replace('HOST', repr(host)).replace('ROOT',repr(root)).replace('PREFIX',repr(prefix))
        if host == 'local':
            output = subprocess.run(['python', '-'], input=script, text=True, capture_output=True, check=True, timeout=30).stdout
        else:
            output = fill.remote_python(host, script, timeout=30)
        return json.loads(output)
    reports = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        jobs = {pool.submit(inspect, host, gpus): host for host, gpus in hosts.items()}
        for future in as_completed(jobs):
            try:reports.append(future.result())
            except Exception as exc:reports.append(dict(host=jobs[future],error=str(exc)))
    result = dict(updated_unix=time.time(),hosts=reports)
    fill.save(BASE / 'gpu_floor_status.json', result)
    with (BASE / 'gpu_floor_history.jsonl').open('a') as stream:
        stream.write(json.dumps(result)+'\n')
    return result


def watch():
    with (BASE / 'capacity_controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        last_gpu_audit = 0
        while True:
            try:
                recover_identical_source_metadata();retry_failures();retry_remote_failures();prioritize_checkpoints();results=sweep();progress=audit()
                if time.time() - last_gpu_audit >= 60:
                    gpu_floor_audit();last_gpu_audit=time.time()
                print(json.dumps(dict(time=time.time(),states=[(r['name'],r['status']) for r in results],
                    done=progress['done_tasks'],remaining=progress['remaining_admitted_tasks'])),flush=True)
            except Exception as exc:print(repr(exc),flush=True)
            time.sleep(30)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','sweep','audit','watch']);a=p.parse_args()
    result=globals()[a.mode]()
    if result is not None:print(json.dumps(result,indent=2))
