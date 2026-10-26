#!/usr/bin/env python3
"""Mirror every intermediate EMA, admit locked evaluations, and refresh full curves."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

ROOT = Path('/mnt/huawei_deepcad/dinov3')
BASE = ROOT / 'outputs/02_eval_runs/l20tb_selected29_fill_20261021'
OLD = ROOT / 'outputs/02_eval_runs/l20tb_selected29_20261019'
SNAPSHOT = Path('/mnt/huawei_deepcad/dinov3_l20tb_selected29_snapshot_20261019')
PYTHON = '/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'
REMOTE = '/data/xuzijing/l20tb_selected29_dense_20261021'
REMOTE_CAMPAIGN = REMOTE + '/campaign'
REMOTE_SOURCE = REMOTE + '/source'
LYX = '5090-lyx-xr'
HXW = '5090-hxw-xzj'
HXW_LW5_ROOT = '/home/xzj/storage/merged/lw5_selected29_fill_20261021'
HXW_LW5_CAMPAIGN = HXW_LW5_ROOT + '/campaign'
HXW_LW5_SOURCE = HXW_LW5_ROOT + '/source'
RUNS = ROOT / 'outputs/01_training_runs'
ARMS = {
    'l20tb_domain': dict(local=RUNS / 'l20tb_domain_recoverability_from47823_len19520_20261018',
        remote='/home/xzj/storage/merged/l20tb_domain_recoverability_from47823_len19520_hxw_20261019',
        sparse='/data/xuzijing/cooldown_20261013_eval/20tb_domain_recoverability_hxw_20261020'),
    'l20tb_complement_bio5tb': dict(local=RUNS / 'l20tb_complement_bio5tb_from47823_len19520_qi_20261018',
        remote='/home/xzj/storage/merged/l20tb_complement_bio5tb_from47823_len19520_hxw_20261019',
        sparse='/data/xuzijing/cooldown_20261013_eval/20tb_complement_bio5tb_hxw_20261020'),
}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(16 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ssh(host, args, timeout=120):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
                           shlex.join([str(x) for x in args])], check=True, text=True,
                          capture_output=True, timeout=timeout)


def remote_python(host, code, timeout=120):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, 'python3', '-'],
                          input=code, check=True, text=True, capture_output=True, timeout=timeout).stdout


def remote_meta(host, path, hashed=True):
    code = '''import hashlib,json,time
from pathlib import Path
p=Path(PATH)
if not p.is_file() or p.stat().st_size<1000000 or time.time()-p.stat().st_mtime<120:
 print('null')
else:
 s=p.stat();h=hashlib.sha256()
 if HASHED:
  with p.open('rb') as f:
   for b in iter(lambda:f.read(16<<20),b''):h.update(b)
  assert p.stat().st_mtime_ns==s.st_mtime_ns and p.stat().st_size==s.st_size
 print(json.dumps(dict(path=str(p),bytes=s.st_size,sha256=h.hexdigest() if HASHED else None)))
'''.replace('PATH', repr(str(path))).replace('HASHED', repr(hashed))
    return json.loads(remote_python(host, code, timeout=600 if hashed else 120))


def rsync(source, destination, resume=False):
    subprocess.run(['rsync', '-aL', '--partial', *(['--append-verify'] if resume else []), '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                    str(source), str(destination)], check=True, capture_output=True, timeout=14400)


def tasks(assets, datasets):
    priority = {'breastmnist': 0, 'bbbc013': 1, 'nct-crc-he-1k': 2, 'cellpose': 3,
                'pannuke': 4, 'tissuenet': 5, 'hpa-subcellular': 6, 'rxrx1-cross': 7}
    result = []
    for spec in sorted(datasets, key=lambda d: priority.get(d['dataset'], 10)):
        for asset in assets:
            key = f"{asset['arm']}_ck{asset['checkpoint_id']}__{spec['task']}__{spec['dataset']}"
            if spec['task'] == 'segmentation':
                key += '__primary-last__' + spec['split']
            result.append(dict(key=key, asset=asset, dataset=spec))
    return result


def prepare():
    BASE.mkdir(parents=True, exist_ok=True)
    template = json.loads((OLD / 'campaign_manifest.json').read_text())
    for name in ('lw5', 'hxw'):
        path = BASE / name / 'campaign_manifest.json'
        if path.exists():
            continue
        m = copy.deepcopy(template)
        m.update(checkpoint_assets=[], tasks=[], execution_hosts={}, online_checkpoints=True,
                 isolate_runtime_failures=True, created_unix=time.time(),
                 explicit_user_authorization='2026-10-21: fill every intermediate result; hxw methods on lyx; lw5 on 3090-qi and deepcad',
                 allocation=('5090-lyx-xr' if name == 'hxw' else '3090-qi and deepcad'),
                 deepcad_authorized_gpu_indices=[1, 6], deepcad_authorized_max_gpus=2)
        m['source_snapshot_path'] = REMOTE_SOURCE if name == 'hxw' else str(SNAPSHOT)
        m['launcher_source_sha256'] = sha(ROOT / 'scripts/run_20tb_selected29_fill_worker_20261021.py')
        if name == 'hxw':
            m['benchmark_root'] = '/data/xuzijing/benchmark'
            for d in m['datasets']:
                d['source_hashes'] = {p.replace('/mnt/huawei_deepcad/benchmark', '/data/xuzijing/benchmark')
                    .replace('/mnt/huawei_deepcad/dinov3_retest_snapshot_20260918', REMOTE_SOURCE): h
                    for p, h in d['source_hashes'].items()}
        save(path, m)
        if name == 'hxw':
            rsync(path, f'{LYX}:{REMOTE_CAMPAIGN}/campaign_manifest.json')
        (BASE / name / '_state').mkdir(exist_ok=True)
    for arm, spec in ARMS.items():
        target = BASE / 'configs' / f'{arm}.yaml'
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(spec['local'] / 'candidate.yaml', target)
        ssh(LYX, ['mkdir', '-p', REMOTE_CAMPAIGN + '/configs'])
        rsync(target, f'{LYX}:{REMOTE_CAMPAIGN}/configs/{arm}.yaml')
    target = BASE / 'lw5/config.yaml'
    if not target.exists():
        shutil.copy2(OLD / 'assets/20tb_distill_lw5/config.yaml', target)
    done = {('l20tb_domain', 50263), ('l20tb_domain', 52703), ('20tb_distill_lw5', 79543)}
    points = [dict(arm='20tb_distill_lw5', checkpoint=ck, campaign='lw5') for ck in range(75151, 79544, 488)]
    points += [dict(arm=arm, checkpoint=ck, campaign='hxw')
               for ck in range(48311, 67344, 488) for arm in ARMS]
    for item in points:
        item['status'] = 'REUSED_COMPLETE' if (item['arm'], item['checkpoint']) in done else 'WAITING_EXPORT'
    if not (BASE / 'plan.json').exists():
        save(BASE / 'plan.json', dict(points=points, cadence=488, selected29_tasks_per_checkpoint=29,
             expected_total_checkpoints=90, already_complete_checkpoints=3, new_checkpoints=87,
             maximum_new_tasks=87*29, only_ema_exports=True))
    print(BASE)


def ensure_asset(item):
    arm, ck = item['arm'], item['checkpoint']
    if item['campaign'] == 'lw5_hxw':
        source = f'/home/xzj/storage/merged/cooldown_20261013/20tb_distill_cooldown_len4880_lw5/eval/training_{ck}/teacher_checkpoint.pth'
        # Local assets are hashed and inspected by the locked worker's
        # checkpoint_record before any evaluation. Avoid reading each large
        # checkpoint twice on the training host's busy disks.
        expected = remote_meta(HXW, source, hashed=False)
        if expected is None:
            return None
        save(BASE / 'verified_exports' / f'{arm}_{ck}.json', dict(source=source, source_host=HXW,
            destination=source, transfer='NONE_LOCAL_CHECKPOINT',
            checksum_stage='locked_evaluator_before_start', **expected_metadata(expected)))
        return dict(arm=arm, checkpoint_id=str(ck), path=source,
                    config=HXW_LW5_CAMPAIGN+'/config.yaml', kind='dinov3', model_id='', reserve_mib=4096)
    proof = BASE / 'verified_exports' / f'{arm}_{ck}.json'
    config = str(BASE / 'lw5/config.yaml') if item['campaign'] == 'lw5' else f'{REMOTE_CAMPAIGN}/configs/{arm}.yaml'
    if proof.exists():
        record = json.loads(proof.read_text())
        return dict(arm=arm, checkpoint_id=str(ck), path=record['destination'], config=config,
                    kind='dinov3', model_id='', reserve_mib=4096)
    if item['campaign'] == 'lw5':
        raise RuntimeError('User stopped lw5 weight transfers; schedule remaining checkpoints as lw5_hxw')
    else:
        spec = ARMS[arm]
        existing = f"{spec['sparse']}/eval/training_{ck}/teacher_checkpoint.pth"
        expected = remote_meta(LYX, existing)
        if expected is not None:
            destination = existing
            save(proof, dict(source=existing, source_host=LYX, destination=destination, **expected_metadata(expected)))
        else:
            local = spec['local'] / 'eval' / f'training_{ck}' / 'teacher_checkpoint.pth'
            destination = f'{REMOTE_CAMPAIGN}/assets/{arm}/training_{ck}/teacher_checkpoint.pth'
            ssh(LYX, ['mkdir', '-p', str(Path(destination).parent)])
            temp = destination + '.partial'
            if local.is_file() and local.stat().st_size > 1000000 and time.time()-local.stat().st_mtime > 120:
                # Main-building storage is shared. The user prohibits moving
                # these checkpoints to the AI building without explicit approval.
                return dict(arm=arm, checkpoint_id=str(ck), path=str(local),
                    config=str(spec['local'] / 'candidate.yaml'), kind='dinov3',
                    model_id='', reserve_mib=4096, _campaign_override='lw5')
            else:
                source = f"{spec['remote']}/eval/training_{ck}/teacher_checkpoint.pth"
                expected = remote_meta(HXW, source)
                if expected is None:
                    return None
                ssh(LYX, ['rsync', '-a', '--partial', '--append-verify', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                          f'{HXW}:{source}', temp], timeout=14400)
                host = HXW
            actual = remote_meta(LYX, temp)
            if actual is None or actual['sha256'] != expected['sha256'] or actual['bytes'] != expected['bytes']:
                raise RuntimeError('hxw EMA checksum mismatch')
            ssh(LYX, ['mv', temp, destination])
            save(proof, dict(source=source, source_host=host, destination=destination, **expected_metadata(expected)))
    return dict(arm=arm, checkpoint_id=str(ck), path=destination, config=config,
                kind='dinov3', model_id='', reserve_mib=4096)


def expected_metadata(expected):
    return dict(sha256=expected['sha256'], bytes=expected['bytes'], verified_unix=time.time())


def admit(campaign, asset):
    if campaign == 'lw5':
        path = BASE / campaign / 'campaign_manifest.json'
        with (BASE / campaign / '_state/manifest.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            m = json.loads(path.read_text())
            if any((a['arm'], a['checkpoint_id']) == (asset['arm'], asset['checkpoint_id']) for a in m['checkpoint_assets']):
                return
            m['checkpoint_assets'].append(asset)
            m['tasks'] = tasks(m['checkpoint_assets'], m['datasets'])
            if m.get('checkpoint_first'):
                m['tasks'].sort(key=lambda t: int(t['asset']['checkpoint_id']))
            save(path, m)
    else:
        code = '''import fcntl,json,os
from pathlib import Path
root=Path(ROOT);(root/'_state').mkdir(exist_ok=True)
with (root/'_state/manifest.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);p=root/'campaign_manifest.json';m=json.loads(p.read_text());asset=ASSET
 if not any((a['arm'],a['checkpoint_id'])==(asset['arm'],asset['checkpoint_id']) for a in m['checkpoint_assets']):
  m['checkpoint_assets'].append(asset)
  tasks=[]
  priority={'breastmnist':0,'bbbc013':1,'nct-crc-he-1k':2,'cellpose':3,'pannuke':4,'tissuenet':5,'hpa-subcellular':6,'rxrx1-cross':7}
  for d in sorted(m['datasets'],key=lambda d:priority.get(d['dataset'],10)):
   for a in m['checkpoint_assets']:
    key=f"{a['arm']}_ck{a['checkpoint_id']}__{d['task']}__{d['dataset']}"
    if d['task']=='segmentation':key+='__primary-last__'+d['split']
    tasks.append(dict(key=key,asset=a,dataset=d))
  if m.get('checkpoint_first'):tasks.sort(key=lambda t:int(t['asset']['checkpoint_id']))
  m['tasks']=tasks;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2)+'\\n');tmp.replace(p)
'''.replace('ROOT', repr(HXW_LW5_CAMPAIGN if campaign == 'lw5_hxw' else REMOTE_CAMPAIGN)).replace('ASSET', repr(asset))
        remote_python(HXW if campaign == 'lw5_hxw' else LYX, code)


def sync_remote(campaign='hxw'):
    host = HXW if campaign == 'lw5_hxw' else LYX
    remote_campaign = HXW_LW5_CAMPAIGN if campaign == 'lw5_hxw' else REMOTE_CAMPAIGN
    mirror = BASE / campaign
    subprocess.run(['rsync', '-a', '--exclude=assets/***', '--exclude=configs/***',
        '--exclude=_state/done/***', '--exclude=_state/claims/***',
        '--exclude=cache/', '--exclude=checkpoint_links/', '--include=*/', '--include=*.json', '--exclude=*',
        f'{host}:{remote_campaign}/', str(mirror)+'/'], check=True, capture_output=True, timeout=120)
    # Live state is a mirror, not a history: remove claims/failures that have
    # disappeared remotely so completed tasks are not shown as still running.
    for name in ('running', 'failed_resource'):
        destination = mirror / '_state' / name
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync', '-a', '--delete',
            f'{host}:{remote_campaign}/_state/{name}/', str(destination)+'/'],
            check=True, capture_output=True, timeout=60)
    raw = mirror / '_remote_done';raw.mkdir(exist_ok=True)
    rsync(f'{host}:{remote_campaign}/_state/done/', str(raw)+'/')
    normalized = mirror / '_state/done';normalized.mkdir(exist_ok=True)
    for marker in raw.glob('*.json'):
        destination = normalized / marker.name
        if destination.exists():
            continue
        report = json.loads(marker.read_text())
        if report.get('status') != 'VALID_COMPLETE':
            continue
        expected = report['result_sha256']
        if isinstance(expected, dict):
            relocated = {str(mirror / Path(p).relative_to(remote_campaign)): h for p, h in expected.items()}
        else:
            relocated = {str(mirror / 'cells' / marker.stem / 'component_result.json'): expected}
        if any(not Path(p).exists() or sha(p) != h for p, h in relocated.items()):
            continue
        if isinstance(expected, dict):
            report['result_sha256'] = relocated
        report['mirror_provenance'] = dict(remote_campaign=remote_campaign,
            original_report_sha256=sha(marker), original_report=str(marker),
            result_bytes_unchanged=True, synced_unix=time.time())
        save(destination, report)


def sync_hxw():
    sync_remote('hxw')
    if (BASE/'lw5_hxw/campaign_manifest.json').exists():
        sync_remote('lw5_hxw')


def refresh_training():
    code = '''import json,time,subprocess
from pathlib import Path
paths=PATHS;arms={}
for arm,path in paths.items():
 p=Path(path);r=json.loads(subprocess.check_output(['tail','-n','1',str(p/'raw_loss_metrics.jsonl')],text=True))
 arms[arm]=dict(run=path,latest=r['optimizer_update'],metric_time=r['time_unix'],exports=sorted(int(x.name.split('_')[-1]) for x in (p/'eval').glob('training_*')))
print(json.dumps(dict(captured=time.time(),host='5090-hxw-xzj',arms=arms)))
'''.replace('PATHS', repr(dict(domain=ARMS['l20tb_domain']['remote'], complement=ARMS['l20tb_complement_bio5tb']['remote'],
            lw5='/home/xzj/storage/merged/cooldown_20261013/20tb_distill_cooldown_len4880_lw5')))
    value = json.loads(remote_python(HXW, code))
    save(ROOT / 'outputs/00_reports/l20tb_domain_7b_baseline_20261018/overview_20261021/full_figure_training.json', value)


def launch(host_filter=None):
    specs = [
        (LYX, 'lyx:dense', [6], 'segmentation', 1),
        (LYX, 'lyx:dense45', [4, 5], 'segmentation', 1),
        (LYX, 'lyx:frozen', [4, 5, 7], 'frozen', 3),
        ('3090-qi', 'qi:lw5dense', [6], 'segmentation', 1),
        ('3090-qi', 'qi:lw5frozen', [0, 1, 2, 3, 4, 5, 7], 'frozen', 3),
        ('deepcad', 'deepcad:lw5dense', [1], 'segmentation', 1),
        ('deepcad', 'deepcad:lw5frozen', [6], 'frozen', 2),
        (HXW, 'hxw:lw5parallel', list(range(8)), 'frozen', 16),
        (HXW, 'hxw:lw5dense', list(range(8)), 'segmentation', 16),
        (HXW, 'hxw:lw5cache', list(range(8)), 'frozen', 0),
    ]
    records = []
    if host_filter and (BASE / 'launched_workers.json').exists():
        records = [r for r in json.loads((BASE / 'launched_workers.json').read_text()) if r['host'] != host_filter]
    for host, name, gpus, family, count in specs:
        if host_filter and host != host_filter:
            continue
        lyx = host == LYX
        python = '/data/xuzijing/eval_envs/hs6_protocol_v2/bin/python' if lyx else PYTHON
        script = REMOTE + '/worker.py' if lyx else str(ROOT / 'scripts/run_20tb_selected29_fill_worker_20261021.py')
        output = REMOTE_CAMPAIGN if lyx else str(BASE / 'lw5')
        source = REMOTE_SOURCE if lyx else str(SNAPSHOT)
        if host == HXW:
            python = '/home/xzj/eval_envs/hs6_protocol_v2/bin/python'
            script = HXW_LW5_ROOT + '/worker.py'
            output, source = HXW_LW5_CAMPAIGN, HXW_LW5_SOURCE
        command = [python, '-u', script, '--snapshot', source, '--output', output, '--host', name,
                   '--family', family, '--max-jobs', str(count), '--gpus', *map(str, gpus)]
        if host == HXW:
            command += ['--allow-training-share', '--memory-headroom-mib', '2048', '--max-global-jobs', '16',
                        '--target-per-gpu', '2',
                        '--checkpoint-cache', '/dev/shm/lw5_selected29_fill_20261021']
            if name == 'hxw:lw5cache':
                command.append('--prefetch-only')
        logfile = f'{output}/launcher_logs/{name.replace(":", "_")}.log'
        code = '''import json,subprocess
from pathlib import Path
command=COMMAND;log=Path(LOG);log.parent.mkdir(parents=True,exist_ok=True)
with log.open('a') as f:
 p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,command=command,log=str(log))))
'''.replace('COMMAND', repr(command)).replace('LOG', repr(logfile))
        record = json.loads(remote_python(host, code))
        records.append(dict(host=host, name=name, **record))
        print(host, name, record['pid'], flush=True)
    save(BASE / 'launched_workers.json', records)


def supervise():
    """Restart absent launchers only after ruling out live orphan evaluations."""
    path = BASE / 'launched_workers.json'
    if not path.exists():
        return []
    records = json.loads(path.read_text())
    results = []
    for record in records:
        if record['host'] == LYX and (BASE / 'lyx_safe_admission.json').exists():
            # The capacity supervisor drains these legacy launchers and owns
            # their validation/recovery; never restart an unbounded old copy.
            continue
        command = record['command']
        output = command[command.index('--output') + 1]
        code = '''import fcntl,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(OUTPUT);name=NAME;state=root/'_state';command=COMMAND
lock=(state/'workers'/f'{name}.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 result=dict(host=name,status='ALIVE')
 activity=state/'activity'/f'{name}.json'
 if activity.exists():
  phase=json.loads(activity.read_text());result['activity']=phase
  if phase.get('phase')=='EVALUATING':
   child=Path('/proc',str(phase.get('child_pid',0)),'cmdline')
   result['status']='GPU_EVALUATING' if child.exists() and b'dinov3.eval.' in child.read_bytes() else 'WAITING_RESOURCE_OR_QUEUE'
  else:result['status']=phase.get('phase','ALIVE')
 worker_status=state/'workers'/f'{name}.json'
 evaluation_pids=[]
 if worker_status.exists():
  worker=json.loads(worker_status.read_text());pid=str(worker['pid'])
  children=Path('/proc',pid,'task',pid,'children')
  if children.exists():
   for child_pid in children.read_text().split():
    try:cmd=Path('/proc',child_pid,'cmdline').read_bytes()
    except OSError:continue
    if b'dinov3.eval.' in cmd:evaluation_pids.append(int(child_pid))
 if evaluation_pids:result.update(status='GPU_EVALUATING',evaluation_pids=evaluation_pids,active_evaluations=len(evaluation_pids))
 print(json.dumps(result));raise SystemExit
if (state/'PAUSED.json').exists():
 print(json.dumps(dict(host=name,status='PAUSED',reason=json.loads((state/'PAUSED.json').read_text()))));raise SystemExit
owned=[]
for p in (state/'running').glob('*.json'):
 if json.loads(p.read_text()).get('host')==name:owned.append(p)
commands=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:commands.append(p.read_bytes().replace(bytes([0]),b' ').decode(errors='replace'))
 except OSError:pass
if any(str(root/'cells'/p.stem) in c for p in owned for c in commands):
 print(json.dumps(dict(host=name,status='LIVE_ORPHAN_EVALUATION',tasks=[p.stem for p in owned])));raise SystemExit
archive=state/'recovery'/str(int(time.time()));archive.mkdir(parents=True,exist_ok=True)
for p in owned:
 claim=state/'claims'/p.stem
 if claim.exists():shutil.move(str(claim),str(archive/claim.name))
 p.rename(archive/p.name)
log=Path(LOG)
with log.open('a') as f:
 child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(host=name,status='RESTARTED',pid=child.pid)))
'''.replace('OUTPUT', repr(output)).replace('NAME', repr(record['name'])).replace('COMMAND', repr(command)).replace('LOG', repr(record['log']))
        try:
            result = json.loads(remote_python(record['host'], code))
            if result.get('pid'):
                record['pid'] = result['pid']
                record['restarted_unix'] = time.time()
            results.append(result)
        except Exception as exc:
            results.append(dict(host=record['name'], status='UNREACHABLE', error=repr(exc)))
    save(path, records)
    save(BASE / 'supervision.json', dict(updated_unix=time.time(), workers=results))
    return results


def watch():
    with (BASE / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((BASE / 'plan.json').read_text())
        # Recover only our interrupted transfers; completed queue entries persist.
        for item in plan['points']:
            if item['status'] in ('TRANSFERRING', 'VERIFYING_LOCAL'):item['status'] = 'WAITING_EXPORT'
        pool = ThreadPoolExecutor(max_workers=2)
        active = {}
        last_sync = last_plot = last_supervise = 0
        while True:
            for campaign, (future, item) in list(active.items()):
                if not future.done():continue
                try:
                    asset = future.result()
                    if asset:
                        placement = asset.pop('_campaign_override', campaign)
                        admit(placement, asset)
                        item.update(status='ADMITTED', campaign=placement, asset=asset, admitted_unix=time.time())
                    else:
                        item.update(status='WAITING_EXPORT', retry_after=time.time()+180)
                except Exception as exc:
                    item['failures'] = item.get('failures', 0)+1
                    item.update(status='FAILED_THREE_ATTEMPTS' if item['failures'] >= 3 else 'WAITING_EXPORT',
                                error=repr(exc), retry_after=time.time()+60)
                del active[campaign]
                save(BASE / 'plan.json', plan)
            for campaign in ('lw5_hxw', 'hxw'):
                if campaign in active:continue
                candidates = [p for p in plan['points'] if p['campaign'] == campaign
                              and p['status'] == 'WAITING_EXPORT' and p.get('retry_after', 0) <= time.time()]
                candidates.sort(key=lambda p: (0 if (p['arm'],p['checkpoint']) == ('l20tb_complement_bio5tb',50263) else 1,
                                               p['checkpoint'], p['arm']))
                if candidates:
                    item = candidates[0];item.update(status='VERIFYING_LOCAL' if campaign == 'lw5_hxw' else 'TRANSFERRING', started_unix=time.time())
                    active[campaign] = (pool.submit(ensure_asset, item.copy()), item)
                    save(BASE / 'plan.json', plan)
            state = dict(pid=os.getpid(), updated_unix=time.time(), transfer_jobs=list(active),
                         statuses={s: sum(p['status']==s for p in plan['points']) for s in sorted({p['status'] for p in plan['points']})})
            if time.time()-last_supervise > 120:
                state['supervision'] = supervise()
                last_supervise = time.time()
            if time.time()-last_sync > 60:
                try:
                    sync_hxw();state['mirror'] = 'OK'
                except Exception as exc:state['mirror_error'] = repr(exc)
                last_sync=time.time()
            for campaign in ('lw5','hxw','lw5_hxw'):
                state[campaign] = dict(done_cells=len(list((BASE/campaign/'_state/done').glob('*.json'))),
                                       failed_cells=len(list((BASE/campaign/'_state/failed_resource').glob('*.json'))),
                                       activity=[json.loads(p.read_text()) for p in (BASE/campaign/'_state/activity').glob('*.json')],
                                       workers=[json.loads(p.read_text()) for p in (BASE/campaign/'_state/workers').glob('*.json') if '.resources.' not in p.name])
            if time.time()-last_plot > 300:
                try:
                    refresh_training()
                    result=subprocess.run(['python', str(ROOT/'scripts/plot_hs6_five_scales_selected29_20261020.py')],
                        capture_output=True,text=True,timeout=180)
                    state['plot_returncode']=result.returncode
                    if result.returncode:state['plot_error']=result.stderr[-2500:]
                    else:save(BASE/'last_plot.json',dict(updated_unix=time.time(),summary=result.stdout))
                except Exception as exc:state['plot_error']=repr(exc)
                last_plot=time.time()
            save(BASE/'status.json',state)
            time.sleep(15)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','watch','sync','launch'])
    args=parser.parse_args()
    {'prepare':prepare,'watch':watch,'sync':sync_hxw,'launch':launch}[args.mode]()
