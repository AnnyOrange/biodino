#!/usr/bin/env python3
"""Minute watchdog, result-only HXW synchronization, and explicit coverage."""
import concurrent.futures
import csv
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/02_eval_runs/screening45_8k_20261024'


def ensure(record):
    code = '''import json,subprocess,socket,os,time,signal
from pathlib import Path
r=RECORD
campaign=Path(r['command'][r['command'].index('--campaign')+1])
status=campaign/'_state/workers'/(socket.gethostname()+'.json')
d=json.loads(status.read_text()) if status.exists() else {}
pid=d.get('pid')
if pid:
 try:
  cmd=Path('/proc',str(pid),'cmdline').read_bytes()
  if b'run_screening45_8k_20261024.py' in cmd and b'--task' not in cmd:
   age=time.time()-d.get('updated',0)
   if age<180:
    print(json.dumps({'action':'ALIVE','status':d,'heartbeat_age_seconds':age,'remote_time':time.time()}));raise SystemExit
   # Kill only the stalled supervisor; separately launched evaluators survive.
   os.kill(pid,signal.SIGTERM)
   for _ in range(20):
    try:
     if not Path('/proc',str(pid),'cmdline').read_bytes():break
    except FileNotFoundError:break
    time.sleep(.1)
 except FileNotFoundError:pass
if d.get('status')=='ALL_COMPLETE':
 print(json.dumps({'action':'ALL_COMPLETE','status':d}));raise SystemExit
Path(r['log']).parent.mkdir(parents=True,exist_ok=True)
with open(r['log'],'a') as stream:
 p=subprocess.Popen(r['command'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'action':'STARTED','pid':p.pid,'previous':d}))
'''.replace('RECORD', repr(record))
    try:
        p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6', record['host'],
                            'python3 -c ' + shlex.quote(code)], capture_output=True, text=True, timeout=20)
        return dict(host=record['host'], result=json.loads(p.stdout), stderr=p.stderr[-500:])
    except Exception as e:
        return dict(host=record['host'], error=repr(e))


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / 'watchdog.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (BASE / 'STOP_AUTOMATION').exists():
            return
        records = json.loads((BASE / 'deployment.json').read_text())
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            hosts = list(pool.map(ensure, records))
        remote = next((r for r in records if r['site'] == 'hxw'), None)
        sync = None
        if remote:
            campaign = remote['command'][remote['command'].index('--campaign') + 1]
            mirror = BASE / 'hxw_results'
            mirror.mkdir(exist_ok=True)
            command = ['rsync', '-a', '--timeout=20', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=6',
                       '--exclude=cache/', '--exclude=features/', '--exclude=adapters/', '--exclude=_runtime/',
                       '--include=*/', '--include=*.json', '--include=*.log', '--include=*.csv', '--exclude=*',
                       f'{remote["host"]}:{campaign}/', str(mirror) + '/']
            try:
                r = subprocess.run(command, capture_output=True, text=True, timeout=40)
                sync = dict(returncode=r.returncode, error=r.stderr[-500:])
            except Exception as e:
                sync = dict(error=repr(e))
            # rsync intentionally preserves durable results. Running states are
            # ephemeral, so remove mirrored entries once their done record exists.
            for p in (mirror / '_state/running').glob('*.json'):
                if (mirror / '_state/done' / p.name).exists():
                    p.unlink()
            for p in (mirror / '_state/failures').glob('*.json'):
                if (mirror / '_state/done' / p.name).exists():
                    p.unlink()
                    continue
                archived = mirror / '_state/repair_20261026_archived_failures' / p.name
                if archived.exists():
                    old = json.loads(archived.read_text())
                    current = json.loads(p.read_text())
                    if old.get('failed_at', 0) >= current.get('failed_at', 0):
                        p.unlink()
        coverage = []
        score_rows = []
        curves = []
        blocked = []
        for site, dirname in [('main', 'main'), ('hxw', 'hxw_results')]:
            m = json.loads((BASE / site / 'campaign_manifest.json').read_text())
            root = BASE / dirname
            for asset in m['checkpoint_assets']:
                tasks = [t for t in m['tasks'] if (t['arm'], t['checkpoint']) == (asset['arm'], asset['checkpoint'])]
                scores = dict(asset.get('reused_scores', {}))
                provenance = {k: asset.get('reused_source', {}).get('path', 'historical') for k in scores}
                complete = 0
                failed = 0
                exhausted = 0
                for t in tasks:
                    p = root / '_state/done' / (t['key'] + '.json')
                    if p.exists():
                        d = json.loads(p.read_text())
                        if d.get('status') == 'VALID_COMPLETE':
                            complete += 1
                            scores.update(d['scores'])
                            provenance.update({k: str(p) for k in d['scores']})
                            continue
                    failure = root / '_state/failures' / (t['key'] + '.json')
                    if failure.exists():
                        problem = json.loads(failure.read_text())
                        failed += 1
                        if problem.get('attempts', 0) >= 3:
                            exhausted += 1
                            blocked.append(dict(site=site, task=t['key'], error=problem.get('error'),
                                                attempts=problem.get('attempts')))
                correction = m.get('multimodal_correction_authorized', False)
                mm_key = 'segmentation/multimodal_cellseg'
                complete45 = correction and len(scores) == 45 and mm_key in scores
                for key, value in scores.items():
                    family, dataset = key.split('/', 1)
                    score_rows.append(dict(site=site, arm=asset['arm'], checkpoint=asset['checkpoint'],
                        family=family, dataset=dataset, value=value, source=provenance[key]))
                if complete45:
                    family_scores = {}
                    for family, n in [('classification',25), ('regression',4), ('retrieval',4), ('clustering',4), ('segmentation',8)]:
                        values = [v for k,v in scores.items() if k.startswith(family+'/')]
                        if len(values) != n:
                            raise ValueError(f'Unexpected screening45 inventory: {family} {len(values)}')
                        family_scores[family] = 100 * sum(values) / len(values)
                    curves.append(dict(arm=asset['arm'], checkpoint=asset['checkpoint'],
                        protocol=m['corrected_protocol_id'], **family_scores,
                        overall=sum(family_scores.values())/5))
                coverage.append(dict(site=site, arm=asset['arm'], checkpoint=asset['checkpoint'],
                    validated_jobs=complete, expected_jobs=len(tasks), validated_metrics=len(scores),
                    reused_metrics=len(asset.get('reused_scores', {})),
                    failed_jobs=failed, exhausted_jobs=exhausted,
                    screening45_expected=45, complete_screening45=complete45,
                    withheld='' if correction else 'multimodal_cellseg_pending_user_decision',
                    multimodal_status='VALID_COMPLETE' if mm_key in scores else
                        ('QUEUED_CORRECTED_SPLIT' if correction else 'PENDING_USER_DECISION')))
        with (BASE / 'coverage.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=coverage[0])
            w.writeheader()
            w.writerows(coverage)
        for filename, rows, fields in [
            ('corrected_scores.csv', score_rows, ['site','arm','checkpoint','family','dataset','value','source']),
            ('corrected_curve.csv', curves, ['arm','checkpoint','protocol','classification','regression','retrieval','clustering','segmentation','overall'])]:
            with (BASE/filename).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader();w.writerows(rows)
        tmp = BASE / 'status.tmp.json'
        tmp.write_text(json.dumps(dict(updated=time.time(), hosts=hosts, hxw_result_sync=sync,
            coverage=coverage, attention_required=bool(blocked), blocked_tasks=blocked,
            note='Results/logs only. Corrected Multimodal and matched LC25000 reference fits; old mismatched scores excluded. Corrected curve includes complete 45-item points only. ACTIVE supervisor alone does not mean every pending task is progressing.'), indent=2))
        tmp.replace(BASE / 'status.json')
        print(json.dumps(dict(hosts=len(hosts), validated_jobs=sum(c['validated_jobs'] for c in coverage),
                             expected_jobs=sum(c['expected_jobs'] for c in coverage), sync=sync)))


if __name__ == '__main__':
    main()
