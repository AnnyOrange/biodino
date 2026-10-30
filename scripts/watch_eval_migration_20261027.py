"""Keep user-authorized migration workers on deepcad / 5090-zxr."""
import concurrent.futures
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/02_eval_runs/eval_migration_20261027'

def ensure(record):
    command=record['command']
    from unified_paper_gpu_policy import excluded
    gpu = int(command[command.index('--gpu')+1]) if '--gpu' in command else -1
    if excluded(record['host'], gpu):
        return dict(host=record['host'], result=dict(ident=record['ident'], status='DISABLED_BY_HOST_POLICY'))
    if any(Path(arg).name=='unified_paper_queue.py' for arg in command):
        def option(name,default):return command[command.index(name)+1] if name in command else default
        ident=f"{option('--host-label',record['host'])}_gpu{option('--gpu','-1')}_slot{option('--slot','0')}"
        from unified_paper_gpu_policy import worker_disabled,excluded
        if worker_disabled(ident) or excluded(record['host'],int(option('--gpu','-1'))):
            return dict(host=record['host'],result=dict(ident=record['ident'],status='DISABLED_BY_CAPACITY_POLICY'))
    code='''import json,os,subprocess,time
from pathlib import Path
r=RECORD
state=Path(r['state']);d=json.loads(state.read_text()) if state.exists() else {}
pid=d.get('pid');alive=False
if pid:
 try:
  a=Path('/proc',str(pid),'cmdline').read_bytes().split(b'\\0')
  ticks=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
  alive=all(s.encode() in a for s in r['identity']) and ticks==d.get('start_ticks')
 except OSError:pass
if alive:
 print(json.dumps(dict(ident=r['ident'],status='ALIVE',pid=pid)))
elif time.time()-d.get('started',0)<300:
 print(json.dumps(dict(ident=r['ident'],status='RESTART_BACKOFF',previous=d)))
else:
 with open(r['log'],'a') as f:
  p=subprocess.Popen(r['command'],cwd=r['cwd'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 ticks=Path('/proc',str(p.pid),'stat').read_text().rsplit(')',1)[1].split()[19]
 d=dict(pid=p.pid,start_ticks=ticks,started=time.time(),host=r['host'],command=r['command'])
 tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(state)
 print(json.dumps(dict(ident=r['ident'],status='STARTED',pid=p.pid)))
'''.replace('RECORD',repr(record))
    command=['python3','-c',code]
    if record['host']=='deepcad':command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=6','deepcad','python3 -c '+shlex.quote(code)]
    try:
        r=subprocess.run(command,capture_output=True,text=True,timeout=20)
        return dict(host=record['host'],result=json.loads(r.stdout),error=r.stderr[-500:])
    except Exception as e:return dict(host=record['host'],ident=record['ident'],error=repr(e))

def main():
    with (BASE/'watch.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        records=json.loads((BASE/'deployment.json').read_text())
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(ensure,records))
        p=BASE/'watch_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(updated=time.time(),workers=rows),indent=2));t.replace(p)
        print(json.dumps(rows))
if __name__=='__main__':main()
