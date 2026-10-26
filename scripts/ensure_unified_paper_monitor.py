"""Cron recovery of unified controllers; never interrupts evaluation jobs."""
import fcntl, json, os, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
PY='/home/lxy/miniconda3/envs/dinov3/bin/python'

def main():
    with (OUT/'locks/unified_monitor_watchdog.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        commands=[]
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:commands.append((int(path.parent.name),path.read_bytes().split(b'\0')))
            except OSError:pass
        states=[]
        for name in ['watch_unified_paper.py','watch_unified_native_fleet.py','snapshot_unified_paper_resources.py']:
            script=ROOT/'scripts'/name;live=[pid for pid,parts in commands if any(x==str(script).encode() or x==('scripts/'+name).encode() for x in parts)]
            if name=='snapshot_unified_paper_resources.py':live=[pid for pid,parts in commands if b'--watch'in parts and str(script).encode()in parts]
            log=OUT/'logs'/('maintenance.log' if name=='watch_unified_paper.py' else 'native_fleet_guard.log')
            if name=='snapshot_unified_paper_resources.py':log=OUT/'logs/resource_observer.log'
            if not live:
                args=[PY,'-u',str(script)]+(['--watch']if name=='snapshot_unified_paper_resources.py'else[])
                with log.open('a') as f:p=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                live=[p.pid];status='restarted'
            else:status='live'
            states.append(dict(script=str(script),pids=live,status=status,log=str(log)))
        p=OUT/'states/unified_monitor_watchdog.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'updated':time.time(),'controllers':states},indent=2)+'\n');tmp.replace(p)

if __name__=='__main__':main()
