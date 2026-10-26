"""Reload named analysis/controllers only at their verified idle polling sleep."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
ALLOWED={'prepare_unified_ai_rxrx1.py','watch_unified_paper.py','prepare_unified_center_probes.py','unified_paper_dense_statistics.py','admit_unified_additional_retrieval.py','admit_unified_hpa_retrieval.py','admit_unified_rxrx3.py','prepare_unified_rxrx1.py','unified_paper_clustering_statistics.py','unified_paper_retrieval_statistics.py','manage_unified_ai_branches.py','watch_unified_native_fleet.py'}
ALLOWED.add('repair_unified_duplicate_executions.py')
ALLOWED.add('refresh_unified_idle_workers.py')
ALLOWED.add('unified_paper_segmentation_statistics.py')
ALLOWED.add('snapshot_unified_paper_resources.py')
ALLOWED.add('unified_paper_clustering_statistics.py')
ALLOWED.add('unified_paper_global_statistics.py')
def main():
 p=argparse.ArgumentParser();p.add_argument('--pid',type=int,nargs='+',required=True);a=p.parse_args();pending={};done=[]
 for pid in a.pid:
  proc=Path(f'/proc/{pid}');args=[v.decode()for v in(proc/'cmdline').read_bytes().split(b'\0')if v];script=next((Path(v).name for v in args if Path(v).name in ALLOWED),None);assert script,(pid,args)
  pending[pid]=dict(args=args,script=script,start_ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19])
 while pending:
  for pid,item in list(pending.items()):
   proc=Path(f'/proc/{pid}')
   try:
    args=[v.decode()for v in(proc/'cmdline').read_bytes().split(b'\0')if v]
    if args!=item['args']or(proc/'stat').read_text().rsplit(')',1)[1].split()[19]!=item['start_ticks']:raise ProcessLookupError()
    if(proc/'wchan').read_text()!='hrtimer_nanosleep':continue
    children=(proc/'task'/str(pid)/'children').read_text().split()
    if children and item['script'].endswith('_statistics.py'):continue
   except (OSError,ProcessLookupError):done.append(dict(old_pid=pid,status='original_process_already_exited'));pending.pop(pid);continue
   os.kill(pid,signal.SIGTERM)
   for _ in range(100):
    if not(proc/'cmdline').exists()or not(proc/'cmdline').read_bytes():break
    time.sleep(.1)
   env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
   with(OUT/'logs'/f"{Path(item['script']).stem}_recovery.log").open('a')as f:child=subprocess.Popen(item['args'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   done.append(dict(old_pid=pid,new_pid=child.pid,script=item['script'],status='reloaded_at_idle_poll'));pending.pop(pid)
  state=OUT/'states/idle_service_reload.json';tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(dict(pid=os.getpid(),updated=time.time(),pending=list(pending),completed=done),indent=2));tmp.replace(state)
  if pending:time.sleep(5)
if __name__=='__main__':main()
