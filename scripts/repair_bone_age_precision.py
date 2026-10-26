"""Recompute only legacy float32 Ridge fits from identical cached features."""
import argparse,fcntl,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';PY='/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 with (OUT/'locks/bone_age_precision.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  while True:
   for f in (OUT/'results').glob('*/bone_age/component_result.json'):
    raw=json.loads(f.read_text())
    if raw.get('solver_precision')=='float64':continue
    subprocess.run([PY,str(ROOT/'scripts/run_unified_paper_bone_age.py'),'--model',raw['model'],'--rescore'],check=True)
   if not a.watch:return
   time.sleep(60)
if __name__=='__main__':main()
