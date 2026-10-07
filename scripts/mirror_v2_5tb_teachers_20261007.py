#!/usr/bin/env python3
"""Execute the explicitly approved 118-teacher transfer plan; retain source files."""
import argparse,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/huawei_deepcad/dinov3')
PLAN=ROOT/'outputs/00_reports/deepcad_method_20260927/progress_20261007/TRANSFER_PLAN.json'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-approved-plan',action='store_true');a=p.parse_args()
 plan=json.loads(PLAN.read_text())
 if not a.execute_approved_plan:print(json.dumps({k:v for k,v in plan.items() if k!='rows'},indent=2));return
 dest=ROOT/'outputs/02_eval_inputs/v2_5tb_teachers_20261007';dest.mkdir(parents=True,exist_ok=True)
 log=dest/'transfer_journal.jsonl'
 for row in plan['rows']:
  target=Path(row['target']);target.parent.mkdir(parents=True,exist_ok=True)
  remote_hash=subprocess.check_output(['ssh',row['host'],'sha256sum -- '+shlex.quote(row['path'])],text=True).split()[0]
  if target.exists():assert digest(target)==remote_hash
  else:
   temp=target.with_suffix('.pth.partial')
   subprocess.run(['rsync','-aL','--partial','--append-verify','--bwlimit=51200',row['host']+':'+row['path'],str(temp)],check=True)
   assert temp.stat().st_size==row['bytes'] and digest(temp)==remote_hash
   after=subprocess.check_output(['ssh',row['host'],'sha256sum -- '+shlex.quote(row['path'])],text=True).split()[0]
   assert after==remote_hash
   os.replace(temp,target)
  source_config=str(Path(row['path']).parents[2]/'config.yaml');config=target.parents[1]/'config.yaml'
  text=subprocess.check_output(['ssh',row['host'],'cat -- '+shlex.quote(source_config)])
  if config.exists():assert config.read_bytes()==text
  else:config.write_bytes(text)
  record=dict(row,sha256=remote_hash,config=str(config),time=time.time())
  with log.open('a') as f:f.write(json.dumps(record)+'\n')
  # Publish only hash-verified files; the full-v4 watcher admits each checkpoint.
  assets_path=ROOT/'outputs/02_eval_runs/v2_full_v4_20261007/5tb_assets.json'
  assets=json.loads(assets_path.read_text()) if assets_path.exists() else []
  resident='/data/xuzijing/hs6_l5_v2_recovery_eval_20260930' if row['host']=='5090-lyx-xr' else '/data/hs6_l5_v2_recovery_eval_20260930'
  asset=dict(arm='5tb_'+row['arm'],checkpoint_id=str(row['step']),path=str(target),config=str(config),kind='dinov3',model_id='',reserve_mib=4096,missing_only=True,resident_component_root=row['host']+':'+resident+'/'+row['arm'])
  if not any(x['path']==str(target) for x in assets):
   assets.append(asset);tmp=assets_path.with_suffix('.tmp');tmp.write_text(json.dumps(assets,indent=2)+'\n');tmp.replace(assets_path)
  print('VERIFIED',row['arm'],row['step'],flush=True)
 save=dest/'COMPLETE.json';save.write_text(json.dumps(dict(plan_sha256=digest(PLAN),count=len(plan['rows']),finished=time.time()),indent=2)+'\n')
if __name__=='__main__':main()
