import json,time,pathlib
r=pathlib.Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/unified_paper');key='cytoself__native_box__bccd__seed0';attempt=r/'states'/f'{key}_attempts.json';result=r/'results/cytoself/native_box/bccd/seed0/component_result.json'
while not result.exists():
 if attempt.exists():
  a=json.loads(attempt.read_text())
  if a['attempts']>=2 and not (r/'locks/cytoself.claimdir').exists():
   old=r/'corrections/cytoself_native_prior_attempts.json';old.write_text(json.dumps(a,indent=2));a['attempts']=0;a['repair']='Shared invalid-training candidate policy implemented; retry preserves completed LR0.001 head and original failure evidence';t=attempt.with_suffix('.tmp');t.write_text(json.dumps(a,indent=2));t.replace(attempt);print('REQUEUED after shared policy fix',flush=True);break
 time.sleep(15)
