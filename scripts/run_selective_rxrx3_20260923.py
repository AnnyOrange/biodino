#!/usr/bin/env python3
"""Validated adapter to the existing full eligible-gene v4 RxRx3 evaluator."""
import argparse,csv,json,os,subprocess
from pathlib import Path
import selective_retention_eval_queue_20260923 as q
SRC=Path('/mnt/huawei_deepcad/dinov3_selective_retention_snapshot_20260923')
CACHE=q.REPO/'outputs/02_eval_inputs/formal_v3/rxrx3-core'
PROTOCOL='crispr-query-guide-plate-disjoint-all-eligible-genes-v1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--config',required=True);a=p.parse_args()
 meta=json.loads((CACHE/'metadata.json').read_text());rows=meta['rows']
 assert meta['protocol_id']==PROTOCOL and sum(r['split']=='query' for r in rows)==734 and sum(r['split']=='gallery' for r in rows)==734
 spec=json.loads((q.REPO/'Evaluation Rules/protocol_v4.json').read_text())
 assert q.sha(CACHE/'split_manifest.jsonl')==spec['retrieval_splits']['rxrx3-core']['manifest_sha256']
 out=q.ROOT/'retrieval/rxrx3-core'/a.arm;out.mkdir(parents=True,exist_ok=True)
 manifest={'protocol_id':'bio-eval-union-v4','checkpoint':a.checkpoint,'checkpoint_sha256':q.sha(a.checkpoint),
           'config':a.config,'config_sha256':q.sha(a.config),'fixed_split_protocol_id':PROTOCOL,
           'batch_size':64,'teacher_branch':True,'source':str(SRC),'query':734,'gallery':734}
 q.atomic(out/'campaign_manifest.json',manifest)
 cmd=['/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python',str(SRC/'scripts/run_external4_fixedbudget_model.py'),
      '--model',a.arm,'--campaign',str(out),'--cache-root',str(CACHE.parent),'--datasets','rxrx3',
      '--checkpoint',a.checkpoint,'--train-config',a.config,'--device','cuda:0','--batch-size','64']
 subprocess.run(cmd,cwd=SRC,check=True)
 data=json.loads((out/'models'/a.arm/'results.json').read_text());res=data['tests']['rxrx3']
 assert data['status']=='VALID_COMPLETE' and res['status']=='FORMAL' and not res['proxy']
 assert data['batch_size']==64 and data['teacher_branch']=='teacher'
 assert res['protocol_id']==PROTOCOL and res['n_query']==734 and res['n_gallery']==734
 row={'dataset':'rxrx3-core','task':'retrieval/clustering','arm':a.arm,'protocol':PROTOCOL,
      'result_file':str(out/'models'/a.arm/'results.json'),'status':'VALID_COMPLETE'}
 with (out/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
if __name__=='__main__':main()
