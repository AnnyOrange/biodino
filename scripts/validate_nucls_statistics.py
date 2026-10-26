"""Reconstruct NuCLS metrics after true image resampling and paired swaps."""
import sys,time,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
import unified_paper_nucls_statistics as stats
from dinov3.eval.bio_detection.nucls import native_metrics
class Dataset:
 def __init__(self,records):self.records=records
 def __len__(self):return len(self.records)
def main():
 start=time.time();torch.set_num_threads(2)
 result=stats.OUT/'results/biodino_1tb_b/nucls_qc/fold2/seed0/component_result.json'
 full=stats.load([result]);raw=json.loads(result.read_text());payload=torch.load(raw['prediction_path'],map_location='cpu',weights_only=False)
 # Select diverse genuine masks and ambiguous annotations from actual FOVs.
 selection=np.linspace(0,len(payload['records'])-1,24,dtype=int)
 a={**payload,'records':[payload['records'][i] for i in selection],'predictions':[payload['predictions'][i] for i in selection],'sample_metrics':[payload['sample_metrics'][i] for i in selection]}
 b=copy.deepcopy(a)
 for p in b['predictions']:
  p.update({k:p[k][::2] for k in ['boxes','labels','scores']});p['mask_rle']=p['mask_rle'][::2]
 _,b['sample_metrics']=native_metrics(Dataset(b['records']),b['predictions'])
 ac,bc=stats.prepare(a),stats.prepare(b);rng=np.random.default_rng(17);checks=[]
 for mode in ['original','bootstrap','paired_swap','bootstrap_paired_swap']:
  indices=rng.integers(24,size=24) if 'bootstrap' in mode else np.arange(24)
  groups=np.array([r['group_id'] for r in a['records']]);unique,inverse=np.unique(groups,return_inverse=True)
  swaps=rng.integers(2,size=len(unique))[inverse].astype(bool) if 'swap' in mode else np.zeros(24,bool)
  records=[a['records'][i] for i in indices];preds=[b['predictions'][i] if swaps[i] else a['predictions'][i] for i in indices]
  expected,_=native_metrics(Dataset(records),preds);actual=stats.one_metric(ac,indices,bc,swaps)
  truth=np.array([expected[k] if expected[k] is not None else np.nan for k in stats.METRICS])
  assert np.allclose(actual,truth,atol=1e-12,rtol=0,equal_nan=True),(mode,actual,truth)
  checks.append({'mode':mode,'max_absolute_error':float(np.nanmax(np.abs(actual-truth)))})
 # A self-comparison must be exactly null, irrespective of numeric outcome.
 fa={**full,'caches':[ac],'ids':[r['sample_id'] for r in a['records']],'groups':[r['group_id'] for r in a['records']]}
 delta,p,valid=stats.permutation({2:fa},{2:fa},12,17)
 assert np.array_equal(delta,np.zeros(7)) and np.array_equal(p,np.ones(7))
 # Patient bootstrap indices must preserve every FOV in each selected cluster.
 groups=stats.group_members(fa)
 assert sorted(np.concatenate(groups).tolist())==list(range(24))
 counts=np.zeros((1,8));counts[0,6]=1
 empty={**ac,'counts':counts}
 # Undefined mask/classification metrics remain NaN, never fabricated zeros.
 null=stats.intervals(np.full((10,7),np.nan));assert all(x['ci_low'] is None for x in null)
 output={'status':'passed','full_test_images':len(full['ids']),'full_test_patients':len(set(full['groups'])),'reconstructed_full_values':full['values'][0].tolist(),'metric_order':stats.METRICS,'checks':checks,'self_comparison_p':p.tolist(),'patient_membership_partition_verified':True,'seconds':time.time()-start,'benchmark_result':False}
 stats.native.dump(stats.OUT/'validation/nucls_patient_statistics.json',output);print(json.dumps(output,indent=2))
if __name__=='__main__':main()
