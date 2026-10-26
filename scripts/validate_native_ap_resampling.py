"""Check cached AP resampling against fresh official COCO matching/aggregation."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
from unified_paper_native_statistics import prepare,ap,METRICS,OUT,dump
from dinov3.eval.bio_detection.native import coco_bbox_metrics
def main():
 torch.set_num_threads(2)
 paths=[OUT/'results'/m/'native_box/bccd/seed0/test_predictions.pth' for m in ['biodino_1tb_b','cytoself']]
 a,b=[torch.load(p,map_location='cpu',weights_only=False) for p in paths];ca,cb=prepare(a),prepare(b);n=len(a['records']);rng=np.random.default_rng(20261021);checks=[];start=time.time()
 class Dataset:
  def __len__(self):return len(self.records)
 for mode in ['original','bootstrap','paired_swap','bootstrap_paired_swap']:
  indices=np.arange(n) if mode in ['original','paired_swap'] else rng.integers(n,size=n)
  swaps=rng.integers(2,size=n).astype(bool) if 'swap' in mode else np.zeros(n,bool)
  # Give each duplicated image a fresh evaluator ID through native evaluator.
  ds=Dataset();ds.records=[a['records'][i] for i in indices]
  preds=[(b if swaps[i] else a)['predictions'][i] for i in indices]
  official=coco_bbox_metrics(ds,preds,a['classes']);cached=ap(ca,indices,cb,swaps)
  expected=np.array([official[k] for k in METRICS]);assert np.allclose(cached,expected,atol=1e-12,rtol=0),(mode,cached,expected)
  checks.append({'mode':mode,'metrics':dict(zip(METRICS,cached.tolist())),'max_absolute_error':float(np.max(np.abs(cached-expected)))})
 result={'status':'passed','images':n,'checks':checks,'seconds':time.time()-start,'method':'Per-image COCO matches plus exact stable global score sorting and101point AP; duplicate images get independent matches; swapped groups retain all local detections','benchmark_result':False}
 dump(OUT/'validation/native_ap_resampling.json',result);print(result,flush=True)
if __name__=='__main__':main()
