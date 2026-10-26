"""Fix the secondary silhouette subset across fitted seeds; retain original records."""
import argparse,fcntl,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def dump(p,d):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');temp.replace(p)
def run():
 corrected=0
 for base in (OUT/'results').glob('*/retrieval_clustering/*'):
  files=sorted(base.glob('seed*/clustering_result.json'));pending=[p for p in files if json.loads(p.read_text()).get('silhouette_evaluation_seed')!=0]
  if not pending:continue
  first=json.loads(pending[0].read_text());x=np.load(first['feature_cache'])['features'].astype(np.float32)
  # Reproduce existing retrieval normalization followed by clustering normalization.
  x/=np.linalg.norm(x,axis=1,keepdims=True)+1e-12;x/=np.linalg.norm(x,axis=1,keepdims=True)+1e-12
  indices=np.random.RandomState(0).permutation(len(x))[:min(5000,len(x))];distance=cosine_distances(x[indices]);changes=[]
  for path in pending:
   raw=json.loads(path.read_text());prediction=np.load(raw['prediction_path']);cluster=prediction['cluster'];score=float(silhouette_score(distance,cluster[indices],metric='precomputed'))
   if raw['seed']==0:assert abs(score-raw['metrics']['silhouette_cosine'])<1e-6,(score,raw['metrics']['silhouette_cosine'])
   archive=path.with_name('clustering_result.original_metric_sampling.json')
   if not archive.exists():archive.write_bytes(path.read_bytes())
   changes.append({'seed':raw['seed'],'old_silhouette':raw['metrics']['silhouette_cosine'],'new_silhouette':score,'original':str(archive),'corrected':str(path)})
   raw['metrics']['silhouette_cosine']=score;raw['silhouette_evaluation_seed']=0;raw['seed_semantics']='MiniBatchKMeans training randomness: initialization and minibatch sampling; fixed silhouette evaluation subset seed0';raw['metric_sampling_correction']=str(base/'silhouette_subset_correction.json');dump(path,raw);corrected+=1
  note=base/'silhouette_subset_correction.json';previous=json.loads(note.read_text()) if note.exists() else {'changes':[]}
  previous.update(reason='Keep evaluation objects fixed across five clustering fits; original v4 tied secondary silhouette subsampling to training seed. NMI, ARI, cluster accuracy and every fitted cluster assignment remain exactly unchanged.',evaluation_seed=0,n_evaluation_samples=len(indices),member_index_path=str(base/'silhouette_evaluation_indices.npy'),updated=time.time())
  previous['changes']+=changes;np.save(base/'silhouette_evaluation_indices.npy',indices);dump(note,previous);print('FIXED SILHOUETTE SUBSET',first['model'],first['dataset'],len(changes),flush=True)
 return corrected
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 with (OUT/'locks/clustering_metric_sampling.local.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   count=run();dump(OUT/'states/clustering_metric_sampling.json',{'status':'watching' if a.watch else 'complete','corrected_this_pass':count,'updated':time.time()})
   if not a.watch:break
   time.sleep(30)
