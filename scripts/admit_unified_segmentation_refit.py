"""Refit missing-cache cells under the unchanged common frozen-probe recipe."""
import argparse,json,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def admit():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dataset',required=True);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
 evidence=json.loads(a.evidence.read_text());assert evidence['pass'] is False and evidence['inference_device'].startswith('cuda')
 inventory=OUT/'fm_segmentation_inventory.json';data=json.loads(inventory.read_text());index=next(i for i,c in enumerate(data['cells']) if (c['model'],c['dataset'])==(a.model,a.dataset));cell=data['cells'][index]
 assert 'original_test_cache' in cell,'Only admitted missing-cache recovery cells'
 correction=OUT/'corrections'/f'{a.model}__{a.dataset}__fresh_cache_refit.json'
 if correction.exists():return
 dump(correction,{'created':time.time(),'historical_cell':cell,'evidence':str(a.evidence),'reason':'Original feature cache unavailable; regenerated features cannot reproduce retained validation-selected head score within fixed1e-6 tolerance. Do not relax tolerance or select by old test score. Fresh five-seed heads use unchanged shared B32/E20/E50/AdamW/cosine/val-mIoU recipe on fixed regenerated split caches; original encoder remains frozen. Original metrics and rejected prediction artifacts preserved.','changes_to_training_protocol':False})
 source=OUT/'segmentation_cache_recovery'/cell['model']/'segmentation'/cell['dataset']/cell['split']
 cell.update(requires_fresh_training=True,result_family='segmentation_refit',refit_evidence=str(correction),caches={part:str(source/'cache'/cell['dataset']/cell['model']/f'{part}.npz') for part in ['train','val','test']})
 dump(inventory,data)
 # Keep obsolete attempts and logs, but do not dispatch further head-only recovery.
 q=OUT/'segmentation_recovery_queue.json';jobs=json.loads(q.read_text());jobs['tasks']=[j for j in jobs['tasks'] if (j['model'],j['dataset'])!=(a.model,a.dataset)];dump(q,jobs)
 q=OUT/'segmentation_cache_queue.json';jobs=json.loads(q.read_text());key=f'{a.model}__segmentation_cache__{a.dataset}__all_original_recipe'
 jobs['tasks'].insert(0,{'key':key,'model':a.model,'dataset':a.dataset,'task':'segmentation_cache_recovery','entrypoint':'run_new_fm_v4_segmentation_20261016.py','args':['--model',a.model,'--dataset',a.dataset,'--extract-cache-only','--output-root',str(OUT/'segmentation_cache_recovery')],'result':cell['caches']['val'],'log':str(OUT/'logs'/f'{key}.log'),'claim_key':'seg_cache_'+a.model,'min_free_mib':18000 if a.model=='genbio_pathfm' else 9000});dump(q,jobs)
 q=OUT/'segmentation_queue.json';jobs=json.loads(q.read_text());jobs['tasks']=[j for j in jobs['tasks'] if (j['model'],j['dataset'])!=(a.model,a.dataset)]
 for budget in [20,50]:
  for seed in range(5):
   key=f'{a.model}__segmentation_refit__{a.dataset}__E{budget}_seed{seed}'
   result=OUT/'results'/a.model/'segmentation_refit'/a.dataset/cell['split']/f'E{budget}'/f'seed{seed}'/'component_result.json'
   jobs['tasks'].append({'key':key,'model':a.model,'dataset':a.dataset,'task':'segmentation','budget':budget,'seed':seed,'fold':'fixed','split':cell['split'],'entrypoint':'run_unified_paper_segmentation.py','args':['--inventory-file',str(inventory),'--cell-index',str(index),'--budget',str(budget),'--seed',str(seed)],'result':str(result),'requires':list(cell['caches'].values()),'log':str(OUT/'logs'/f'{key}.log'),'claim_key':'seg_refit_'+a.model+'_'+a.dataset,'min_free_mib':7000,'execution_mode':'fresh_downstream_training_after_missing_cache'})
 dump(q,jobs);print('ADMITTED FRESH FROZEN HEADS',a.model,a.dataset,'10 fits; preserved original results')
if __name__=='__main__':
 with (OUT/'locks/model_admission.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);admit()
