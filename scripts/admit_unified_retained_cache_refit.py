"""Refit an affected cell after retained-head GPU reproduction fails.

Keep historical fits and rejected predictions, and reuse the exact frozen caches.
"""
import argparse, fcntl, json, time
from pathlib import Path
from admit_unified_paper_20tb_best import ROOT, OUT, read, write

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dataset',required=True);p.add_argument('--split',required=True);p.add_argument('--inventory-file',type=Path,default=OUT/'biodino_segmentation_inventory.json');a=p.parse_args()
    with (OUT/'locks/model_admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=a.inventory_file;inventory=read(path)
        index=next(i for i,c in enumerate(inventory['cells']) if (c['model'],c['dataset'],c['split'])==(a.model,a.dataset,a.split));cell=inventory['cells'][index]
        if cell.get('requires_fresh_training'):return
        evidence=[]
        for f in (OUT/'results'/a.model/'segmentation'/a.dataset/a.split).glob('E*/seed*/prediction_recovery_audit.json'):
            audit=read(f)
            if audit['inference_device'].startswith('cuda') and not audit['pass']:evidence.append({'path':str(f),'audit':audit})
        assert evidence,'A failed GPU oracle is required before refitting'
        assert all(Path(f).is_file() for f in cell['caches'].values())
        correction=OUT/'corrections'/f'{a.model}__{a.dataset}__{a.split}__retained_cache_refit.json'
        write(correction,{'created':time.time(),'historical_cell':cell,'evidence':evidence,'reason':'Original retained head fails fixed 1e-6 reproduction even on GPU. Preserve all old outputs; refit all five seeds for each budget using unchanged frozen caches, optimizer, initialization/shuffle semantics and validation-only head selection. No tolerance relaxation or seed selection.','changes_to_training_protocol':False})
        cell.update(requires_fresh_training=True,result_family='segmentation_refit',refit_evidence=str(correction));write(path,inventory)
        def affected(j):return (j['model'],j['dataset'],j.get('split'))==(a.model,a.dataset,a.split)
        for name in ['segmentation_recovery_queue','segmentation_queue']:
            q=OUT/(name+'.json');jobs=read(q);jobs['tasks']=[j for j in jobs['tasks'] if not affected(j)];write(q,jobs)
        q=OUT/'segmentation_queue.json';jobs=read(q)
        for budget in [20,50]:
            for seed in range(5):
                key=f'{a.model}__segmentation_refit__{a.dataset}__{a.split}__E{budget}_seed{seed}'
                jobs['tasks'].append(dict(key=key,model=a.model,dataset=a.dataset,task='segmentation',budget=budget,seed=seed,split=a.split,fold=a.split if a.dataset=='pannuke' else 'fixed',cell_index=index,entrypoint='run_unified_paper_segmentation.py',args=['--inventory-file',str(path),'--cell-index',str(index),'--budget',str(budget),'--seed',str(seed)],result=str(OUT/'results'/a.model/'segmentation_refit'/a.dataset/a.split/f'E{budget}'/f'seed{seed}'/'component_result.json'),requires=list(cell['caches'].values()),log=str(OUT/'logs'/f'{key}.log'),claim_key='segmentation_'+a.model+'_'+a.dataset+'_'+a.split,min_free_mib=3500,execution_mode='fresh_downstream_training_after_retained_head_gpu_mismatch'))
        write(q,jobs);print('ADMITTED',a.model,a.dataset,a.split,'10 fresh fits; original outputs retained')

if __name__=='__main__':main()
