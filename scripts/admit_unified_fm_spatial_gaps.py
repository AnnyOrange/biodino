"""Admit missing established FM segmentation cells using the common probe."""
import argparse,fcntl,json
from pathlib import Path
from admit_unified_paper_20tb_best import ROOT,OUT,read,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dataset',required=True);a=p.parse_args()
    with (OUT/'locks/model_admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);cfg=read(OUT/'config.json');spec=next(m for m in cfg['models'] if m['model']==a.model);assert spec['kind']=='fm'
        source=read(OUT/'biodino_segmentation_inventory.json')['cells'];templates=[c for c in source if c['model']=='biodino_1tb_l' and c['dataset']==a.dataset];assert templates
        path=OUT/'fm_segmentation_inventory.json';inventory=read(path);known={(c['model'],c['dataset'],c['split']) for c in inventory['cells']};cacheq=read(OUT/'segmentation_cache_queue.json');headq=read(OUT/'segmentation_queue.json');added=[]
        for ref in templates:
            ds,split=ref['dataset'],ref['split']
            if (a.model,ds,split) in known:continue
            invocation=read(Path(ref['invocation']));cmd=invocation['command'];arg=lambda k:cmd[cmd.index(k)+1]
            folder=OUT/'segmentation_cache_recovery'/a.model/'segmentation'/ds/split;folder.mkdir(parents=True,exist_ok=True);recipe=folder/'unified_recipe.json';expected=read(Path(ref['validation']))['expected_counts']
            write(recipe,dict(model=spec,dataset=ds,split=split,expected_counts=expected,template_invocation=ref['invocation'],template_validation=ref['validation'],feature_readout='last spatial layer; existing published-model normalization/channel/size adapter',feature_batch_size=2,probe_batch_size=32,probe_budgets=[20,50],training_seeds=[0,1,2,3,4],seed_semantics='head initialization, dropout and shuffle; fixed cached members'))
            done=folder/'cache_complete.json';index=len(inventory['cells']);cell=dict(model=a.model,dataset=ds,split=split,cell=str(folder),invocation=str(recipe),validation=ref['validation'],existing=[],cache_ready=False,cache_completion=str(done),input_spec=dict(data_root_base=arg('--data-root-base'),image_size=int(arg('--feature-img-size')),resize_mode=arg('--resize-mode')),class_weight_mode=arg('--probe-class-weight-mode'),caches={part:str(folder/'cache'/ds/a.model/f'{part}.npz') for part in ['train','val','test']})
            inventory['cells'].append(cell);key=f'{a.model}__spatial_cache__{ds}__{split}'
            # Published224/B2 spatial extraction has measured NVML peaks below
            # 6GiB for OpenPath/GenBio and below0.5GiB for DistillPath/MorphEm.
            # Global/full-resolution model estimates do not describe this task.
            cacheq['tasks'].insert(0,dict(key=key,model=a.model,dataset=ds,task='segmentation_cache_recovery',split=split,entrypoint='run_unified_paper_fm_segmentation_cache.py',args=['--cell-index',str(index)],claim_key=key,min_free_mib=8000 if a.model in ['openpath','genbio_pathfm'] else 6000,expected_peak_mib=7000 if a.model in ['openpath','genbio_pathfm'] else 2048,result=str(done),log=str(OUT/'logs'/f'{key}.log')))
            for budget in [20,50]:
                for seed in range(5):
                    key=f'{a.model}__segmentation__{ds}__{split}__E{budget}_seed{seed}'
                    headq['tasks'].insert(0,dict(key=key,model=a.model,dataset=ds,task='segmentation',split=split,fold=split if ds=='pannuke' else 'fixed',budget=budget,seed=seed,entrypoint='run_unified_paper_segmentation.py',args=['--inventory-file',str(path),'--cell-index',str(index),'--budget',str(budget),'--seed',str(seed)],requires=[str(done)],claim_key=f'segmentation_{a.model}_{ds}_{split}',min_free_mib=3500,result=str(OUT/'results'/a.model/'segmentation'/ds/split/f'E{budget}'/f'seed{seed}'/'component_result.json'),log=str(OUT/'logs'/f'{key}.log')))
            added.append(dict(model=a.model,dataset=ds,split=split,cell_index=index,head_fits=10))
        write(path,inventory);write(OUT/'segmentation_cache_queue.json',cacheq);write(OUT/'segmentation_queue.json',headq);write(OUT/'corrections'/f'{a.model}__{a.dataset}__spatial_gaps.json',{'added':added});print(json.dumps(added))

if __name__=='__main__':main()
