"""Fresh common fits after author split/label correction; preserve old artifacts."""
import argparse,fcntl,json
from pathlib import Path
from admit_unified_biodino_spatial_gaps import ROOT,OUT,read,write
PROTOCOL='neurips22-author-public52-v2'
def run(names):
    ready=read(OUT/'multimodal_official_readiness.json');assert ready['protocol']==PROTOCOL
    with (OUT/'locks/model_admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);cfg=read(OUT/'config.json');cacheq=read(OUT/'segmentation_cache_queue.json');headq=read(OUT/'segmentation_queue.json');added=[]
        for model in cfg['models']:
            name=model['model']
            if name not in names or model.get('execution_host'):continue
            kind=model['kind'];path=OUT/('fm_segmentation_inventory.json' if kind=='fm' else 'biodino_segmentation_inventory.json');inventory=read(path)
            if any(c['model']==name and c['dataset']=='multimodal_cellseg' and c['split']==PROTOCOL for c in inventory['cells']):continue
            for c in inventory['cells']:
                if c['model']==name and c['dataset']=='multimodal_cellseg':c['superseded_protocol']=PROTOCOL
            folder=OUT/'segmentation_cache_recovery'/name/'multimodal_cellseg'/PROTOCOL;folder.mkdir(parents=True,exist_ok=True)
            recipe=folder/'unified_recipe.json';write(recipe,dict(model=model,dataset='multimodal_cellseg',split=PROTOCOL,expected_counts=ready['counts'],source_readiness=str(OUT/'multimodal_official_readiness.json'),template_invocation=str(OUT/'multimodal_official_readiness.json'),feature_readout='last spatial layer; published normalization/channel adapter',feature_batch_size=2,probe_batch_size=32,probe_budgets=[20,50],training_seeds=list(range(5)),seed_semantics='head initialization, dropout and shuffle; fixed cached members'))
            done=folder/'cache_complete.json';index=len(inventory['cells'])
            cachebase=folder/'cache'/'multimodal_cellseg'/name if kind=='fm' else folder/'cache'
            cell=dict(model=name,dataset='multimodal_cellseg',split=PROTOCOL,cell=str(folder),invocation=str(recipe),validation=str(OUT/'multimodal_official_readiness.json'),existing=[],caches={part:str(cachebase/f'{part}.npz') for part in ['train','val','test']},cache_ready=False,cache_completion=str(done),requires_fresh_training=True,result_family='segmentation_refit',correction_reason=ready['correction'],input_spec=dict(data_root_base='/mnt/huawei_deepcad/benchmark/segmentation',image_size=512,resize_mode='pad'),class_weight_mode='none')
            inventory['cells'].append(cell);write(path,inventory);key=f'{name}__spatial_cache__multimodal_cellseg__{PROTOCOL}'
            cacheq['tasks'].insert(0,dict(key=key,model=name,dataset='multimodal_cellseg',task='segmentation_cache_recovery',split=PROTOCOL,entrypoint=f'run_unified_paper_{"fm" if kind=="fm" else "biodino"}_segmentation_cache.py',args=['--cell-index',str(index)],claim_key=key,min_free_mib=max(8000,model['min_free_mib']),expected_peak_mib=6500 if name=='genbio_pathfm' else 4000,result=str(done),log=str(OUT/'logs'/f'{key}.log')))
            for budget in [20,50]:
                for seed in range(5):
                    key=f'{name}__segmentation__multimodal_cellseg__{PROTOCOL}__E{budget}_seed{seed}'
                    headq['tasks'].insert(0,dict(key=key,model=name,dataset='multimodal_cellseg',task='segmentation',split=PROTOCOL,fold='fixed',budget=budget,seed=seed,entrypoint='run_unified_paper_segmentation.py',args=['--inventory-file',str(path),'--cell-index',str(index),'--budget',str(budget),'--seed',str(seed)],requires=[str(done)],claim_key=f'segmentation_{name}_multimodal_cellseg_{PROTOCOL}',min_free_mib=3500,result=str(OUT/'results'/name/'segmentation_refit/multimodal_cellseg'/PROTOCOL/f'E{budget}'/f'seed{seed}'/'component_result.json'),log=str(OUT/'logs'/f'{key}.log')))
            added.append(name)
        write(OUT/'segmentation_cache_queue.json',cacheq);write(OUT/'segmentation_queue.json',headq)
        print('OFFICIAL MULTIMODAL ADMITTED',added,flush=True);return added
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);a=p.parse_args();run(a.models)
