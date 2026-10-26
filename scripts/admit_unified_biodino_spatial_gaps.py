"""Append missing model-specific caches and five common head fits per budget."""
import argparse, fcntl, json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'

def read(p):return json.loads(p.read_text())
def write(p,d):
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);args=parser.parse_args()
    with (OUT/'locks/model_admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cfg=read(OUT/'config.json');spec=next(m for m in cfg['models'] if m['model']==args.model);assert spec['kind']=='biodino'
        path=OUT/'biodino_segmentation_inventory.json';inventory=read(path)
        known={(c['model'],c['dataset'],c['split']) for c in inventory['cells']}
        cacheq=read(OUT/'segmentation_cache_queue.json');headq=read(OUT/'segmentation_queue.json');added=[]
        templates=[c for c in inventory['cells'] if c['model']=='biodino_1tb_l']
        for ref in sorted(templates,key=lambda c:(c['dataset']!='monuseg',c['dataset'])):
            ds,split=ref['dataset'],ref['split']
            if (args.model,ds,split) in known:continue
            source=read(Path(ref['invocation']));cmd=source['command'];arg=lambda k:cmd[cmd.index(k)+1]
            cell_dir=OUT/'segmentation_cache_recovery'/args.model/ds/split;cell_dir.mkdir(parents=True,exist_ok=True)
            invocation=cell_dir/'invocation_manifest.json';counts=read(Path(ref['validation']))['expected_counts']
            write(invocation,dict(model=spec,template_invocation=ref['invocation'],template_validation=ref['validation'],
                expected_counts=counts,split=split,policy='Same shared data loader, fixed split, normalization, frozen final spatial layer and common downstream recipe; feature microbatch2'))
            caches={part:str(cell_dir/'cache'/f'{part}.npz') for part in ['train','val','test']}
            done=cell_dir/'cache_complete.json';index=len(inventory['cells'])
            cell=dict(model=args.model,dataset=ds,split=split,cell=str(cell_dir),invocation=str(invocation),
                validation=ref['validation'],existing=[],caches=caches,cache_ready=False,cache_completion=str(done),
                input_spec={'data_root_base':arg('--data-root-base'),'image_size':int(arg('--feature-img-size')),'resize_mode':arg('--resize-mode')},
                class_weight_mode=arg('--probe-class-weight-mode'))
            inventory['cells'].append(cell)
            key=f'{args.model}__spatial_cache__{ds}__{split}'
            cacheq['tasks'].append(dict(key=key,model=args.model,dataset=ds,task='segmentation_cache_recovery',split=split,
                entrypoint='run_unified_paper_biodino_segmentation_cache.py',args=['--cell-index',str(index)],
                claim_key=f'spatial_cache_{args.model}_{ds}_{split}',min_free_mib=10000,result=str(done),
                log=str(OUT/'logs'/f'{key}.log')))
            for budget in [20,50]:
                for seed in range(5):
                    key=f'{args.model}__segmentation__{ds}__{split}__E{budget}_seed{seed}'
                    headq['tasks'].append(dict(key=key,model=args.model,dataset=ds,task='segmentation',split=split,
                        budget=budget,seed=seed,cell_index=index,entrypoint='run_unified_paper_segmentation.py',
                        args=['--cell-index',str(index),'--budget',str(budget),'--seed',str(seed)],requires=[str(done)],
                        claim_key=f'segmentation_{args.model}_{ds}_{split}',min_free_mib=2500,
                        result=str(OUT/'results'/args.model/'segmentation'/ds/split/f'E{budget}'/f'seed{seed}'/'component_result.json'),
                        log=str(OUT/'logs'/f'{key}.log')))
            added.append({'dataset':ds,'split':split,'cell_index':index,'cache_result':str(done),'head_fits':10})
        write(path,inventory);write(OUT/'segmentation_cache_queue.json',cacheq);write(OUT/'segmentation_queue.json',headq)
        write(OUT/'corrections'/f'{args.model}_spatial_admission.json',{'model':args.model,'added':added})
        print(json.dumps(added,indent=2))

if __name__=='__main__':main()
