"""Idempotently add the already selected lw5 representative without replacing jobs."""
import copy, fcntl, hashlib, json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/02_eval_runs/unified_paper'
REPORT = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
MODEL = 'biodino_20tb_lw5_best'
BASE = 'biodino_20tb_l'
ASSET = ROOT / 'outputs/02_eval_runs/l20tb_selected29_20261019/assets/20tb_distill_lw5'

def read(p):
    return json.loads(p.read_text())

def write(p, value):
    temp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(p)

def admit_segmentation(spec):
    model=spec['model']
    from dinov3.eval.bio_frozen_eval.external_fm_protocol import validate_best_head
    inventory_path=OUT/'biodino_segmentation_inventory.json'
    inventory=read(inventory_path)
    known={(c['model'],c['dataset'],c['split']) for c in inventory['cells']}
    source=ROOT/'outputs/02_eval_runs/l20tb_selected29_20261019/cells'
    for cell in sorted(source.glob(spec.get('historical_cell_prefix','20tb_distill_lw5_ck79543' if model == MODEL else f"{model}_ck{spec['step']}")+'__segmentation__*')):
        _,_,dataset,_,split=cell.name.split('__')
        if (model,dataset,split) in known:continue
        invocation=read(cell/'invocation_manifest.json');validation=read(cell/'validation_report.json')
        assert invocation['checkpoint']['sha256']==spec['checkpoint_sha256']
        assert validation['status']=='VALID_COMPLETE'
        cmd=invocation['command']
        assert cmd[cmd.index('--probe-batch-size')+1]=='32'
        assert cmd[cmd.index('--probe-lr')+1]=='0.001'
        existing=[]
        for path in sorted((cell/'results').rglob('results.json')):
            raw=read(path);validate_best_head(raw['_meta'])
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest in validation['result_sha256'].values()
            existing.append(dict(seed=raw['_meta']['seed'],budget=raw['_meta']['probe_epochs'],
                                 value=raw['test']['mDice'],source=str(path),sha256=digest))
        caches={}
        for split_name in ['train','val','test']:
            found=list((cell/'cache').rglob(f'{dataset}_{split_name}_*last1*.npz'))
            if len(found)==1:caches[split_name]=str(found[0])
        inventory['cells'].append(dict(model=model,dataset=dataset,split=split,cell=str(cell),
            invocation=str(cell/'invocation_manifest.json'),validation=str(cell/'validation_report.json'),
            existing=existing,caches=caches,cache_ready=len(caches)==3))
    write(inventory_path,inventory)
    counts={}
    for recovery in [False,True]:
        name='segmentation_recovery_queue' if recovery else 'segmentation_queue'
        path=OUT/(name+'.json');queue=read(path);known_jobs={t['key'] for t in queue['tasks']};add=[]
        for index,cell in enumerate(inventory['cells']):
            if cell['model']!=model or not cell['cache_ready']:continue
            for budget in [20,50]:
                for seed in range(5):
                    old=any(x['budget']==budget and x['seed']==seed for x in cell['existing'])
                    if old!=recovery:continue
                    key=f"{model}__segmentation__{cell['dataset']}__{cell['split']}__E{budget}_seed{seed}"
                    if recovery:key+='__recover'
                    if key in known_jobs:continue
                    args=['--cell-index',str(index),'--budget',str(budget),'--seed',str(seed)]
                    if recovery:args.append('--recover-existing')
                    add.append(dict(key=key,model=model,task='segmentation',dataset=cell['dataset'],
                        split=cell['split'],budget=budget,seed=seed,cell_index=index,
                        claim_key='segmentation_'+model+'_'+cell['dataset']+'_'+cell['split'],
                        entrypoint='run_unified_paper_segmentation.py',min_free_mib=2500,args=args,
                        execution_mode='retained_head_prediction_recovery' if recovery else 'downstream_training',
                        result=str(OUT/'results'/model/'segmentation'/cell['dataset']/cell['split']/f'E{budget}'/f'seed{seed}'/'component_result.json'),
                        log=str(OUT/'logs'/f'{key}.log')))
        queue['tasks']=add+queue['tasks'];write(path,queue);counts[name]=len(add)
    return counts

def main():
    with (OUT/'locks/model_admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verified = read(ASSET/'verified_copy.json')
        for name, info in verified['files'].items():
            digest = hashlib.file_digest((ASSET/name).open('rb'), 'sha256').hexdigest()
            assert digest == info['sha256'], name
        config = read(OUT/'config.json')
        old = next(m for m in config['models'] if m['model'] == BASE)
        spec = dict(model=MODEL, checkpoint=str(ASSET/'teacher_checkpoint.pth'),
                    train_config=str(ASSET/'config.yaml'), kind='biodino', step=79543,
                    min_free_mib=8000, display_name='BioDINO 20TB lw5 distill best (79543)',
                    checkpoint_sha256=verified['files']['teacher_checkpoint.pth']['sha256'],
                    selection_note='Previously recorded best complete selected29 method point: lw5 distill 79543. Retrospective test-selected anchor; selection fixed before new unified runs, not a confirmatory independent checkpoint selection.',
                    selection_source=str(REPORT/'l20tb_scale_methods_29_45_20261020_manifest.json'))
        archive=OUT/'corrections/before_20tb_method_best_config.json'
        if not archive.exists():write(archive,config)
        old['display_name']='BioDINO 20TB no-GRAM (74663)'
        old['method']='no-GRAM mainline; fixed late anchor, not globally best across all method branches'
        if MODEL not in {m['model'] for m in config['models']}:config['models'].append(spec)
        config['statistics']['native_box_definition']=config['statistics']['native_box_definition'].replace('primary6x18','all configured BioDINO anchors x18')
        write(OUT/'config.json',config)
        count={}
        for name in ['queue','global_queue','detection_queue','native_detection_queue','nucls_queue','retrieval_queue']:
            path=OUT/(name+'.json');queue=read(path);existing={t['key'] for t in queue['tasks']};add=[]
            for task in queue['tasks']:
                if task['model']!=BASE:continue
                if name=='queue' and '--input-fix-v2' not in task['args']:continue
                raw=json.dumps(task).replace(old['checkpoint'],spec['checkpoint']).replace(old['train_config'],spec['train_config']).replace(BASE,MODEL)
                new=json.loads(raw)
                if new['key'] in existing:continue
                add.append(new);existing.add(new['key'])
            queue['tasks'].extend(add)
            if 'models' in queue and MODEL not in {m['model'] for m in queue['models']}:queue['models'].append(spec)
            write(path,queue);count[name]=len(add)
        count.update(admit_segmentation(spec))
        stats=read(OUT/'nucls_statistics_config.json')
        family=sum(m['kind']=='biodino' for m in config['models'])*sum(m['kind']=='fm' for m in config['models'])
        for key in ['primary_family','secondary_families','paired_test']:
            stats[key]=stats[key].replace('6 BioDINO anchors','7 BioDINO anchors').replace('108',str(family))
        write(OUT/'nucls_statistics_config.json',stats)
        record={'model':spec,'retained_no_gram_model':BASE,'retained_no_gram_step':74663,
                'admitted_at':time.time(),'new_jobs_this_invocation':count,'holm_family_size':family,
                'weight_transfer_performed':False,'artifact_verification':verified,
                'scope_note':'Both no-GRAM and method best retained. Other branch-specific best representatives await user scope steering; no historical checkpoint sweep.',
                'remaining_adapters':['segmentation cells absent from selected29 cache inventory','CTC awaits common native pilot completion','same unresolved dataset/group metadata as all models']}
        record['total_registered_jobs']={name:sum(j['model']==MODEL for j in read(OUT/(name+'.json'))['tasks']) for name in count}
        write(OUT/'scope_20tb_best.json',record);write(REPORT/'20tb_checkpoint_scope.json',record)
        print(json.dumps(record,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
