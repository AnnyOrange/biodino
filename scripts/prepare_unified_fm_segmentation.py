"""Index verified FM scores and surviving frozen caches/heads for prediction recovery."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';REPORT=ROOT/'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
def main():
    audit=json.loads((ROOT/'outputs/02_eval_runs/fm14_segmentation_five_seed_20261019/audit.json').read_text())['cells']
    fm={m['model'] for m in json.loads((OUT/'config.json').read_text())['models'] if m['kind']=='fm'};cells={}
    for row in csv.DictReader((REPORT/'paper_results.csv').open()):
        m,d=row['model'],row['dataset']
        if m not in fm or row['task']!='segmentation' or d not in ['cellpose','monuseg','conic']:continue
        paths=json.loads(row['raw_result_path']) if row['raw_result_path'].startswith('[') else []
        if not paths:continue
        source=Path(paths[0]);old=next((x for x in audit if x['model']==m and x['dataset']==d),None)
        if old:
            recipe_path=Path(old['recipe']);caches=old['caches'];split=old['split_protocol'];base=Path(old['source_root'])
        else:
            base=source.parents[2];recipe_path=source.with_name('recipe.json');split=base.name
            caches={s:str(base/'cache'/d/m/f'{s}.npz') for s in ['train','val','test']}
        recipe=json.loads(recipe_path.read_text());key=(m,d,split)
        cell=cells.setdefault(key,{'model':m,'dataset':d,'split':split,'cell':str(base),'invocation':str(recipe_path),'input_spec':{'data_root_base':'/mnt/huawei_deepcad/benchmark/segmentation','image_size':recipe['image_size'],'resize_mode':recipe['resize_mode']},'class_weight_mode':recipe['class_weight_mode'],'caches':caches,'existing':[],'recipe':str(recipe_path),'recipe_sha256':hashlib.sha256(recipe_path.read_bytes()).hexdigest()})
        for name in paths:
            p=Path(name);raw=json.loads(p.read_text());meta=raw['_meta'];budget=int(row['budget']);seed=int(meta['seed'])
            assert meta['probe_batch_size']==32 and meta['probe_epochs']==budget
            cell['existing'].append({'seed':seed,'budget':budget,'value':raw['test']['mDice'],'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'head_available':p.with_name('best_head.pth').is_file()})
    items=list(cells.values())
    for cell in items:
        cell['cache_ready']=Path(cell['caches']['test']).is_file();cell['cache_parts_present']={s:Path(p).is_file() for s,p in cell['caches'].items()}
        cell['ready_recovery_runs']=sum(r['head_available'] for r in cell['existing']) if cell['cache_ready'] else 0
    target=OUT/'fm_segmentation_inventory.json';target.write_text(json.dumps({'cells':items,'scope':'Existing18 FM; initially3 source-group-certified segmentation datasets; other tasks stay in canonical scope','method':'No additional training; recover fixed validation-selected heads with score reproduction and ordered-label verification'},indent=2)+'\n')
    print(json.dumps({'cells':len(items),'recoverable_runs':sum(c['ready_recovery_runs'] for c in items),'cache_missing':[(c['model'],c['dataset']) for c in items if not c['cache_ready']],'head_missing':sum(not r['head_available'] for c in items for r in c['existing'])}))
if __name__=='__main__':main()
