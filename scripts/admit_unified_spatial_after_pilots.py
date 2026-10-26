"""Expand missing spatial tasks only after a real matching-adapter head completes."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';PROTOCOL='neurips22-author-public52-v2'
def finished(model,dataset,split,family='segmentation'):
    for path in (OUT/'results'/model/family/dataset/split).glob('E*/seed*/component_result.json'):
        r=json.loads(path.read_text())
        if r.get('sample_manifest_status')=='verified' and r.get('_meta',{}).get('test_evaluations')==1 and r.get('_meta',{}).get('probe_epochs') in [20,50]:return str(path)
    return None
def run():
    models=json.loads((OUT/'config.json').read_text())['models'];new={'distillpath','morphem','genbio_pathfm','openpath'}
    legacy=finished('mae','tissuenet','formal-static-v1') or finished('dinov2','tissuenet','formal-static-v1')
    modern=finished('distillpath','multimodal_cellseg',PROTOCOL,'segmentation_refit')
    bio=finished('biodino_1tb_splus','multimodal_cellseg',PROTOCOL,'segmentation_refit');added=[]
    for m in models:
        if m['kind']!='fm' or m.get('execution_host'):continue
        gate=modern if m['model'] in new else legacy
        if not gate:continue
        cells=json.loads((OUT/'fm_segmentation_inventory.json').read_text())['cells']
        for dataset in ['livecell','pannuke','tissuenet']:
            if any(c['model']==m['model'] and c['dataset']==dataset for c in cells):continue
            proc=subprocess.run([sys.executable,str(ROOT/'scripts/admit_unified_fm_spatial_gaps.py'),'--model',m['model'],'--dataset',dataset],text=True,capture_output=True,check=True)
            added.append(dict(model=m['model'],dataset=dataset,pilot=gate,output=proc.stdout))
    names=[m['model'] for m in models if not m.get('execution_host') and (bio if m['kind']=='biodino' else modern if m['model'] in new else legacy and modern)]
    if names:
        from admit_unified_multimodal import run as admit
        admitted=admit(names);added.extend(dict(model=name,dataset='multimodal_cellseg',pilot=modern if name in new else bio if next(m for m in models if m['model']==name)['kind']=='biodino' else legacy) for name in admitted)
    state={'updated':time.time(),'pilots':{'fm14':legacy,'new_fm':modern,'biodino_official_multimodal':bio},'added':added,'policy':'Real prescribed head completion and ordered source-label validation before adapter-family expansion; fixed budget/seeds preserved'}
    p=OUT/'states/spatial_expansion.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p)
    return len(added)
if __name__=='__main__':print('SPATIAL CELLS ADDED',run())
