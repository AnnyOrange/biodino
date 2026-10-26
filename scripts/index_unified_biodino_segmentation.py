#!/usr/bin/env python3
"""Reuse only selected BioDINO anchors' audited primary-last dense results and caches."""
import csv,json,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';REPORT=ROOT/'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
from dinov3.eval.bio_frozen_eval.external_fm_protocol import validate_best_head

def main():
 anchors={'biodino_1tb_splus':('hs6_splus','9224'),'biodino_1tb_b':('hs6_b','11274'),'biodino_1tb_l':('hs6_l','15374'),'biodino_1tb_hplus':('hs6_hplus','15374'),'biodino_5tb_l':('hs6_l_5tb_no_gram','26351')}
 candidates=set()
 for model,(alias,step) in anchors.items():
  cells=json.loads((REPORT/'v4_models'/f'{alias}.json').read_text())['checkpoints'].get(step,{}).get('cells',{}).get('segmentation',{})
  for d,cell in cells.items():
   for obs in cell.get('observations',[]):
    for path in obs.get('source','').split(';'):
     if path and 'primary-last' in path:candidates.add((model,d,path))
 for name in ['paired_component_scores.csv','full_v4_extension_scores.csv']:
  for r in csv.DictReader((REPORT/'20tb_route2_all_checkpoints_v4_20261012'/name).open()):
   if r.get('checkpoint')=='74663' and r.get('family')=='segmentation' and r.get('metric')=='mDice' and r.get('status')=='VALID_COMPLETE':candidates.add(('biodino_20tb_l',r['dataset'],r['source']))
 cells={};rejected=[]
 for model,d,path in sorted(candidates):
  p=Path(path)
  if not p.exists():continue
  cell=next((x for x in p.parents if '__segmentation__' in x.name),None)
  if cell is None:continue
  manifest=cell/'invocation_manifest.json';audit=cell/'validation_report.json'
  if not manifest.exists() or not audit.exists():rejected.append({'model':model,'path':path,'reason':'Missing source invocation/validation record'});continue
  inv=json.loads(manifest.read_text());val=json.loads(audit.read_text())
  if val.get('status')!='VALID_COMPLETE':continue
  if d=='monuseg' and val.get('expected_counts')!={'train':30,'val':7,'test':14}:
   rejected.append({'model':model,'path':path,'reason':'Historical MoNuSeg membership not common30/7/14'});continue
  split=cell.name.split('__')[-1];key=(model,d,split)
  if key in cells:continue
  result_paths=sorted((cell/'results').rglob('results.json'));valid=[]
  for r in result_paths:
   raw=json.loads(r.read_text());meta=raw.get('_meta',{})
   try:validate_best_head(meta)
   except (KeyError,ValueError):continue
   if meta.get('probe_batch_size')!=32 or meta.get('learning_rate')!=1e-3:continue
   digest=hashlib.sha256(r.read_bytes()).hexdigest()
   recorded=val.get('result_sha256',{}).get(str(r))
   # Mirrors retain original absolute paths in the validation map; match exact content hash.
   if digest not in val.get('result_sha256',{}).values():continue
   valid.append({'seed':meta['seed'],'budget':meta['probe_epochs'],'value':raw['test']['mDice'],'source':str(r),'sha256':digest})
  if not valid:continue
  caches={}
  for part in ['train','val','test']:
   found=list((cell/'cache').rglob(f'{d}_{part}_*last1*.npz'))
   if len(found)==1:caches[part]=str(found[0])
  cells[key]={'model':model,'dataset':d,'split':split,'cell':str(cell),'invocation':str(manifest),'validation':str(audit),'existing':valid,'caches':caches,'cache_ready':len(caches)==3}
 payload={'cells':list(cells.values()),'rejected':rejected,'scope':'Fixed selected 1TB/5TB/20TB anchors only; primary-last; existing validation-result hashes and best-val-head rules checked'}
 (OUT/'biodino_segmentation_inventory.json').write_text(json.dumps(payload,indent=2)+'\n')
 print('valid cells',len(cells),'cached cells',sum(c['cache_ready'] for c in cells.values()),'existing seed results',sum(len(c['existing']) for c in cells.values()),'rejected',len(rejected))
if __name__=='__main__':main()
