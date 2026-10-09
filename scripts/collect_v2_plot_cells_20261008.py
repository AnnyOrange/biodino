#!/usr/bin/env python3
"""Read resident result JSON only; canonical row choice and validated dense fits."""
import argparse,hashlib,json,math,statistics,time,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',choices=['lyx','hxw'],required=True);p.add_argument('--arms',nargs='+');p.add_argument('--min-checkpoint',type=int,default=29280);p.add_argument('--max-checkpoint',type=int,default=35135);p.add_argument('--output',type=Path);a=p.parse_args()
root=Path('/data/xuzijing/hs6_l5_v2_recovery_eval_20260930' if a.site=='lyx' else '/data/hs6_l5_v2_recovery_eval_20260930')
arms=['global','global_local','global_cls','global_slow','global_w03','global_cls_slow','global_cls_slow2','global_cls_w3','global_cls_slow2_w3','global_cls_slow2_w03','global_cls_w03','global_cls_early','gram_ext']
roots={arm:root/arm for arm in arms}
if a.arms:roots={arm:root/arm for arm in a.arms}
if a.site=='hxw':roots['noGRAM']=Path('/data/hs6_l_5tb_nogram_eval_20260921')
rows=[];warnings=[]
def emit(model,ck,fam,ds,metric,value,paths,protocol):
 if isinstance(value,(int,float)) and math.isfinite(value):
  rows.append(dict(model=model,checkpoint=ck,family=fam,dataset=ds,metric=metric,value=float(value),source=';'.join(str(x) for x in paths),source_host=a.site,sha256=';'.join(hashlib.sha256(x.read_bytes()).hexdigest() for x in paths),protocol=protocol))
for arm,r in roots.items():
 model='GRAM' if arm=='gram_ext' else arm
 for f in (r/'results').glob('point_*/*/bio_*/*/*/last_result.json'):
  ck=int(f.parent.name)
  if not a.min_checkpoint<=ck<=a.max_checkpoint:continue
  ds=f.parent.parent.name;fam=f.parent.parent.parent.name[4:];obj=json.loads(f.read_text())
  if fam in ('classification','regression'):
   for metric in (['macro_f1','balanced_accuracy','macro_auc'] if fam=='classification' else ['spearman','r2']):emit(model,ck,fam,ds,metric,obj.get(metric),[f],obj.get('split',''))
  elif fam=='retrieval':
   entries=obj.get('rows',[obj])
   for family,metric in [('retrieval','map_at_5'),('retrieval','recall_at_1'),('clustering','nmi')]:
    if ds=='hpa-subcellular':chosen=[x for x in entries if x.get('aggregation')==('global' if family=='retrieval' else 'location') and (family=='retrieval' or x.get('n_classes')==41)]
    elif ds=='rxrx1-cross':chosen=[x for x in entries if x.get('aggregation')==('global' if family=='retrieval' else 'global-perturbation')]
    else:chosen=[x for x in entries if x.get(metric) is not None]
    if len(chosen)==1:emit(model,ck,family,ds,metric,chosen[0].get(metric),[f],chosen[0].get('protocol',''))
    elif chosen:warnings.append(dict(model=model,ck=ck,dataset=ds,metric=metric,error='ambiguous rows'))
 folds={}
 for cell in (r/'v3/cells').glob('point_*__*'):
  parts=cell.name.split('__');ck=int(parts[0][6:]);ds=parts[1]
  if not a.min_checkpoint<=ck<=a.max_checkpoint or ds=='monuseg':continue
  report=cell/'validation_report.json'
  if not report.exists():continue
  v=json.loads(report.read_text())
  if v.get('status')!='VALID_COMPLETE':continue
  files=[]
  for rel,expected in v.get('result_sha256',{}).items():
   if '/budget20/' not in rel:continue
   f=cell/rel
   if hashlib.sha256(f.read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Validated result changed: {f}')
   x=json.loads(f.read_text());m=x.get('_meta',{})
   if m.get('probe_epochs')==20 and m.get('seed') in [0,1,2] and m.get('probe_batch_size')==32:files.append((f,x))
  if len(files)!=3 or {x['_meta']['seed'] for f,x in files}!={0,1,2}:continue
  if ds=='pannuke':folds.setdefault(ck,[]).append((parts[2],files));continue
  emit(model,ck,'segmentation',ds,'mDice',statistics.mean(x['test']['mDice'] for f,x in files),[report]+[f for f,x in files],'E20/B32/3seeds/primary-last/'+parts[2])
 for ck,ff in folds.items():
  if len(ff)!=3 or len({k for k,fs in ff})!=3:continue
  emit(model,ck,'segmentation','pannuke','mDice',statistics.mean(x['test']['mDice'] for k,fs in ff for f,x in fs),[f for k,fs in ff for f,x in fs],'E20/B32/3seeds/3rotations/primary-last')
 print('COLLECTED',a.site,arm,len(rows),file=sys.stderr,flush=True)
 for f in (r/'v4/detection_b8').glob('point_*/*/results_bio_detection.json'):
  ck=int(f.parent.parent.name[6:]);x=json.loads(f.read_text())
  if not a.min_checkpoint<=ck<=a.max_checkpoint or x.get('batch_size')!=8 or x.get('epochs')!=5 or x.get('seed')!=0:continue
  if x['dataset']=='conic' and x.get('conic_split_protocol')!='official-baseline-fold0-nested-v1':continue
  emit(model,ck,'detection',x['dataset'],'test_patch_f1',x.get('test_patch_f1',float('nan'))/100,[f],'B8/5ep/seed0')
result=json.dumps(dict(site=a.site,time=time.time(),rows=rows,warnings=warnings),separators=(',',':'))
if a.output:a.output.write_text(result+'\n')
else:print(result)
