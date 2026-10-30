"""Strict same-55-cell aggregation; no imputation, no partial-family grand mean."""
import csv,fcntl,hashlib,json,math,os,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/seven_family_100tb_1pb_20261029'
DEST=Path('/home/lxy/100tb_1pb_seven_family_completion_20261029')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)
def build():
 cfg=read(OUT/'config.json');inv=read(OUT/'biodino_segmentation_inventory.json')['cells']
 reference=Path('/home/lxy/20tb_expanded_tasks_20261029/all55_scores.csv');refrows=list(csv.DictReader(reference.open()));sets={}
 for row in refrows:sets.setdefault(row['family'],[]).append(row['dataset'])
 assert [len(v) for v in sets.values()]==[25,4,7,7,7,2,3]
 evidence=[];rows=[]
 def checked(path,m,f,d,seed=None):
  r=read(path);assert r['model']==m and r['dataset']==d
  if seed is not None:assert r['seed']==seed
  assert Path(r['prediction_path']).is_file()
  manifest=r.get('canonical_manifest',r.get('sample_manifest',r.get('manifest')))
  assert manifest and Path(manifest).is_file()
  expected=r.get('sample_manifest_sha256',r.get('manifest_sha256'))
  if expected:assert sha(manifest)==expected
  if f=='classification':
   if d=='lc25000':assert r['protocol']=='lc25000-clean-original-tile-80-20-seed0-v1' and r['split_sha256']=='5d7db052ca78639514cff2776c4af659e07d51f95b14aa2be2bb635d75cdbafa'
   value=100*r['macro_auc' if d=='chestmnist' else 'balanced_accuracy']
  elif f=='regression':value=100*r['r2']
  elif f in ['retrieval','clustering']:value=100*r['metrics']['recall_at_1' if f=='retrieval' else 'nmi']
  elif f=='segmentation':
   meta=r['_meta'];assert r['budget']==meta['probe_epochs']==20 and meta['probe_batch_size']==32
   assert meta['test_evaluations']==1 and meta['selection_metric']=='val_mIoU' and r['sample_manifest_status']=='verified'
   if d=='multimodal_cellseg':assert r['split']=='neurips22-author-public52-v2'
   value=100*r['test']['mDice']
  elif f=='native_detection':
   assert r['epochs']==10 and r['batch_size']==2
   selection=read(r['selection']);assert selection['test_used'] is False and selection['config_sha256']==sha(OUT/'native_detection_config.json')
   value=100*r['metrics']['bbox_ap_50_95']
  else:
   assert r['epochs']==5 and r['batch_size']==8
   value=r['test_patch_f1']
  assert math.isfinite(value)
  evidence.append(dict(model=m,family=f,dataset=d,seed=seed,value=value,path=str(path),sha256=sha(path)))
  return value
 for spec in cfg['models']:
  m=spec['model'];base=OUT/'results'/m;values={};missing=[]
  for f,datasets in sets.items():
   values[f]={}
   for d in datasets:
    try:
     if f in ['classification','regression']:v=checked(base/'global'/d/'component_result.json',m,f,d)
     elif f in ['retrieval','clustering']:
      b=read(base/'retrieval_clustering'/d/'component_result.json')
      if f=='retrieval':v=checked(b['retrieval_result'],m,f,d)
      else:
       paths={read(p)['seed']:p for p in b['clustering_results']};assert set(paths)==set(range(5))
       v=statistics.mean(checked(paths[s],m,f,d,s) for s in range(5))
     elif f=='segmentation':
      cells=[c for c in inv if c['model']==m and c['dataset']==d];assert len(cells)==(3 if d=='pannuke' else 1)
      v=statistics.mean(statistics.mean(checked(base/c['result_family']/d/c['split']/'E20'/f'seed{s}'/'component_result.json',m,f,d,s) for s in range(5)) for c in cells)
     else:
      folder='native_box' if f=='native_detection' else f
      v=statistics.mean(checked(base/folder/d/f'seed{s}'/'component_result.json',m,f,d,s) for s in range(5))
     values[f][d]=v
    except (OSError,KeyError,AssertionError,ValueError,TypeError) as error:missing.append(dict(family=f,dataset=d,reason=f'{type(error).__name__}: {error}'))
  means={f:statistics.mean(values[f].values()) for f in sets if len(values[f])==len(sets[f])}
  rows.append(dict(model=m,checkpoint=spec['checkpoint'],step=spec['step'],dataset_scores=values,family_scores=means,complete_cells=sum(map(len,values.values())),required_cells=55,complete=not missing,seven_family_score=statistics.mean(means.values()) if not missing else None,missing=missing))
 references=[]
 for label in [k for k in refrows[0] if k not in ['family','dataset']]:
  means={family:statistics.mean(float(r[label]) for r in refrows if r['family']==family) for family in sets}
  references.append(dict(model=label,family_scores=means,seven_family_score=statistics.mean(means.values()),complete_cells=55,complete=True))
 result=dict(time=time.time(),definition='55 fixed task-dataset cells; within-dataset five prescribed seeds and PanNuke three folds, then within-family dataset mean, finally seven equal family means. Incomplete families and grand means remain null; no OOD/CTC.',reference_csv=str(reference),reference_sha256=sha(reference),reference_scores=references,scores=rows,evidence=evidence,selection='Fixed broad-coverage historical checkpoints; not proof of each scale best. Comparison scope/weighting was explored after observing prior test results.')
 write(DEST/'scores.json',result);write(OUT/'scores.json',result)
 with (DEST/'scores.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['family','dataset',*[r['model'] for r in rows]])
  for family,datasets in sets.items():
   for d in datasets:w.writerow([family,d,*[r['dataset_scores'][family].get(d,'') for r in rows]])
 with (DEST/'comparison.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['method/checkpoint',*sets,'complete_cells','seven_family_score'])
  for r in references+rows:w.writerow([r['model'],*[r['family_scores'].get(k,'') for k in sets],r['complete_cells'],r['seven_family_score'] if r['complete'] else ''])
 lines=['# 100TB / 1PB 七类补测','',time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),'', '与此前1TB/5TB/20TB表使用同一55项；七类等权。缺项留空，所有规定重复完成并校验后才计算总分。','', '| checkpoint | 已完整任务单元 | 七类分数 |','|---|---:|---:|']
 for r in references+rows:lines.append(f"| {r['model']} | {r['complete_cells']}/55 | {format(r['seven_family_score'],'.4f') if r['complete'] else '未齐，不计算'} |")
 lines+=['','分割E20、原生框检测E10、检测代理E5；规定的seed0–4平均，PanNuke先合并三折。修正版LC25000和Multimodal。保留旧头必须通过原分数重现，否则相同协议重训。','', '这两个checkpoint是既有覆盖较齐的固定代表点，不是新七类最佳点。结果未出齐前不推断100TB/1PB的七类均值。', '',f'执行状态：{OUT}/status.json；逐项缺口、结果路径及SHA256见scores.json。']
 (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps([{k:r[k] for k in ['model','complete_cells','seven_family_score']} for r in rows]))
def main():
 DEST.mkdir(parents=True,exist_ok=True)
 with (DEST/'report.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  build()
if __name__=='__main__':main()
