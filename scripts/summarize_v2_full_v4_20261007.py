#!/usr/bin/env python3
"""Summarize validated paired v4 extensions, keeping task metrics separate."""
import csv,json,statistics,time
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path('/mnt/huawei_deepcad/dinov3')
RUN=ROOT/'outputs/02_eval_runs/v2_full_v4_20261007'
OUT=ROOT/'outputs/00_reports/deepcad_method_20260927/progress_20261007'

def metrics(t):
 d=t.get('done',{});p=Path(d.get('path','/nonexistent'));ds=t['dataset'];fam=t['family']
 if fam=='ood':
  x=json.loads(p.read_text());return [('ood',ds,k,x[f'{ds}_ood_{k}']) for k in ('auroc','average_precision')]
 if fam=='cell_tracking':
  x=json.loads(p.read_text());rows=x['domain_rows'];return [('cell_tracking',ds,k,statistics.mean(float(r['ctc_metrics'][k]) for r in rows)) for k in ('TRA','SEG')]
 if fam=='detection_proxy':
  # center_probe writes F1 on a 0..100 scale; normalize before the common delta x100.
  x=json.loads(p.read_text());return [('detection_proxy',ds,'test_patch_f1',x['test_patch_f1']/100)]
 if d.get('type')=='summary_csv':
  rows=[r for r in csv.DictReader(p.open()) if not r.get('error')];assert len(rows)==1;x=rows[0]
  if ds=='rxrx3-core':x=json.loads(Path(x['result_file']).read_text())['tests']['rxrx3']
  result=[]
  for k in ('balanced_accuracy','r2','recall_at_1','nmi'):
   if x.get(k) not in (None,''):
    f='regression' if k=='r2' else 'clustering' if k=='nmi' else 'retrieval' if k=='recall_at_1' else 'classification'
    result.append((f,ds,k,float(x[k])))
  return result
 return []

def main():
 points={};failures=[];counts=Counter()
 for p in (RUN/'tasks').glob('*.json'):
  t=json.loads(p.read_text());s=RUN/'claims'/t['id']/'status.json'
  state=json.loads(s.read_text())['state'] if s.exists() else 'QUEUED';counts[state]+=1
  if state!='VALID_COMPLETE':continue
  try:
   for fam,ds,k,v in metrics(t):points[(t['asset']['arm'],int(t['asset']['checkpoint_id']),fam,ds,k)]=v
  except Exception as e:failures.append(dict(task=t['id'],error=str(e)))
 rows=[]
 for (arm,ck,fam,ds,k),v in points.items():
  other=('noGRAM20tb',ck,fam,ds,k)
  if arm=='cls_slow2_20tb' and other in points:rows.append(dict(checkpoint=ck,family=fam,dataset=ds,metric=k,baseline=points[other],method=v,delta_x100=(v-points[other])*100))
 groups=defaultdict(list)
 for r in rows:groups[(r['family'],r['dataset'],r['metric'])].append(r['delta_x100'])
 lines=['# v4 补测配对结果','',f'UTC {time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())}；执行状态：{dict(counts)}。仅纳入双方已验证的相同步数。','', '|任务|数据集|指标|配对数|平均 Δ×100|','|---|---|---|---:|---:|']
 for key,values in sorted(groups.items()):lines.append('|'+ '|'.join(key)+f'|{len(values)}|{statistics.mean(values):+.4f}|')
 lines+=['','LC25000 分类仍是 provisional；回归为 ΔR²×100。此表只补充原 20TB 共享组件报告，未宣称完整 v4 胜出。',f'解析异常：{failures}']
 OUT.mkdir(exist_ok=True);(OUT/'V4_EXTENSION_RESULTS.md').write_text('\n'.join(lines)+'\n')
 with (OUT/'v4_extension_pairs.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['checkpoint','family','dataset','metric','baseline','method','delta_x100']);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
