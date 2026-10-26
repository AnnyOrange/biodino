#!/usr/bin/env python3
"""Update fixed paper CSV/Excel entry with scores, provenance and honest gaps."""
import csv,json,math,statistics,os
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
REPORT=ROOT/'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'

def csvread(path):
 return list(csv.DictReader(path.open())) if path.exists() else []

def main(publish=True):
 cfg=json.loads((OUT/'config.json').read_text());queue=json.loads((OUT/'queue.json').read_text());rows={};seeds=[]
 models={m['model']:m for m in cfg['models']}
 def add(m,t,d,budget='',fold='fixed',**kw):
  key=(m,t,d,str(budget),str(fold))
  base=rows.get(key,dict(model=m,checkpoint=models[m]['checkpoint'],task=t,dataset=d,budget=budget,fold=fold,metric='',scale='0..1',direction='higher',value='',sd='',ci_low='',ci_high='',display='未启动',required_seeds=0,completed_seeds=0,data_complete='尚未逐项核对',status='待执行',reason='未启动；待读取匹配结果及样本输出',job_id='',log='',result='',predictions='',protocol='bio-eval-union-v4',seed_semantics='not_applicable'))
  base.update(kw);rows[key]=base;return base
 for model in models:
  for task in cfg['tasks']:
   t=task['task'];d=task['dataset'];budgets=[20,50] if t=='segmentation' else [''];folds=range(5) if t=='cell_tracking' else task.get('fold_ids',['fixed'])
   for b in budgets:
    for f in folds:
     add(model,t,d,b,f,required_seeds=task['required_seeds'],seed_semantics=task.get('seed_semantics','task_specific'),protocol=task['protocol'],reason=task.get('implementation_status','未启动；待读取匹配结果及样本输出'))
 # Existing v4 component evidence: deterministic scores remain single solves.
 alias={'biodino_1tb_splus':('hs6_splus','9224'),'biodino_1tb_l':('hs6_l','15374'),'biodino_1tb_hplus':('hs6_hplus','15374'),'biodino_5tb_l':('hs6_l_5tb_no_gram','26351')}
 for m in models:
  name,step=alias.get(m,('fm_'+m,'pretrained'))
  p=REPORT/'v4_models'/f'{name}.json'
  if not p.exists():continue
  d=json.loads(p.read_text())['checkpoints'].get(step,{}).get('cells',{})
  for task,cells in d.items():
   for dataset,cell in cells.items():
    value=cell.get('preferred')
    if not value or task not in ['classification','regression','retrieval','clustering']:continue
    if dataset=='lc25000' and task=='classification':continue
    if value.get('excluded_reason') or value.get('superseded_by_split'):continue
    metric=value['metric'];v=float(value['value'])
    add(m,task,dataset,metric=metric,value=v,display=f'{v:.6f}（CI待计算）',result=value['source'],status='待执行',reason='已有单次指标；需样本级输出与分组统计',completed_seeds=1 if task=='clustering' else 0,data_complete='历史结果已运行；成员待与共享清单对齐',direction='lower' if metric in ['mae','rmse'] else 'higher',scale='native' if metric in ['mae','rmse'] else '0..1')
 summaries=[('fm14_detection_multiseed_20261016/seed_summary.csv','detection_proxy','test_patch_f1','mean_f1','sd_f1'),('fm14_segmentation_five_seed_20261019/seed_summary.csv','segmentation','mDice','mean_mDice','sd_mDice'),('new_fm_v4_matched_20261016/detection_seed_summary.csv','detection_proxy','test_patch_f1','mean','sd'),('new_fm_v4_matched_20261016/segmentation_seed_summary.csv','segmentation','mDice','mean','sd')]
 for relative,task,metric,mean_key,sd_key in summaries:
  p=ROOT/'outputs/02_eval_runs'/relative
  for r in csvread(p):
   m=r['model'];d=r['dataset']
   if m not in models:continue
   if not r.get(mean_key):continue
   mean=float(r[mean_key]);sd=float(r[sd_key]);n=int(r['n_seeds']);budget=r.get('budget','');split=r.get('split_protocol',r.get('split','fixed'))
   fold=split if d=='pannuke' else 'fixed'
   row=add(m,task,d,budget,fold,metric=metric,value=mean,sd=sd,display=f'{mean:.6f} ± {sd:.6f}'+('' if n==5 else f'（{n}/5）'),required_seeds=5,completed_seeds=n,scale='0..100' if task=='detection_proxy' else '0..1',data_complete='是（已有完整训练记录）',status='待执行',reason='五次训练已完成；样本预测及配对统计尚缺' if n==5 else f'已完成 {n}/5 seed；余下训练和统计待执行',result=str(p),protocol=split,seed_semantics='training_randomness')
   for seed in range(5):
    if r.get(f'seed{seed}') not in (None,''):seeds.append({'model':m,'checkpoint':models[m]['checkpoint'],'task':task,'dataset':d,'budget':budget,'fold':fold,'seed':seed,'metric':metric,'value':float(r[f'seed{seed}']),'source':str(p)})
 # Native readiness only; missing implementations are not resource queues.
 for key,row in rows.items():
  if row['task']=='cell_tracking':row.update(data_complete='是（20域11706文件已核对）',status='待执行',reason='FM原生HoVerNet适配与五fold×五训练seed尚未运行；不能用计数或proxy替代')
  if row['dataset']=='lc25000' and row['task']=='classification':row.update(status='待执行',reason='Published original-tile groups and fixed group split verified; shared deterministic refit pending. Legacy image-random scores excluded.',display='未完成：共同分组协议待运行')
  if row['dataset']=='nucls_qc':row.update(status='待执行',data_complete='是（QC1744，发布方fold及患者隔离已核对）',reason='共享混合标注适配器已通过全量几何及原生指标回代检查；自身模型试跑后批量接入；仅实际入队项标为已排队')
  if row['dataset']=='xray_bone':row.update(status='阻塞',data_complete='否',reason='缺官方validation图像及年龄标签；train12611/test200已在本地')
  if row['model']=='openpath' and row['dataset']=='pcam':row.update(display='不作独立泛化比较：预训练重叠',value='',status='阻塞',reason='已知预训练重叠；排除排名与显著性检验')
 for task in queue['tasks']:
  if '__groupfix_v3' not in task['key']:continue
  m,d=task['model'],task['dataset'];result=Path(task['result']);statepath=OUT/'states'/f"{task['key']}.json"
  state=json.loads(statepath.read_text()) if statepath.exists() else {}
  row=add(m,'ood',d,metric='auroc',data_complete='固定名单已锁定；逐模型运行时校验',required_seeds=0,seed_semantics='not_applicable',log=task['log'],result=str(result))
  if result.exists():
   v=json.loads(result.read_text());score=v['metrics']['auroc'];row.update(value=score,display=f'{score:.6f}（CI待计算）',data_complete='是',status='待执行',reason='确定性指标与逐样本评分完成；统计待完成',predictions=v['prediction_path'])
   ci=OUT/'statistics/ood_groupfix_v3'/f'{m}__{d}__ci.json'
   if ci.exists():
    stat=json.loads(ci.read_text());metric=stat['metrics']['auroc'];lo,hi=metric['ci_low'],metric['ci_high'];row.update(ci_low=lo,ci_high=hi,display=f'{score:.6f} [{lo:.6f}, {hi:.6f}]',reason='确定性指标及分组CI完成；配对比较待其余模型',status='待执行')
  else:
   claim=OUT/'locks'/f'{m}.claimdir'/'owner.json'
   try:owner=json.loads(claim.read_text())
   except (FileNotFoundError,json.JSONDecodeError):owner={}
   if owner:
    row.update(status='运行中' if owner['task']==task['key'] else '已排队',reason='正在提取特征/评分' if owner['task']==task['key'] else '同一模型的前一任务运行中',job_id=f"{owner['host']}:{owner['pid']}")
   elif state.get('state')=='failed':row.update(status='阻塞',reason=state['error'],job_id=f"{state['host']}:{state['pid']}")
   else:row.update(status='已排队',reason='已进入运行中的持久队列；等待worker领取')
 # Explicit native CTC pilot is a real training process, not a completed fold.
 ctc=OUT/'states/ctc_pilot.json'
 if ctc.exists():
  d=json.loads(ctc.read_text());result=Path(d['result'])
  add(d['model'],'cell_tracking','ctc','',d['fold'],status='运行中' if not result.exists() else '待执行',completed_seeds=1 if result.exists() else 0,required_seeds=5,job_id=f"{d['host']}:{d['pid']}",log=d['log'],result=d['result'],reason='原生HoVerNet 50epoch、B8、冻结编码器；当前seed0，完成后仍需其余seed',data_complete='是（固定20域/5fold）')
 # Cross-model comparisons; missing family members are p=1 for conservative interim Holm.
 pairs=[]
 for p in list((OUT/'statistics').glob('*__vs__*.json'))+list((OUT/'statistics/ood_groupfix_v3').glob('*__vs__*.json'))+list((OUT/'statistics/native_box').glob('*__vs__*.json'))+list((OUT/'statistics/retrieval').glob('*__vs__*.json'))+list((OUT/'statistics/clustering').glob('*__vs__*.json'))+list((OUT/'statistics/nucls').glob('*__vs__*.json'))+list((OUT/'statistics/film').glob('*__vs__*.json')):
  d=json.loads(p.read_text())
  if d.get('task','ood')=='ood' and d.get('input_revision')!='cyclops_raw_float_percentile_0p5_99p5_v2':continue
  grouping_verified=d.get('primary_grouping_verified',d.get('primary_independent_groups_verified',not (d.get('task') in ['retrieval','clustering'] and d['dataset'] in ['nct-crc-he-1k','crc-val-he-7k','nct-crc-he-100','lc25000'])))
  if d['dataset']=='livecell'and d.get('task')in ['segmentation','detection_proxy']:
   from unified_paper_livecell_groups import VERSION
   grouping_verified=grouping_verified and d.get('grouping_version')==VERSION
  for metric,v in d['metrics'].items():pairs.append({**{k:d[k] for k in ['candidate','reference','dataset','n_test','n_groups']},'task':d.get('task','ood'),'fold':str(d.get('fold','fixed')),'budget':str(d.get('budget','')),'metric':metric,**v,'source':str(p),
   'test_definition':d.get('test_definition','See original statistical output'),'paired_test_method':d.get('paired_test_method','See original statistical output'),'primary_grouping_verified':grouping_verified,
   'statistical_scope':'Primary grouped comparison' if grouping_verified else 'Exploratory source-image conditional comparison; patient/slide mapping unavailable'})
 pairdf=pd.DataFrame(pairs)
 if not pairdf.empty:
  family=sum(m['kind']=='fm' for m in models.values())*sum(m['kind']=='biodino' for m in models.values())
  pairdf['p_holm']=float('nan');pairdf['correction_status']='interim; uncomputed planned tests treated as p=1'
  for _,g in pairdf.groupby(['task','dataset','budget','fold','metric'],dropna=False):
   g=g[pd.to_numeric(g['p_two_sided'],errors='coerce').notna()]
   if len(g)==family:pairdf.loc[g.index,'correction_status']='complete declared family'
   order=g.sort_values('p_two_sided').index;previous=0
   for rank,idx in enumerate(order):
    previous=max(previous,min(1.,float(pairdf.loc[idx,'p_two_sided'])*(family-rank)));pairdf.loc[idx,'p_holm']=previous
 frames={'paper_results':pd.DataFrame(rows.values()),'execution_status':pd.DataFrame(rows.values()),'per_seed_metrics':pd.DataFrame(seeds),'model_comparisons':pairdf,'task_inventory':pd.DataFrame(cfg['tasks']),'model_inventory':pd.DataFrame(cfg['models'])}
 if not publish:
  # The final builder enriches these rows in memory. Never replace the public
  # table/workbook with an incomplete intermediate snapshot during refresh.
  for name in ['model_comparisons','task_inventory','model_inventory']:
   tmp=REPORT/f'{name}.tmp';frames[name].to_csv(tmp,index=False);tmp.replace(REPORT/f'{name}.csv')
  return frames
 wide=frames['paper_results'].pivot_table(index=['task','dataset','budget','fold','metric','scale'],columns='model',values='display',aggfunc='first',fill_value='未启动/见状态表').reset_index()
 frames['paper_results_wide']=wide
 for name,frame in frames.items():
  tmp=REPORT/f'{name}.tmp';frame.to_csv(tmp,index=False);tmp.replace(REPORT/f'{name}.csv')
 tmp=REPORT/'paper_results.tmp.xlsx'
 with pd.ExcelWriter(tmp,engine='openpyxl') as writer:
  for name,frame in frames.items():frame.to_excel(writer,sheet_name=name[:31],index=False)
  notes=pd.DataFrame({'notes':['训练随机性报告样本标准差(ddof=1)，不以seed作为独立测试样本。','确定性任务不适用训练seed；CI为2000次按来源分层的独立组bootstrap，条件于固定ID bank。','X-ray按发布的sample_id；Cryo按4个project；BloodMNIST缺donor ID，仅能按image ID，限制需保留。','主要指标值按scale列解释；缺项绝不补0。','历史checkpoint经回顾性test选点，既定权重上的条件比较不消除选择偏差。','五fold不等于五seed；CTC原生追踪和检测proxy分栏。','撤回24域survey及变动test的LC25000重复不进入本表。','新旧结果的原始路径保留，暂无样本输出的旧指标明确标统计待完成。']});notes.to_excel(writer,sheet_name='Notes',index=False)
 tmp.replace(REPORT/'paper_results.xlsx')
 counts=frames['execution_status'].status.value_counts().to_dict();summary={'generated_utc':datetime.now(timezone.utc).isoformat(),'rows':len(rows),'status_counts':counts,'ood_metrics_completed':len(list((OUT/'results').glob('*/ood_groupfix_v3/*_result.json'))),'legacy_ood_excluded_runs':len(list((OUT/'results').glob('*/ood/*_result.json'))),'configuration':str(OUT/'config.json'),'queue':str(OUT/'queue.json')}
 (REPORT/'unified_execution_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n');print(summary,flush=True)
if __name__=='__main__':
 # Keep scheduled refreshes on the same audited tables used by Fig. 3 / Fig. 4.
 import runpy
 runpy.run_path(str(ROOT/'plot/build_paper_figure_tables.py'),run_name='__main__')
