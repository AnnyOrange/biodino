#!/usr/bin/env python3
"""Reproduce fixed selected-29 curves and matched six-family diagnostic deltas."""
import csv,json,math,statistics,hashlib,shutil
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/mnt/huawei_deepcad/dinov3')
OLD=ROOT/'plot/fig2/hs6_l5_5tb_selected29_gram_nogram_20260924'
OUT=OLD/'v2_update_20261008';OUT.mkdir(exist_ok=True)
REPORT=ROOT/'outputs/00_reports/deepcad_method_20260927/progress_20261008'
MODELS=['noGRAM','GRAM','global_cls','global_cls_w3','global_cls_slow2']
LABELS={'noGRAM':'no-GRAM','GRAM':'GRAM','global_cls':'CLS w=1','global_cls_w3':'CLS w=3','global_cls_slow2':'Slow CLS (EMA .9998)'}
COLORS=dict(zip(MODELS,['#747D88','#D19C33','#1875B9','#CC4F63','#268D78']))
FAMILIES=['classification','regression','retrieval','clustering','segmentation']
METRICS=dict(zip(FAMILIES,['macro_f1','spearman','map_at_5','nmi','mDice']))
TITLES=dict(zip(FAMILIES,['Classification · Macro-F1','Regression · Spearman','Retrieval · mAP@5','Clustering · NMI','Segmentation · mDice']))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
history=list(csv.DictReader((OLD/'per_dataset_source_evidence.csv').open()))
sets={f:sorted({r['dataset'] for r in history if r['family']==f}) for f in FAMILIES}
assert [len(sets[f]) for f in FAMILIES]==[16,2,4,2,5]
records={};conflicts=[]
def add(r):
 r=dict(r);r['checkpoint']=int(r['checkpoint']);r['value']=float(r['value'])
 key=tuple(r[k] for k in ['model','checkpoint','family','dataset','metric'])
 if key in records and abs(records[key]['value']-r['value'])>1e-9:
  conflicts.append({'key':key,'first':records[key],'second':r});return
 records.setdefault(key,r)
for r in history:
 if r['metric']=='mDice_E20_primary_last':r['source_metric']=r['metric'];r['metric']='mDice'
 r['model']={'5tb_no_gram':'noGRAM','5tb_gram12687':'GRAM'}[r['model']];r['source_host']='historical shared report';add(r)
new_inputs=[REPORT/'hxw_cells.json',REPORT/'lyx_cells.json',REPORT/'hxw_nogram_tail_cells.json']
for source_file in new_inputs:
 data=json.loads(source_file.read_text())
 if data['warnings']:raise RuntimeError(data['warnings'])
 for r in data['rows']:add(r)
if conflicts:
 (OUT/'conflicts.json').write_text(json.dumps(conflicts,indent=2))
 for conflict in conflicts:records.pop(tuple(conflict['key']),None)
 print(f'Excluded {len(conflicts)} conflicting metric cells; see conflicts.json')
def dump(name,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
dump('all_source_evidence.csv',list(records.values()))
steps=sorted({r['checkpoint'] for r in records.values()})
curve=[]
for m in MODELS:
 for ck in steps:
  vals=[]
  for fam in FAMILIES:
   rr=[records.get((m,ck,fam,ds,METRICS[fam])) for ds in sets[fam]]
   good=[r for r in rr if r]
   v=statistics.mean(r['value'] for r in good) if len(good)==len(rr) else float('nan')
   vals.append(v);curve.append(dict(model=m,checkpoint=ck,family=fam,value=v,n=len(good),required=len(rr)))
  curve.append(dict(model=m,checkpoint=ck,family='overall',value=statistics.mean(vals),n=sum(math.isfinite(v) for v in vals),required=5))
dump('selected29_curve.csv',curve)
lookup={(r['model'],r['checkpoint'],r['family']):r['value'] for r in curve}
for old in csv.DictReader((OLD/'task_family_curve.csv').open()):
 model={'5tb_no_gram':'noGRAM','5tb_gram12687':'GRAM'}[old['model']]
 family=old['task_family']
 if family not in FAMILIES:family='overall'
 assert abs(lookup[model,int(old['checkpoint']),family]-float(old['raw_macro_mean']))<1e-10,old
def trajectories(zoom):
 fig,axs=plt.subplots(2,3,figsize=(15,8.7))
 for ax,fam in zip(axs.flat,FAMILIES+['overall']):
  for m in MODELS:
   xs=[ck for ck in steps if (ck>=28791 if zoom else True) and (m in ['noGRAM','GRAM'] or ck>29279)]
   ys=[lookup[m,ck,fam] for ck in xs]
   ax.plot(xs,ys,label=LABELS[m],color=COLORS[m],lw=1.8,marker='o' if zoom else None,ms=3)
  ax.axvline(29279,color='#AAA',ls=':',lw=1)
  ax.axvline(35135,color='#BBB',ls='--',lw=.9)
  ax.set_title(TITLES.get(fam,'Equal mean of 5 families')+(f' ({len(sets[fam])} datasets)' if fam!='overall' else ' (29 datasets)'))
  ax.grid(alpha=.18);ax.set_xlabel('Training step');ax.ticklabel_format(axis='x',style='sci',scilimits=(3,3))
  if zoom:ax.set_xlim(28950,42000)
  ax.margins(y=.15)
 fig.legend(*axs.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=5,frameon=False)
 fig.suptitle('5 TB · fixed selected-29 evaluation'+(' · continuation after step 29,279' if zoom else ' · historical curves and new CLS methods'),fontsize=16,y=.985)
 fig.text(.5,.016,'Exploratory: 29 datasets were selected post hoc using test results. Fixed dataset sets; incomplete points are gaps.\nDotted: branch at 29,279. Dashed: last shared step at 35,135; no-GRAM continues to 41,479. No smoothing.',ha='center',fontsize=9,color='#555')
 fig.tight_layout(rect=[0,.06,1,.89])
 stem='selected29_continuation' if zoom else 'selected29_history_and_v2'
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{stem}.{ext}',dpi=190)
 plt.close(fig)
trajectories(False);trajectories(True)
def late_followup():
 fig,(ax,delta_ax)=plt.subplots(2,1,figsize=(13,8.2),gridspec_kw={'height_ratios':[1.85,1]})
 for m in MODELS:
  points=[(ck,lookup[m,ck,'overall']) for ck in steps if 29279<=ck<=41479 and math.isfinite(lookup[m,ck,'overall'])]
  if m not in ('noGRAM','GRAM'):points=[(ck,v) for ck,v in points if ck>29279]
  ax.plot([ck for ck,v in points],[v for ck,v in points],color=COLORS[m],lw=2.1,
          marker='o',ms=3.3,label=LABELS[m])
 ax.axvspan(35135,42000,color='#F2F4F5',zorder=-2)
 ax.axvline(35135,color='#777',ls='--',lw=1)
 ax.text(35300,.96,'Last matched step',transform=ax.get_xaxis_transform(),
         ha='left',va='top',fontsize=9,color='#555')
 for ck in (40015,41479):
  value=lookup['noGRAM',ck,'overall']
  ax.scatter([ck],[value],s=42,facecolor='white',edgecolor=COLORS['noGRAM'],lw=1.8,zorder=5)
 ax.text(36400,.7555,f'no-GRAM late points\n40,015: {lookup["noGRAM",40015,"overall"]:.6f}  |  41,479: {lookup["noGRAM",41479,"overall"]:.6f}',
         fontsize=9,color='#414952',va='top')
 ax.set(xlim=(29100,42000),ylabel='Equal mean of five task families',
        title='Selected-29 trajectory | same-step comparison ends at 35,135')
 ax.grid(alpha=.18)
 ax.legend(loc='lower left',ncol=3,frameon=False,fontsize=9)
 families=FAMILIES
 names=['Classification','Regression','Retrieval','Clustering','Segmentation']
 baseline=[lookup['noGRAM',35135,f] for f in families]
 late_rows=[dict(checkpoint=ck,family=f,baseline_checkpoint=35135,
                 baseline_score=lookup['noGRAM',35135,f],score=lookup['noGRAM',ck,f],
                 delta_pp=100*(lookup['noGRAM',ck,f]-lookup['noGRAM',35135,f]))
            for ck in (40015,41479) for f in families+['overall']]
 dump('nogram_late_family_deltas.csv',late_rows)
 for i,ck in enumerate((40015,41479)):
  changes=[100*(lookup['noGRAM',ck,f]-base) for f,base in zip(families,baseline)]
  x=[j+(-.19 if i==0 else .19) for j in range(len(families))]
  delta_ax.bar(x,changes,width=.35,color='#64737F' if i==0 else '#2C917D',
               label=f'no-GRAM {ck:,}')
  for xpos,change in zip(x,changes):
   delta_ax.annotate(f'{change:+.3f}',(xpos,change),xytext=(0,4 if change>=0 else -4),
                     textcoords='offset points',ha='center',va='bottom' if change>=0 else 'top',fontsize=8)
 delta_ax.axhline(0,color='#555',lw=.9)
 delta_ax.set_ylim(-.5,3.45)
 delta_ax.set_xticks(range(len(families)),names)
 delta_ax.set_ylabel('Change from no-GRAM 35,135 (pp)')
 delta_ax.set_title('Later no-GRAM changes by task family | each compared with its own 35,135 score')
 delta_ax.grid(axis='y',alpha=.17)
 delta_ax.legend(frameon=False,ncol=2,fontsize=9)
 fig.suptitle('5 TB continuation: the original no-GRAM run reaches 41,479',fontsize=16,y=.99)
 fig.text(.5,.012,'Five families weighted equally; fixed post-hoc selected-29 datasets. Later no-GRAM points have no same-step CLS/GRAM comparator. No interpolation.',
          ha='center',fontsize=9,color='#555')
 fig.tight_layout(rect=[0,.045,1,.945],h_pad=2.2)
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'selected29_late_followup.{ext}',dpi=190)
 plt.close(fig)
late_followup()
# Six-family comparison: same (step,dataset) cells across EVERY plotted model.
primary={'classification':'balanced_accuracy','regression':'r2','retrieval':'recall_at_1','clustering':'nmi','segmentation':'mDice','detection':'test_patch_f1'}
paired=[];summary=[]
for family,metric in primary.items():
 keys=[(ck,ds) for (m,ck,f,ds,mt) in records if m=='noGRAM' and ck>29279 and f==family and mt==metric and all((other,ck,f,ds,mt) in records for other in MODELS)]
 for ref in ['noGRAM','GRAM']:
  for m in MODELS[2:]:
   dd=[]
   for ck,ds in keys:
    v=records[m,ck,family,ds,metric]['value'];b=records[ref,ck,family,ds,metric]['value'];delta=100*(v-b);dd.append(delta)
    paired.append(dict(model=m,reference=ref,checkpoint=ck,family=family,dataset=ds,metric=metric,value=v,baseline=b,delta_x100=delta))
   summary.append(dict(model=m,reference=ref,family=family,metric=metric,n=len(keys),datasets=len({ds for ck,ds in keys}),steps=len({ck for ck,ds in keys}),mean_delta_x100=statistics.mean(dd) if dd else float('nan')))
dump('six_family_matched_cells.csv',paired);dump('six_family_matched_summary.csv',summary)
fig,axs=plt.subplots(2,3,figsize=(14,8))
for ax,(family,metric) in zip(axs.flat,primary.items()):
 for i,m in enumerate(MODELS[2:]):
  for j,ref in enumerate(['noGRAM','GRAM']):
   rr=next(r for r in summary if r['family']==family and r['model']==m and r['reference']==ref)
   ax.bar(i+(j-.5)*.34,rr['mean_delta_x100'],width=.31,color=COLORS[m],alpha=1 if j==0 else .4,hatch=None if j==0 else '//')
 ax.axhline(0,color='#777',lw=.8);ax.set_xticks(range(3),['CLS w=1','CLS w=3','Slow CLS']);ax.set_title(f'{family.title()} · {metric}\n{rr["n"]} paired cells / {rr["datasets"]} datasets');ax.set_ylabel('Score difference × 100');ax.grid(axis='y',alpha=.15)
fig.suptitle('5 TB · six-task trade-offs on identical available cells',fontsize=16,y=.985)
fig.text(.5,.035,'Solid: vs no-GRAM. Hatched: vs GRAM. Same cells for all three methods and both controls within each task.\nExploratory repeated checkpoints; no confidence interval or full-v4 completion claim. R² differences ×100 are not relative percentages.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.09,1,.95])
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'six_task_matched_deltas.{ext}',dpi=190)
plt.close(fig)
manifest={'created_utc':'2026-10-08','posthoc_test_selected':True,'selected_datasets':sets,'models':LABELS,'new_branch_step':29279,'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD/'per_dataset_source_evidence.csv',*new_inputs]},'aggregation':'Family fixed-set complete cases; overall equal-family mean; no interpolation. Six-task bars use intersection of available cells across all 5 models.','limitations':['No uncertainty estimate; repeated checkpoints are correlated.','Incomplete full-v4, especially CTC/OOD/RxRx3.','GRAM training recipe differs.','Selected-29 is posthoc test selected; not independent validation.']}
manifest['conflicting_metric_cells_excluded']=len(conflicts)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
addendum=ROOT/'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/5tb_selected29_late_20261008'
addendum.mkdir(exist_ok=True)
for stem in ('selected29_history_and_v2','selected29_continuation','selected29_late_followup'):
 for ext in ('png','pdf','svg'):
  shutil.copy2(OUT/f'{stem}.{ext}',addendum/f'{stem}.{ext}')
for name in ('selected29_curve.csv','nogram_late_family_deltas.csv','manifest.json'):
 shutil.copy2(OUT/name,addendum/name)
print(json.dumps(summary,indent=2));print('OUT',OUT)
