#!/usr/bin/env python3
"""Plot fixed posthoc 29-cell GRAM/no-GRAM curves in the original Fig2 style."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SEARCH=ROOT/'outputs/00_reports/v4_subset_scaling_fm14_search_20260924'
SELECT=SEARCH.with_name(SEARCH.name+'_both_refined')
OUT=ROOT/'plot/fig2/hs6_l5_5tb_selected29_gram_nogram_20260924'
FAMILIES=('classification','regression','retrieval','clustering','segmentation')
PANELS=FAMILIES+('five_family_mean',)
COLORS=dict(zip(PANELS,('#2563A6','#9A4EAE','#16807A','#D17A22','#3A8F5C','#C64B4B')))
LABELS=dict(zip(PANELS,('Classification','Regression','Retrieval','Clustering','Segmentation','Five-family mean')))
METRICS=dict(zip(PANELS,('Macro-F1','Spearman rho','mAP@5','NMI','mDice (E20, 3 seeds)','Descriptive equal-family mean')))
MODELS=('5tb_no_gram','5tb_gram12687')
NAMES={'5tb_no_gram':'no-GRAM','5tb_gram12687':'GRAM (anchor ck12687)'}

def read(p):return list(csv.DictReader(p.open()))

def save(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inventory=read(SELECT/'per_dataset_selection_audit.csv')
    chosen={(r['family'],r['dataset']) for r in inventory if r['selected']=='True'}
    assert len(chosen)==29
    counts=Counter(f for f,d in chosen)
    assert dict(counts)==dict(classification=16,regression=2,retrieval=4,clustering=2,segmentation=5)
    evidence=[r for r in read(SEARCH/'search_input_evidence.csv') if r['model'] in MODELS and (r['family'],r['dataset']) in chosen]
    points=defaultdict(dict)
    for r in evidence:
        key=r['family'],r['dataset'];pkey=r['model'],int(r['checkpoint'])
        assert key not in points[pkey]
        points[pkey][key]=float(r['value'])
    assert Counter(m for m,ck in points)==Counter({'5tb_no_gram':60,'5tb_gram12687':36})
    assert all(set(p)==chosen for p in points.values())
    curve=[]
    for (model,ck),p in sorted(points.items()):
        means={f:float(np.mean([v for (ff,d),v in p.items() if ff==f])) for f in FAMILIES}
        means['five_family_mean']=float(np.mean(list(means.values())))
        for f,v in means.items():
            curve.append(dict(model=model,checkpoint=ck,task_family=f,datasets=counts[f] if f in FAMILIES else 29,
                              metric=METRICS[f],raw_macro_mean=v))
    curves={(m,f):sorted([r for r in curve if r['model']==m and r['task_family']==f],key=lambda r:r['checkpoint']) for m in MODELS for f in PANELS}
    peaks=[]
    for m in MODELS:
        for f in PANELS:
            r=max(curves[m,f],key=lambda r:(r['raw_macro_mean'],-r['checkpoint']))
            peaks.append(dict(model=m,task_family=f,datasets=r['datasets'],peak_checkpoint=r['checkpoint'],
                              peak_value=r['raw_macro_mean'],endpoint_checkpoint=curves[m,f][-1]['checkpoint'],
                              endpoint_value=curves[m,f][-1]['raw_macro_mean']))
    pmap={(r['model'],r['task_family']):r for r in peaks}
    for expected in read(SELECT/'selected_models_task_means.csv'):
        m=expected['model']
        if m in MODELS:
            peak=pmap[m,'five_family_mean']
            assert peak['peak_checkpoint']==int(expected['checkpoint'])
            assert abs(peak['peak_value']-float(expected['selected_mean']))<1e-12
    save('task_family_curve.csv',curve);save('task_family_peaks.csv',peaks)
    save('selected_29_datasets.csv',[dict(family=f,dataset=d) for f,d in sorted(chosen)])
    save('per_dataset_source_evidence.csv',evidence)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#27313A',
                         'text.color':'#20272D','xtick.color':'#44515C','ytick.color':'#44515C',
                         'axes.titleweight':'bold'})
    ranges={}
    for f in PANELS:
        vals=[r['raw_macro_mean'] for m in MODELS for r in curves[m,f]]
        pad=max((max(vals)-min(vals))*.18,.001)
        ranges[f]=(min(vals)-pad,max(vals)+pad)
    for stem,models in [('nogram',MODELS[:1]),('gram',MODELS[1:]),('gram_vs_nogram',MODELS)]:
        overlay=len(models)==2
        fig,axes=plt.subplots(2,3,figsize=(15.5,8.2),sharex=True)
        for ax,f in zip(axes.flat,PANELS):
            for m in models:
                rr=curves[m,f];xx=[r['checkpoint'] for r in rr];yy=[r['raw_macro_mean'] for r in rr]
                gram=m==MODELS[1];color=COLORS[f]
                ax.plot(xx,yy,color=color,lw=1.8,ls='--' if gram and overlay else '-',alpha=.72 if gram and overlay else .94,zorder=2)
                ax.scatter(xx,yy,s=16,marker='s' if gram else 'o',facecolor='white' if gram and overlay else color,
                           edgecolor=color if gram and overlay else 'white',lw=.6 if gram and overlay else .35,zorder=3)
                peak=pmap[m,f];px=peak['peak_checkpoint'];py=peak['peak_value']
                starcolor='#923C81' if gram and overlay else '#E33D3D'
                ax.scatter([px],[py],marker='*',s=155,color=starcolor,edgecolor='white',linewidth=.9,zorder=5)
                if overlay:
                    ax.annotate(f'{"GRAM" if gram else "no-GRAM"} peak ck{px}',xy=(px,py),xycoords='data',
                                xytext=(.98,.10 if gram else .20),textcoords='axes fraction',ha='right',fontsize=8,
                                color=starcolor,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1),
                                arrowprops=dict(arrowstyle='-',color=starcolor,lw=.6))
                else:
                    dx=12 if px<19000 else -10
                    ax.annotate(f'peak ck{px}',(px,py),xytext=(dx,-20),textcoords='offset points',
                                ha='left' if dx>0 else 'right',fontsize=8.5,fontweight='bold',color='#9D2727',
                                arrowprops=dict(arrowstyle='-',color='#C85757',lw=.7))
            ax.axvline(12687,color='#687D97',ls=(0,(4,3)),lw=1.15,alpha=.75)
            if not overlay:ax.axvline(curves[models[0],f][-1]['checkpoint'],color='#7A858E',ls=':',lw=1,alpha=.8)
            ax.set_ylim(*ranges[f]);ax.set_xlim(0,31000)
            ax.set_xticks(np.arange(0,30001,5000),['0','5k','10k','15k','20k','25k','30k'])
            ax.set_title(f'{LABELS[f]}  |  {counts[f]} datasets' if f in FAMILIES else 'Overall  |  five tasks equally weighted',fontsize=11)
            ax.set_ylabel(METRICS[f]);ax.grid(axis='y',color='#D9DEE3',lw=.8,alpha=.75);ax.grid(axis='x',color='#EDF0F2',lw=.55,alpha=.7)
        for ax in axes[1]:ax.set_xlabel('5TB SSL checkpoint (optimizer updates)')
        title='GRAM vs no-GRAM' if overlay else NAMES[models[0]]
        fig.suptitle(f'HS6-L 5TB {title}: task-family checkpoint peaks',fontsize=17,fontweight='bold',y=.985)
        support='60 no-GRAM / 36 GRAM checkpoints' if overlay else f'{len(curves[models[0],PANELS[0]])} complete checkpoints'
        fig.text(.5,.941,f'Fixed 29-dataset subset (16 / 2 / 4 / 2 / 5); {support}; raw task means',ha='center',fontsize=10.5,color='#4B5862')
        handles=[]
        if overlay:
            handles.extend([Line2D([0],[0],color='#4B5862',lw=1.8,marker='o',label='no-GRAM'),
                            Line2D([0],[0],color='#4B5862',lw=1.8,ls='--',marker='s',markerfacecolor='white',label='GRAM')])
        handles.extend([Line2D([0],[0],color='#687D97',ls=(0,(4,3)),lw=1.2,label='GRAM resume / anchor ck12687'),
                        Line2D([0],[0],marker='*',color='none',markerfacecolor='#E33D3D',markersize=12,label='Task-family peak')])
        if not overlay:handles.append(Line2D([0],[0],color='#7A858E',ls=':',label=f'Endpoint ck{curves[models[0],PANELS[0]][-1]["checkpoint"]}'))
        fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.008))
        fig.text(.5,.053,'POSTHOC TEST-SELECTED SUBSET. Overall is descriptive; peaks use each branch\'s available checkpoints. No smoothing.',ha='center',fontsize=9,color='#68747D')
        fig.subplots_adjust(left=.07,right=.985,bottom=.13,top=.89,hspace=.30,wspace=.25)
        for ext in ('png','svg','pdf'):fig.savefig(OUT/f'hs6_l5_5tb_selected29_{stem}_task_family_raw_macro_curves.{ext}',dpi=200,facecolor='white')
        plt.close(fig)
    manifest=dict(selection_source=str(SELECT/'per_dataset_selection_audit.csv'),selection_sha256=hashlib.sha256((SELECT/'per_dataset_selection_audit.csv').read_bytes()).hexdigest(),
                  style_reference=str(ROOT/'outputs/00_reports/hs6_l5_5tb_task_peak_curves_20260914'),
                  family_counts=dict(counts),models={m:dict(checkpoints=sorted(ck for mm,ck in points if mm==m),count=sum(mm==m for mm,ck in points)) for m in MODELS},
                  gram_anchor=12687,posthoc_test_selected=True,smoothing=False,
                  overall='equal mean of five family means; not average of separate family peaks',same_y_axis_limits_for_each_panel_across_all_three_figures=True)
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    md=['# 5TB GRAM / no-GRAM：固定29项子集曲线','',
        '沿用20260914参考图的2×3布局、任务配色、折线/散点及峰值星标。前五格对应分类16、回归2、检索4、聚类2、分割5；第六格为五任务等权的描述性均值。', '',
        '数据集固定为上一轮29项方案，未为GRAM重新选择。指标：macro-F1、Spearman、mAP@5、NMI、E20 primary-last mDice；每个点所有29项齐全。所有图中同一任务使用相同y轴范围，无平滑。', '',
        'no-GRAM有60点，ck487–29279；GRAM有36点，ck13175–30255，恢复点/锚点为ck12687。GRAM恢复前留空，没有把no-GRAM历史点当作GRAM观测。峰值按各分支全部可用点选择，范围不同已在清单记录。', '',
        '**此29项子集根据test分数事后选择；图中已标注POSTHOC TEST-SELECTED。不代表完整v4。**', '',
        '|任务|no-GRAM峰值checkpoint|no-GRAM峰值|GRAM峰值checkpoint|GRAM峰值|','|---|---:|---:|---:|---:|']
    for f in PANELS:
        a,b=(pmap[m,f] for m in MODELS)
        md.append(f'|{LABELS[f]}|{a["peak_checkpoint"]}|{a["peak_value"]:.6f}|{b["peak_checkpoint"]}|{b["peak_value"]:.6f}|')
    md+=['','SVG可编辑，PNG用于预览，PDF用于导出；三个版本分别为nogram、gram、gram_vs_nogram。', '',
         '`task_family_curve.csv`含576个任务/均值点；`task_family_peaks.csv`含12个峰值；`per_dataset_source_evidence.csv`含2784条原始来源记录；`selected_29_datasets.csv`固定入选清单。']
    (OUT/'README.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))

if __name__=='__main__':main()
