#!/usr/bin/env python3
"""Verify and visualize the explicitly posthoc V4-subset search candidates."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_fig2_dataset_scales_fm14_20260924 import value

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/00_reports/v4_subset_scaling_fm14_search_20260924'
FAMILIES=('classification','regression','retrieval','clustering','segmentation')

def rows(p):return list(csv.DictReader(p.open()))

def main():
    comparison=[]
    for suffix in ('','_both_refined','_both_margin28'):
        out=BASE.with_name(BASE.name+suffix)
        if not (out/'summary.json').exists():continue
        s=json.loads((out/'summary.json').read_text())
        data=rows(out/'selected_models_task_means.csv')
        evidence=rows(out/'selected_source_evidence.csv')
        inventory=rows(out/'per_dataset_selection_audit.csv')
        allpoints=rows(out/'all_checkpoint_scores.csv')
        lookup={r['model']:r for r in data}
        bypoint={}
        checked={}
        for r in evidence:
            paths=[Path(p) for p in r['source'].split(';')]
            objects=[json.loads(p.read_text()) for p in paths]
            if r['family']=='segmentation':
                raw=statistics.mean(o['test']['mDice'] for o in objects)
            else:_,raw=value(objects[0],r['family'],r['dataset'])
            assert abs(raw-float(r['value']))<1e-12
            bypoint.setdefault((r['model'],r['checkpoint']),{}).setdefault(r['family'],[]).append(raw)
            for p in paths:
                parts=p.parts;i=parts.index('cells');report=Path(*parts[:i+2])/'validation_report.json'
                z=json.loads(report.read_text());assert z['status']=='VALID_COMPLETE',(p,z['status'])
                digest=hashlib.sha256(p.read_bytes()).hexdigest()
                hashes=z.get('result_sha256',{})
                if isinstance(hashes,dict) and str(p) in hashes:assert hashes[str(p)]==digest
                checked[str(p)]=dict(source=str(p),sha256=digest,status=z['status'])
        for r in data:
            fs=bypoint[r['model'],r['checkpoint']]
            actual=statistics.mean(statistics.mean(fs[f]) for f in FAMILIES)
            assert abs(actual-float(r['selected_mean']))<1e-12
        one=float(lookup['hs6_l']['selected_mean']);five=float(lookup['5tb_no_gram']['selected_mean'])
        fm=max(float(r['selected_mean']) for r in data if r['model'].startswith('fm_'))
        assert five-one>=.007-1e-10 and five>fm
        if s['both_hs6_required_above_fm']:assert one>fm
        for m in ('hs6_l','5tb_no_gram'):
            assert abs(max(float(r['selected_mean']) for r in allpoints if r['model']==m)-float(lookup[m]['selected_mean']))<1e-12
        audit=dict(source_files=len(checked),selected_evidence_rows=len(evidence),all_sources_valid_complete=True,
                   raw_recalculation_max_tolerance=1e-12,best_checkpoint_and_target_checks_pass=True)
        (out/'verification.json').write_text(json.dumps(audit,indent=2))
        with (out/'verified_source_hashes.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['source','sha256','status']);w.writeheader();w.writerows(checked.values())
        label='36-cell maximum coverage' if not suffix else f'{s["selected_cells"]}-cell both HS6 above FM14'
        ordered=[lookup['hs6_l'],lookup['5tb_no_gram']]+sorted([r for r in data if r['model'].startswith('fm_')],key=lambda r:-float(r['selected_mean']))
        names=[('HS6-L 1TB ck'+r['checkpoint']) if r['model']=='hs6_l' else ('HS6-L 5TB ck'+r['checkpoint']) if r['model']=='5tb_no_gram' else r['model'][3:] for r in ordered]
        fig,axes=plt.subplots(2,3,figsize=(15,10),constrained_layout=True)
        for ax,metric in zip(axes.flat,(*FAMILIES,'selected_mean')):
            vals=[float(r[metric]) for r in ordered]
            ax.barh(range(len(vals)),vals,color=['#4e79a7','#f28e2b']+['#9aa8b6']*14)
            ax.set_yticks(range(len(vals)),names,fontsize=7);ax.invert_yaxis()
            ax.set_title(metric.replace('_',' '));ax.grid(axis='x',alpha=.2)
        fig.suptitle(label+' | POSTHOC TEST-SELECTED | Five-family equal mean',fontsize=12)
        for ext in ('png','svg'):fig.savefig(out/('task_and_fm14_comparison.'+ext),dpi=160)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
        for m,color,label2 in (('hs6_l','#4e79a7','HS6-L 1TB'),('5tb_no_gram','#f28e2b','HS6-L 5TB no-GRAM'),('5tb_gram12687','#59a14f','HS6-L 5TB GRAM')):
            rr=sorted([r for r in allpoints if r['model']==m],key=lambda r:int(r['checkpoint']))
            ax.plot([int(r['checkpoint']) for r in rr],[float(r['selected_mean']) for r in rr],'.-',label=label2,color=color,lw=1)
        ax.axhline(one+.007,color='#4e79a7',linestyle='--',label='1TB best + 0.7 pp')
        ax.axhline(fm,color='#777777',linestyle=':',label='Strongest FM14')
        ax.set_xlabel('Checkpoint step (training schedules differ)');ax.set_ylabel('Selected five-family mean')
        ax.set_title(label+' | POSTHOC TEST-SELECTED');ax.legend(fontsize=8);ax.grid(alpha=.2)
        for ext in ('png','svg'):fig.savefig(out/('all_checkpoint_curves.'+ext),dpi=160)
        plt.close(fig)
        md=['# v4共享组件的事后子集搜索','',
            '**此子集根据测试结果搜索得到，只用于探索/展示，不是预先指定的benchmark，也不支持完整v4的总体领先结论。**','',
            '范围：40项共同有效的v3/v4共享任务，分类24、回归2、检索4、聚类4、分割6。未引入provisional LC25000分类、NCT100低样本项、未完成组件或检测observation。指标固定为macro-F1、Spearman、mAP@5、NMI、E20 primary-last mDice。', '',
            '先在每个任务族内按入选数据集等权，再对五个任务族等权；至少保留'+str(s['min_per_family'])+'项/任务族。每个checkpoint算完整子集分数，1TB在15个、5TB no-GRAM在60个可用checkpoint里取best，FM14逐个对比。GRAM36点另列，不参与no-GRAM子集的目标选择。', '',
            f'保留 **{s["selected_cells"]}/40项**；5TB−1TB = **{100*(five-one):.6f} pp**；5TB−最强FM14 = **{100*(five-fm):.6f} pp**；1TB−最强FM14 = **{100*(one-fm):.6f} pp**。', '',
            ('60个目标checkpoint对应的MILP均已求得最优或证实不可行，因此36是本候选范围/指标/约束下的最大保留数。该证明仅针对覆盖数量，不是对领先幅度的全局最优证明。' if s.get('global_optimality_proven') else '这是已找到的可行方案；未证明所有约束下的全局最大覆盖。'), '',
            '|模型|checkpoint|分类|回归|检索|聚类|分割|五任务均值|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in ordered:md.append('|'+r['model']+'|'+r['checkpoint']+'|'+'|'.join(f'{float(r[k]):.6f}' for k in (*FAMILIES,'selected_mean'))+'|')
        md+=['','## 选择清单','']
        for f in FAMILIES:
            kept=[r['dataset'] for r in inventory if r['family']==f and r['selected']=='True']
            dropped=[r['dataset'] for r in inventory if r['family']==f and r['selected']=='False']
            md+=['- '+f+'，保留'+str(len(kept))+'项：'+', '.join(kept)+'。排除：'+(', '.join(dropped) if dropped else '无')+'。']
        md+=['','## 完整40项参照','']
        for m in ('hs6_l','5tb_no_gram'):
            best=max((r for r in allpoints if r['model']==m),key=lambda r:float(r['full40_mean']))
            md.append(f'- {m} 完整40项best：ck{best["checkpoint"]}，{float(best["full40_mean"]):.6f}。')
        bf=max((r for r in allpoints if r['model'].startswith('fm_')),key=lambda r:float(r['full40_mean']))
        md.append(f'- FM14 完整40项最高：{bf["model"]}，{float(bf["full40_mean"]):.6f}。')
        md+=['','完整40项上5TB best没有超过1TB best；选择子集后排序改变。所有checkpoint选择与子集选择均使用test读数；本次没有重跑推理，也没有独立holdout确认。','',
             '## 文件','',
             '- `per_dataset_selection_audit.csv`：全部40项、入选/排除、每模型原始分数。',
             '- `selected_models_task_means.csv`：各任务均值与综合均值，FM14完整列出。',
             '- `all_checkpoint_scores.csv`：15个1TB、60个5TB no-GRAM、36个GRAM和FM14的全部子集/完整40项分数。',
             '- `selected_source_evidence.csv`、`verified_source_hashes.csv`：来源和复核哈希。',
             '- `task_and_fm14_comparison.png/svg`、`all_checkpoint_curves.png/svg`：任务图与checkpoint曲线。',
             '- `milp_search_log.csv`：求解状态、覆盖数量上界和gap。', '',
             f'复核：{len(evidence)}条入选分数从原始JSON重新提取一致；{len(checked)}个原始文件的组件验证状态均为VALID_COMPLETE；best及阈值独立检查通过。']
        (out/'README.md').write_text('\n'.join(md)+'\n')
        comparison.append(dict(directory=str(out),cells=s['selected_cells'],one=one,five=five,best_fm=fm,
                               five_minus_one_pp=100*(five-one),five_minus_fm_pp=100*(five-fm),one_minus_fm_pp=100*(one-fm),
                               min_per_family=s['min_per_family']))
    with (BASE/'candidate_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    print(json.dumps(comparison,indent=2))

if __name__=='__main__':main()
