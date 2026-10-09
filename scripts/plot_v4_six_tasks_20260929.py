"""Six fixed v4 task families; no changing-denominator or test-selected subsets."""
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT/'outputs/02_eval_runs'
REPORT = ROOT/'outputs/00_reports/deepcad_method_20260927'
OUT = REPORT/'v4_six_tasks_20260929'
COLORS = {'Vanilla (5TB no-GRAM)':'#647080','GRAM (5TB)':'#d58a20','Adaptive':'#07858b'}
N,G,A = COLORS


def write_csv(path, rows):
    if not rows: path.write_text(''); return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    spec=json.loads((ROOT/'Evaluation Rules/protocol_v4.json').read_text())
    expected={}
    for fam in ('classification','regression','retrieval','segmentation'):
        expected[fam]=spec['tier_a'][fam]+spec['tier_b'][fam]+spec['union_extension'].get(fam,[])
    expected['clustering']=expected['retrieval'].copy()
    expected['detection_proxy']=spec['union_extension']['detection_proxy']
    families=['classification','regression','retrieval','clustering','segmentation','detection_proxy']
    cells={}; duplicates=[];missing_sources=[];known_steps={m:set() for m in COLORS}
    def put(method,step,family,dataset,value,source,budget=0,provisional=False):
        if family not in expected or dataset not in expected[family]:return
        if value is None or not math.isfinite(float(value)):return
        key=(method,int(step),family,dataset,int(budget)); known_steps[method].add(int(step))
        value=float(value)
        if key in cells:
            if abs(cells[key]['value']-value)>1e-6:
                duplicates.append(dict(key=str(key),retained=cells[key]['value'],alternate=value,source=source))
            return
        metric={'classification':'macro_auc' if dataset=='chestmnist' else 'balanced_accuracy','regression':'r2','retrieval':'recall_at_1','clustering':'nmi','segmentation':'mDice','detection_proxy':'patch_f1'}[family]
        cells[key]=dict(method=method,checkpoint=int(step),family=family,dataset=dataset,budget=int(budget),metric=metric,value=value,provisional=provisional,source=source)
    def scalar(method,step,ds,obj,source):
        if ds in expected['classification']:
            metric='macro_auc' if ds=='chestmnist' else 'balanced_accuracy'
            if obj.get(metric) not in (None,''):put(method,step,'classification',ds,float(obj[metric]),source,provisional=ds=='lc25000')
        if ds in expected['regression'] and obj.get('r2') not in (None,''):put(method,step,'regression',ds,float(obj['r2']),source)
    def retrieval(method,step,ds,rows,source):
        if ds not in expected['retrieval']:return
        if ds=='rxrx3-core':
            row=rows[0]
            for fam,metric in [('retrieval','recall_at_1'),('clustering','nmi')]:
                if row.get(metric) not in (None,''):put(method,step,fam,ds,float(row[metric]),source)
            return
        for row in rows:
            ag=row.get('aggregation','');task=row.get('task','')
            good_ret=ag=='global' if ds in ('hpa-subcellular','rxrx1-cross') else ag in ('class','global')
            good_cluster=(ag=='location' and str(row.get('n_classes'))=='41') if ds=='hpa-subcellular' else ag=='global-perturbation' if ds=='rxrx1-cross' else ag in ('class','global')
            if good_ret and row.get('recall_at_1') not in (None,''):put(method,step,'retrieval',ds,float(row['recall_at_1']),source)
            if good_cluster and row.get('nmi') not in (None,''):put(method,step,'clustering',ds,float(row['nmi']),source)
    def dense(method,step,ds,paths,budget):
        if ds not in expected['segmentation']:return
        paths=sorted(set(paths)); values=[];identities=set()
        for path in paths:
            try:obj=json.loads(Path(path).read_text())
            except (OSError,ValueError):missing_sources.append(str(path));continue
            meta=obj.get('_meta',{})
            if meta.get('probe_epochs')!=budget or meta.get('seed') not in (0,1,2) or meta.get('probe_batch_size')!=32:continue
            if 'mDice' not in obj.get('test',{}):continue
            # Path prefix differentiates the three PanNuke rotations.
            identity=(str(path).split(f'/budget{budget}/')[0],meta['seed'])
            if identity in identities:continue
            identities.add(identity);values.append(obj['test']['mDice'])
        if len(values)==(9 if ds=='pannuke' else 3):put(method,step,'segmentation',ds,statistics.mean(values),';'.join(map(str,paths)),budget)

    # Reuse the prior audit's identified original-model sources, not its subsets.
    previous=json.loads((REPORT/'original_5tb_comparison_20260929/DATA_AND_AUDIT.json').read_text())
    name_map={'5TB no-GRAM':N,'5TB GRAM':G}
    for e in previous['evidence']:
        if e['method'] not in name_map:continue
        method=name_map[e['method']];step=e['checkpoint'];metric=e['metric'];src=e['source']
        if e['group']=='classification':put(method,step,'classification',metric,e['value']/100,src)
        elif metric=='ChestMNIST macro AUC (%)':put(method,step,'classification','chestmnist',e['value']/100,src)
        elif metric.endswith(' R2'):put(method,step,'regression',metric[:-3],e['value'],src)
        elif metric.endswith(' patch F1 (%)'):put(method,step,'detection_proxy',metric[:-13],e['value']/100,src)
        elif ' mIoU (%)' in metric and not src.startswith('{'):
            match=re.fullmatch(r'(.+) E(20|50) mIoU \(%\)',metric)
            if match:dense(method,step,match[1],src.split(';'),int(match[2]))
    print('Historical cached classification/regression/dense loaded',flush=True)
    # Complete retrieval AND clustering; include the locked LC25000 diagnostic.
    for row in csv.DictReader((EVAL/'old_v3_protocol_union/results.csv').open()):
        if row['model'] not in ('5tb_no_gram','5tb_gram12687'):continue
        if row['family']!='retrieval' and not (row['family']=='classification' and row['dataset']=='lc25000'):continue
        if not row['source'].endswith('component_result.json'):continue
        valid=row['protocol']=='v3' and row['evidence_status']=='VALID_COMPLETE'
        extension=row['protocol']=='old_union_extension' and row['evidence_status']=='VALIDATED_LEGACY_COMPONENT'
        if not (valid or extension):continue
        p=Path(row['source']);obj=json.loads(p.read_text());method=N if row['model']=='5tb_no_gram' else G;step=int(row['checkpoint']);ds=row['dataset']
        if row['family']=='classification':scalar(method,step,ds,obj,str(p))
        else:retrieval(method,step,ds,obj.get('rows',[obj]),str(p))
    native=EVAL/'rxrx3_core_formal_l5_all49_v3_single3090_20260911'
    validation=json.loads((native/'validation_report.json').read_text());manifest=json.loads((native/'campaign_manifest.json').read_text())
    assert validation['status']=='VALID_COMPLETE'
    assert manifest['dataset_protocol']['manifest_sha256']==spec['retrieval_splits']['rxrx3-core']['manifest_sha256']
    assert manifest['evaluation']['logical_batch_size']==64
    for r in csv.DictReader((native/'checkpoint_metrics.csv').open()):retrieval(N,int(r['checkpoint']),'rxrx3-core',[r],str(native/'checkpoint_metrics.csv'))
    print('Historical retrieval/clustering and native RxRx3 loaded',flush=True)

    # Read all completed tasks, including fields the old plotting collector omitted.
    extra_status=[]
    for root in [EVAL/'hs6_l5_selective_retention_v4_20260923',EVAL/'hs6_l5_deepcad_method_v4_20260927']:
        for path in sorted((root/'tasks').glob('*.json')):
            task=json.loads(path.read_text());arm=task['arm'];method=None;step=None
            if arm.startswith(('adaptive_formal_','adaptive_continue_')):method=A;step=int(arm.rsplit('_ck',1)[1])
            elif arm in ('baseline_E','baseline_M','baseline_L'):method=N;step={'baseline_E':12687,'baseline_M':20007,'baseline_L':29279}[arm]
            elif re.fullmatch('[GN][0-9]+',arm):method=G if arm[0]=='G' else N;step=int(arm[1:])
            if method is None:continue
            known_steps[method].add(step)
            status_path=root/'claims'/task['id']/'status.json'
            state=json.loads(status_path.read_text()).get('state') if status_path.exists() else 'QUEUED'
            if task['family'] in ('cell_tracking','ctc','ood'):extra_status.append(dict(method=method,checkpoint=step,family=task['family'],dataset=task['dataset'],state=state,source=str(path)))
            if state!='DONE':continue
            done=task['done'];ds=task['dataset'];kind=done['type']
            if kind=='summary_csv':
                p=Path(done['path']);rows=[r for r in csv.DictReader(p.open()) if not r.get('error')]
                for r in rows:scalar(method,step,ds,r,str(p))
                if ds=='rxrx3-core':
                    for r in rows:
                        if r.get('result_file'):
                            obj=json.loads(Path(r['result_file']).read_text());rx=obj['tests']['rxrx3']
                            assert rx['protocol_id']==spec['retrieval_splits']['rxrx3-core']['protocol']
                            retrieval(method,step,ds,[rx],r['result_file'])
                else:retrieval(method,step,ds,rows,str(p))
            elif kind=='detection_json':
                p=Path(done['path']);obj=json.loads(p.read_text())
                assert obj['batch_size']==8 and obj['epochs']==5 and obj['image_size']==224
                put(method,step,'detection_proxy',ds,obj['test_patch_f1']/100,str(p))
            elif kind=='seg_results':
                bases=list(Path(done['root']).glob(done['run_prefix']+'*'))
                for budget in (20,50):
                    paths=[]
                    for base in bases:paths+=list(base.glob(f'**/budget{budget}/seed*/{ds}/{done["ckpt_id"]}/results.json'))
                    dense(method,step,ds,paths,budget)
    print('Completed v4 task artifacts loaded',flush=True)
    write_csv(OUT/'CELLS.csv',list(cells.values()))
    write_csv(OUT/'DUPLICATE_DISCREPANCIES.csv',duplicates)
    write_csv(OUT/'CTC_OOD_STATUS.csv',extra_status)

    summary=[];lookup={}
    for method,steps in known_steps.items():
        for step in sorted(steps):
            for family in families:
                for budget in ((20,50) if family=='segmentation' else (0,)):
                    included=[cells[method,step,family,d,budget] for d in expected[family] if (method,step,family,d,budget) in cells]
                    absent=[d for d in expected[family] if (method,step,family,d,budget) not in cells]
                    complete=not absent
                    strict=[c['value'] for c in included if not c['provisional']]
                    mean=statistics.mean(c['value'] for c in included) if complete else None
                    state='PROVISIONAL_LC25000' if complete and family=='classification' else 'ALL_EXPECTED_OBSERVED' if complete else 'INCOMPLETE'
                    row=dict(method=method,checkpoint=step,family=family,budget=budget,n=len(included),expected=len(expected[family]),mean=mean,status=state,missing=';'.join(absent),classification24_mean=statistics.mean(strict) if family=='classification' and len(strict)==24 else None)
                    summary.append(row);lookup[method,step,family,budget]=row
    write_csv(OUT/'TASK6_MEANS_AND_COVERAGE.csv',summary)
    gaps=[]
    for row in summary:
        for ds in filter(None,row['missing'].split(';')):
            gaps.append({k:row[k] for k in ('method','checkpoint','family','budget')} | {'missing_dataset':ds})
    write_csv(OUT/'MISSING_CELLS.csv',gaps)
    latest=max(known_steps[A]);complete_adaptive=[s for s in sorted(known_steps[A]) if all(lookup[A,s,f,20 if f=='segmentation' else 0]['mean'] is not None for f in families)]
    meta=dict(time=datetime.now(timezone.utc).isoformat(),protocol=spec['protocol_id'],expected=expected,latest_registered_adaptive=latest,latest_six_families_observed=max(complete_adaptive) if complete_adaptive else None,missing_sources=missing_sources,duplicates=len(duplicates),aggregation='unweighted dataset mean within each task; no six-task composite; no changing denominator',classification='All25 diagnostic is provisional because LC25000; separately report validated24 mean; no claim of strict full-v4 completeness',segmentation='mDice; E20 and E50 independent; equal datasets after equal seeds/folds',rxrx3='Recall@1 for retrieval, NMI for clustering; never substitute mAP or a proxy')
    (OUT/'SUMMARY.json').write_text(json.dumps(meta,indent=2)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    pdf=PdfPages(OUT/'V4_TASK6_CURVES.pdf')
    names={'classification':'Classification | 25 datasets*','regression':'Regression | 4 datasets','retrieval':'Retrieval | 7 datasets','clustering':'Clustering | 7 datasets','segmentation':'Segmentation | 7 datasets','detection_proxy':'Detection proxy | 3 datasets'}
    labels={'classification':'Mean BA / macro AUC (%)','regression':'Mean R-squared','retrieval':'Mean Recall@1 (%)','clustering':'Mean NMI (%)','segmentation':'Mean mDice (%)','detection_proxy':'Mean patch F1 (%)'}
    for budget in (20,50):
        for zoom in (False,True):
            fig,axes=plt.subplots(2,3,figsize=(15,9))
            for ax,family in zip(axes.flat,families):
                b=budget if family=='segmentation' else 0
                for method,steps in known_steps.items():
                    grid=sorted(s for s in steps if not zoom or 12687<=s<=latest)
                    vals=[lookup[method,s,family,b]['mean'] for s in grid]
                    scale=1 if family=='regression' else 100
                    y=[v*scale if v is not None else math.nan for v in vals]
                    ax.plot(grid,y,color=COLORS[method],label=method,marker='o',ms=3.5 if method==A else 2.5,lw=1.7,ls='--' if family=='classification' else '-')
                ax.set(title=names[family]+(f' | E{budget}' if family=='segmentation' else ''),xlabel='Optimizer update',ylabel=labels[family])
                ax.grid(alpha=.18);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
                ax.set_xlim(12500,latest+300) if zoom else ax.set_xlim(0,30500)
                if not any(math.isfinite(v) for line in ax.lines for v in line.get_ydata()):ax.text(.5,.5,'No complete family at these checkpoints',ha='center',transform=ax.transAxes)
            axes.flat[0].legend(fontsize=8)
            fig.suptitle(f'V4 task families: original 5TB Vanilla / GRAM vs Adaptive (segmentation E{budget})',fontsize=14,y=.995)
            fig.text(.5,.014,'Fixed dataset lists; plot only complete family means. Missing cells create gaps. No six-task composite.\n* Classification25 is provisional due to LC25000; validated24 values are reported separately. One FM seed; descriptive results.',ha='center',fontsize=8)
            fig.tight_layout(rect=[0,.065,1,.95]);name=f'TASK6_E{budget}'+('_ZOOM' if zoom else '')
            fig.savefig(OUT/(name+'.png'),dpi=175);fig.savefig(OUT/(name+'.svg'));pdf.savefig(fig);plt.close(fig)
    # Coverage is part of the main deliverable, not hidden behind a subset score.
    steps=sorted(known_steps[A]);fig,axes=plt.subplots(3,1,figsize=(15,11),sharex=True)
    for ax,method in zip(axes,COLORS):
        counts=np.array([[lookup.get((method,s,f,20 if f=='segmentation' else 0),{}).get('n',0)/len(expected[f]) for s in steps] for f in families])
        ax.imshow(counts,aspect='auto',vmin=0,vmax=1,cmap='YlGnBu')
        ax.set_yticks(range(6),[names[f].split(' |')[0] for f in families]);ax.set_title(method,loc='left')
        for i,f in enumerate(families):
            for j,s in enumerate(steps):
                n=lookup.get((method,s,f,20 if f=='segmentation' else 0),{}).get('n',0)
                ax.text(j,i,f'{n}/{len(expected[f])}',ha='center',va='center',fontsize=8,color='white' if counts[i,j]>.7 else '#222222')
    axes[-1].set_xticks(range(len(steps)),steps,rotation=30)
    fig.suptitle('V4 task-family coverage at Adaptive checkpoints',fontsize=15)
    fig.text(.5,.01,'Counts are matched observed dataset metrics, not a declaration of full-v4 validity. LC25000 classification remains provisional; CTC/OOD tracked separately.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,.96]);fig.savefig(OUT/'TASK6_COVERAGE.png',dpi=175);pdf.savefig(fig);plt.close(fig);pdf.close()
    lines=['# v4六任务汇总曲线','',f'采集时间：{meta["time"]}。','',
           '固定清单：classification25、regression4、retrieval7、clustering7、segmentation7、detection proxy3。'
           '不再用共同9项或事后挑选的数据集替代任务族。每个数据集等权；某点缺任一必需数据集，整项均值留空，不补零。','',
           '指标：分类BA/ChestMNIST macro AUC；回归R²；检索Recall@1；聚类NMI；分割mDice；检测代理patch F1。'
           '沿用历史六任务图的指标定义，但不沿用其事后选出的37项子集。分割E20、E50各成一版，不混合两个预算。'
           'PanNuke先平均3 rotations各3 seeds，再作为一个数据集进入7项均值。','',
           '分类25项包含LC25000的固定随机split诊断，其来源独立性尚未验证。图中虚线表示这项汇总仍是provisional，'
           '不能称为严格完整v4分类分数；同一CSV另列其余24项的固定均值。不因LC25000状态而缩小v4目标清单。','',
           '模型：原始5TB no-GRAM/GRAM全程基线，Adaptive从ck12687接入。'
           '旧版raw dense与不同B4检测结果不补入。原始RxRx3独立49点正式评测已纳入，而不是只查old_v3索引。','',
           f'Adaptive最新已登记评测点：{latest}。六任务全部数据集数值齐全的最新点（仍含LC provisional）：{meta["latest_six_families_observed"]}。','',
           '|checkpoint|分类|回归|检索|聚类|分割E20|检测代理|','|---|---|---|---|---|---|---|']
    for s in steps:lines.append('|'+str(s)+'|'+'|'.join(f"{lookup[A,s,f,20 if f=='segmentation' else 0]['n']}/{len(expected[f])}" for f in families)+'|')
    lines+=['','全程主图：TASK6_E20.png；放大图：TASK6_E20_ZOOM.png；E50独立版本同名替换E20。'
            'PDF包含4张六任务图与覆盖率图。TASK6_MEANS_AND_COVERAGE.csv列出每个checkpoint每个任务的均值和缺失数据集。'
            'CELLS.csv保留逐项来源；MISSING_CELLS.csv逐行列缺项；CTC_OOD_STATUS.csv单独列扩展任务状态。六任务图不等于整个v4的CTC/OOD也已完成。','',
            '重复来源按固定优先级保留历史已验证值，后续v4定点评测只填空缺，不根据分数大小挑来源。'
            f'发现{len(duplicates)}条超过1e-6的重复评测数值差异，完整保留在DUPLICATE_DISCREPANCIES.csv；'
            '这些差异尚未全部归因，因此图用于描述，不据小差异作显著性或因果结论。','']
    (OUT/'README.md').write_text('\n'.join(lines))
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
