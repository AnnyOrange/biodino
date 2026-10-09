"""Original 5TB trajectories versus Adaptive; retain source/protocol identities."""
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

import report_methods_20260929 as report

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/00_reports/deepcad_method_20260927/original_5tb_comparison_20260929'
INDEX = ROOT / 'outputs/02_eval_runs/old_v3_protocol_union/results.csv'
COLORS = {'5TB no-GRAM': '#626b78', '5TB GRAM': '#d58920', 'Adaptive': '#087f8c'}


def dump_csv(path, rows):
    if not rows:
        path.write_text(''); return
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    trajectories = {k: {} for k in COLORS}
    evidence, conflicts, omitted, audit = [], [], [], []
    def model(name, step):
        return trajectories[name].setdefault(step, dict(classification={}, metrics={}, sources={}))
    def add(name, step, group, key, value, source, protocol, metadata=None):
        m = model(name, step)
        if key in m[group]:
            if abs(m[group][key] - value) > 1e-7:
                conflicts.append(dict(method=name, step=step, group=group, metric=key,
                                      retained=m[group][key], alternate=value, alternate_source=source))
            return
        m[group][key] = value; m['sources'][group + ':' + key] = source
        evidence.append(dict(method=name, checkpoint=step, group=group, metric=key,
                             value=value, source=source, source_protocol=protocol,
                             metadata=metadata or {}))

    # Only archive-validated shared cells and explicitly labeled union extensions.
    # Old raw dense scores are never used here.
    rows = list(csv.DictReader(INDEX.open()))
    segmentation = defaultdict(dict)
    for row in rows:
        if row['model'] not in ('5tb_no_gram', '5tb_gram12687'): continue
        name = '5TB no-GRAM' if row['model'] == '5tb_no_gram' else '5TB GRAM'
        valid = row['protocol'] == 'v3' and row['evidence_status'] == 'VALID_COMPLETE'
        extension = row['protocol'] == 'old_union_extension' and row['evidence_status'] in ('VALIDATED_LEGACY_COMPONENT', 'OBSERVATIONAL')
        if not (valid or extension): continue
        source = Path(row['source']); family = row['family']; ds = row['dataset']; step = int(row['checkpoint'])
        if family == 'segmentation':
            if not valid or source.name != 'results.json' or '__primary-last__' not in row['campaign_key']: continue
            match = re.search(r'/budget(20|50)/seed([012])/', str(source))
            if not match: continue
        elif source.name != 'component_result.json': continue
        try: obj = json.loads(source.read_text())
        except (OSError, ValueError) as exc:
            omitted.append(dict(source=str(source), reason=str(exc))); continue
        meta = {k: obj[k] for k in ('split', 'image_size', 'resize_size', 'batch_size', 'seed', 'n_train', 'n_test', 'ridge_alpha', 'channel_policy', 'channel_tta_samples') if k in obj}
        if family == 'segmentation':
            budget, seed = map(int, match.groups()); sm = obj.get('_meta', {})
            if sm.get('probe_epochs') != budget or sm.get('seed') != seed or sm.get('probe_batch_size') != 32: continue
            segmentation[name, step, ds, budget][row['split'], seed] = (obj, str(source))
        elif family == 'classification' and ds != 'lc25000':
            if 'balanced_accuracy' in obj:
                add(name, step, 'classification', ds, 100*obj['balanced_accuracy'], str(source), row['protocol'], meta)
            if ds == 'chammi-cp-task3':
                add(name, step, 'metrics', 'CP Task3 accuracy (%)', 100*obj['accuracy'], str(source), row['protocol'], meta)
            if ds == 'chestmnist' and obj.get('macro_auc') is not None:
                add(name, step, 'metrics', 'ChestMNIST macro AUC (%)', 100*obj['macro_auc'], str(source), row['protocol'], meta)
        elif family == 'regression' and 'r2' in obj:
            add(name, step, 'metrics', ds+' R2', obj['r2'], str(source), row['protocol'], meta)
        elif family == 'retrieval':
            rr = obj.get('rows', [obj])
            accepted = 'global' if ds in ('hpa-subcellular', 'rxrx1-cross') else 'class'
            rec = next((r for r in rr if r.get('aggregation') == accepted and 'recall_at_1' in r), None)
            if rec: add(name, step, 'metrics', ds+' R@1 (%)', 100*rec['recall_at_1'], str(source), row['protocol'], meta)
        elif family == 'detection':
            if obj.get('batch_size') != 8 or obj.get('image_size') != 224 or obj.get('epochs') != 5: continue
            add(name, step, 'metrics', ds+' patch F1 (%)', obj['test_patch_f1'], str(source), 'v4-matched B8 observation', meta)
    for (name, step, ds, budget), cells in segmentation.items():
        splits = {split for split, seed in cells}
        expected = 3 if ds == 'pannuke' else 1
        if len(splits) != expected or any({seed for split2, seed in cells if split2 == split} != {0,1,2} for split in splits):
            continue
        val = statistics.mean(x[0]['test']['mIoU']*100 for x in cells.values())
        add(name, step, 'metrics', f'{ds} E{budget} mIoU (%)', val,
            ';'.join(x[1] for x in cells.values()), 'v3 validated; shared v4 budget', {'probe_seeds':[0,1,2], 'budget':budget, 'splits': sorted(splits)})

    # Supplement sparse native RxRx3 and other missing metrics using original-weight
    # v4 arms. Explicit arm identities prevent substitution of rerun controls.
    old_models = report.collect(report.OLD)
    original_arms = {'baseline_E': ('5TB no-GRAM',12687), 'baseline_M': ('5TB no-GRAM',20007), 'baseline_L': ('5TB no-GRAM',29279)}
    for arm in old_models:
        if re.fullmatch(r'[GN]\d+', arm): original_arms[arm] = ('5TB GRAM' if arm[0]=='G' else '5TB no-GRAM', int(arm[1:]))
    for arm, (name, step) in original_arms.items():
        if arm not in old_models: continue
        m = old_models[arm]
        for group in ('classification','metrics'):
            for key, value in m[group].items():
                if key not in model(name,step)[group]:
                    add(name,step,group,key,value,json.dumps(m['sources']), 'original-checkpoint v4 supplement')

    new_models = report.collect(report.NEW)
    combined = {**old_models, **new_models}
    adaptive_arms = {}
    for arm, m in combined.items():
        if not (arm.startswith('adaptive_formal_') or arm.startswith('adaptive_continue_')): continue
        step = int(arm.rsplit('_ck',1)[1]); adaptive_arms[step] = arm
        for group in ('classification','metrics'):
            for key, value in m[group].items():
                add('Adaptive',step,group,key,value,json.dumps(m['sources']), 'method v4')
        # E20 is an independent probe budget; only include complete task seeds/folds.
        for task, paths in m['sources'].items():
            if not isinstance(paths,list): continue
            ds = task.split('__')[1]; paths20 = [Path(p.replace('/budget50/','/budget20/')) for p in paths]
            if len(paths20) != (9 if ds=='pannuke' else 3) or not all(p.is_file() for p in paths20): continue
            values = [json.loads(p.read_text()) for p in paths20]
            if any(x.get('_meta',{}).get('probe_epochs') != 20 for x in values): continue
            add('Adaptive',step,'metrics',f'{ds} E20 mIoU (%)',statistics.mean(x['test']['mIoU']*100 for x in values),';'.join(map(str,paths20)),'method v4 E20')
        for task, path in m['sources'].items():
            if not isinstance(path,str) or not path.endswith('.csv'): continue
            for r in csv.DictReader(Path(path).open()):
                if r.get('error'): continue
                if r.get('dataset')=='chestmnist' and r.get('macro_auc'):
                    add('Adaptive',step,'metrics','ChestMNIST macro AUC (%)',float(r['macro_auc'])*100,path,'method v4')
                # Actual output settings versus both historical references.
                for name in ('5TB no-GRAM','5TB GRAM'):
                    key = r.get('dataset','') if r.get('balanced_accuracy') else r.get('dataset','')+' R2'
                    candidates = [e for e in evidence if e['method']==name and e['checkpoint']==step and e['metric']==key]
                    if not candidates: continue
                    meta = candidates[0]['metadata']; mismatch = {}
                    for k,v in meta.items():
                        if k not in r or r[k]=='': continue
                        try: equal = float(r[k])==float(v)
                        except (ValueError,TypeError): equal = str(v)==str(r[k])
                        if not equal: mismatch[k] = [v,r[k]]
                    audit.append(dict(step=step,dataset=r.get('dataset'),baseline=name,metadata_fields=list(meta),mismatch=mismatch))
    # Verify evaluator implementation used by the archived validated campaign.
    historical = Path('/mnt/huawei_deepcad/dinov3_retest_snapshot_20260918_fm_bound')
    current = ROOT/'outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921'
    code_audit = []
    for module in ['run_classification.py','encoder.py','datasets.py','probes.py','registry.py','retrieval_clustering.py','run_retrieval_clustering.py']:
        rel = 'dinov3/eval/bio_frozen_eval/'+module
        a,b = historical/rel,current/'source_snapshot'/rel
        code_audit.append(dict(file=rel,historical_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),method_sha256=hashlib.sha256(b.read_bytes()).hexdigest(),identical=a.read_bytes()==b.read_bytes()))
    for rel in ['dinov3/eval/bio_segmentation/linear_probe.py','dinov3/eval/bio_segmentation/feature_extractor.py','dinov3/eval/bio_segmentation/preprocessing.py','dinov3/eval/bio_detection/center_probe.py']:
        a,b=historical/rel,current/'source_snapshot_dense_v4'/rel
        code_audit.append(dict(file=rel,historical_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),method_sha256=hashlib.sha256(b.read_bytes()).hexdigest(),identical=a.read_bytes()==b.read_bytes()))
    assert all(r['identical'] for r in code_audit)
    assert not any(r['mismatch'] for r in audit), [r for r in audit if r['mismatch']]

    steps = sorted(trajectories['Adaptive']); latest = max(steps)
    common = sorted(set.intersection(*(set(trajectories[m][s]['classification']) for m in COLORS for s in steps)))
    allclass = sorted(set.union(*(set(v['classification']) for v in trajectories['Adaptive'].values())))
    common_key = f'Classification fixed {len(common)}-task BA (%)'
    for mapping in trajectories.values():
        for m in mapping.values():
            if all(d in m['classification'] for d in common):
                m['metrics'][common_key] = statistics.mean(m['classification'][d] for d in common)
    deltas = []
    for step in steps:
        a = trajectories['Adaptive'][step]
        for name in ('5TB no-GRAM','5TB GRAM'):
            b = trajectories[name].get(step,{})
            for group in ('classification','metrics'):
                for key in sorted(a[group].keys() & b.get(group,{}).keys()):
                    deltas.append(dict(checkpoint=step,group=group,metric=key,baseline=name,baseline_value=b[group][key],adaptive=a[group][key],delta=a[group][key]-b[group][key]))
    dump_csv(OUT/'TASK_DELTAS.csv',deltas)
    dump_csv(OUT/'CURVE_VALUES.csv',[{k:v for k,v in e.items() if k!='metadata'} for e in evidence])
    timestamp = datetime.now(timezone.utc).isoformat()
    (OUT/'DATA_AND_AUDIT.json').write_text(json.dumps(dict(time=timestamp,trajectories=trajectories,common_classification=common,evidence=evidence,metadata_audit=audit,evaluator_code_audit=code_audit,conflicts=conflicts,omitted=omitted),indent=2)+'\n')

    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    pdf=PdfPages(OUT/'ORIGINAL_5TB_VS_ADAPTIVE.pdf')
    def curve(ax,group,key,zoom=False):
        for name,mapping in trajectories.items():
            grid=sorted(mapping)
            values=[mapping[s][group].get(key,math.nan) for s in grid]
            ax.plot(grid,values,color=COLORS[name],marker='o' if name=='Adaptive' else '.',markersize=4 if name=='Adaptive' else 2,linewidth=1.8,label=name)
        ax.axvline(12687,color='#bbbbbb',lw=.8,ls=':')
        ax.set_title(key,fontsize=10);ax.grid(alpha=.18)
        ax.set_xlabel('Optimizer update');ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.set_ylabel('R-squared' if key.endswith(' R2') else 'Score (%)')
        ax.set_xlim(12400,latest+300) if zoom else ax.set_xlim(0,30500)
    def save(fig,name,title,note):
        fig.suptitle(title,fontsize=15,y=.995);fig.text(.5,.008,note,ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.045,1,.95]);fig.savefig(OUT/name,dpi=175);pdf.savefig(fig);plt.close(fig)
    overview=[common_key,'bbbc038 patch F1 (%)','hpa-subcellular R@1 (%)','cellpose E50 mIoU (%)','bbbc005 R2','bbbc013 R2']
    for zoom,filename in [(False,'FULL_TRAJECTORY.png'),(True,'ADAPTIVE_INTERVAL.png')]:
        fig,axes=plt.subplots(2,3,figsize=(15,8.6))
        for ax,key in zip(axes.flat,overview):curve(ax,'metrics',key,zoom)
        axes.flat[0].legend(fontsize=8)
        save(fig,filename,'Original 5TB no-GRAM / GRAM trajectories vs Adaptive',f'Adaptive evaluated through {latest}; fixed {len(common)} classification tasks across all plotted Adaptive checkpoints. Missing cells remain gaps. Partial v4; one FM seed.')
    for i in range(math.ceil(len(allclass)/12)):
        fig,axes=plt.subplots(4,3,figsize=(15,15));subset=allclass[i*12:(i+1)*12]
        for ax,key in zip(axes.flat,subset):curve(ax,'classification',key,True)
        for ax in list(axes.flat)[len(subset):]:ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        save(fig,f'CLASSIFICATION_{i+1}.png','Individual classification: original trajectories vs Adaptive','No-GRAM/GRAM are the original 5TB trajectories, not newly branched vanilla/Gram controls. Gaps indicate incomplete Adaptive tests.')
    keys=sorted(set.union(*(set(m['metrics']) for m in trajectories['Adaptive'].values()))-{common_key})
    for i in range(math.ceil(len(keys)/9)):
        fig,axes=plt.subplots(3,3,figsize=(15,12));subset=keys[i*9:(i+1)*9]
        for ax,key in zip(axes.flat,subset):curve(ax,'metrics',key,True)
        for ax in list(axes.flat)[len(subset):]:ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        save(fig,f'OTHER_METRICS_{i+1}.png','Task curves: shared evaluation settings, original 5TB references','E20/E50 are separate segmentation probes. Detection is the B8 patch proxy. Native RxRx3 baseline coverage is sparse; no proxy substitution.')
    # A single reference across time exposes regressions without mean cancellation.
    fig,ax=plt.subplots(figsize=(13,11))
    grid=np.full((len(allclass),len(steps)),np.nan)
    for i,key in enumerate(allclass):
        for j,step in enumerate(steps):
            aa=trajectories['Adaptive'][step]['classification'];bb=trajectories['5TB no-GRAM'][step]['classification']
            if key in aa and key in bb:grid[i,j]=aa[key]-bb[key]
    cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#dddddd')
    im=ax.imshow(grid,aspect='auto',cmap=cmap,vmin=-3,vmax=3)
    ax.set_xticks(range(len(steps)),steps,rotation=40);ax.set_yticks(range(len(allclass)),allclass,fontsize=8)
    for i in range(len(allclass)):
        for j in range(len(steps)):
            v=grid[i,j];ax.text(j,i,f'{v:+.2f}' if np.isfinite(v) else 'N/A',ha='center',va='center',fontsize=7,color='white' if abs(v)>1.8 else '#222222')
    fig.colorbar(im,ax=ax,label='BA difference (percentage points); colors clipped at +/-3')
    save(fig,'ADAPTIVE_MINUS_ORIGINAL_NOGRAM.png','Adaptive minus ORIGINAL 5TB no-GRAM at the same checkpoint','Blue = improvement; red = regression; gray = missing Adaptive evaluation. Raw differences, not significance tests.')
    pdf.close()
    lines=['# 原始5TB轨迹与Adaptive：更正后的对照','','采集时间：'+timestamp,'',
           '纠正：此前图中的vanilla_formal/gram_formal是从12687重新分支的短程对照，'
           '不能代表原始5TB no-GRAM/GRAM。此前“后期没有对照”的说法不成立；历史索引中有对应结果。','',
           '|轨迹|本次纳入的首/末checkpoint|数量|','|---|---|---:|']
    for name,m in trajectories.items():lines.append(f'|{name}|{min(m)} / {max(m)}|{len(m)}|')
    lines+=['','实际评测首点为ck487，不凭空补step0。GRAM从12687分支。'
            'Adaptive曲线止于其已有下游评测结果，训练日志步数不是已完成评测步数。','',
            '历史来源为old_v3_protocol_union中的VALID_COMPLETE v3共享cell及匹配的union扩展；'
            '旧raw dense、B4检测、tracking代理未混入。保留原协议标签，不改称每个checkpoint已有完整56项v4。'
            '与1TB、FM14一起补测的统一索引确实存在，不需要重训历史基线。','',
            f'主图分类固定共同{len(common)}项：'+', '.join(common)+'。23项分类逐项图另列，不能把共同子集均值叫全分类结果。','',
            f'已核对{len(code_audit)}个评测核心源码文件逐字一致，{len(audit)}组分类/回归输出设置无记录字段差异。'
            '分割按E20/E50分别要求3个probe seeds；PanNuke要求3 rotations。'
            '这是复用已有有效结果的描述性比较，不是对完整v4或全部数据内容重新认证。'
            '训练数据流、优化器恢复和卡数布局不同，不能只用同step比较证明因果方法效果。','',
            f'## 最新可比较点 ck{latest}','', '|指标|原始no-GRAM|原始GRAM|Adaptive|Adaptive−no-GRAM|','|---|---:|---:|---:|---:|']
    a=trajectories['Adaptive'][latest];n=trajectories['5TB no-GRAM'][latest];g=trajectories['5TB GRAM'][latest]
    for group in ('classification','metrics'):
        for key in sorted(a[group].keys() & n[group].keys() & g[group].keys()):
            lines.append(f'|{key}|{n[group][key]:.6f}|{g[group][key]:.6f}|{a[group][key]:.6f}|{a[group][key]-n[group][key]:+.6f}|')
    lines+=['','全量曲线PDF：ORIGINAL_5TB_VS_ADAPTIVE.pdf；全程总览：FULL_TRAJECTORY.png；'
            'Adaptive区间放大：ADAPTIVE_INTERVAL.png；逐项差值：TASK_DELTAS.csv；'
            '来源、代码hash与设置核查：DATA_AND_AUDIT.json。','',
            '仍然以逐项超过原始no-GRAM并对比GRAM为目标。部分指标更好不能抵消其他任务退步。','']
    (OUT/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(OUT),latest=latest,common=common,values=len(evidence),conflicts=len(conflicts),omitted=len(omitted),metadata_checks=len(audit)),indent=2))


if __name__=='__main__':main()
