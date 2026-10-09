"""Audit a frozen observation snapshot; never select checkpoints or hyperparameters."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/00_reports/deepcad_method_20260927/results_20260929'


def main():
    snapshot = json.loads((OUT / 'RESULTS.json').read_text())
    models = snapshot['models']
    rows, summary = [], []
    for step in (13175, 13663, 14151, 14639):
        baseline = models[f'vanilla_formal_ck{step}']
        for name, arm in (
            ('Adaptive', f'adaptive_formal_ck{step}'),
            ('C', f'ck_c_e12687_formal_ck{step}'),
            ('K', f'ck_k_e12687_formal_ck{step}'),
        ):
            if arm not in models:
                continue
            method = models[arm]
            for group in ('classification', 'metrics'):
                shared = sorted(baseline[group].keys() & method[group].keys())
                deltas = []
                for metric in shared:
                    ref, val = baseline[group][metric], method[group][metric]
                    delta = val - ref
                    deltas.append(delta)
                    rows.append(dict(snapshot_time=snapshot['time'], checkpoint=step,
                                     method=name, group=group, metric=metric,
                                     vanilla=ref, value=val, delta=delta,
                                     unit='R2' if metric.endswith(' R2') else 'percentage_points',
                                     direction='higher_is_better',
                                     comparison='historical_two_rank_reference' if name != 'Adaptive'
                                     else 'historical_matched_continuation'))
                if group == 'classification' and deltas:
                    summary.append(dict(checkpoint=step, method=name, n=len(deltas),
                                        wins=sum(d > 0 for d in deltas),
                                        losses=sum(d < 0 for d in deltas),
                                        ties=sum(d == 0 for d in deltas),
                                        mean_delta_pp=statistics.mean(deltas)))
    with (OUT / 'VANILLA_TASK_DELTAS.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [
        '# 相对 vanilla 的逐项核查', '',
        f"固定结果快照：{snapshot['time']}。不是实时全量评测报告。", '',
        '结论：Adaptive、C、K均未证明所有任务优于vanilla。分类平均增益不能抵消任何单项退步。', '',
        '对照是从12687分支的vanilla续训，不等同于原始完整5TB no-Gram轨迹。'
        'C/K单rank，历史对照双rank；相同有效batch不代表完全匹配的数据流与数值路径。'
        '全量v4尚未完成；单个FM训练seed，数值胜负不等于统计显著。', '',
        '## 单标签分类：各配对的共同已完成任务', '',
        'BA以百分点计；不同分母不可直接比较平均增益。排除LC25000与无BA的ChestMNIST。', '',
        '| checkpoint | 方法 | 共同任务数 | 增长 | 下降 | 持平 | 平均差值 |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in summary:
        text.append('| {checkpoint} | {method} | {n} | {wins} | {losses} | {ties} | {mean_delta_pp:+.6f} |'.format(**row))
    text += ['', '## ck13663：Adaptive与C的共同已完成非BA指标', '',
             '这里逐行报告，禁止将R²与百分点混合平均；CP accuracy与同数据集BA不是两个独立任务。', '',
             '| 指标 | Adaptive−vanilla | C−vanilla | 单位 |',
             '|---|---:|---:|---|']
    selected = {(r['method'], r['metric']): r for r in rows
                if r['checkpoint'] == 13663 and r['group'] == 'metrics'}
    keys = sorted({key for name, key in selected if name == 'Adaptive'} &
                  {key for name, key in selected if name == 'C'})
    for key in keys:
        a, c = selected['Adaptive', key], selected['C', key]
        text.append(f"| {key} | {a['delta']:+.6f} | {c['delta']:+.6f} | {a['unit']} |")
    text += ['', '全部已观测逐项差值见同目录VANILLA_TASK_DELTAS.csv；缺失项仍然缺失。', '',
             '## 机制解释与验证边界', '',
             '恢复旧特征是能力保持的代理目标，不等价于提高每一个下游任务。'
             '历史最低误差只能下降，可能受有利噪声影响而逐渐收紧预算；'
             '当前gate饱和也可能反映真实恢复困难，尚不能单独归因为噪声。'
             'C用独立冻结校准预算避免这一收紧机制，但其低gate也意味着接近无正则续训。'
             '现有结果不能区分精确保留与单纯减弱约束的贡献。', '',
             '若训练总梯度为g_ssl + λg_rec，则一阶SSL损失变化为'
             '−η||g_ssl||² − ηλ〈g_ssl,g_rec〉。内积为负时，恢复项削弱该步SSL下降。'
             '这是可测的梯度冲突动机，不是下游退步的因果证明；'
             'AdamW需在实际预条件更新空间检查。', '',
             '## 后续判定与实验顺序', '',
             '1. 主要目标为预先确定的全量v4逐项优于vanilla；同时报告Gram。'
             '任何负增益必须列出，平均分、历史regret和C>K均不能替代这个目标。'
             '所有任务包括ID和OOD；分割各既定预算分列。',
             '2. 同一步数、同分支、同有效batch和同读出协议比较；'
             '保留原始5TB no-Gram轨迹这一独立参照，不与续训vanilla混称。'
             '最终复核多个FM seed并给出不确定性；无显著退步只是中间检查，不等于全部提高。',
             '3. 当前C/K完成原预设训练和v4，不根据本轮test临时修改超参或终点。'
             '在新增方法长训前，先设计同单rank、同样本流、同优化器的vanilla零恢复损失和原Adaptive对照，'
             '将训练布局与正则贡献分离；不是重新训练整条5TB基线。',
             '4. 在独立train/validation样本上测量分来源恢复误差、预算误触发、最坏恢复方向、'
             'SSL/恢复项梯度冲突及实际更新幅度，验证哪种退步机制成立。'
             '随后才设计C预算下的受限恢复更新；SSL一阶下降保护不保证全部下游任务改善。',
             '5. 本批test是探索性观察，不能再称独立最终确认。新超参和checkpoint选择只用train/validation；'
             '如需确认性结论，使用事先留出的独立数据并冻结方案。', '']
    (OUT / 'VANILLA_TASK_AUDIT.md').write_text('\n'.join(text))
    print(json.dumps(dict(rows=len(rows), classification_summary=summary), indent=2))


if __name__ == '__main__':
    main()
