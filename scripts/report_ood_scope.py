#!/usr/bin/env python3
"""Publish OOD scope from indexed evidence, without running any evaluation.

Numeric rows are not protocol acceptance or completed uncertainty estimates.
Unknown future datasets fail closed instead of silently becoming native OOD.
"""
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
DATASETS = {
    'xray': ('X-ray/PXCT tomography', 'ood_detection', '固定 ID 参考库的来源判别'),
    'cryo': ('Cryo-EM', 'ood_detection', '固定 ID 参考库的来源判别'),
    'film_age': ('FILM', 'native_downstream', '年龄分类'),
    'xray_bone': ('RSNA Bone Age', 'native_downstream', '骨龄回归（官方 validation）'),
    'xray_chest': ('RSNA Pneumonia', 'native_downstream', '肺部异常框检测'),
    'bccd': ('BCCD', 'native_downstream', '细胞框检测'),
    'bbbc041': ('BBBC041', 'native_downstream', '疟疾细胞框检测'),
    'nucls_qc': ('NuCLS', 'native_downstream', '核检测、匹配实例分割、检测匹配条件下分类'),
    'spheroid_lsfm': ('Spheroid LSFM', 'native_downstream', '3D 质心检测；独立性冲突阻塞'),
}


def role(dataset):
    return DATASETS.get(dataset, ('', 'unclassified', ''))[1]


def numeric(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def atomic_text(path, content):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name + '.', delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def csv_text(rows, columns):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def refresh(report=REPORT):
    source = report / 'paper_results.csv'
    payload = source.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig')))
    columns = reader.fieldnames
    rows = list(reader)
    ood = [row for row in rows if row['domain'] == 'OOD']
    groups = {}
    for row in ood:
        groups.setdefault(row['dataset'], []).append(row)
    scope_rows = []
    for dataset in dict.fromkeys([*DATASETS, *sorted(groups)]):
        entries = groups.get(dataset, [])
        named, evaluation_role, task = DATASETS.get(dataset, (dataset, 'unclassified', '范围待审查'))
        counts = Counter(row.get('metric_status', '') for row in entries)
        scope_rows.append(dict(
            dataset=dataset, name=named, evaluation_role=evaluation_role, task=task,
            indexed_models=len({r['model'] for r in entries}), indexed_metric_rows=len(entries),
            numeric_metric_rows=sum(numeric(r['value']) for r in entries),
            complete_metric_rows=sum(numeric(r['value']) and r.get('metric_status') == 'complete' for r in entries),
            partial_metric_rows=counts['partial'], missing_metric_rows=counts['missing'],
            numeric_models=len({r['model'] for r in entries if numeric(r['value'])}),
        ))
    scoped = [dict(r, evaluation_role=role(r['dataset']),
                   inference_scope=('fixed_ID_reference_source_discrimination_only'
                                    if role(r['dataset']) == 'ood_detection'
                                    else 'dataset_task_specific; pretraining_overlap_not_certified'
                                    if role(r['dataset']) == 'native_downstream'
                                    else 'unreviewed; no_generalization_claim')) for r in ood]
    atomic_text(report / 'ood_results_scoped.csv', csv_text(scoped, columns + ['evaluation_role', 'inference_scope']))
    atomic_text(report / 'ood_dataset_status.csv', csv_text(scope_rows, list(scope_rows[0])))
    split_path = report / 'paper_results_ood.csv'
    split = list(csv.DictReader(io.StringIO(split_path.read_text()))) if split_path.exists() else []
    coverage_path = report / 'ood_coverage_crosswalk.csv'
    coverage = list(csv.DictReader(io.StringIO(coverage_path.read_text()))) if coverage_path.exists() else []
    candidates = [r for r in coverage if r['authorized_execution'].lower() != 'true']
    snapshot_path = report / 'fig4_plotted_snapshot.csv'
    figure_snapshot = list(csv.DictReader(io.StringIO(snapshot_path.read_text()))) if snapshot_path.exists() else []
    now = datetime.now(timezone.utc).isoformat()
    summary = dict(
        generated_utc=now, source=str(source), source_sha256=hashlib.sha256(payload).hexdigest(),
        source_models=len({r['model'] for r in rows}), ood_indexed_models=len({r['model'] for r in ood}),
        ood_metric_rows=len(ood), indexed_dataset_families=len(groups),
        ood_detection_dataset_families=sum(role(d) == 'ood_detection' for d in groups),
        native_downstream_dataset_families=sum(role(d) == 'native_downstream' for d in groups),
        unclassified_dataset_families=sorted(d for d in groups if role(d) == 'unclassified'),
        candidate_families_without_admitted_protocol=len({r['actual_dataset'] for r in candidates}),
        all_audited_source_families=len({r['actual_dataset'] for r in coverage}),
        fm_split_file='paper_results_ood.csv', fm_split_models=len({r['model'] for r in split}),
        figure_snapshot_file='fig4_plotted_snapshot.csv',
        figure_snapshot_models=len({r['model'] for r in figure_snapshot}),
        figure_snapshot_mtime_utc=(datetime.fromtimestamp(snapshot_path.stat().st_mtime, timezone.utc).isoformat()
                                  if snapshot_path.exists() else None),
        full_candidate_inventory_evaluated=False,
        overall_ood_generalization_claim_supported=False,
        count_definition='Rows are model/task/fold/metric entries, not independent experiments. '
                         'Numeric/complete status is inherited from the source table, not revalidated; '
                         'it does not certify completed CI/tests or pretraining disjointness.',
        datasets=scope_rows,
    )
    atomic_text(report / 'ood_scope_status.json', json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# OOD 范围与结论边界',
        f'范围审计时间：{now}。来源：`paper_results.csv`；SHA256：`{summary["source_sha256"]}`。这里只读取已落盘结果，未运行新实验。',
        '**X-ray tomography 与 Cryo-EM 只是 OOD detection 子集，不能代表全部 OOD，更不能据此归纳跨领域下游泛化能力。** 两者衡量与固定 ID 参考库的可区分性；高 AUROC/AP 不等于外部分类、回归或分割准确率高。',
        f'当前总表登记 {len(groups)} 个数据族：{summary["ood_detection_dataset_families"]} 个来源判别数据族、{summary["native_downstream_dataset_families"]} 个扩充原生任务数据族、{len(summary["unclassified_dataset_families"])} 个范围未分类族。登记包含缺测占位；不能表述为全部完成。以下覆盖是总表索引状态，未重新验证原始预测或统计结果。',
        '| 数据族 | 评测角色 | 任务 | 有数值模型/登记模型 | 有数值指标行/登记行 | complete 指标行 | partial 指标行 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in scope_rows:
        lines.append(f'| {row["name"]} | {row["evaluation_role"]} | {row["task"]} | '
                     f'{row["numeric_models"]}/{row["indexed_models"]} | '
                     f'{row["numeric_metric_rows"]}/{row["indexed_metric_rows"]} | '
                     f'{row["complete_metric_rows"]} | {row["partial_metric_rows"]} |')
    lines += [
        '',
        'NuCLS 的任务、fold、指标分列，Cryo 四个 project 的来源行不算四个新增数据族；上述指标行和模型数均不是独立重复次数。`complete` 也不表示置信区间与配对检验已经完成，须另查原表统计状态。',
        '扩充任务按数据集和任务报告。是否属于新模态、来源迁移或任务迁移，须相对于具体模型的预训练来源判断；目录的 `OOD` 标签不是全部模型均未见过该数据的证明。BCCD/BBBC041/NuCLS 等不可仅因放入此表便称为全新模态。',
        f'候选清单合计 {summary["all_audited_source_families"]} 个来源族，其中 {summary["candidate_families_without_admitted_protocol"]} 个尚未纳入冻结的执行协议；它们包括仍待审查、已知重叠或重复来源，并非全部已认证 OOD。原始 36 来源审查清单保留在 [覆盖交叉表](ood_coverage_crosswalk.md)。九个已登记族不是用户扩充范围的全集；撤回的 24 域 survey 分数不恢复。',
        'Spheroid 仍因训练/测试来自相同完整 spheroid 且 patch 重叠而阻塞，不补零、不当作已完成。未登记的 OOD retrieval/clustering 协议也不能借 ID 对应结果宣称已完成。',
        'X-ray/Cryo 当前采用 `ood_groupfix_v3`：同一批 3,000 ID 图像按 FOV 组固定为 2,098 bank / 902 ID test，cosine kNN10；X-ray 992 slices / 124 volumes，Cryo 80,000 particles / 4 projects。`ood/` 与 `ood_inputfix_v2/` 的旧分数仅供追溯。此协议和“条件于固定 ID bank”的区间解释仅适用于来源判别子集。',
        '**允许的结论：** 在指定 ID 参考库下区分 X-ray/Cryo；在各扩充数据集及其冻结协议下的分类、回归、检测等表现。**不支持的结论：** 由两个来源判别分数推断全部 OOD 泛化，或将这些异质指标直接平均成一个总体 OOD 分数。',
        '## 文件口径',
        f'- `paper_results.csv` / `paper_results.xlsx`：全量模型/anchor 总表；本次读取 {summary["source_models"]} 个模型条目。',
        '- [ood_results_scoped.csv](ood_results_scoped.csv)：从全量总表筛选 OOD，保留全部原始列和值，并增加 `evaluation_role` 与 `inference_scope`；适合后续分任务汇总。',
        f'- `paper_results_ood.csv`：FM 发布脚本写入的子表，本次 {summary["fm_split_models"]} 个 FM，不能当成全量 BioDINO + FM 覆盖。',
        '- [ood_dataset_status.csv](ood_dataset_status.csv) / [ood_scope_status.json](ood_scope_status.json)：本次逐数据族覆盖及输入哈希。',
        f'- `fig4_plotted_snapshot.csv`、`paper_figures.xlsx` 及 Fig.4 图片属于历史渲染快照；CSV 的修改时间为 {summary["figure_snapshot_mtime_utc"]}，包含 {summary["figure_snapshot_models"]} 个固定主图模型。本次仅修正文档和范围表，未重绘，不代表当前进度。',
        '- `current_biodino_ood.csv` 是固定 BioDINO 主图模型的历史任务摘要，已包含扩充数据族，但不是全模型实时进度。',
        '- `ood_score_trace.csv`、`ood_score_by_id_source.csv` 只覆盖 X-ray/Cryo 判别审计。`unified_execution_summary.json` 的旧 `ood_metrics_completed` 也是这个子集的历史计数。',
        '',
        '生成入口：`scripts/report_ood_scope.py`；覆盖审计和当前成绩刷新脚本也调用此入口。图形构建/渲染主脚本当前为空文件，修复前不能用旧 `--refresh` 命令宣称已重绘。deepcad 评测继续保持用户要求的停止状态。',
    ]
    # Keep table/list rows contiguous; separate prose paragraphs for Markdown.
    markdown = []
    for line in lines:
        if not line:
            continue
        if markdown and not (line.startswith('|') and markdown[-1].startswith('|')) and not (
                line.startswith('- ') and markdown[-1].startswith('- ')):
            markdown.append('')
        markdown.append(line)
    atomic_text(report / 'OOD_SCOPE.md', '\n'.join(markdown) + '\n')
    return summary


if __name__ == '__main__':
    status = refresh()
    print(json.dumps({k: status[k] for k in ('generated_utc', 'indexed_dataset_families',
                                           'ood_detection_dataset_families', 'native_downstream_dataset_families')}))
