#!/usr/bin/env python3
"""Refresh local/shared deployment evidence without changing training jobs."""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / 'outputs/01_training_runs/hs6_l_20tb_v2_recovery_fork38063_20261009'
OUT = ROOT / 'outputs/00_reports/20tb_resampling_adaptive_v2_plan_20261009'
BASE = ROOT / 'outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_mix009_021_0604_0096_8x5090zxr_20260924'


def state(name):
    run = GROUP / name
    path = run / 'raw_loss_metrics.jsonl'
    latest = None
    if path.exists():
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size-65536))
            for line in reversed(f.read().decode().splitlines()):
                try:
                    latest = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass
    return dict(run=str(run), latest=latest,
                full_checkpoints=sorted(int(p.parent.name) for p in (run / 'ckpt').glob('*/checkpoint.pth')
                                        if p.stat().st_size > 6_000_000_000))


def main():
    c = state('fixed_cls_w1_anchor38063')
    b = state('resampling_only_38063_8x3090qi')
    pause = json.loads((BASE / 'USER_PAUSE_FOR_ADAPTIVE_V2_20261009.json').read_text())
    actual = OUT / 'actual_source_index'
    manifest = json.loads((actual / 'manifest.json').read_text()) if (actual / 'manifest.json').exists() else None
    preflight = json.loads((actual / 'PREFLIGHT.json').read_text()) if (actual / 'PREFLIGHT.json').exists() else None
    tail_path = OUT / 'r9_micro_tail_audit.json'
    r9_tail = json.loads(tail_path.read_text()) if tail_path.exists() else None
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = dict(updated=now, original_pause=pause, C=c, B=b,
                  actual_sampling_manifest=manifest, actual_sampling_preflight=preflight, r9_micro_tail=r9_tail)
    (OUT / 'deployment_status.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    progress = lambda s: str(s['latest']['optimizer_update']) if s['latest'] else '尚无 optimizer update 记录'
    concentration_text = '实际索引尚未完成。'
    if manifest:
        concentration_text = '|实际池|样本数|来源ESS：原→重采样|最高频1%来源占比：原→重采样|\n|---|---:|---:|---:|\n'
        for item in manifest['concentration']:
            concentration_text += (f"|{item['pool']}|{item['patches']:,}|"
                                   f"{item['baseline_source_ess']:,.0f} → {item['target_source_ess']:,.0f}|"
                                   f"{100*item['baseline_top1pct_source_probability']:.3f}% → "
                                   f"{100*item['target_top1pct_source_probability']:.3f}%|\n")
    tail_text = ''
    if r9_tail:
        tail_text = f"""真实r0的来源分散程度显著好于历史整份15TB显微清单。两者之差位于r9尾部：{r9_tail['r9_micro_samples']:,}个显微tile来自{r9_tail['r9_micro_sources']:,}个来源，其tile频率对应来源ESS仅{r9_tail['r9_micro_source_ess']:.1f}，最大单来源占该显微子集{100*r9_tail['largest_source_probability_within_r9_micro']:.2f}%；最高频来源为idr0116。它们由正式15TB显微清单减去实际r0成员数得到，详见`r9_micro_tail_audit.json`。

本轮B按计划保留r9流，因此**没有纠正这部分集中度**。下一轮值得单独比较：在r9内部区分病理与显微尾部、保持两者采样份额，仅均衡显微尾部来源。当前B/C先作为固定配方的首轮对照，不把r0的ESS改善误写成全20TB来源均衡已解决。
"""
    text = f'''# 20TB adaptive-v2 与重采样执行记录

更新时间：{now}。下列步数来自训练 JSONL，不将进程启动或 teacher 导出当作完整训练/下游评测完成。

|实验|机器|起点|采样|方法|最新记录|
|---|---|---|---|---|---|
|C|本机 8×5090|完整 ck38063|原配比|固定 CLS anchor38063，w1|{progress(c)}|
|B|3090-qi 8×3090|完整 ck38063|六池 source 重采样|原 No-GRAM|{progress(b)}|

deepcad 已运行第二条 boundary 20TB 分支，启动前仅约91GiB可用内存；3090-qi八卡空闲、约679GiB可用内存，且已有全部数据路径，因此把 B 放在3090-qi。数据图片不复制、不重打包；约17GB旧 SQLite 元数据缓存到3090-qi本地盘用于加快索引，原checkpoint直接读现有共享目录。

## 状态与保留策略

- 原本机20TB进程及其自动重启 watcher 已停止。最后观察到74900，最新完整可恢复断点为 **74663**（357个optimizer state）；最后237次更新没有额外完整落盘。所有原有checkpoint和teacher导出保留。
- C 的完整fork已验证iteration38063、357个optimizer state、Adam step38064；原student、EMA teacher和AdamW动量连续，只新增固定锚点和recovery统计buffer。
- ck26351 teacher单独保留于 `{GROUP / 'anchor26351_control/teacher_checkpoint.pth'}`。它用于后续同样从完整38063恢复的锚点对照；该对照尚未启动。
- B/C保持61×4098原LR/WD/teacher日程、8卡×16×累积8、有效batch1024、全局microbatch128及输入归一化。`max_updates=43920`，完成5856次新更新后结束首轮。
- 每488次更新保存teacher及 **model+optimizer**，`checkpointing.max_to_keep=null`，不自动删旧断点。首个新完整断点为38551；正式比较点为40015/41967/43919。
- C已保存完整断点：{c['full_checkpoints']}；B：{b['full_checkpoints']}。

## 实际重采样实现

顶层比例依次为：旧1TB **9%**、旧4TB **21%**、r0 **45.81%**、r9 **14.19%**、boundary旧OID **4%**、boundary新OID **6%**。旧1TB/4TB/r9使用原streaming读取；另外三个池从当前正式tar的实际成员清单重算概率。

池内保持成像模态边际，source质量为`sqrt(min(n,64))`；项目概率限制在原值0.5–2倍，项目内混合一半旧tile分布、一半source质量。项目调节强度受最高频1%来源占比、最大单来源概率、sum(p²)不升的三个约束。它们是**每个索引池内**约束，不声称已经证明全20TB或旧流共享OID的总集中度下降。

实际运行采用source概率CDF抽样，再选一个crop；worker内按随机起点/互质步长无放回轮换，跨worker使用不同seed。**跨worker仍可能选中相同source/crop，不承诺全局无重复。** 2265818条训练来源记录的OID均唯一；boundary旧/新按旧5TB OID表划分。

与初始设计中的顺序扫描设想不同，最终使用本地SQLite和numpy元数据索引，从原tar按payload偏移读取。每worker预取32个样本，按archive/offset安排4路并发读取，再恢复原随机抽样顺序；最多缓存64个文件句柄。是否能运行以真实解码和吞吐预检为准。索引只验证成员结构及唯一key；不是对千万级所有图片的穷尽解码，运行中发现无效样本会报错停止，避免悄悄改变概率。

实际项目/模态命中与独立OID数记录在 B 的 `sampler_audit/`。详细概率及预检见 `actual_source_index/`（若尚未生成，该阶段仍在执行）。

{concentration_text}

上述ESS描述采样概率，不是模型性能或患者独立性。

{tail_text}

## 验证与效果边界

新sampler的4项单元验证涵盖10万级来源抽样、每source裁块轮换、独立seed、tar payload读序、解码一致性及archive变化检查。正式启动前，对每个索引池再做30万次概率抽样、16个独立tar文件头核验、512张实际解码，并验证1024张六池混合样本及读取速度。

这些检查证明数据与训练接入，不等于模型效果提高。当前没有新分支完整v4结果；后续需按一致协议覆盖分类、回归、检索、聚类、分割、检测等任务，不能只看分类或将training的teacher导出称为下游test完成。已有历史A为参考；D（重采样+固定CLS）保留为后续组合对照，尚未启动。

运行目录：

- C：`{c['run']}`
- B：`{b['run']}`
- 两个独立runtime快照及源码hash：`{GROUP}`
- 3090-qi实际索引：`/home/bbnc/20tb_resampling_20261009/index/`
'''
    (OUT / 'DEPLOYMENT.md').write_text(text)
    print(json.dumps(dict(updated=now, C=progress(c), B=progress(b), preflight=preflight and preflight['status'])))


if __name__ == '__main__':
    main()
