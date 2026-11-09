# 7B：从 21503 恢复的四组 CLS 辅助蒸馏

2026-11-09。代码交接包；未启动训练，未完成真实教师加载及完整 DCP 恢复验证。不包含权重、评测数据或实际训练配置。

## 实验与建议预算

所有组从同一份 **21503 完整 student / EMA / heads / AdamW checkpoint** 恢复，下一步 21504，保留 DINO/iBOT 主损失。13311 仅为 B 组的冻结辅助教师。

| 组 | 冻结辅助教师 | 建议首次续训更新数 | 最后 checkpoint |
|---|---|---:|---:|
| A | 官方 DINOv3-7B | 2048 | 23551 |
| B | 历史 7B EMA 13311 | 2048 | 23551 |
| C | 官方 DINOv3-7B + H-optimus-0 | 2048 | 23551 |
| D | 官方 DINOv3-7B + GenBio-PathFM | 2048 | 23551 |

建议第一轮四组各跑一个训练 seed、相同数据流与训练预算，共 8192 次 optimizer 更新。不是四组各重新预训练，也不是每组立即做五个 backbone seed。

预算依据：合作方此前记录实际 global batch 为 4096；旧 L 的 4880 更新 × global batch 1024 = 4,997,120 次图像采样，在 7B 上对应 **1220 更新**。推荐的每组 2048 更新约为该采样预算的 1.68 倍；四组共 33,554,432 次图像采样。实际配置若改变 global batch，必须重新换算。不能据此推断 GPU 小时，双教师组的开销也不同。

2048 是建议首轮预算，不是已证明最优的训练长度。若要严格匹配旧 L 的采样预算，四组统一设 `--updates 1220`，最后 checkpoint 为 22723。首轮验证仍在改善时，可以统一延长至总计 4096 更新（最终 25599）；只延长胜出组属于探索，不能作为四组等预算结论。

保存原始 21503 的基线结果。建议通过训练验证集观察分割与分类，最终点按固定 FM ID 协议比较；不要反复按最终 test 选 checkpoint。不同下游 seed 与 backbone 训练 seed 分开记录。仅与起点比较不能分离继续训练本身的收益；若要证明辅助蒸馏的因果收益，建议另增加一个相同预算、无辅助教师的续训对照（第五组，尚未加入准备器）。

准备器已设置 `checkpointing.period=1024`、`evaluation.eval_period_iterations=1024`、`checkpointing.period_origin_iteration=21503`。每累计 1024 次 optimizer 更新保存完整 DCP 和导出 EMA，2048 更新预算对应 **22527、23551**；中途恢复仍按固定起点计数，不重置周期。最终点即使不足周期也保存，和周期重合只保存一次。`PREPARED.json` 记录预期保存点。这里的 `do_test()` 只导出 EMA，**不会自动执行 FM ID 测试**，合作方应让现有评测流程接入这两个导出点。预算结束是提前停止，保留原始 LR、EMA、teacher 温度日程，**不是旧 L 的晚期 cooldown**。

## 必须注意：当前配置不等于旧 lw2/lw5

直接复用旧 `ExternalRecoveryStream` 的 CLS ridge loss 核心，但本包默认总辅助权重 1、双教师平均、256 更新渐入、momentum 0.98，只蒸馏 CLS。旧 lw2/lw5 同时蒸馏 CLS 与 patch、教师项求和、总权重 2/5；核对的 lw2 momentum 为 0.995。

本次同步未擅自改变这些配置，也不把它命名为严格 lw2/lw5 复现。详见 [核对记录](CLS_LW2_LW5_AUDIT.md)。四组默认使用相同辅助损失总系数，以便先比较教师组合。它不保证梯度幅度相同，更不保证 CLS-only 能修复分割。

## 集成代码

目标基线为合作方记录的 commit `1b1c9b9b1f4ad9ba16a7fbfe897083086c01e476`。

- `integration.patch`：完整交接补丁，包含三个既有文件的最小修改和七个新增文件。
- `files/`：七个新增文件的独立副本，方便阅读；应用补丁后不必再复制。
- `PROVENANCE.json`：基线及修改后源码 SHA256。旧 `external_distill.py` 与该基线完全一致。
- `SHA256SUMS`：包内文件校验清单。

在合作方工作副本保留其本地修改，并先检查补丁冲突：

```bash
git apply --check /path/to/BioDINO_7B_CLS_4ARM_20261109/integration.patch
git apply /path/to/BioDINO_7B_CLS_4ARM_20261109/integration.patch
```

如有冲突，人工合并相关 hunk，尤其保留 S3 数据读取及本地 checkpoint 保存修改；不要用当前主楼整份训练源码覆盖合作方代码。

## 准备四组目录

需要真实运行的配置（启动覆盖之后、自动 LR scaling 之前）、21503 完整 DCP、13311 已合并的 EMA 权重及其配置、四位教师的本地路径与真实 SHA256 manifest。具体字段见 [四组定义](CLS_ONLY_FOUR_ARMS.md)。这些路径目前尚未绑定；占位路径不能直接开训。

```bash
python scripts/prepare_7b_global_distillation.py \
  --source-config /actual_7b_run/config.yaml \
  --source-checkpoint /actual_7b_run/ckpt/21503 \
  --expected-origin-iteration 21503 \
  --teacher-manifest /local_verified/teacher_manifest.json \
  --output-root /new_runs/7b_cls_four_arms_from21503 \
  --updates 2048
```

此命令只准备四组独立目录，不启动训练。正式启动需沿用合作方实际分布式入口，并核对首次恢复 `--expected-start-iteration 21504`。冻结 7B 教师当前逐 GPU 复制；正式分配卡数前需要测量真实显存、吞吐及完整恢复。此前 CPU 合成输入核对不覆盖这些事项。

原 `BioDINO_7B/` 交接目录与 ZIP 均保留；本包不改动已冻结的 FM 评测协议。

## 2026-11-09 保存周期更新

已更新准备器、训练入口、默认配置及交接补丁。尚未应用旧补丁的工作副本使用 `integration.patch`；已应用旧包的工作副本使用 `save_interval_1024_update.patch`（不要将两个补丁重复应用）。本次未运行训练或测试；旧包的基线补丁检查不代表此更新已经过运行验证。

## GitHub 交付

分支：`collaborator/7b-cls-four-arms-20261109`（AnnyOrange/biodino）。此分支的源码已包含完整补丁，直接使用该分支时不要再次应用 `integration.patch` 或增量补丁。若在自己的定制工作副本合并，请保留本地 S3/保存改动，并按前文选择对应补丁。
