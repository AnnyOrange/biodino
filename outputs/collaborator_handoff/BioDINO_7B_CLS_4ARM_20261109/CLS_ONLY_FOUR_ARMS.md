# 21503 恢复：四组 CLS-only 辅助教师对照

2026-11-09，按用户纠正更新。代码与四组配置模板已写入；随后已完成合成输入的 CPU CLS loss/梯度核对，见 [与 lw2/lw5 的差异及验证](CLS_LW2_LW5_AUDIT.md)。尚未进行真实教师、完整恢复或训练验证。

## 当前实验定义

**四组学生、在训 EMA、训练 heads 和 AdamW 均从同一份 `ckpt/21503/` 恢复，下一步为 21504。** 13311 只提供 B 组的冻结辅助教师，不作为任一组的学生/optimizer 恢复起点。

| 组 | 冻结辅助教师 | CLS 辅助损失 | overlay |
|---|---|---|---|
| A | 官方 DINOv3-7B | `L_official` | `dinov3/configs/train/7b_cls_dinov3.yaml` |
| B | 历史 7B 第 13311 步 EMA | `L_history13311` | `dinov3/configs/train/7b_cls_history13311.yaml` |
| C | 官方 DINOv3-7B + H-optimus-0 | `0.5 L_official + 0.5 L_hopt` | `dinov3/configs/train/7b_cls_dinov3_hoptimus0.yaml` |
| D | 官方 DINOv3-7B + GenBio-PathFM | `0.5 L_official + 0.5 L_genbio` | `dinov3/configs/train/7b_cls_dinov3_genbio_pathfm.yaml` |

用户选择 21503 作为总体最佳恢复点，希望考察历史 13311 的分类/分割优势能否作为辅助知识保留。B 组采用此前评测对应的 EMA 分支，保持固定，不随学生更新。

原 DINO/iBOT 主损失、在训 EMA 更新继续保留。新增蒸馏只使用 CLS：`global_readout=cls`，`patchmean_weight=0`、`patch_weight=0`。这是 CLS 辅助表征约束；不能仅据历史分割成绩推断它必然改善局部分割。

## 实现与比较口径

- 每位教师有独立 ridge decoder 和统计，只有学生接收梯度；不使用 CorrCA 目标。
- 官方 7B 和历史 7B 的 CLS 为 4096 维，H-optimus-0 为 1536 维，GenBio RGB CLS 为 4608 维。
- 冻结教师 BF16、永久 eval、分块推理；CLS 模式跳过 patch pooling、拼接及 patch 蒸馏统计。原 ViT/iBOT 仍计算 patch tokens。
- 同一学生第一张 global crop，保留原 iBOT masking；辅助教师看该增强图像的无 mask 输入。
- 历史 13311 使用其原训练配置中的 RGB normalization 和 global crop 尺寸（`native_preprocess=checkpoint_config`）；官方教师使用其发布模型的 normalization。学生均保留 21503 的配置。
- 四组附加 update 数、batch、累积、LR/温度/EMA 日程、数据与增强保持一致。总辅助系数默认 1，256 次 optimizer update 渐入；双教师取等权平均。这是初始实验设置，尚无效果或梯度范数等效的验证。
- 恢复检查绑定模型/AdamW、分片、起点、教师文件指纹和读出设置；不同组的蒸馏统计不能互相恢复。每 1024 次 optimizer 更新保存完整 DCP 与 EMA，以固定 21503 为计数起点，中途恢复不重置；非周期末步也保存。2048 更新预算对应 22527、23551。EMA 导出不自动执行 FM ID，需接入合作方评测队列。

## 准备入口与材料

`scripts/prepare_7b_global_distillation.py` 现在准备四组，源目录编号仅接受 21503。所有 overlay 固定 `resume_origin_iteration=21503`、`start_update=21504`。

```bash
python scripts/prepare_7b_global_distillation.py \
  --source-config /actual_7b_run/config.yaml \
  --source-checkpoint /actual_7b_run/ckpt/21503 \
  --expected-origin-iteration 21503 \
  --teacher-manifest /local_verified/teacher_manifest.json \
  --output-root /new_runs/7b_cls_four_arms_from21503 \
  --updates ADDITIONAL_UPDATE_BUDGET
```

路径与附加预算仍是占位；本轮未执行准备器或生成实际运行目录。首次启动核对 `--expected-start-iteration 21504`；后续恢复按各组最新完整保存点继续。

需要分别绑定两种历史文件：

1. **学生恢复：**21503 的完整 DCP（student、EMA、heads、AdamW、iteration）和实际运行配置。
2. **B 组教师：**13311 的已合并 EMA/backbone 权重文件及对应原训练配置。无需使用 13311 的 optimizer。当前教师 loader 读取权重文件，不能直接把 `sharded_teacher_checkpoint/` 目录传给它；若仅有分片，需先正确合并并核验导出身份。

教师 manifest 包含四个有权重/配置 SHA256 的条目：

- `dinov3_7b`：`kind=dinov3`，官方 7B，`provenance.checkpoint_origin=official_dinov3_release`。
- `history_7b_13311`：`kind=dinov3`，`checkpoint_branch=teacher`、`native_preprocess=checkpoint_config`，`provenance.checkpoint_origin=historical_7b_ema`、整数 `provenance.checkpoint_iteration=13311`；`img_size` 与该历史配置的 global crop 尺寸一致。
- `hoptimus0`、`genbio_pathfm`：沿用已登记的模型身份和本地文件；GenBio 自定义源码也需哈希。

两个 7B 条目的 manifest `output_dim=8192` 是共享 loader 默认 CLS+patch mean 接口的维度；本轮显式 CLS 读出为 4096。各条目继续要求 image-only 自监督、source、method 来源记录。准备器核对历史模型的配置、分支、声明步数和文件类型；这些检查不能代替原始导出记录与权重 SHA 的对应关系核验。

## 完成状态

- 四组组合、21503 恢复点、13311 辅助教师角色：用户已明确，配置已更新。
- CLS-only 多教师接口、四组模板、准备器：已写入并人工阅读；已做合成教师输出的 CPU 数值对照，未运行真实模型或完整测试套件。
- 21503 完整 DCP、13311 合并 EMA、实际配置与最终 manifest：实际路径/身份尚未绑定到本轮运行。
- 训练、显存峰值、吞吐、恢复验证及 FM 评测：未运行。7B 冻结辅助教师当前按 GPU 复制，实际资源预算仍需测量。

旧 `CLS_ONLY_THREE_ARMS.md` 的“从 13311 恢复”是已纠正的理解，不能再用于本轮启动。
