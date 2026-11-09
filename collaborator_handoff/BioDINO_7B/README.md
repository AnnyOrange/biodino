# 7B 训练交付说明

历史训练基线为 `1b1c9b9b1f4ad9ba16a7fbfe897083086c01e476`。本分支现已包含 FM 评测及从 21503 恢复的四组 CLS 辅助蒸馏更新；请保留合作方 S3 和本地 DCP 适配，合并后用于新的续训目录。

## 当前执行：21503 恢复，四组 CLS 辅助蒸馏

完整代码已在 `collaborator/fm-id-ood-20261102` 分支中，无需下载另一条训练分支。

| 组 | 冻结辅助教师 | 学生恢复点 | 新增 optimizer 更新 |
|---|---|---:|---:|
| A | 官方 DINOv3-7B | 21503 | 2048 |
| B | 历史 7B EMA 13311 | 21503 | 2048 |
| C | 官方 DINOv3-7B + H-optimus-0 | 21503 | 2048 |
| D | 官方 DINOv3-7B + GenBio-PathFM | 21503 | 2048 |

保留 DINO/iBOT 主损失，新增蒸馏只使用 CLS。13311 只作为 B 组辅助教师，四组都恢复 21503 的完整 student、EMA、heads 和 AdamW。

- **保存与评测：**从恢复点起每 1024 次 optimizer 更新保存完整 checkpoint 并导出 EMA，编号 **22527、23551**；中途恢复不重置周期。请将两个 EMA 导出点接入现有 FM ID 评测流程，使用一致协议，先统一下游 seed0，优先观察分类及分割并与起点比较。训练入口不会自动运行 FM ID。
- **准备与合并步骤：**[四组续训操作说明](train/cls_distillation/README.md)。已使用当前分支的代码时，不要重复应用补丁；保留自定义工作副本时按说明合并。
- **损失口径：**[与旧 lw2/lw5 的核对](train/cls_distillation/CLS_LW2_LW5_AUDIT.md)。当前为教师损失平均、总权重 1、256 更新渐入，不等同于旧完整配方。
- **实际启动前：**绑定真实配置、21503 完整 DCP、13311 合并 EMA 及教师 manifest，先确认真实教师加载、完整恢复与显存。当前未完成真实 7B GPU 训练验证。
- **评测协议：**[FM ID/OOD 交付入口](../FM_evaluation/README.md)。

## 原训练配置与历史依据

- train/7b_b200.yaml：原8卡GB1024基准配置；合作方64卡GB4096覆盖参数另行记录，已获接受。不把此配置误称为64卡实际解析配置。
- attachments/provenance/launch_7b_20tb_hxw_20261015.sh.txt：历史配方依据，只读，不执行。
- attachments/provenance/rgb_normalization_verification_20261102.json：与20TB L实际mean/std的核查记录。
- attachments/provenance/stage_2_train_labels.csv：RSNA来源标签；评测使用固定manifest，不重新划分。
- [完整评测入口及步骤](../FM_evaluation/README.md)。不用额外代码ZIP。

训练环境要求Python>=3.11；不要求立即中断当前Python3.10运行或原地升级。下一次启动/恢复先用独立环境验证完整DCP。最新选点和保存安排见评测README，原“全部永久保留”不再是最终清理要求，但未测齐候选与最新可恢复状态必须保留到验收完成。
