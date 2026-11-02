# 7B 训练交付说明

训练基线保持1b1c9b9b1f4ad9ba16a7fbfe897083086c01e476，保留合作方S3和本地DCP补丁。本分支新增的是评测交付，不要求覆盖正在训练的checkout。

- train/7b_b200.yaml：原8卡GB1024基准配置；合作方64卡GB4096覆盖参数另行记录，已获接受。不把此配置误称为64卡实际解析配置。
- attachments/provenance/launch_7b_20tb_hxw_20261015.sh.txt：历史配方依据，只读，不执行。
- attachments/provenance/rgb_normalization_verification_20261102.json：与20TB L实际mean/std的核查记录。
- attachments/provenance/stage_2_train_labels.csv：RSNA来源标签；评测使用固定manifest，不重新划分。
- [完整评测入口及步骤](../FM_evaluation/README.md)。不用额外代码ZIP。

训练环境要求Python>=3.11；不要求立即中断当前Python3.10运行或原地升级。下一次启动/恢复先用独立环境验证完整DCP。最新选点和保存安排见评测README，原“全部永久保留”不再是最终清理要求，但未测齐候选与最新可恢复状态必须保留到验收完成。
