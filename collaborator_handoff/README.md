# 合作方交付入口

代码和说明直接从本GitHub分支获取，不再上传额外代码ZIP。

- [FM完整ID/OOD评测、Spheroid入口与验收步骤](FM_evaluation/README.md)
- [7B 四组 CLS 续训、训练配置及配方依据](BioDINO_7B/README.md)
- [补传评测数据及实时上传状态](https://huggingface.co/datasets/annyorange/4tb/tree/main/dataset_transfer_fm_id_ood_supplement_20261102)
- [已有CTC、RxRx1等数据](https://huggingface.co/datasets/annyorange/4tb/tree/main/dataset_transfer_formal_v3_remaining_20260920)

本分支基于训练 commit `1b1c9b9b1f4ad9ba16a7fbfe897083086c01e476`，现已包含 7B 四组 CLS 续训更新。评测使用独立环境；新续训按 7B README 合并，保留合作方 S3/本地 DCP 适配及原训练目录。数据上传进行中，具体文件以HF的UPLOAD_STATUS.json为准。
