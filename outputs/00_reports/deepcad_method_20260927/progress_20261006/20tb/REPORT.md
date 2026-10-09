# 20TB 同步数评测：阶段结果

UTC 2026-10-09 11:00:58；已校验 1334/1558 个任务。

仅包含双方均完成且通过校验的相同步数结果。队列仍在运行；以下不是完整 v4 结论，也不能用于挑选最佳 checkpoint。

| 任务 | 数据集 | 指标 | 设置 | 配对数 | 均值 Δ×100 | 胜/负 |
|---|---|---|---|---:|---:|---:|
| classification | bbbc048-cellcycle | balanced_accuracy |  | 15 | +3.818 | 15/0 |
| classification | bloodmnist | balanced_accuracy |  | 15 | +0.003 | 10/5 |
| classification | breastmnist | balanced_accuracy |  | 16 | -1.864 | 2/13 |
| classification | chammi-allen-task1 | balanced_accuracy |  | 15 | +1.143 | 11/4 |
| classification | chammi-allen-task2 | balanced_accuracy |  | 15 | +0.780 | 11/4 |
| classification | chammi-cp-task1 | balanced_accuracy |  | 15 | +0.389 | 14/1 |
| classification | chammi-cp-task2 | balanced_accuracy |  | 15 | -0.302 | 5/10 |
| classification | chammi-cp-task3 | balanced_accuracy |  | 15 | +1.225 | 11/4 |
| classification | chammi-hpa-task1 | balanced_accuracy |  | 15 | +0.435 | 15/0 |
| classification | chammi-hpa-task2 | balanced_accuracy |  | 15 | +0.485 | 12/3 |
| classification | chestmnist | macro_auc |  | 15 | +1.105 | 15/0 |
| classification | cyclops-protein-loc | balanced_accuracy |  | 15 | +1.812 | 15/0 |
| classification | dermamnist | balanced_accuracy |  | 15 | +0.607 | 10/5 |
| classification | midog25-atypical | balanced_accuracy |  | 15 | +0.517 | 10/5 |
| classification | nct-crc-he | balanced_accuracy |  | 15 | -9.743 | 0/15 |
| classification | octmnist | balanced_accuracy |  | 15 | -0.393 | 3/10 |
| classification | organamnist | balanced_accuracy |  | 15 | +0.184 | 11/4 |
| classification | organcmnist | balanced_accuracy |  | 15 | +0.208 | 13/2 |
| classification | organsmnist | balanced_accuracy |  | 15 | -0.518 | 2/13 |
| classification | pathmnist | balanced_accuracy |  | 15 | -0.737 | 2/13 |
| classification | pcam | balanced_accuracy |  | 15 | -0.053 | 6/9 |
| classification | pneumoniamnist | balanced_accuracy |  | 16 | +0.692 | 12/4 |
| classification | retinamnist | balanced_accuracy |  | 15 | +2.396 | 14/1 |
| classification | tissuemnist | balanced_accuracy |  | 15 | +0.600 | 15/0 |
| regression | bbbc005 | r2 |  | 15 | -0.446 | 2/13 |
| regression | bbbc013 | r2 |  | 16 | -2.961 | 1/15 |
| retrieval | crc-val-he-7k | nmi | class | 15 | -2.213 | 3/12 |
| retrieval | crc-val-he-7k | recall_at_1 | class | 15 | -0.091 | 0/15 |
| retrieval | hpa-subcellular | nmi | location | 30 | -2.026 | 0/30 |
| retrieval | hpa-subcellular | recall_at_1 | global | 15 | -0.478 | 1/14 |
| retrieval | nct-crc-he-1k | nmi | class | 16 | +4.122 | 15/1 |
| retrieval | nct-crc-he-1k | recall_at_1 | class | 16 | -0.263 | 1/13 |
| retrieval | rxrx1-cross | nmi | global-perturbation | 15 | +1.606 | 15/0 |
| retrieval | rxrx1-cross | recall_at_1 | HEPG2 | 15 | -0.271 | 2/12 |
| retrieval | rxrx1-cross | recall_at_1 | HUVEC | 15 | -1.179 | 0/15 |
| retrieval | rxrx1-cross | recall_at_1 | RPE | 15 | -0.845 | 0/15 |
| retrieval | rxrx1-cross | recall_at_1 | U2OS | 15 | -0.087 | 1/11 |
| retrieval | rxrx1-cross | recall_at_1 | global | 15 | -0.532 | 0/15 |
| retrieval | rxrx1-cross | recall_at_1 | macro-cell-type | 15 | -0.596 | 0/15 |
| segmentation | cellpose | mDice | E20 | 15 | +1.041 | 14/1 |
| segmentation | cellpose | mDice | E50 | 15 | +0.771 | 13/2 |
| segmentation | conic | mDice | E20 | 15 | -0.016 | 9/6 |
| segmentation | conic | mDice | E50 | 15 | +0.187 | 12/3 |
| segmentation | livecell | mDice | E20 | 15 | +0.187 | 14/1 |
| segmentation | livecell | mDice | E50 | 15 | +0.189 | 14/1 |
| segmentation | multimodal_cellseg | mDice | E20 | 15 | -0.154 | 2/13 |
| segmentation | multimodal_cellseg | mDice | E50 | 15 | -0.131 | 3/12 |
| segmentation | pannuke | mDice | E20 | 45 | -0.143 | 19/26 |
| segmentation | pannuke | mDice | E50 | 45 | -0.090 | 16/29 |
| segmentation | tissuenet | mDice | E20 | 15 | -0.081 | 6/9 |
| segmentation | tissuenet | mDice | E50 | 15 | -0.162 | 3/12 |

分类/检索/分割的 Δ×100 为百分点；R² 的该列只是差值乘100，不是准确率百分点。跨 checkpoint 的结果相关；本表未进行显著性检验。
