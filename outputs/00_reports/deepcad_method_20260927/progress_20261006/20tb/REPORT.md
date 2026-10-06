# 20TB 同步数评测：阶段结果

UTC 2026-10-06 03:14:52；已校验 126/532 个任务。

仅包含双方均完成且通过校验的相同步数结果。队列仍在运行；以下不是完整 v4 结论，也不能用于挑选最佳 checkpoint。

| 任务 | 数据集 | 指标 | 设置 | 配对数 | 均值 Δ×100 | 胜/负 |
|---|---|---|---|---:|---:|---:|
| classification | bloodmnist | balanced_accuracy |  | 4 | +0.022 | 3/1 |
| classification | breastmnist | balanced_accuracy |  | 6 | -3.227 | 0/5 |
| classification | dermamnist | balanced_accuracy |  | 6 | +1.219 | 6/0 |
| classification | organcmnist | balanced_accuracy |  | 6 | +0.131 | 4/2 |
| classification | pathmnist | balanced_accuracy |  | 1 | -0.856 | 0/1 |
| classification | pneumoniamnist | balanced_accuracy |  | 6 | +0.691 | 5/1 |
| classification | retinamnist | balanced_accuracy |  | 6 | +2.447 | 6/0 |
| regression | bbbc013 | r2 |  | 6 | -0.720 | 1/5 |
| retrieval | nct-crc-he-1k | nmi | class | 6 | +3.620 | 5/1 |
| retrieval | nct-crc-he-1k | recall_at_1 | class | 6 | +0.017 | 1/3 |
| segmentation | cellpose | mDice | E20 | 4 | +0.531 | 3/1 |
| segmentation | cellpose | mDice | E50 | 4 | +0.242 | 2/2 |

分类/检索/分割的 Δ×100 为百分点；R² 的该列只是差值乘100，不是准确率百分点。跨 checkpoint 的结果相关；本表未进行显著性检验。
