# 02 — Dataset and Split Rules

状态：APPROVED（2026-09-09）。这里区分“已有实现”和“是否适合正式发表”；发现有泄漏风险时，阻断评测，不沿用错误 split。

## 1. 固定数据集层级

### Tier A：论文主比较（所有 baseline / matched / shuffled 必须完整）

- classification/multilabel（15）：`bloodmnist pathmnist tissuemnist breastmnist organamnist organcmnist organsmnist dermamnist octmnist pneumoniamnist retinamnist chestmnist bbbc048-cellcycle cyclops-protein-loc midog25-atypical`
- regression（1）：`bbbc005`
- retrieval/clustering（2，LC25000 退出后）：`nct-crc-he-1k crc-val-he-7k`
- segmentation（7）：`bbbc038 cellpose conic livecell monuseg pannuke tissuenet`
- detection（1）：`livecell`

Tier A 对应既有 HS6 cross-model 主口径，但删除了 LC25000。聚合名称应改为 `Ret2`，不能继续叫旧的 `Ret3`。

### Tier B：完整生物学与跨成像扩展（正式 full suite 必须单独报告）

- classification：`pcam nct-crc-he chammi-allen-task1 chammi-allen-task2 chammi-cp-task1 chammi-cp-task2 chammi-cp-task3 chammi-hpa-task1 chammi-hpa-task2`
- regression：`bbbc013 conic-cell-count livecell-cell-count`
- retrieval/clustering：`hpa-subcellular rxrx1-cross`
- segmentation：`multimodal_cellseg`
- detection：`bbbc038 conic`
- OOD：`xray cryo`

Tier A 和 Tier B 都要跑，分别聚合；禁止用扩展集替换主集中的缺失数据。

### 禁止进入正式结果

- `lc25000`：已退出所有正式 classification/retrieval/clustering 评测。
- `nct-crc-he-100`：样本过小，只允许 smoke/debug。
- CHAMMI `hpa-task3`、`cp-task4`：Train 中不存在测试标签，是 open-set task；在专门 open-set protocol 完成前禁止用 closed-set logistic regression。
- 任意 `max_samples` / `max_per_class` / `SMOKE=1` 结果。

## 2. Classification / multilabel split

| 数据集 | 合法 train → test | 状态 |
|---|---|---|
| 12 个 MedMNIST | NPZ `train_images/labels` → `test_images/labels`；不把 val 并入 test | 合法 official split |
| nct-crc-he | NCT-CRC-HE-100K-NONORM train → CRC-VAL-HE-7K different-patient test | 合法 cross-patient split |
| pcam | parquet official train → official test | 合法 official split |
| CHAMMI 7 个 closed-set task | 对应 segment 的 `Train` → `Task_one/two/three` | 合法 official held-out split；启动前验证 test labels 是 train labels 子集 |
| cyclops-protein-loc | committed group split；group=`filename stem 去掉末尾 channel` | 合法 source-group split |
| bbbc048-cellcycle | committed group split；group=`文件名前缀 well/field id` | 合法 source-group split |
| midog25-atypical | committed group split；group=CSV source slide filename | 合法 slide-group split |

固定 group split 文件与当前 hash：

- `cyclops-protein-loc.json`: `886c960016d30fe011a45d8fbf94520fae97f8d7b4b8dcc287ea9087859b9d8c`
- `bbbc048-cellcycle.json`: `c8ea224712cc6e0564c36cebc28d33ce11fb659afb82568b25e1c5114d703f31`
- `midog25-atypical.json`: `f5c1806316e1cbfa45888059cf1ff1394723bd0fb03c7a1089d3c4ebe159011d`

任何 split 文件变化都创建新 protocol version，不能覆盖旧结果。

## 3. Regression split

| 数据集 | 合法协议 | 状态 |
|---|---|---|
| bbbc005 | committed 80/20 group split；group=`plate/count/field`；7675 train / 1925 test，seed 0 | 合法；split SHA256 `6bcfd65a7bd38e9a2e919f850409bc5c59eb59b6beece3ef62ce5d15840575b4` |
| bbbc013 | 按 compound 分开，对 log1p dose 做 leave-one-replicate-row-out；每 compound 4 folds，每 fold 36 train/12 test | 合法；不能改成普通随机 80/20 |
| conic-cell-count | prepared source-image-grouped CoNIC-10fold-v1 train/test | 合法，启动前验证 manifest hash |
| livecell-cell-count | LIVECell official COCO train → official test | 合法 official split |

BBBC013 的 `bbbc013.json` group split 文件存在，但正式 BBBC013 regression 采用 compound OOF 特例；不得误用普通 group 80/20。

## 4. Retrieval / clustering protocol

retrieval/clustering 不训练 supervised probe；“split”指固定样本集合或 disjoint query/gallery。

| 数据集 | 协议 | 状态 |
|---|---|---|
| nct-crc-he-1k | 固定 1K 集合，within-set leave-one-out；self-match 屏蔽 | Tier A 合法 |
| crc-val-he-7k | 固定 7K validation patient 集合，within-set leave-one-out；self-match 屏蔽 | Tier A 合法 |
| hpa-subcellular | `hpa_same_gene_query_gallery.csv` 的 disjoint query/gallery；clustering 使用 single-location manifest | Tier B 合法，manifest 必须 hash 固定 |
| rxrx1-cross | `rxrx1_official_cross_experiment_core.csv` 的跨 experiment query/gallery；默认 balanced core，不用 full | Tier B 合法，manifest 必须 hash 固定 |

统一 feature 为 final CLS + final patch mean、L2 normalized、224 crop、batch 64。retrieval 主指标 Recall@1，clustering 主指标 NMI；同时保存 Recall@5/10、mAP、MRR、ARI 和 cluster accuracy。

## 5. Segmentation / detection split 审计

| 数据集 | 当前 train / val / test | 审计结论 |
|---|---|---|
| LIVECell | official COCO train / val / test | 合法；segmentation 与 detection 共用 |
| TissueNet | official train / val / test NPZ | 合法 |
| MoNuSeg | official train 中固定 seed42 的 20% 做 val；official test 做 test | 合法；`monuseg_val_indices.npy` 必须固定 |
| Cellpose | public train pool 的确定性前 80%/后 20% 为 train/val；official test 为 test | 可用；必须固定排序与是否包含 `train_cyto2` |
| BBBC038 | 有 mask 的 stage1_train 做 seed42 70/15/15；官方 test 无 GT 不使用 | 可用但不是 official test；必须固定 `bbbc038_splits.npz` |
| Multimodal_CellSeg | train/val CSV；`test_source_heldout.csv` 只含 held-out Tuning source | 合法 source-heldout；WSI/超大图过滤规则必须固定 |
| CoNIC | official-baseline outer fold 0 + nested validation | **正式 v1**：按 source image 分组、cohort 分层；外层 80/20、seed 5、10 splits 取 fold 0；外层 train 内再以 seed 5 按 source 分层划 87.5/12.5，形成约 70/10/20；公开 20% 只能称 development holdout，不能称隐藏 challenge test |
| PanNuke | 官方 3-fold 轮换 | **正式 v1**：分别跑 `F1 train/F2 val/F3 test`、`F2 train/F1 val/F3 test`、`F3 train/F2 val/F1 test`；三组均报告并取均值，禁止 val=test |

当前已定位的固定文件：

- BBBC038 split SHA256：`4eb72dc1e58453126893261fedff661b7eb58dc9863ae04d991e7a5282f57ac4`（100 val、100 test，其余 train）。
- CoNIC legacy random index（3984/498/499）仅用于识别旧结果，禁止正式使用。正式协议标识为 `official-baseline-fold0-nested-v1`；本地得到 3469 train / 494 val / 1018 development holdout，index 与 source 均两两不相交。划分由 `patch_info.csv` 和确定性算法运行时生成，campaign 必须记录该 CSV 的 SHA256。
- PanNuke 本地 fold 大小固定为 F1=2656、F2=2523、F3=2722；三个正式协议标识分别为 `pannuke-fold1-train-fold2-val-fold3-test`、`pannuke-fold2-train-fold1-val-fold3-test`、`pannuke-fold3-train-fold2-val-fold1-test`。
- MoNuSeg val index SHA256：`932a09d0e936bd2ee83438145f6e6955dac224b747f2536ae06d31c764ff5c91`（7 个 val）。
- Multimodal split：876 train、176 val、101 source-heldout test；test source 为 `Tuning`，与 train/val source 集合分开。三个 CSV 的 SHA256 必须由 campaign manifest 完整记录。

## 6. Split 正确性的启动前自动检查

每个 campaign 必须通过：

1. train/val/test 路径或 sample id 两两无 overlap；
2. group-aware 数据集的 group id 两两无 overlap；
3. classification test labels 是 train labels 的子集（open-set 专项除外）；
4. 文件数量、类别数、split hash 与本规则/manifest 相同；
5. retrieval query/gallery 角色不重复，且每个 query label 在 gallery 中存在；
6. result 中记录的 `n_train/n_test/n_query/n_gallery` 与 manifest 相同；
7. PanNuke 必须展开为三个 protocol 并分别输出；CoNIC 必须使用 `official-baseline-fold0-nested-v1`，否则结果标记 `INVALID_PROTOCOL`。

## 7. 官方依据

- PanNuke 原论文说明数据被随机分成三个 training/validation/testing folds，组织类型在三份中均分，并对三个 split 的结果取平均：<https://arxiv.org/abs/2003.10778>。
- CoNIC 官方 baseline 的 `generate_split.py` 使用 `patch_info.csv` 的 source 前缀分组、cohort 分层、`SEED=5`、`StratifiedShuffleSplit(n_splits=10, train_size=.8, test_size=.2)`；官方 baseline 训练示例使用 fold 0：<https://github.com/vqdang/hover_net/blob/conic/generate_split.py>、<https://github.com/vqdang/hover_net/tree/conic>。
- CoNIC challenge test 标签由 Grand Challenge 持有；本地公开 4981 patch development data 的 20% 不得写成官方 challenge test：<https://github.com/TissueImageAnalytics/CoNIC>。
