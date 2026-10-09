# 10 — Union v4 非分割任务合作人执行协议

状态：**FROZEN FOR COLLABORATOR HANDOFF**（2026-09-23）

本文件是 `protocol_v4.json` 的人工可读执行说明，覆盖 classification 的旧版扩展项，以及 regression、retrieval、clustering、detection proxy 的完整 v4 数据划分。它不把旧任务重新定义成 v3 Tier A/B；所有结果仍需分栏报告。

发生冲突时按以下优先级处理：

1. `protocol_v4.json`；
2. 本文件；
3. `01_Model_Task_Protocol.md` 与 `02_Dataset_Split_Rules.md`；
4. 历史脚本、历史结果目录和旧 README。

禁止根据测试结果修改 split、样本集合、超参数或 checkpoint。任何必需 manifest/hash 不匹配时，状态写为 `BLOCKED_INPUT_MISMATCH`，不得临时随机重切。

## 1. 结果范围与命名

新结果统一记录：

- `protocol_id=bio-eval-union-v4`；
- `protocol_source=historical-25-4-6-8-3_union_bio-eval-formal-v3`；
- `model/checkpoint/config/code/data/split` 的路径与 SHA256；
- `teacher_branch=true`；
- `seed=0`；
- `max_samples=null`、`max_per_class=null`、`smoke=false`。

分栏固定为：

- `v3-tier-a`；
- `v3-tier-b`；
- `v4-union-extension`；
- `v4-detection-proxy`。

旧协议独有数据集只能进入 extension/proxy 栏，不得并入或改写 v3 aggregate。

## 2. 全局冻结的模型与特征设置

除 detection proxy 外，所有任务统一使用：

- checkpoint 的 teacher/EMA 分支；
- BF16 feature extraction；
- feature batch 64；
- seed 0；
- final block 的 `CLS token || mean(final patch tokens)`，拼接后 L2 normalize；
- RGB backbone 使用冻结的 `channel_policy=auto`；不得逐模型选择通道；
- 不使用 last4/even4/纯 CLS 替代；
- 不允许 sample cap、class cap 或 smoke 输出进入结果表。

Regression 使用 train-only `StandardScaler + Ridge(alpha=1.0)`。Retrieval 使用 cosine similarity。Clustering 使用仓库固定的 `MiniBatchKMeans`、真实标签类别数 `K`、seed 0，并用 Hungarian alignment 计算 cluster accuracy；主指标分别为 Recall@1 与 NMI。

## 3. Classification 旧版扩展项

### 3.1 LC25000 classification

| 字段 | 固定值 |
|---|---|
| 状态 | `PROVISIONAL_LEGACY_ONLY`，进入 `v4-union-extension`，不得进入 v3 aggregate |
| 数据根目录 | `/mnt/huawei_deepcad/benchmark/Retrieval_Clustering/LC25000/images/lung_colon_image_set` |
| 类别 | `colon_aca`, `colon_n`, `lung_aca`, `lung_n`, `lung_scc` |
| 样本 | 25,000；五类各 5,000 |
| split | 固定分层 80/20、seed 0；20,000 train / 5,000 test；五类各 4,000/1,000 |
| 锁定文件 | `/mnt/huawei_deepcad/benchmark_model/benchmark_runs/hs6_5tb_protocol_union_nonseg_20260921/locked_inputs/lc25000_classification_split.json` |
| SHA256 | `84edcdf2d30bb2b989d7d4dcc42dad1e058c59f49e167693626bf72bdcc85fd8` |
| probe | train-only StandardScaler + balanced LogisticRegression，`max_iter=10000`, seed 0 |
| 主指标 | balanced accuracy；同时保存 macro-F1、accuracy |

必须逐条使用锁定 JSON 中的 `(absolute image path, label)`；允许在另一台机器做根路径前缀映射，但相对路径、标签、顺序和 SHA256 对应的集合不得变化。该 split 尚未证明 source/group-disjoint，因此论文中必须保留 `PROVISIONAL_LEGACY_ONLY` 标记。

## 4. Regression：完整四项

所有目标均保持原始连续值；只有 BBBC013 对 dose 做 `log1p`。主指标 R2，同时保存 MAE 和 Spearman。

| 数据集 | 固定 train/test 或 OOF | 输入/目标 | 报告栏 |
|---|---|---|---|
| `bbbc005` | 固定 group 80/20：group=`plate/count/field`，seed 0；7,675 train / 1,925 test；split SHA256 `6bcfd65a7bd38e9a2e919f850409bc5c59eb59b6beece3ef62ce5d15840575b4` | `w1` 图像；目标为文件名 `_C<n>_` 中的 cell count | `v3-tier-a` |
| `bbbc013` | 分 compound 的 leave-one-replicate-row-out；Wortmannin=A/B/C/D，LY294002=E/F/G/H；每 compound 四折，每折 36 train / 12 test | Channel1；目标 `log1p(dose)`；不得使用现存普通 group 80/20 JSON | `v3-tier-b` |
| `conic-cell-count` | `CoNIC-10fold-v1`：source-image-grouped、source/count-stratified 10-fold，seed 42；fold 2–9 train、fold 1 val、fold 0 test；3,940/524/517 | 官方中央 224×224 区域；目标为该区域六类 instance count 之和 | `v4-union-extension` |
| `livecell-cell-count` | official LIVECell COCO train/val/test manifest；Ridge 只用 official train 拟合并在 official test 报告，3,253/570/1,564；不得合并或重切 | 全图 mean-color letterbox 成正方形后 resize 224；目标为 COCO instance 数 | `v4-union-extension` |

BBBC013 先在每个 compound 内生成四折 OOF prediction，再分别计算 compound 指标；最终报告必须保留两个 compound 的结果和宏观汇总，不能把两种化合物的原始 dose 混成一次随机回归。

CoNIC 与 LIVECell count 是派生 proxy，只进入 v4 extension；它们不是 v3 regression，也不能替代对应的 segmentation。

两项 count manifest 必须固定为：

```text
8640e1fd6365a5bd51269bb107725d86538a3cbfce59a79a9df3292a0c2ac0d1  /mnt/huawei_deepcad/benchmark/Regression/CoNIC_Cell_Count/conic_cell_count.csv
ea9073c3edd5efc575c15732a2fb4c0b772f462b4c9cc56bbd52f43a536717d0  /mnt/huawei_deepcad/benchmark/Regression/LIVECell_Cell_Count/livecell_cell_count.csv
```

## 5. Retrieval 与 clustering：同一组七个数据身份

Retrieval 与 clustering 是两个独立任务，但必须复用完全相同的冻结样本身份。统一 resize 256、center crop 224、batch 64、L2 normalized features。

### 5.1 四个 within-set 数据集

| 数据集 | 固定样本集合 | 固定协议 | 状态 |
|---|---|---|---|
| `lc25000` | 上述完整 25,000 张、5 类；不得使用 classification 的 20k/5k split | 所有样本既作 query 又作 gallery；每个 query 屏蔽自身；clustering 在同一 25,000 张上执行 | v4 extension |
| `nct-crc-he-100` | `nct_crc_he_100-00000-of-00001-25a54abad9e9e379.parquet`；SHA256 `f145a1e5c6aaa23bbab1743903b12cc784275550d5d51a98e32fdc4dce66af77`；99 个可用样本、9 类 | within-set leave-one-out，屏蔽自身；clustering 使用同一 99 样本 | `LOW_N` v4 extension，必须单列警示 |
| `nct-crc-he-1k` | `nct_crc_he_1k-00000-of-00001-5b5590ca11070fef.parquet`；SHA256 `9f212328c843e7b3a6d51c70cac7688b4ad45706cd3f24fcc0a03a0eb10de7f6` | 固定 1K 集合，within-set leave-one-out，屏蔽自身 | v3 Tier A |
| `crc-val-he-7k` | 固定 3 个 CRC-VAL-HE-7K parquet shards；各 SHA256 见本节下方 | 固定独立患者 7,180 张集合，within-set leave-one-out，屏蔽自身 | v3 Tier A |

CRC-VAL-HE-7K 三个 shard 的 SHA256：

```text
b5a43c55c528bceb0efe6147ccb62425083c1c97c0ee36dad83bfa1f6cd42a0c  crc_val_he_7k-00000-of-00003-a44b36e006c0b9b1.parquet
270d04a6ed81dfac10e1e51dc4040fee75d7ef31c25bf9dcd1835b1babf5bcef  crc_val_he_7k-00001-of-00003-99f6a50aeb3e1a12.parquet
8cc8a7ed0997b0b5b2316a2e6f8f76fb293ef2d2d944a4a33f7fc74284f534ff  crc_val_he_7k-00002-of-00003-3d70705e64186eb5.parquet
```

### 5.2 三个显式 query/gallery 数据集

| 数据集 | Retrieval split | Clustering split | manifest / gate |
|---|---|---|---|
| `hpa-subcellular` | disjoint query/gallery；按 gene label 匹配；1,786 query + 1,786 gallery，937 genes | `hpa_single_location_clustering.csv`：1,458 样本/41类；主表使用 all-41，ge10-34 可作 secondary | retrieval SHA `10e668157d5313200e7906606fa8f0ee90bd3056c7ea2258bc914168fdc6f9a0`；clustering SHA `19998bf581242612be1538f145614ce6739b759bc6ffd15ae225e680270a77b2` |
| `rxrx1-cross` | 固定 balanced core：8,864 query + 8,864 gallery；cross-experiment；1,108 siRNA 类；query label 必须存在于 gallery | 将同一 core query+gallery 合并，在 perturbation label 上 clustering | `rxrx1_official_cross_experiment_core.csv`，SHA `c18cdf8e29ecb3c90921a1b9421e17e4b8e673f7d42fe91836a31672ffb98741`；禁止改用 112,824-view full |
| `rxrx3-core` | 每个 eligible gene 确定性选择一个 query well 和一个 gallery well；gene 内 plate-disjoint；734 query + 734 gallery | 使用同一冻结身份做 gene clustering | `crispr-query-guide-plate-disjoint-all-eligible-genes-v1`；`split_manifest.jsonl` SHA `94d570cb66d71de20e9ded203d8727623fe85a807d8b5b9cdbe0de3fce1318f5`；128-gene quick screen 无效 |

HPA/RxRx1 manifest 根目录固定为：

`/mnt/huawei_deepcad/benchmark/Retrieval_Clustering/protocols/v1`

RxRx3 的六个 Cell Painting 通道先分别做 p01/p99 normalization，再固定 pair-mean `(1,2),(3,4),(5,6)` 成 RGB。禁止按模型更换 channel mapping。

### 5.3 Retrieval 指标

- cosine similarity；
- within-set 任务必须把相似度矩阵对角线设为无效，严禁 self-match；
- 显式 query/gallery 任务的两个角色不得有相同 sample identity；
- 主指标 Recall@1；同时保存 Recall@5/10、mAP@1/5/10、MRR；
- 结果必须记录 `n_query/n_gallery/n_classes`，within-set 还要记录 `n_samples`。

### 5.4 Clustering 指标

- `K =` 当前冻结集合的真实类别数；
- MiniBatchKMeans、seed 0；
- 主指标 NMI；同时保存 ARI、Hungarian cluster accuracy、cosine silhouette；
- 禁止用 test metric 挑 seed、K、PCA 维数或 checkpoint；
- `nct-crc-he-100` 必须显示 `LOW_N`，不得无警示地与 1K/7K 等权解释。

## 6. Detection proxy：完整三项

这些任务是 frozen center-to-patch linear probe，**不是原生 object detection，也不是 COCO bbox AP**。统一进入 `v4-detection-proxy` 分栏。

### 6.1 统一 head 与超参数

- 输入 224×224 stretch；patch size 16，得到 14×14 patch grid；
- backbone teacher/EMA frozen，使用 final block spatial patch map；
- 每个实例以 mask centroid 或 COCO bbox center 映射到对应 patch，生成二元 patch label；
- linear head：每个 patch feature 接一个共享 `Linear(d,1)`；
- loss：`BCEWithLogitsLoss(pos_weight=clip(negative/positive,1,100))`，pos_weight 只从 train 估计；
- AdamW，lr `1e-3`、weight decay `1e-4`；
- batch 8，BF16，workers 2，seed 0，5 epochs；
- threshold 0.5；
- 主指标 test patch F1，同时保存 patch precision、recall、accuracy；
- 禁止使用 B4 历史结果，禁止把 patch F1 称作 detection AP。

### 6.2 三项固定划分

| 数据集 | train / val / test | label 来源 | 强制检查 |
|---|---|---|---|
| `bbbc038` | 有标签的 stage1_train 固定 seed42 70/15/15；当前锁定为 470 train / 100 val / 100 test | instance mask centroid | `bbbc038_splits.npz` SHA `4eb72dc1e58453126893261fedff661b7eb58dc9863ae04d991e7a5282f57ac4`；官方无标签 test 不使用 |
| `conic` | `official-baseline-fold0-nested-v1`：3,469 train / 494 val / 1,018 development holdout | instance mask centroid | train/val/test 的 source-image group 与 index 两两不相交；禁止 legacy random 3984/498/499 |
| `livecell` | official COCO train / val / test：3,253 / 570 / 1,564 image records；3,188 / 569 / 1,512 unique filenames | COCO bbox center | 必须逐字节匹配 v3 锁定 annotation SHA；保留官方 JSON 自带重复 record 与 train/val 30 个同名文件，不自行去重 |

Detection proxy 使用和对应 segmentation 相同的数据身份/split，但只消费 center-to-patch labels；不得把 segmentation mask 指标混进 detection proxy 汇总。

## 7. 合作人最小执行清单

如果 v3 共享 cell 已有且通过指纹核对，只需要新跑以下十个 extension/proxy task-dataset cell：

1. classification：`lc25000`；
2. regression：`conic-cell-count`, `livecell-cell-count`；
3. retrieval+clustering：`lc25000`, `nct-crc-he-100`（每个数据集同时产生两类结果）；
4. detection proxy：`bbbc038`, `conic`, `livecell`。

若合作人负责完整 regression/retrieval/clustering 表，则必须按本文分别得到 4/7/7 项；不得用 retrieval 结果代替 clustering 行，也不得只报四个历史 within-set retrieval 数据集而漏掉 HPA、RxRx1、RxRx3。

## 8. 启动前和交付验收

启动前必须保存：

- 本文件和 `protocol_v4.json` 的 SHA256；
- checkpoint/config/code commit 与 SHA256；
- 每个 split/manifest/data shard 的 SHA256；
- 样本数、类别数、query/gallery 或 train/test 交集审计；
- 实际 batch、输入分辨率、feature 定义、probe 和 seed。

以下任一情况不得记为完成：

- 数据同名但 manifest/hash 不同；
- classification LC25000 重新随机切分；
- retrieval 出现 self-match；
- RxRx1 使用 full 而不是 balanced core；
- RxRx3 使用 128-gene screen；
- BBBC013 使用普通 80/20；
- CoNIC 使用 legacy random split；
- LIVECell 自行去重或重切；
- detection 使用 B4、非 224 stretch 或把 patch F1 写作 AP；
- 使用 student branch、sample cap、smoke 或缺少结果指纹。

合格结果状态写为 `VALID_COMPLETE`；输入或协议不满足时写明确 blocker，不得补零，也不得从 aggregate 中静默删除。
