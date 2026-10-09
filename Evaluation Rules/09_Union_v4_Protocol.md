# v4：旧矩阵 ∪ v3；segmentation 例外

状态：2026-09-21 用户确定评测**目标范围**；可执行性依赖各 cell 的下述准入检查。
机器可读清单：[protocol_v4.json](protocol_v4.json)。本文件与清单仅作用于**新建的 v4 campaign**；既有 v1/v2/v3 campaign、结果和历史决策不被追溯改名，v3 的 Tier A/B 统计仍可单独复现。v4 不自动启动评测，也不授权把旧结果改标为新结果。

交给合作人的逐数据集非分割执行细则、固定 manifest/hash 和验收条件见
[10_Union_v4_NonSeg_Collaborator_Protocol.md](10_Union_v4_NonSeg_Collaborator_Protocol.md)。

## 1. 去重后的目标清单

按 `(task, dataset)` 去重，旧版与 v3 同名 cell 采用 v3 的 split、特征、probe、指标与数据身份；旧版独有 cell 列在 v4 扩展栏。retrieval 和 clustering 各有同一组七个数据集，但作为两个任务分别出结果。PanNuke 三个 rotation 属于同一 segmentation 数据集的三个必需执行 cell，不是三个新数据集。

| 家族 | v4 数据集数 | 在 v3 之外新增 | 报告说明 |
|---|---:|---|---|
| Classification（含 multilabel） | **25** | `lc25000` | 24 个 v3 + 1 个旧版；LC25000 的历史随机 split 须单列有效性 |
| Regression | **4** | `conic-cell-count`, `livecell-cell-count` | 两个 count proxy 标注来源，不等同于原生 segmentation |
| Retrieval / clustering | **7 / 7** | `lc25000`, `nct-crc-he-100` | RxRx3 沿用 v3；NCT100 低样本标注 |
| Segmentation | **7** | **无** | 只用 v3 正式七个；BBBC038 继续单列 v3 observation，不计入 7 |
| Detection proxy | **3*** | `bbbc038`, `conic`, `livecell` | 星号表示 frozen center-patch proxy，并非原生 detection |
| CTC / OOD | **1 / 2** | 无 | v3 规定的 CTC 与 X-ray/Cryo OOD |

上述数字是**预期 inventory**，不是已完成结果。`classification25` 不代表 LC25000 已通过 v4 的来源独立性审查；任务被阻断时保留 `BLOCKED_NOT_TESTED` 行，不能把总数缩回 24、补零或者声称完整 v4。

## 2. 沿用的评测设置

- Teacher/EMA；相同样本、输入、checkpoint 分支及代码/配置/数据 hash。全局 frozen 任务 B64、BF16、seed0、final CLS 与 final patch mean 拼接后 L2；classification 用 train-only StandardScaler + balanced logistic，regression 用 train-only StandardScaler + Ridge（BBBC013 延续 compound/log1p OOF），retrieval cosine，clustering 固定 KMeans/seed。输入尺寸、各数据集 split 和指标沿用 [01_Model_Task_Protocol.md](01_Model_Task_Protocol.md) 与 [02_Dataset_Split_Rules.md](02_Dataset_Split_Rules.md) 中 v3 共享 cell 的定义。
- Segmentation 完全照 v3 与 2026-09-15 probe-budget amendment：七个数据集、PanNuke 三轮、native-final/last1 primary，B32 特征/B32 probe，独立 E20 与 E50、每 epoch 验证、seeds0/1/2，最优 validation mIoU 选 epoch、test 只评一次。BBBC038 mask segmentation 仍仅供独立 observation；旧版 split、旧版 val=test 与 E50 单次终点不可充当 v4 正式分数。
- 三项 detection proxy 统一执行 2026-09-20 的**matched B8** 观察设置：224 stretch、final spatial patch map、BF16、workers2、seed0、frozen center-to-patch linear head、AdamW lr1e-3/wd1e-4、5 epochs，主指标 test patch F1。历史 B4 不能直接并入；这些分数从不称为原生 detection/AP。
- LC25000 retrieval 是完整 25000 张五类 within-set leave-one-out/self-match 屏蔽；NCT100 是原始固定 parquet 内 99 个可用样本（实际数量需重新核验）的同型任务，单列低样本警示及不确定性，不允许把 smoke/sample-cap 的输出算成完成。它们和 v3 的五项共用七项清单。

## 3. 新增 cell 的准入与冲突处理

1. **LC25000 classification 暂仅算 legacy/provisional。** 当前 `registry.py` 中 train/test 均返回同一 image folder，且不在 `NATIVE_TEST_SPLIT_DATASETS`；不得把其完整目录分别当训练/测试。2026-09-21 补测 campaign 已另存固定的 20000/5000 分层随机 80/20 路径与标签清单（`hs6_5tb_protocol_union_nonseg_20260921/locked_inputs/lc25000_classification_split.json`，campaign 登记 SHA256），可用于明确标记的 legacy 诊断；仍未证明 source/group-disjoint，不能据此声称 v4 完整分类能力已获严格验证。后续需独立审查来源与拆分方案；禁止根据 checkpoint/test 指标选择划分。
2. CoNIC count regression 沿用 `CoNIC-10fold-v1` source-image grouped split，LIVECell count regression 沿用 official COCO train/test；分别记录 provenance/目标定义，不把旧的 v3 excluded 状态悄悄改写为 v3 Tier A/B。检测 proxy 使用其各自原生来源身份和固定拆分，重复实例/源图不可跨独立训练测试而不披露。
3. NCT100 保留在 v4 目标 7 项里；先核验固定 manifest、实际可用数量、每类支持与 self-match 排除。小样本不能被视作与 1K/7K 等精度相同，报告独立 per-dataset 分数、波动与样本量。
4. `rxrx3-core` 使用 v3 全 eligible-gene manifest，不以 128-gene quick screen 代替；CTC 只有 native TRA/SEG head+linker 与验证通过后才能记有效；MoNuSeg 需通过 v3 official 样本身份 gate；OOD 每个 task 保留自己的 readiness gate。未通过分别标记 blocked。
5. CHAMMI open-set task3/task4、MIDOG++、额外 BloodCell 不属于本次**旧矩阵 ∪ v3**；要纳入须另起明确的协议版本，不因文件可访问而自动加进 v4。

## 4. 报告与复用

新 campaign 必须记录 `protocol_id=bio-eval-union-v4`、两来源矩阵、完整 25/4/7/7/7/3*/1/2 期望清单、任务-数据集 split/hash/probe/seed/模型与代码身份，以及每个 cell 的 `VALID_COMPLETE`、`BLOCKED_NOT_TESTED` 或具体失败原因。旧结果仅在 dataset、split、样本清单、teacher、特征、batch、probe、指标、seed 和版本指纹完全匹配并通过独立 validator 时可复用；仅数据集同名不能复用。既有 v3 Tier A/B 结果和新增 union extension 分栏展示；仅全部相关 cell 有效时才计算其预先声明的 v4 并集汇总，proxy 检测单独带星号，不混入原生 detection 或 v3 aggregate。

对于 5TB Early–Late coexistence diagnostic，五种冻结表示 `E/M/L/E+L/M+L` 在**同一 v4 dataset/split**上成对评估；classification 目标 25 项而不是 v3 的 24 项。PCA 仅在每个任务的训练侧拟合；对无监督 within-set retrieval 的降维必须先冻结独立、无 downstream-test 选择的拟合集和变换。诊断可以使用既有 checkpoint 观察选锚，但结果不充当未来正式训练方法的无偏 checkpoint 选择依据。

## 5. 2026-09-29 MoNuSeg 与 detection 比较口径补充

用户选择以历史 37 张训练池作为 HS0/HS6 1TB、FM14、HS6 5TB no-GRAM/GRAM 及新方法的**共同 MoNuSeg 比较口径**。从下载的训练压缩包中固定 30 train / 7 val，另用独立的 14 张 test；全部 51 张图像和 XML 的身份及哈希、旧验证索引哈希均锁在 `outputs/02_eval_inputs/monuseg37_v4_20260929/manifest.json`，其 SHA256 见 `protocol_v4.json` 的 `comparison_amendment_20260929`。训练与测试样本 ID 无交集。经典挑战赛原始训练池为 30 张，现有 `dense_splits.monuseg` 的 24/6/14 结果继续保留独立标签，不与 30/7/14 的结果接成同一曲线。37 张池含经典30张之外的7张；这组比较结果不得称为经典挑战赛训练协议分数。

新的共同结果沿用分割 E20 与 E50 分开、B32、seeds 0/1/2、验证集 mIoU 选 epoch、PanNuke 三轮等固定设置。所有目标模型的 MoNuSeg 需要逐点核验已有原始结果能否满足**同一锁定身份和 probe 指纹**；不符合者重测，旧结果保留来源但退出新的匹配均值。HS0/HS6 1TB 的全部已有点、FM14、HS6-L/H+ 5TB 以及方法分支都列入覆盖清单，远端 checkpoint 只在权重仍可访问时执行。

Detection proxy 的 CoNIC 统一 `official-baseline-fold0-nested-v1`、B8；历史随机 patch split 或 B4 结果重测后才替换比较单元。GRAM 检索/聚类七项的缺口优先补 `rxrx3-core`，再补末端缺少的 `lc25000`、`nct-crc-he-100`；检测三项逐点补 BBBC038、CoNIC、LIVECell。检索和聚类来自同次数据集评测，但仍分别记录 Recall@1 与 NMI。仅由已验证的完整固定数据集清单产生任务均值；保留缺项、失败与权重不可访问的状态，不用其它 checkpoint 推算。
