# 01 — Model and Task Protocol

状态：APPROVED（2026-09-09）。

## 1. 比较单元

一个合法的 matched comparison cell 定义为：

`(model size, base checkpoint/epoch, continuation budget, method arm, task, dataset, split version, resolution, embedding layers, channel policy, probe, seed, code commit)`。

本轮 SKDT 三臂为：

- baseline：原 HS6-L，同一目标 epoch 的原 checkpoint；若历史评测协议不一致则重新评测。
- matched SKDT：真实 scout relation target。
- shuffled SKDT：只打乱配对关系的 causal control，其余训练和评测设置相同。

所有 consolidated checkpoint 默认读取 `teacher`/EMA state。manifest 必须同时记录 checkpoint 实路径、文件大小、mtime、配置路径及 SHA256（可在启动前只计算一次）。

## 2. 模型大小与 embedding 层

层号全部为 **0-based block index**。

| 模型 | arch | depth | 非 dense 任务 | segmentation 的 `even4` | segmentation 的 `last1` | detection |
|---|---|---:|---|---|---|---|
| S+ | `vit_small` | 12 | final block，`n_last_blocks=1` | `[2,5,8,11]` | block 11 | final patch tokens |
| B | `vit_base` | 12 | final block，`n_last_blocks=1` | `[2,5,8,11]` | block 11 | final patch tokens |
| L | `vit_large` | 24 | final block，`n_last_blocks=1` | `[4,11,17,23]` | block 23 | final patch tokens |
| H+ | `vit_huge2` | 32 | final block，`n_last_blocks=1` | `[7,15,23,31]` | block 31 | final patch tokens |
| 7B（若评） | `vit_7b` | 40 | final block，`n_last_blocks=1` | `[9,19,29,39]` | block 39 | final patch tokens |

非 dense frozen feature 固定为 `final CLS || mean(final patch tokens)`，然后 L2 normalize。不能按模型大小改成 last4、even4 或纯 CLS。segmentation 是否用 `even4` 由数据集决定，模型大小只负责把 `even4` 映射为上表中的实际层号。

## 3. 通用预处理

- autocast：`bf16`。
- patch size：16；输入边长必须兼容 patch size。
- channel policy：`auto`，`channel_tta_samples=8`，`channel_policy_seed=0`。
- 对真正的 multichannel backbone，`auto` 使用 native channels；对普通 RGB backbone，`auto` 等价于确定性的 first-3/不足 3 通道重复补齐。
- split protocol：`current`；禁止回退到旧的 sample-level `internal-80-20`，除非数据集规则明确允许。
- 评测 seed：0；同一正式表不得混用其他 seed。需要方差时另建 multi-seed campaign，不能覆盖主结果。
- smoke/debug 的 sample cap 结果必须单独目录并带 `SMOKE`，不能进入正式 summary。

## 4. 任务级固定超参数

| 任务 | feature batch | 输入 | embedding | probe / metric 计算 | 主指标 |
|---|---:|---|---|---|---|
| classification | **64** | dataset-best，见下表 | final CLS + final patch mean | `StandardScaler + LogisticRegression(class_weight=balanced,max_iter=10000,n_jobs=1)` | balanced accuracy |
| multilabel | **64** | dataset-best | 同上 | `StandardScaler + OneVsRest balanced LogisticRegression` | macro AUC；同时保留 micro AUC/F1/AP |
| regression | **64** | best/fallback；count 全图不 crop | 同上 | `StandardScaler + Ridge(alpha=1.0)`；BBBC013 例外见数据规则 | R2；同时保留 MAE/Spearman |
| retrieval/clustering | **64** | resize 256 + center crop 224 | 同上，L2 normalized | cosine retrieval；MiniBatchKMeans + Hungarian alignment，seed 0 | Recall@1 和 NMI |
| segmentation | **32** | dataset-specific，见下表 | dataset-specific last1/even4 patch tokens | frozen encoder + linear probe，50 epochs，probe batch 32，eval every 50，seed 0 | test mDice；同时保留 mIoU/AJI/AP/bPQ |
| detection proxy | **8** | 224 stretch | final block patch map | frozen center-to-patch linear head，AdamW，5 epochs，seed 0 | test patch F1 |
| OOD | **64** | resize 256 + crop 224 | final CLS + patch mean | fixed ID/OOD protocol，seed 0 | AUROC（与 ID 均值分开） |

注意：这里 batch 64 指 frozen classification/regression/retrieval/OOD。segmentation 的 feature/probe batch 32 和 detection batch 8 是任务协议本身，不是为了挤显存临时降低。

## 5. Classification / regression resolution

`resolution_protocol=best`。未列出的 classification/regression 数据集使用 crop 224、resize 256。

| 数据集 | crop | pre-resize | 说明 |
|---|---:|---:|---|
| bloodmnist | 384 | 439 | 既有五数据集 resolution sweep 的 best |
| bbbc048-cellcycle | 512 | 585 | 同上 |
| cyclops-protein-loc | 224 | 256 | 同上 |
| midog25-atypical | 384 | 439 | 同上 |
| chestmnist | 512 | 585 | 同上 |
| conic-cell-count | 224 | 224 | 保留完整计数区域，不做 center crop |
| livecell-cell-count | 224 | 224 | 保留完整计数区域，不做 center crop |
| 其他合法 classification/regression | 224 | 256 | 固定 fallback；不能按机器改变 |

## 6. Segmentation dataset-best protocol

| 数据集 | feature resolution | resize | layers | class weighting |
|---|---:|---|---|---|
| bbbc038 | 512 | pad | even4 | none |
| cellpose | 512 | pad | last1 | none |
| conic | 256 | stretch | even4 | sqrt-inverse |
| livecell | 512 | pad | even4 | none |
| monuseg | 768 | pad | last1 | none |
| multimodal_cellseg | 512 | pad | last1 | none |
| pannuke | 256 | stretch | even4 | none |
| tissuenet | 256 | stretch | last1 | none |

`protocol=best` 必须在 log 和 result path 中解析成上表的真实参数；只记录字符串 `best` 而不记录展开后的 resolution/layers/resize/weight，不算合格结果。

## 7. 结果可比性验收

复用 baseline 前逐条验证：

- checkpoint epoch/iteration 与 method arm 相同；
- teacher branch；
- code commit 与 dataset registry version 相同；
- dataset 和 split hash 相同；
- batch、resolution、resize、layers、avgpool、dtype、channel policy、seed 相同；
- probe 类型和 probe 超参数相同；
- result 无 `error`，样本数符合 split manifest；
- 聚合时每个模型包含完全相同的数据集集合；缺项不能静默跳过。

## 8. External FM dense addendum

PanNuke/CoNIC 的 external-FM 比较使用同一 dataset split、256 stretch、feature batch 32、probe batch 32、50 epochs、eval every 50、seed 0；CoNIC 同样使用 `sqrt_inverse` class weighting。固定位置网格的 ViT 在 256 输入上显式插值 spatial positional embedding，禁止悄悄退回 model-native 224 crop。

外部架构没有与 DINO block `[4,11,17,23]` 一一对应的中间层定义，因此统一使用各模型最后一个 spatial dense map，并在结果中记录 `feature_layers=external-final-dense-map`。该表属于 external-FM dense comparison，不能伪称为 DINO even4 layer ablation。正式模型集合固定为：`dinov2 mae siglip2 bioclip cytoself jump_cp cytoimagenet pe uni conch phikon2 virchow2 gigapath hoptimus0`。某模型在 24 GiB、batch 32 下 OOM 时记录 `FAILED_RESOURCE`，禁止私自降低 batch 后混入主表。

CytoImageNet 的公开 EfficientNet-B0 wrapper 原本将输入张量固定为 `224x224`。正式评测须以完全相同的 no-top 架构和原始权重重建 `256x256` 输入接口，禁止将数据暗中缩回 224。对于 patch size 不能整除 256 的 ViT，仅允许按通用规则吸附到最近的合法 patch grid，并在 manifest 中记录实际 encoder input size。
