[返回 7B 训练交付入口](../../README.md)

# 7B：E4880（三教师 CLS+patch 蒸馏 + 4880 步退火）与 SSL 重训

2026-11-16。代码交接包。我们没有在 7B 上运行过这里的配置；只在本地做了准备器的配置合并试跑（假 DCP 目录）和补丁应用检查。不含权重、评测数据。

## 为什么做这两件事

CLS 蒸馏四组（C/D 最好）把 7B 在分类、检索、聚类、分割上拉到了与我们的 H+ E、L lw5 基本持平，落后的只剩**检测代理**（C 23551 为 84.1，L lw5 85.6）和**回归**（尤其 livecell 计数：C 23551 112.9，L lw5 91.3）。这两项依赖 patch 特征，只蒸馏 CLS 管不到。（7B 的数字来自你们的评测流程，H+ / L 来自我们的流程，属于跨流程的桥接比较。）

我们的证据（六类等权，分割 E20，seed0）：

| 模型 | 分割 | 检测代理 | 六类 |
|---|---:|---:|---:|
| H+ 纯 SSL 19519 | 70.2 | 82.8 | 73.62 |
| H+ C：只蒸馏 CLS（官方 H+ + H-optimus-0） | 70.3–70.4 | 83.0–83.1 | 74.40（最好点） |
| H+ D：CLS+patch，只用官方 H+ 一个教师，2048 步退火 | 70.1 | 83.1 | 73.52 |
| H+ E2048：CLS+patch，三教师，2048 步退火 | 71.2 | 85.0 | 74.59 |
| **H+ E4880：同上，4880 步退火** | **71.4** | **85.3** | **75.03** |
| H+ F4880：同 E4880，但用 PHI-S 平衡（权重 0.6，压力对齐） | 71.3 | 84.9 | 74.68 |
| L 纯 SSL 74663 / 纯退火 77103 | 68.9 / 68.7 | 83.6 / 83.7 | 73.69 / 73.67 |
| L lw5 79543（CLS+patch，官方 L + H-optimus-0，4880 步退火） | 70.8 | 85.6 | 75.02 |

结论：patch 级的提升来自「patch 蒸馏 + 会分割的教师」；只退火、只 CLS、或只用官方模型当教师都不够；4880 步比 2048 步好；lw5 平衡比 PHI-S 好。

## 实验一（优先）：从 21503 做 E4880

与 H+ 的 E4880 完全相同的目标，只是第一位教师换成官方 7B。

| 项目 | 设置 |
|---|---|
| 起点 | 21503 的**完整** DCP（student / EMA / heads / AdamW），下一步 21504 |
| 退火 | `optim.cooldown`：从 21504 起 **4880 次更新**线性退火，LR 从当时值（约 4.73e-5）到 **1e-6**（显式设置，原 run 的 min_lr 是 2e-6）；EMA momentum 升到 1；teacher 温度冻结在当时值；weight decay 照原计划；最后一步 26383 |
| 蒸馏 | `external_distill.mode: phis_spatial_recovery`，`weighting: inverse_variance`（= lw5 平衡） |
| 教师 | `dinov3_7b`（官方，img_size **256**）、`hoptimus0`（224）、`genbio_pathfm`（224），各 1.0；就是 CLS 四组用的同一份 FrozenJointTeacher manifest |
| 损失 | loss_weight 5，cls_weight 1，patch_weight 1，每图 64 个 patch，ridge 0.1，解码器 momentum 0.995，warmup 32，每 8 步求解一次 |
| 不变 | batch 4096、数据配比、增广、归一化、clip 3.0、DINO/iBOT/KoLeo、不开 Gram、seed |
| 保存 | 每 976 次更新存完整 DCP + 导出 EMA：**22479、23455、24431、25407、26383**；全部保留 |

**为什么是 4880 次更新而不是按样本数换算（1220）**：退火斜率、EMA momentum 的上升、ridge 解码器的求解次数（每 8 次更新一次，momentum 0.995）都是按更新次数定义的。4880 次更新复现的是我们验证过的动态（解码器求解 610 次）；1220 次只求解 152 次，是没测过的状态。代价是样本数为我们的 4 倍（约 2000 万），请先评估算力。

**显存**：三个教师在每张卡上各存一份（没有分片），官方 7B 教师 bf16 约 13GB；请在日志里记录每卡显存峰值和每步耗时。

**判定**：终点的检测代理和回归能否追上 L lw5，同时其他四类不掉。各导出点都用同一套 FM ID 评测，和 21503、C 23551 对比。

准备（只生成目录和配置，不启动训练）：

```bash
python scripts/prepare_7b_e4880_distillation.py \
  --source-config /actual_7b_run/config.yaml \
  --source-checkpoint /actual_7b_run/ckpt/21503 \
  --teacher-manifest /local_verified/teacher_manifest.json \
  --output-root /new_runs/7b_e4880_from21503
```

启动沿用你们现有的分布式入口，并加 `--expected-start-iteration 21504`。日志里应出现一行
`Cooldown override: updates [21504, 26384) shape=linear | lr <21504 处的原 LR，约 4.73e-05> -> 1e-06 | momentum <当时值> -> 1.0000000`。

## 实验二（并行或排在实验一之后）：SSL 重训

从官方 7B 起，和原来的 20TB SSL 完全相同。**只跑 A 组**；B 组只在 A + E4880 之后回归仍明显落后时再考虑。

| 组 | 改动 | 配置覆盖 | 回答的问题 |
|---|---|---|---|
| **A（推荐，只跑这组）** | `layerwise_decay` 0.98 → **0.94**，其余不变 | `dinov3/configs/train/7b_ssl_ld094.yaml` | 底层改动量对齐 L 后，局部任务（分割、检测代理）是否不再越训越差 |
| B（暂不跑，备选） | A + 峰值 LR 6e-5 → **1.2e-4** | `dinov3/configs/train/7b_ssl_ld094_lr12e5.yaml` | 顶层的总改动量补上后，回归能否像 L 那样明显改善 |

- 0.94 的来历：底层/顶层的 LR 之比，L 是 0.9²⁴ ≈ 0.08，7B 原设置 0.98⁴⁰ ≈ 0.45；0.94⁴⁰ ≈ 0.084 与 L 对齐。**不要用 0.9**（0.9⁴⁰ ≈ 0.015，比 L 冻得还狠 5 倍）。
- B 组 1.2e-4 的来历（备选，供以后参考）：按开方规则从 L 的 batch 1024 / 1e-4 换算到 batch 4096 应约 2e-4；原 run 用 6e-5，按「LR × 更新次数」估算，到选点时顶层的总改动量约为 L 的 1/6。1.2e-4 = 6e-5 × √4 是折中。**B 组有不稳的风险**：前 1000 步就要看 loss 和梯度范数，有尖峰就停。
- 时长至少到 3 万步（覆盖原来 21503 的峰值和之后的下降段），每 2048 步导出；每个点先做便宜筛选（分割 cellpose / livecell / multimodal，检测代理，回归 bbbc005 / livecell 计数，几个分类），入围的点再跑完整评测。
- 选出最好的点后，再做一次和实验一相同的 E4880。
- 为什么只跑 A：A 只改一处，结果好坏都能说清原因；B 把 LR 翻倍，有发散风险，而且回归的短板预计主要由 E4880 补上。
- drop_path（0.4，L 是 0.25）和每卡 batch（KoLeo 分组 64，L 是 16）也和 L 不同，这一轮**故意不改**，避免分不清是哪个起作用。

## 集成代码

本分支已直接包含全部改动，直接用本分支时**不要再应用补丁**。

- 新增 `dinov3/train/phis_multi_teacher_distill.py`（E 的目标；复用 `external_distill.py` 的 ridge 解码器和 `joint_teacher_models.py` 的冻结教师加载，这两个文件未改）。
- `dinov3/train/ssl_meta_arch.py` 只增加一个分支：`external_distill.mode == "phis_spatial_recovery"`。
- 新增配置覆盖 3 个（E4880、SSL A、备选 SSL B）、准备器 `scripts/prepare_7b_e4880_distillation.py`。`train.py` 已含 `optim.cooldown`，未改。
- `e4880_update.patch`：在已应用 CLS 四组包的工作副本上使用的增量补丁（基于本分支 c83ed8a，已检查可干净应用）：

```bash
git apply --check collaborator_handoff/BioDINO_7B/train/e4880_distillation/e4880_update.patch
git apply collaborator_handoff/BioDINO_7B/train/e4880_distillation/e4880_update.patch
```

- `files/`：新增文件的独立副本，方便阅读；`SHA256SUMS`：包内校验。
- GenBio-PathFM 需要 transformers 4.57.1（我们训练时用的版本：transformers 4.57.1、huggingface_hub 0.35.3、safetensors 0.6.2）。

H+ 上 E4880 实际运行的代码与本分支的差别：我们的 `ssl_meta_arch.py` 另有一段 joint-teacher（CorrCA）分支和一个 DDP 精度检查，E4880 运行时 joint-teacher 关闭、使用 FSDP，这两段不参与计算，所以没有放进本包。
