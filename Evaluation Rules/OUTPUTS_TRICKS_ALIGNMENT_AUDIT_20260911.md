# Outputs、Tricks 与评测协议对齐审计（2026-09-11）

状态：**READ-ONLY AUDIT；尚未删除任何文件，也未修改训练实现。**

依据：`protocol_v3.json`、R1-R12、现有方法决议、outputs 实际目录和
`/mnt/huawei_deepcad/benchmark_model` 当前入口。容量按 `du` 的已分配空间统计，
会随仍在写入的目录变化。

## 1. 结论先行

1. 第一批最值得清理的不是论文 JSON，而是可重建 feature cache。仅下列四组就约
   **1.81 TiB**：HS6 Fig.2 PanNuke cache 约 1.44 TiB，NRI eval 约 199.7 GiB，
   NCI coverage eval 约 164.4 GiB，Martingale-NCI eval 约 10.7 GiB。后三组中已确认
   `.npz` 占约 374.8 GiB；应保留 JSON/CSV/Markdown/manifest，删除 `.npz`。
2. S0-S3 五个早期 S+ 训练目录合计约 **513.7 GiB**。它们是 HS6 前的配方演进，
   不应再占当前主线，但训练 checkpoint 属于 `NEVER_AUTO_DELETE`：先固定每组的
   最终/被引用 checkpoint 和汇总表，再由用户批准删除其余 checkpoint。
3. NCI 当前实现已有正式停止决议；NRI 的完整对比也不支持继续。NRI 相对 H-S0
   虽在部分 classification 上升，但 BBBC005 R2 为 `-0.0423`，CRC-VAL R@1 为
   `-0.0699`，NCT-CRC-1K R@1 为 `-0.1562`，6 个正式 dense 数据中 4 个下降，
   所以不能称为稳定增益。
4. 当前 `dinov3` 与 `benchmark_model` 都还不能产生 v3 正式全矩阵。最硬的原因是
   CTC native evaluator 和 RxRx3-core 固定全 eligible-gene manifest/hash 仍为
   `PENDING_IMPLEMENTATION`；v3 validator 会主动 fail closed。因此现有 v3 quick
   screen 只能是 observation，不能补进正式 aggregate。
5. 发现一个从 2026-07 起仍存在的旧评测进程，写入
   `outputs/02_eval_runs/20260707_cpu1_mc_chammi_5models`，参数为 batch 16、fp16，
   且包含已禁用的 CHAMMI task3/task4。该目录在停进程前不得清理；本轮未擅自停止。

## 2. 可执行的 outputs 分级

机器可审核清单见 `output_cleanup_audit_20260911.csv`。

### A. 第一批 DELETE_CANDIDATE（高置信度）

| 范围 | 大小 | 原因 | 删除前保留 |
|---|---:|---|---|
| `outputs/02_eval_cache/hs6_fig2_pannuke_cache` | 1.44 TiB | 纯 PanNuke feature cache；可由 checkpoint、config、split 重建 | 对应 result JSON、作图 CSV、命令清单 |
| NRI eval 下全部 `.npz` | 199.68 GiB | 已淘汰方法的 frozen/dense feature cache | `experiment.json`、manifest、Markdown、最终 JSON |
| NCI coverage eval 下全部 `.npz` | 164.4 GiB | 四臂重复 feature cache；单个 PanNuke train cache 约 12 GiB | NCI final decision 与每臂结果 JSON |
| Martingale-NCI eval 下全部 `.npz` | 10.68 GiB | 已停止 NCI 家族的可重建特征 | 最终指标与日志摘要 |
| `outputs/02_eval_cache/tmp`、`outputs/tmp`、`outputs/tmp_eval` | 约 74 MiB | Python multiprocessing/临时文件 | 确认无当前 PID 使用后无需保留 |
| 名称含 `.failed_*`、`.oom_*`、`.wrong_ckpt*`、`.startup_*` 的小型残片 | 至少约 18 MiB | 明确失败且无有效 checkpoint | 失败原因已在目录名/日志记录 |

这里的 NRI/NCI 建议是**删缓存，不先删整个 eval root**。结果文件很小，负结果本身
仍有科研价值；把完整目录整删会破坏决议的证据链。

### B. 第二批 ARCHIVE_THEN_DELETE（需用户批准）

| 范围 | 当前判断 | 原因 |
|---|---|---|
| S0b/S1b/S2b/S3b 训练与对应 eval | 老配方归档 | 已由 S6/HS6 主线取代；约 513.7 GiB 训练权重，但仍被历史图表引用 |
| M3 ChannelViT、M4 Residual-MC | 负/旁支归档 | RGB-pretrained route 在 NCI 决议中明显更强；保留最终对照即可，不保留全 checkpoint 曲线 |
| NCI/Martingale/MCI 训练目录 | 负结果归档，约 142.35 GiB | NCI 决议明确停止；保留 config、final checkpoint（若需复核）、metrics 和 decision |
| NRI 训练目录 | 负结果归档，约 13.34 GiB | full eval 不稳定且在 retrieval/regression 上明显回退 |
| CMGI/DCGI | 负结果归档，约 101.95 GiB | conditional q10 不优于 density-matched uniform，且伤害 HPA task2 |
| DBI/ATPA 早期机制 screens | 负/被后续实验替代，约 29.41 GiB | 只保留决策所引用的最小证据集 |
| acquisition firewall、IFT topology、expert-consensus/global bridge | 受控负结果归档 | 各自决议均未通过 generic/causal/cross-species gate；保留最终审计和小型 summary |
| `biordt_ret4_skdt_e12to15_3090qi_20260909` | INVALID_PROTOCOL | batch16、Ret4、含 LC25000、缺 full matrix；可提取诊断摘要后删除 root |
| `hs6_all_checkpoints_reg4_ret6_cluster6_det3_3090fleet_20260909` | EXPERIMENTAL_NOT_REPORTABLE | 模型间 batch 不同，并含 v3 排除项；只能作诊断 |

### C. KEEP

- HS6 S+/B/L/H+ 当前正式基线的被选 checkpoint、config、训练曲线和 provenance。
- SKDT matched 与 shuffled 当前训练产物。早期 Scout/SGST 中虽有失败变体，但
  1024-step stable/raw delta 对 TissueNet/HPA 存在因果信号，且 SKDT 是正在执行的
  三臂候选；不能把它与 NCI/NRI 一并删除。
- 所有最终 JSON/CSV、campaign/command manifest、validation report、方法决议、
  split/hash、最终作图数据。
- BBBC038 结果可作为 `OBSERVATIONAL` 保留，但必须从正式 aggregate 排除。

## 3. Trick 去留建议

| 模块 | 证据结论 | 主线建议 |
|---|---|---|
| NCI + Martingale NCI | 正式决议为 stop；full-budget NCI 相对 carrier 的 CHAMMI-7 `-0.0025`，且相对 RGB `-0.0477` | 从 active configs 和 `ssl_meta_arch` 主路径移出；只在 legacy tag/目录保留 |
| NRI | 相对 H-S0 严重损失 regression/retrieval，dense 多数下降 | 与 NCI 同样 legacy 化，不再暴露为常规训练选项 |
| CMGI/DCGI | 选择器不胜 uniform control | legacy 化 `conditional_morphology_graph` |
| acquisition orbit / gradient firewall | 机制有效但下游 gate 失败 | 保留分析代码，移出主训练 meta-arch |
| IFT sample/patch topology | 所有 locked gate 失败；patch term 无可测变化 | 删除 active efficacy 路径；若保留 topology，只能标 safety regularizer、默认关 |
| expert consensus / global bridge | 多轮 true/shuffled 均未达阈值，存在 retrieval-clustering trade-off | legacy 化，不再追加 sweep |
| SIGReg | S+ 历史配方的一部分，但不能外推到 L/H+ | 保留复现配置，不作为 HS6 通用默认 trick；FINO heads-warmup 已明确淘汰 |
| Gram | 结论依 checkpoint selection 改变，证据混合 | 不判死刑；从默认配方中保持关闭，仅保留明确 ablation |
| Scout/SGST/SKDT | 部分 budget/spectral 解释被否定，但 sample correspondence 信号存在 | 保留当前 SKDT matched/shuffled；只清早期 smoke、失败 budget 和重复 checkpoint |

当前默认 YAML 中这些新增模块都为 `enabled: false`，因此没有“默认偷偷叠加”的问题；
真正的维护成本在 `ssl_meta_arch.py` 同时承担了 NCI、NRI、CMGI、IFT、orbit、Scout、
expert consensus、global bridge 等大量初始化和 forward 分支。建议在确认清理范围后做一次
独立代码变更：先打 legacy tag，再把已否定模块及专用 config/launcher 移出主树，同时
保留决议 Markdown 和最小复现说明。

## 4. 与 Evaluation Rules 不对齐之处

### 当前 dinov3

- `dinov3/eval/bio_benchmark.py` 和 `scripts/run_bio_benchmark_all.sh` 的 defaults 仍含
  LC25000、CoNIC/LIVECell count、CoNIC/LIVECell detection 和 BBBC038 formal lane。
- `scripts/hs6_short_suite.env` 仍含 LC25000、nct-crc-he-100、BBBC038，并缺
  Cellpose/LIVECell/TissueNet 等完整 Seg6。
- 旧 HS6 workers 仍按模型改变 frozen batch，且包含 Ret6/Reg4/Det3 旧矩阵。
- v3 validator 已存在，但协议中的 CTC/RxRx3 hash 仍 pending；当前 formal launch
  必然失败，这是正确的 fail-closed 行为。
- 大多数历史 output 没有 v3 `campaign_manifest.json`、protocol fingerprint 和
  `validation_report.json`，所以最多是 legacy/observation，不能因 `last_result.json`
  存在就复用。

### `/mnt/huawei_deepcad/benchmark_model`

- `run_id_test_suite.py` 自称 canonical，但默认仍是 Reg4 + Ret6 + Det3：包含两个已排除
  count proxy、LC25000、nct-crc-he-100，以及 formal 禁止的 LIVECell/CoNIC detection；
  BBBC038 也没有独立 observation lane。
- frozen batch 默认 32，而 v3 是 64；`run_benchmark.py` 和 retrieval runner 同样默认 32。
- 普通 classification/regression 仍大量使用随机 `train_fraction=0.8`，不能证明与 v3
  固定 split/hash 一致。
- retrieval choices 没有 RxRx3-core；整个仓库没有 CTC native TRA/SEG evaluator。
- canonical ID suite 没有跑完整 classification、Seg6、multimodal_cellseg 与 OOD 矩阵；
  它只是 regression/retrieval/detection 子集。
- skip 逻辑只检查 JSON 可解析且无 `error`，不校验 batch、split、resolution、layers、
  checkpoint、commit 或 protocol fingerprint。
- 因此 benchmark_model 的既有外部 FM 数值可以作为历史上下文，但在补齐同一 v3
  manifest/validator 并重跑前，不能与新的 HS6/SKDT v3 结果直接做正式公平排名。

## 5. 推荐执行顺序

1. 先冻结本报告、CSV、NCI/NRI/其他负结果 decision 与所有小型最终指标。
2. 确认并停止那个 65 天旧 batch16/fp16 CHAMMI 进程；重新检查所有候选路径无 PID 占用。
3. 第一批只删 cache/`.npz`，预计释放约 1.81 TiB；删除后核对最终 JSON 仍可读取。
4. 为 S0-S3、M3/M4、NCI/NRI 等训练目录生成“保留 checkpoint 白名单”，再做第二批。
5. 完成 v3 实现：CTC native、RxRx3-core manifest、唯一 formal launcher、严格 post-validator。
6. 最后同步改 `benchmark_model`，使外部 FM 与本模型读取同一 protocol JSON；旧 canonical
   suite 改名 `legacy_id_proxy_suite`，避免继续误用。

注意：Scout/SGST 命名的训练目录合计约 139.98 GiB，但其中混有当前 SKDT 的必要祖先
证据，不能按 glob 整批删除；必须依据决议中实际引用的 checkpoint 做白名单。
