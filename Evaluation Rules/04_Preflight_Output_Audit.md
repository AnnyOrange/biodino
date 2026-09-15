# 04 — Preflight, Result Validation, and Output Audit

状态：APPROVED（v3，2026-09-11）。

## 1. 每次开测前的 plan

在 `Evaluation Rules/plans/<campaign>.md` 写清：

- 研究问题和三臂/多模型 comparison grid；
- checkpoint 实路径、epoch/iteration、teacher branch、config；
- Tier A/Tier B 的完整 task-dataset 列表；observation 必须使用独立 plan 和 output，不得混入 formal matrix；
- 每项 resolution、resize、layers、batch、probe、seed、split/hash；
- 哪些 baseline 可以合规复用，证据路径是什么；
- 每台机器只运行本地/共享盘可直接读取的哪些 checkpoint；
- 每张 GPU 初始显存比例和计划并发数；
- output root、log root、skip 条件和失败重试上限；
- code commit 与依赖版本；
- 明确写出“不传输的 checkpoint/data”。

## 2. 启动前硬性 preflight

任何一项失败，launcher 必须退出非零：

1. Git commit 不匹配或 evaluator 工作树包含未登记修改；
2. checkpoint/config 不存在、仍在写入、无法读取或 teacher key 不明确；
3. dataset registry 不是审核通过的完整版本；
4. LC25000、nct-crc-he-100、BBBC038 或 MIDOG++ 出现在 formal dataset list；或出现被排除的 `regression/{conic-cell-count,livecell-cell-count}`、`detection/{conic,livecell,midogpp}`；或 formal segmentation 缺少 CoNIC/LIVECell；
5. frozen batch 不是 64，seg feature/probe batch 不是 32，detection batch 不是 8；
6. split manifest/hash、样本数或 group leakage 检查失败；
7. segmentation best protocol 展开后与规则表不一致；
8. baseline/matched/shuffled 的参数矩阵不完全相同；
9. deepcad 计划占用超过 4 张卡；
10. 远端任务要求临时复制 checkpoint/data，但没有用户授权。
11. v3 缺少 `retrieval/rxrx3-core` 或 `cell_tracking/ctc`，或者任一正式 split manifest/hash 仍是 `PENDING_IMPLEMENTATION`。

## 3. 完成判定

进程退出 0 不等于完成。一个 cell 只有同时满足以下条件才标 `VALID_COMPLETE`：

- 预期 result JSON/CSV 存在且能解析、无 `error`/NaN 主指标；
- result 内 checkpoint、config、dataset、split、sample count、resolution、layers、batch、seed 与 campaign manifest 一致；
- log 包含实际展开后的命令和 protocol；
- 同一三臂矩阵没有缺项；
- validator 生成独立 `validation_report.json`；
- 聚合脚本只读取 `VALID_COMPLETE` cells。

状态枚举固定为：`PLANNED`、`RUNNING`、`VALID_COMPLETE`、`FAILED`、`INVALID_PROTOCOL`、`EXPERIMENTAL_NOT_REPORTABLE`、`OBSERVATIONAL`。

## 4. 已知待审计输出

| output | 当前状态 | 原因 | 当前动作 |
|---|---|---|---|
| `outputs/02_eval_runs/biordt_ret4_skdt_e12to15_3090qi_20260909` | INVALID_PROTOCOL | batch16、Ret4、包含 LC25000、不是 full HS6 matrix | 已停；暂不删除 |
| `outputs/02_eval_runs/hs6_all_checkpoints_reg4_ret6_cluster6_det3_3090fleet_20260909` | EXPERIMENTAL_NOT_REPORTABLE | plan 中按模型改变 frozen batch，且包含 LC25000、CoNIC/LIVECell count regression 与 detection proxy；BBBC038 只能观察 | 保留已完成结果用于诊断；不得进入 v2/v3 aggregate |
| 旧 HS6 full/proxy outputs | NEEDS_AUDIT | batch、registry、split 与当前规则可能不同 | validator 扫描后分类 |

此外，CoNIC/LIVECell 的 cell-count regression 与 detection proxy 历史结果在 v2/v3 下只能标记为
`EXPERIMENTAL_NOT_REPORTABLE`；其 segmentation 结果只有在对应 source-grouped/official split 和完整协议均匹配时才可进入正式表。BBBC038 结果标记为 `OBSERVATIONAL`。被排除或观察的结果可以保留和查看，但聚合器必须排除。

`external4_hplus_fm_fixedbudget_3090qi_20260910` 中 CTC count proxy、RxRx3 128-gene screen 和 MIDOG++ candidate proxy 继续保持 observation。v3 正式纳入的是 CTC 原生 tracking 与 RxRx3-core 全 eligible-gene 固定协议；MIDOG++ 明确排除。

## 5. 测试完成后的全机审计

先只读扫描本机、deepcad、3090-qi、3090cpui、5090-lyx-xr、5090-hxw-xzj、H100：

- 列出所有本项目 train/eval PID、命令、GPU、显存、启动时间；
- 将每个运行中 test 与已审核 campaign manifest 对齐；
- 不在 manifest、参数不符、读错 checkpoint 或跨机违规的任务标红，先停后报告；
- 汇总每个 output root 的大小、mtime、是否有有效结果、是否只有 cache/features/失败日志；
- 不自动删除任何 material output。

## 6. Outputs 清理分类

- `KEEP`：论文原始 JSON/CSV、最终 manifest、validation report、必要日志、最终图表和不可复现的小文件。
- `ARCHIVE`：仍可能用于复核但不进入主结果的旧协议完整结果。
- `DELETE_CANDIDATE`：可重建 feature caches、重复 checkpoint adapters/symlinks、空目录、smoke 临时文件、确认无用的 failed partial results。
- `NEVER_AUTO_DELETE`：训练 checkpoint、唯一原始结果、用户文件、来源不明目录。

最终提供 CSV 清单：`path,size,owner,mtime,campaign,status,reason,recoverability,recommended_action`。只有用户逐项/按明确范围批准后才执行删除，并在删除后给出释放空间和可恢复性报告。
