# 05 — Current Implementation Gap Audit

状态：IMPLEMENTING（规则于 2026-09-09 获批）。本文件记录审核时发现的问题和修复进度。

| ID | 当前实现/脚本 | 与拟定规则的差距 | 启动前动作 |
|---|---|---|---|
| G1 | `dinov3/eval/bio_frozen_eval/registry.py` | 主树是只含 nct-crc-he 和两个 CHAMMI task 的裁剪 registry | 恢复完整 registry，加入测试后 commit |
| G2 | `scripts/run_bio_benchmark_all.sh`、`dinov3/eval/bio_benchmark.py` | default classification/retrieval 仍含 LC25000 | 从 formal defaults 移除；保留显式 legacy/debug 入口也必须标 deprecated |
| G3 | `scripts/run_hs6_l_6m_full_eval_fleet_worker.py` | lanes 含 LC25000；默认 frozen batch 32，dense batch 又被降到 4/16 | 改为审核后的显式任务 batch；不得按模型降 frozen batch |
| G4 | `scripts/run_hs6_all_ckpts_expanded_fleet_worker.py` 及其 plan | 含 Ret6/LC25000，并按 S+/B/L/H+ 使用 64/32/16/4 | 当前 campaign 标 `NEEDS_AUDIT`；不纳入新结果 |
| G5 | `scripts/watch_eval_hs6_l_skdt_arm_3090qi_20260909.sh` | 临时 Ret4、batch16、非 full matrix | 禁止恢复；审核时决定删除或移入 legacy |
| G6 | segmentation PanNuke loader | fold3 同时用于 val 和 test | 按审核决定建立独立 split/version；重测所有比较臂 |
| G7 | segmentation/detection CoNIC loader | 固定 random image-level 80/10/10，未验证 source/patient 隔离 | 建 source audit；若有 group id，改固定 grouped manifest |
| G8 | retrieval `completed()` 与 feature-cache key | cache/skip 判据没有完整覆盖 batch、resolution、n-last-blocks、dtype、code/split hash | formal output 使用 protocol fingerprint；不接受弱 skip |
| G9 | `bio_benchmark._successful_result_exists()` | 只要 JSON 无 `error` 就可能跳过，未验证协议字段 | 接入严格 validator 后才能 skip |
| G10 | segmentation result metadata | 主要依赖目录名表达 resolution/layers；JSON 没有完整 protocol fingerprint | 把展开后的 size/resize/layers/weight/batches/split hash 写入 JSON |
| G11 | 多个历史 launcher | batch、workers、jobs-per-GPU 和 dataset list 各自覆盖，存在口径漂移 | 新建唯一 formal launcher/config，旧 launcher 只读归档 |
| G12 | outputs | 同目录可能包含 partial、cache、旧协议和正式 JSON | 新 campaign 使用全新 output root；完成后统一只读审计 |

## 必须新增的防护

1. 一个 machine-readable protocol 文件，launcher 和 validator 同时读取，避免 Markdown 与命令分叉。
2. 一个 preflight validator：在 GPU 进程启动前检查数据集集合、split hash、batch、layers、resolution、checkpoint 与 commit。
3. 一个 post-run validator：逐 cell 校验 result metadata 和样本数，生成 `validation_report.json`。
4. output root 内写不可变 `campaign_manifest.json` 和 protocol fingerprint；fingerprint 不同不得复用 cache/result。
5. 所有正式 launcher 默认 fail closed：缺字段或无法证明匹配就退出，不“尽量继续”。
