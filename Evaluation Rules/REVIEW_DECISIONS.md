# Evaluation Rules — Review Decisions

用户已于 2026-09-09 批准以下规则，状态均为 `APPROVED`。

| ID | 建议规则 | 当前状态 | 影响 |
|---|---|---|---|
| R1 | LC25000 从 classification/retrieval/clustering 和所有 aggregate 永久移除 | APPROVED | 旧 Ret3 改名 Ret2；不能复用含 LC25000 的 aggregate |
| R2 | frozen classification/regression/retrieval/OOD 固定 batch 64；seg feature/probe=32；detection=8 | APPROVED | 历史 batch 8/16/32 frozen 结果不能直接复用 |
| R3 | Tier A 主比较 + Tier B 生物学/跨成像扩展都跑，但分别聚合 | APPROVED | 避免只挑 retrieval，也避免扩展任务污染主均值 |
| R4 | PanNuke 改为独立 val/test 后，baseline/matched/shuffled 全部重测 | APPROVED | 现有 fold3 同时 val/test 的结果降级为 legacy invalid |
| R5 | CoNIC segmentation/detection 改为 source-image grouped split 后三臂全重测 | APPROVED | 正式协议固定为 official baseline outer fold 0 + nested validation |
| R6 | 完整 dataset registry 正式提交到 GitHub，同一 commit 后再跨机评测 | APPROVED | 消除主树 3-dataset registry 与临时 full registry 的差异 |
| R7 | baseline 只在 validator 证明协议完全一致时复用，否则与两臂同规则重测 | APPROVED | 可能增加评测量，但保证归因有效 |
| R8 | 机器规则：共享 Huawei 组可混合；5090 两机/H100 当机训当机测；deepcad 最多 4 张卡 | APPROVED | 避免大 checkpoint/data 随意跨机传输 |

审核通过后的执行顺序：

1. 根据 R1–R8 修改 evaluator/registry/split manifest；
2. commit + push GitHub；Huawei 共享盘机器直接使用同一工作树，独立文件系统机器 checkout 同一 SHA；
3. 建立 SKDT e12–e15 三臂 full-matrix campaign plan；
4. 运行 preflight validator；
5. 用户已确认，preflight 通过后启动；
6. 结束后生成结果 validator 和全机/output 只读审计；
7. 与用户共同决定删除项。
