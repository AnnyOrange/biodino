# v4 补测配对结果

UTC 2026-10-08 15:29:10；执行状态：{'VALID_COMPLETE': 539, 'FAILED': 59, 'RUNNING': 5}。仅纳入双方已验证的相同步数。

|任务|数据集|指标|配对数|平均 Δ×100|
|---|---|---|---:|---:|
|classification|lc25000|balanced_accuracy|17|-0.0035|
|clustering|lc25000|nmi|17|-12.2945|
|clustering|nct-crc-he-100|nmi|17|-4.7257|
|clustering|rxrx3-core|nmi|17|+0.9929|
|detection_proxy|bbbc038|test_patch_f1|17|+0.3472|
|detection_proxy|conic|test_patch_f1|17|+0.0244|
|detection_proxy|livecell|test_patch_f1|17|+0.1908|
|ood|cryo|auroc|17|+0.0000|
|ood|cryo|average_precision|17|+0.0000|
|ood|xray|auroc|16|-0.0670|
|ood|xray|average_precision|16|-0.1262|
|regression|conic-cell-count|r2|17|+0.7329|
|regression|livecell-cell-count|r2|17|+1.5568|
|retrieval|lc25000|recall_at_1|17|-0.0012|
|retrieval|nct-crc-he-100|recall_at_1|17|+0.6536|
|retrieval|rxrx3-core|recall_at_1|17|-0.4328|

LC25000 分类仍是 provisional；回归为 ΔR²×100。此表只补充原 20TB 共享组件报告，未宣称完整 v4 胜出。
解析异常：[]
