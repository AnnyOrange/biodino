# v4 补测配对结果

UTC 2026-10-07 16:18:16；执行状态：{'VALID_COMPLETE': 75, 'QUEUED': 385, 'RUNNING': 25, 'STARTING': 1}。仅纳入双方已验证的相同步数。

|任务|数据集|指标|配对数|平均 Δ×100|
|---|---|---|---:|---:|
|classification|lc25000|balanced_accuracy|4|+0.0100|
|clustering|lc25000|nmi|3|-10.3290|
|clustering|nct-crc-he-100|nmi|4|-3.1535|
|clustering|rxrx3-core|nmi|4|+0.9059|
|detection_proxy|bbbc038|test_patch_f1|4|+0.0533|
|detection_proxy|conic|test_patch_f1|3|+0.1556|
|detection_proxy|livecell|test_patch_f1|3|+0.1205|
|ood|xray|auroc|4|-0.0214|
|ood|xray|average_precision|4|-0.0230|
|regression|conic-cell-count|r2|4|+0.3240|
|regression|livecell-cell-count|r2|4|+0.0633|
|retrieval|lc25000|recall_at_1|3|+0.0000|
|retrieval|nct-crc-he-100|recall_at_1|4|-1.5152|
|retrieval|rxrx3-core|recall_at_1|4|-0.1362|

LC25000 分类仍是 provisional；回归为 ΔR²×100。此表只补充原 20TB 共享组件报告，未宣称完整 v4 胜出。
解析异常：[]
