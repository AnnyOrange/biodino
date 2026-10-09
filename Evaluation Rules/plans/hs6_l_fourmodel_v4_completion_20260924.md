# HS6-L 四 checkpoint Union-v4 补齐计划（2026-09-24）

目标 checkpoint：1TB/1M/1ep ck1024、5TB/训练 1M ck975、1PB/1M/1ep
ck976、100TB/1M/1ep ck976。已验证的 v3 共享结果仅在原指纹下复用，不改标、
不重跑；新增结果使用 `protocol_id=bio-eval-union-v4`。

每个 checkpoint 新执行 10 个任务：LC25000 classification；CoNIC/LIVECell
count regression；LC25000、NCT100 retrieval+clustering；RxRx3 full 734/734
retrieval+clustering；official MoNuSeg；BBBC038/CoNIC/LIVECell matched-B8
detection proxy。总计 40 个 GPU 任务。

执行资源为共享单卡 RTX 3090 节点，一卡一任务，通过共享目录上的原子 claim
队列调度。固定源码快照：
`/mnt/huawei_deepcad/dinov3_selective_retention_snapshot_20260923`；固定 Python：
`/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python`。

LC25000 classification 保留 `PROVISIONAL_LEGACY_ONLY`；NCT100 保留
`LOW_N=99`。CTC 的 fixed head/linker 尚未准入，X-ray/Cryo OOD 的公共选择和配置
仍未冻结，三项均保留 `BLOCKED_NOT_TESTED`，不得补零或静默移除。只汇报当前
可准入任务的完整覆盖，不在阻塞解除前声称 all-56 aggregate 完成。

输出：
`outputs/02_eval_runs/hs6_l_1tb_5tb_1pb_100tb_v4_completion_20260924`。
