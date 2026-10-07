# Adaptive-v2：全任务进度、纠错与下一轮实践

2026-10-07 UTC。本报告记录已实际执行的工作；队列入选不等于测试完成。

## 方法与结果

核心方法仍是固定权重的线性可恢复性约束：用训练中的锚点 teacher 约束 CLS 信息可恢复，保留原优化器状态。实验已区分 CLS / CLS+patchmean / local patch、固定锚点 / EMA 锚点、权重 0.3 / 1 / 3。当前没有证据可以宣称新的结构性突破或全 v4 最优。

5TB 全部已测点的阶段结果：固定 CLS w1 相对 no-GRAM，分类约 +0.67 pp、回归 ΔR²×100 +0.46、分割 +0.17 pp、检测代理 −0.09 pp，检索接近持平。慢锚点 CLS w1 分割约 +0.18 pp，但检测 −0.12 pp、检索 −0.06 pp。各分支覆盖量不同，不能拿这些均值直接排名。

新增的慢锚点 CLS w0.3 只有早期三个导出点。`MATCHED_COVERAGE.md` 限制在三个分支共同完成的同一步数、同一能力后，固定 CLS w1 / 慢锚点 w1 / 慢锚点 w0.3 的分割分别为 −0.051 / −0.207 / −0.262 pp，回归 ΔR²×100 为 +0.837 / +0.816 / +0.377。降低权重目前不是全面改进；不能用较少的早期结果冒充升级成功。

20TB 已测共享组件显示明显任务冲突：BBBC005/013 回归下降，RxRx1 检索下降，NCT-CRC 分类下降；Cellpose、CoNIC 部分分割设置提高。新增完整 v4 扩展结果见 `V4_EXTENSION_RESULTS.md`：RxRx3 的检索下降但聚类提高，CoNIC/LIVECell 计数回归提高，检测代理有升有降，X-ray OOD 接近饱和且略降。配对数仍少，不做显著性或全面胜出结论。检测输出原本是 0–100，汇总已正确归一化；回归始终报告 ΔR²。

## 本次实际修复与启动

1. 原 20TB 测试队列在 10 月 6 日因新 checkpoint 校验期间仍被写入而全局暂停。修正为首次注册的写入竞争仅延后该权重 300 秒；已注册权重发生变化仍报硬错误。保留原暂停与 claim 证据，恢复本机、3090-qi、deepcad worker，未改 batch、split、seed 或 evaluator。
2. 新建 `outputs/02_eval_runs/v2_full_v4_20261007`。30 个 20TB checkpoint 逐一覆盖预期 56 个 task–dataset cell；额外注册 390 个执行任务，补计数回归、LC25000、NCT100、RxRx3、B8 检测代理、原生 CTC、X-ray/Cryo OOD、MoNuSeg 30/7/14 六次 E20/E50 probe。CTC 使用原生 20 域、5 折、固定 head/linker。没有用计数代理替代 CTC。新增 checkpoint 自动入队。
3. 共享盘上的 5TB no-GRAM 与官方 GRAM 共 24 个现存 checkpoint 已额外登记 RxRx3/CTC/OOD 四项补测，共 96 个执行任务。其余已测家族保持原始来源，清单标记 `REMOTE_COMPONENT_VALIDATION_PENDING`，没有据此虚报完整。
4. lyx 的慢锚点 w0.3 分支在完成 31,196 次更新后发生 NCCL collective timeout。最新完整 checkpoint 是 30,743，已实际恢复优化器并观察到新的更新；需重放 452 个未保存更新。先终止了本队列 62 个并发 dense 任务的进程组，保留结果和缓存供续跑；旧调度器退下，新的 dense 调度仅使用 4–5 卡，统计旧子进程并限制全机并发 12，留出 180 GiB 主机内存余量。6–7 卡有其他用户作业。
5. deepcad 的 cgroup 上限为 400 GiB。之前的 guard 又把可回收文件缓存当成不可用内存，导致测试无法恢复。现在只计入 clean inactive file 的可回收部分，扣除 dirty/writeback，继续保留 64 GiB 余量和 80 GiB 常规评测预算。已启动常规测试、两个原生 CTC 任务，另登记了只跑 MoNuSeg 的 worker；是否正在执行以 worker 状态为准。
6. 新增固定锚点 CLS w0.3 对照：与当前慢锚点 w0.3 使用同一 29,279 完整 fork、相同训练设置和 35,136 更新终点，只改变锚点是否 EMA。串行调度器已部署，当前状态是等待前一训练完成。非分割、dense、MoNuSeg 和汇总均登记了新 arm；尚无新 arm 训练结果。

## 资源与验证

实测资源快照见 `GPU_STATUS.json`。3090-qi 8 卡已超过 70%；本机大部分/全部卡在不同采样时刻超过 70%。deepcad 部分卡已超过 70%，其余受 RAM/加载阶段约束。lyx 0–3 卡训练约 65%，4–5 卡已有并行测试，但显存仍明显不足 70%，并存在较高磁盘 I/O 等待。因此用户的“所有测试卡显存超过 70%”尚未完全达到，未用空占显存伪造利用率。

已经完成 Python 编译、shell 语法检查、30 个 20TB checkpoint 的 56-cell 逐项覆盖校验、扩展命令 checkpoint 身份校验，以及真实 evaluator 的结果验证。MoNuSeg 的完整 source snapshot 和数据 identity 文件逐项核对 SHA256。扩展队列已实际产出 VALID_COMPLETE 的 X-ray、RxRx3、计数回归、检索聚类和检测结果；未完成或等待资源的任务保留状态。状态和扩展结果每 180 秒刷新。

## 后续实验判断标准

主线继续比较固定 CLS w1 与慢锚点 CLS w1，优先追踪检索/聚类分离、检测与分割退化，不以分类均值决定胜者。w0.3 和固定锚点 w0.3 是控制变量实验，尚未证实改进。每个任务用相同步数、相同数据集、完整固定 split 与 probe 预算比较；LC25000 分类保持 provisional，不计算虚假的严格全 v4 总分。保留所有完整优化器 checkpoint。

5TB 方法分支的 CTC/RxRx3/OOD 数据不在 lyx 现有评测数据目录。已列出 lyx/hxw 的 118 个 teacher 权重，合计 154.064 GiB；共享盘剩余约 6.70 TiB。`TRANSFER_PLAN.json` 与 `scripts/mirror_v2_5tb_teachers_20261007.py` 已准备好：限速复制，核对源/目标 SHA256，原子发布，逐点加入缺失任务，不搬 optimizer 或数据集。根据 `Evaluation Rules/03_Machine_Code_Sync_Rules.md` 第 17 行，跨机大权重复制须明确同意；已向用户提出这一个具体确认，尚未执行复制。其他训练与测试持续运行。
