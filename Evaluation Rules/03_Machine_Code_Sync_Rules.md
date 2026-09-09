# 03 — Machine, Placement, and Code Sync Rules

状态：APPROVED（2026-09-09）。

## 1. 数据与 checkpoint 放置

第一原则：**当机 train，当机 test；尽量不跨机传 checkpoint 或数据集。**

| 机器组 | 存储关系 | 规则 |
|---|---|---|
| 本机、deepcad、3090-qi、单卡 3090cpui | 都能访问 Huawei 共享盘 | 可以混合调度；直接读共享 checkpoint/data，使用路径或 symlink，不复制大文件 |
| 5090-lyx-xr | 独立/有限本地存储 | 在该机已有权重与数据上本地评测；默认禁止从 Huawei 随意推 checkpoint/data |
| 5090-hxw-xzj | 独立且存储紧张 | 在该机已有权重与数据上本地评测；默认禁止 staging 大 checkpoint/data |
| H100 | 独立训练存储 | 权重在哪台 H100 就在哪台评；只回传小型结果文件 |

允许跨机同步：代码、yaml、dataset/split manifest、JSON、CSV、Markdown、SVG/PNG、必要日志。
默认禁止跨机同步：`checkpoint.pth`、DCP shard、完整 feature bank、数据集 archive。确有必要时必须先向用户说明源/目标/大小/剩余空间并得到明确同意。

## 2. deepcad 限制

- deepcad 是借用机器，任何时候本项目占用的 **distinct GPU 不得超过 4 张**。
- 允许在这 4 张卡内部叠多个低显存 frozen test，但不能通过启动第 5 张卡扩容。
- 启动前和每 30 分钟记录一次 GPU index、memory used/total、utilization、本项目 PID；发现其他用户新增任务时主动收缩并发。

## 3. 单卡并发规则

按启动前 `memory.used / memory.total` 决定，不按瞬时 utilization 猜测：

- 显存占用 `<60%`：通常安排 **3–5 个 test processes/GPU**。
- 显存占用 `>=60%`：先只增加 **1 个**；确认峰值稳定后最多增加到 **3 个**。
- 任一卡最多 5 个本项目正式 test；dense 高分辨率 segmentation（尤其 MoNuSeg 768）先以 1 个探测峰值，再决定是否叠加。
- 若预计会 OOM：减少同卡 job 数或换卡；**不能降低协议 batch、resolution 或 layer 数**。
- 每个 job 完成后由统一队列领取下一个缺口，不按 task 类型让 GPU 空等。
- CPU/RAM/NFS 同样要限流：每个 test 默认 `num_workers=2`，BLAS threads=1；不得因 GPU 空闲把 NFS 打满。

## 4. GitHub 与代码同步

GitHub `origin` 是代码真源；正式 campaign 只认一个固定 commit SHA。

1. 评测规则或 evaluator 修改完成后，在本机检查 diff、跑 smoke/static validator、commit 并 push。
2. 每台执行机用 `git fetch` 后 checkout 同一 commit；禁止不同机器各自保留未记录的 evaluator patch。
3. 有脏工作树时不得直接 `git pull` 覆盖。创建独立 worktree 或先保存并提交相关修改。
4. 每个 output root 必须写 `campaign_manifest.json`，至少包含：Git commit、`git status --porcelain`、hostname、GPU、Python/torch/sklearn 版本、checkpoint/config 路径及 hash、dataset/split hash、完整命令和环境覆盖。
5. 活跃训练/评测期至少在“启动前、协议代码变化后、当天收工前”同步 GitHub；不是靠 scp 单个 `.py` 文件维持多机一致。
6. 远端只拉代码 commit；结果回传只拉 JSON/CSV/manifest/log，不回传 checkpoint。

## 5. 启动授权和操作边界

- 新 campaign 必须先提交 plan 给用户审核；审核通过才启动。
- 启动命令必须显式写 batch，不依赖按硬件自动推断的默认值。
- 不得擅自传 checkpoint、删 outputs、改 dataset split、换 embedding 层或缩小数据集。
- 出现协议错误立即停队列；错误输出标 invalid，等 outputs 审计阶段再决定删除。
