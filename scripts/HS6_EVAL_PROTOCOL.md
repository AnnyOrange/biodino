# HS6 评测与 Fig.2 口径（2026-09-01 通过；2026-09-04 C-scale 收窄）

json 可跨机拉；**checkpoint.pth 不跨机拷**。

作图准则另见 `scripts/FIG2_PLOT_SPEC.md`。本文只管评测怎么跑。

`plot/` 被 gitignore。跟踪副本是本文件；`plot/fig2/HS6_EVAL_PROTOCOL.md` 若存在应与本文同步。

---

## A. 评测操作（五条）

### 1. 权重在哪，评测在哪

评测必须在 **checkpoint 已经在、数据已经在** 的机器上跑。

| 权重 | 机器 | 评测 |
|---|---|---|
| S+ / B C-scale `prop15` last | **xr** `/data/xuzijing/biodino/outputs/01_training_runs` | **xr** |
| L C-scale `prop15` last | **hxw** `/mnt/data/biodino_fixed_pass/outputs/01_training_runs` | **hxw** |
| H+ C-scale `prop15` last | **H100** `/data_2/suxin/runs` GPU **2,3** | **H100 2,3**（不要动别人的卡） |
| H+ e15；H+ D-scale 0.2 / 0.5 / 1M | H100 `/data_2/suxin/runs` | **H100** |
| B e15 | hxw | **hxw** |

禁止：为了评测 rsync/scp `checkpoint.pth`（H+ ~17G 尤其禁止）。
允许：把 `last_result.json` / `results.json` / `kshot.csv` 拉回 NFS 作图（KB 级）。
作图只认 json，不要求权重在作图机上。

### 2. 卡不空也可以叠

不要等空卡。一张卡只要 **OOM 之前**，至少叠 **2–5** 路 frozen eval。
训完立刻测：当前 duration 的 last ckpt 一写出就开评，下一段 train 之前先把已齐的 last 测完。

叠的时候 **不准为了塞进小显存而把 batch 改小**。显存放不下协议 batch → 换更大的卡，或少叠一路，而不是 `bs=8`。

### 3. 动态补队列，不要按任务类型串行

一个 job 结束立刻从总缺口里取下一个，skip 已有合格 json。
先写计划（第 5 条），开跑后按缺口热更新。

### 4. 跨机协议必须一致

同一 `(模型, 任务, 数据集)` 在所有机器上用同一套超参。默认 frozen：

- classification / retrieval：**batch 64**（显存够则 **128**，但一旦选定，所有机器同一模型同一任务必须相同）
- `image-size 224`，`resolution-protocol best`，`channel-policy auto`，`split-protocol current`，`n-last-blocks 1`
- classification：`--no-save-features`（k-shot 抽完特征、探针写完 csv 后立刻删 npz）
- 禁止：按卡改 batch。

### 5. 开测前写计划

每次新队列先写短计划（`plot/fig2/eval_plans/` 或 `scripts/eval_plans/`）：

- 任务与数据集、ckpt 格子
- 每台机器评哪些 run（权重 pin）
- 协议 batch / 每卡叠几路
- 已有 json 跳过规则
- 不拷什么

---

## B. 2026-09-04 起 C-scale 只测三题

后面 compute scaling **只做**：

- `bloodmnist`
- `tissuemnist`
- `cyclops-protein-loc`

每题：frozen **full linear** + **10-shot**（k=5/10，seed 0/1/2）。
旧 C-scale（`wu0`/`wu1`/`wu3`、无 `prop15`）不进正文 Pareto。

---

## C. Fig.2 三轴（D/N 口径仍有效；正文 C 用 1M duration）

FLOPs：`6 * N_params * (ckpt_iter * 1024) * 514`
`N_params`：S+ 21M，B 86M，L 300M，H+ 840M。
`514 = 2 * ((256/16)^2 + 1)`。

### D：unique data，点 = 该档 8 个 epoch 的 **best**

### C：compute，横轴 **FLOPs**

正文主图 = 固定 1M，只变 duration + 模型（`lineB_1Monly`）。
`prop15` 链：e1 → e2 → e4 → e15（e8 不进 C 链）。

### N：model size，点 = **C-best**
