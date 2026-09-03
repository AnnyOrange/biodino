# Fig.2 作图准则（2026-09-03 锁定）

本文是 **唯一画图标准**（git 跟踪副本；`plot/` 与 `docs/` 都被 gitignore；本文件是 git 跟踪副本，改完后同步到这里再 push）。改图之前先改本文；未改本文就改脚本，一律作废。

参考图：`plot/fig2/fromscratchandcontinue/scalingvit_fig2.png`  
论文：Zhai et al., *Scaling Vision Transformers*, CVPR 2022, Fig.2 + §2.1 + §2.2。  
评测格子、机器 pin、batch：`HS6_EVAL_PROTOCOL.md`（本文不管评测怎么跑）。

---

## 0. 论文原文（口径来源，不是装饰）

Caption：

> **Left/Center**: Representation quality, measured as ImageNet **finetune** and linear **10-shot** error rate, as a function of total training compute. A **saturating power-law approximates the Pareto frontier**. Note that smaller models (blue shading), or models trained on fewer images (**smaller markers**), saturate and **fall off the frontier** when trained for longer.
>
> **Top right**: Representation quality when bottlenecked by **model size**. For each model size, a large dataset and amount of compute is used, so model capacity is the main bottleneck. **Faintly-shaded markers depict sub-optimal runs of each model.**
>
> **Bottom Right**: Representation quality by **dataset size**. For each dataset size, **the model with an optimal size and amount of compute is highlighted**, so dataset size is the main bottleneck.

§2.1：

> For each combination of model size and data size we pre-train for **various numbers of steps**. In Figure 2, **connected points represent the same model trained for a different number of steps**.

§2.2：黑虚线是 Pareto 前沿上的饱和幂律 \(E = a(C+d)^{-b}+c\)，不是把同色点连成折线。

看参考 PNG，四个面板实际长这样：

| 面板 | 论文画了什么 |
|---|---|
| 左 C | ImageNet **finetune** error vs log compute。圆点面积 = 数据量。同 (模型, D) 的多个 **训完 duration** 用同色极淡细线连（附录 G 的 cooldown 终点，不是 cosine 半路 ckpt）。小模型 / 小数据那条线训久了弯上去。**唯一粗线 = 黑虚线饱和幂律**。 |
| 中 C | 同左，Y 换成 **10-shot** error。 |
| 上 N | 每个模型一列 **灰淡点**（该模型所有次优 run），再叠 **一个彩色 best**。灰点不是删掉，是背景云。 |
| 下 D | 每个数据量一列灰淡点，再叠 **一个** 该档最优 (模型 × compute)，颜色 = 拿下该档的模型。点的大小仍随 D 变。 |

C 上看起来「淡」的是 **同 (N,D) 的 duration 轨迹线**（以及相对前沿偏高的那些终点），不是把未退火 ckpt 画进去。caption 的 faintly-shaded 明确写在 N。我们 C 不画中间 ckpt；N/D 灰云才是全部 ckpt。

---

## 1. 一张图四个面板（每个数据集一张）

从左到右、右栏上下：

```
[ C  full linear ]  [ C  10-shot ]  [ N ]
                                    [ D ]
```

- **左 C**：frozen **full linear** error %（整份 train 的线性探针）。对应论文 Left 的 finetune 角色：全量下游。
- **中 C**：**10-shot** error %。对应论文 Center。主量尺。
- **N / D**：Y 轴与 **中 C 相同，用 10-shot**，不用 full linear。

### 为什么 N/D 用 10-shot，不用 full linear / 全量 finetune

1. 论文右栏与 **center** 共用 Y（10-shot）。caption 的 Left 才是 finetune；N/D 写的是 representation quality，没有改成 finetune。
2. 我们没有做下游 **full finetune**（整网 SGD）。有的是 frozen full linear 和 frozen 10-shot。不能把 full linear 叫成论文的 finetune，更不能拿它去填 N/D。
3. 生物题上 full linear 经常贴天花板，N/D 斜率被压平；10-shot 才是这条 scaling 图存在的理由。
4. 左 C 已经单独放了 full linear。N/D 再放一遍 full linear，等于丢掉论文右栏。

缺 10-shot 的格子：**N/D 留空或该点不画**，禁止偷偷换成 full linear。

禁止再出 `*_fullin_fig2.png` / `*_k10_fig2.png` 两套文件。一题一文件：`plot/fig2/scalingvit_fig2match/{stem}_fig2.png`（+ svg）。

---

## 2. 视觉（四个面板共用）

| 项 | 定法 | 禁止 |
|---|---|---|
| marker | **全部圆圈 `o`** | □ ◇ △ 分模型 |
| 颜色 | S+ `#465ECF` / B `#88ABFD` / L `#E97A5F` / H+ `#B40426` | 别的 ramp |
| 面积 | **= unique-D**：0.1M 最小 → 0.2M → 0.5M → 1M 最大。论文 30M/300M/1B/3B 同一逻辑 | 所有圆一样大（2026-09-03 那版错了） |
| C 上的点 | 只有退火 last，实心、alpha=1、面积 = unique-D | 把 e15 / D-scale 中间 ckpt 画进 C |
| N/D 淡点 | 固定灰 `#D5DBE5`，alpha 约 0.45，面积仍按该点 unique-D | 只画 best、没有灰云 |
| 连线（C） | **同 (模型, unique-D)** 的退火 duration 之间：极细、同色、alpha ≤ 0.25。目前只有 1M 够两点 | 把同一模型不同 D 连成一条；把 1M 的 C-scale/e15 与 0.1M 混连；把彩色线当拟合 |
| 拟合（C） | **一条黑虚线**，饱和幂律，只拟合 Pareto | 彩色 zigzag；把同模型点 `plot()` 成折线当 law |
| Y | error %，**不从 0 起**，按该面板点的范围裁。线性轴 | log-y（我们的 10pp 落差会被压没） |
| X | 全部 log（N 用参数量 log，刻度标 S+/B/L/H+；D 用 unique images） | |

面积比例（相对 0.1M = 1）：`0.1M : 0.2M : 0.5M : 1M = 1 : 1.6 : 2.6 : 4`（约按 \(\sqrt{D}\) 或 \(\log D\) 分级，不要按 D 正比，否则 1M 会把 0.1M 吃掉）。具体 `s=` 映射写进脚本常数，四个面板同一套。

图例：左上 C 放 **点大小 = unique-D**（0.1M / 0.2M / 0.5M / 1M）；中上 C 放 **颜色 = 模型**。不要再写 “all markers equal-size”。

---

## 3. C（compute）

### 3.1 论文为什么有连线，30M 和 300M 为什么同时在

连线 **不是** 拟合，也 **不是** 同一条 cosine 的中间 ckpt。

§2.1：*connected points represent the same model trained for a different number of steps.*  
附录 G：每个点一行 `Data Size | Steps | Cooldown`。同一模型、同一 Data Size、不同 Steps 的若干行，就是一条同色、**同样大** 的淡线。

所以图上同时有 30M 和 300M，是因为：

- **颜色** = 模型（L/16 全是橙）
- **面积** = 数据量（30M 小圆，300M 中圆，1B/3B 大圆）
- **淡线** = 固定 (模型, 数据量) 之后，把不同训完 duration 按 step 串起来

L/16 × 30M 自己一条小圆线（20k→4M steps，附录 G 有 8 个 cooldown 终点）；L/16 × 300M 另条中圆线。论文 **不会** 把 30M 连到 300M——那是换 D，不是换 duration。小数据那条线训久了弯上去，就是 caption 的 *fewer images (smaller markers) saturate and fall off*。

黑虚线是所有这些 **退火终点** 的 Pareto 幂律，和彩色淡线无关。

### 3.2 我们有哪些退火终点（C 上只画这些）

中间 ckpt **不进 C**（进 N/D 灰云）。C 只放每条 run 的 last（cosine 已落到 0）。对应论文附录 G 的一行。

ckpt 约定：C-scale e1/e2/e4 last = 1024 / 2049 / 4099；D-scale e8 last ≈ 0.1M 823 / 0.2M 1639 / 0.5M 4103 / 1M 8199；e15 last = 15374。

每个模型：

| unique-D（点面积） | 退火终点 | 条数 | 连线 |
|---|---|---|---|
| 0.1M | D-scale e8 last | 1 | **不连**（单点，像论文 G/14） |
| 0.2M | D-scale e8 last | 1 | 不连 |
| 0.5M | D-scale e8 last | 1 | 不连 |
| 1M（最大圈） | C-scale e1 / e2 / e4 last、D-scale 1M e8 last、e15 last | 最多 5 | 见 3.3 两种连法 |
| （空心） | 1-pass last，画在 e15 ck1024 左侧；与 ck1024 重合则不重复 | 1 | **不连进任何实心线** |

四个模型 × 上表，就是 C 的全部实心点。缺评的 last 就缺那个点，不拿中间 ckpt 顶。

小 D 只有 1 个 duration，所以 **不会** 出现论文 30M 那种「小圈连成一条、训久了掉出前沿」的轨迹。那是数据缺口（没在 0.1/0.2/0.5M 上另训 e1/e2/e4），不是连线规则能造出来的。不要用 D-scale 的 8 个未退火 epoch 去假扮这条轨迹。

### 3.3 连线规则

一条淡线的键 = `(模型, unique-D)`。z-order：淡线 < 黑虚线 < 实心点。淡线不是 law。

两种都画，等选定后删掉另一种（输出文件名带后缀，不是两套口径）：

| 方案 | 1M 上连谁 | 文件 |
|---|---|---|
| **A** `lineA` | e1 → e2 → e4 → **D-scale 1M e8** → e15（ckpt 1024 / 2049 / 4099 / 8199 / 15374） | `{stem}_fig2_lineA.png` |
| **B** `lineB` | 只连与 e15 **同一归一化 schedule** 的点：e1 → e2 → e4 → e15。D-scale 1M e8 **仍画同大实心点，不进线** | `{stem}_fig2_lineB.png` |

**重训完成后默认看 lineB。** e8 的 warmup/freeze/temp 形状跟 e15 不是同一条（37.5% / 12.5% / 26.7% vs 20% / 6.7% / 50%），它属于 D 面板，不进 C 的 duration 链。lineA 只作对照。

两种方案的 **点集相同**（D-scale 四档 last + C-scale last + e15 last 都在），只改 1M 淡线。拟合用的 Pareto 也相同。

C 面板还要出 **两种 D 口径**（文件名再加后缀，不是偷偷换点）：

| 口径 | C 上放哪些退火 last | 文件后缀 | 用来看什么 |
|---|---|---|---|
| `allD` | 0.1/0.2/0.5/1M 全放（现在这版） | `_allD` | 报告「D 已饱和时前沿被小 D 占住」这个反常结论 |
| `1Monly` | **只放 unique-D = 1M**（e1/e2/e4/e15，e8 仍按 lineA/B） | `_1Monly` | 固定最大数据量、只变 duration，才是 Chinchilla Approach 1 的 C |

两口径都出。正文主图用哪张，看完再定，不要只留好看的那张。

共同禁止：跨 0.1M↔1M；跨模型；1-pass；中间 ckpt；**跨过缺测档硬连**（见 §3.6）。0.1/0.2/0.5M 永远单点。

### 3.6 还没训完 / 还没评完

没有 error 就 **不能在图上出现一个点**。论文附录 G 也是缺行就不画（*a few missing rows*），不会用空心点去占一个假的 \(E\)。

| 想做的 | 定法 | 为什么不行 |
|---|---|---|
| 虚线空心圆占位 | **禁止** | 没有 \(E\) 就没有 \(y\)。空心圆已经留给 1-pass。黑虚线留给 \(E\) 拟合。再做一种「虚线点」三者撞车，还像造了数据 |
| 用邻近点插一个 \(E\) | **禁止** | 插值进 Pareto / 拟合会改 law |
| 用未退火中间 ckpt 顶 last | **禁止** | §3.2 |

**只做这三件事：**

1. **点**：该格子缺 last 就不画。C-scale e4 还在训、10-shot 没出 json，图上就是少一个 1M 大圆。
2. **1M 淡线**：按计划顺序（lineA：e1–e2–e4–e8–e15；lineB：e1–e2–e4–e15）只把 **相邻两档都在** 的线段画实淡线。中间缺一档就 **断开**，不要把 e1 直接连到 e15 假装走过 e2/e4。
3. **\(E\) 拟合 / N / D**：只用已有点。D 某档 8 个 epoch 没齐，灰点画已有的，best 在已有里取，**不要**把 last 写成 best-of-8；该档高亮点若 epoch 数 < 8，边框改成 **细灰圈**（仍实心色，表示「best among incomplete」），齐 8 才白边。

图注不逐题报缺。缺哪些格子记在评测计划，不写进每张 png 的 caption。

### 3.4 \(E\) 曲线（饱和幂律）— 给定，不要再改实现细节

论文 §2.2 原文：Pareto 前沿 ≈ \(E = a C^{-b}\)；高端饱和加 \(c\)（任务不可约误差）；低端饱和加 \(d\)（零算力也能蒙对）。合成：

\[
E = a(C + d)^{-b} + c
\]

图上为让系数可读，写成与 Scaling-ViT 面板相同的归一化：

\[
E = c + a\left(\frac{C}{C_0} + d\right)^{-b},\qquad C_0 = 10^{18}\ \text{FLOPs}
\]

\(E\) 的单位是 **error %**（与坐标轴一致，不要用 0–1）。\(C\) 的单位是预训 FLOPs：

\[
C = 6 \cdot N_{\mathrm{params}} \cdot (\mathrm{ckpt} \cdot 1024) \cdot 514
\]

ckpt = 该权重的 iteration。\(N_{\mathrm{params}}\)：S+ 21M / B 86M / L 300M / H+ 840M。

**谁进拟合（输入点）**

- 只用 C 面板上的 **退火 last**（§3.2 那张表，含四档 D-scale last、C-scale last、e15 last）。
- 先做 **Pareto 前沿**：按 \(C\) 升序扫，只保留严格更低 \(E\) 的点（不存在另一 last 满足 \(C'\le C\) 且 \(E'<E\)）。
- 中间 ckpt、1-pass、N/D 灰点 **一律不进**。
- lineA / lineB 拟合点集相同。

**怎么估参数**

不要用「把同色点连起来」当曲线，也不要单独再画一条 Pareto 折线冒充拟合。

1. Pareto 点不足 **4** 个：只画点，不画公式、不画虚线。
2. 网格 \(c \in [0,\ 0.99\min E_{\mathrm{Pareto}}]\)（25 档），\(d \in \{0\}\cup\{10^{-3},\ldots,10^{1}\}\)（相对 \(C_0\)，约 20 档）。
3. 对每对 \((c,d)\)，令 \(y_i = E_i - c\)（若有 \(y_i\le 0\) 则丢弃该 \(c\)），\(x_i = C_i/C_0 + d\)，在 \(\log y = \log a - b \log x\) 上做 OLS。
4. 要求 \(a>0,\ b>0\)。残差用 **线性 \(E\) 的 MSE**（\(\frac1n\sum(\hat E-E)^2\)）选最优；并列时取较小的 \(b\)。
5. 禁止 \(c > \min E_{\mathrm{Pareto}}\)（曲线不得穿过已有最优之下去「更好看」）。

实现：`scripts/plot_hs6_fig2_paper.py` 里的 `fit_saturating_power_law`。不要再调已经对不上的旧 `fit_power_law`。

**怎么画**

- **唯一** 表示 law 的线：黑虚线，在 \([\min C\times 0.85,\ \max C\times 1.15]\) 上 log 取样 200 点，画 \(\hat E(C)\)。线宽约 1.6，z-order 在淡 duration 线之上、圆点之下或之间，但 **绝不是** 彩色折线。
- 水平细实线 \(E=c\)（渐近）。
- 面板内公式：\(E = c + a(C/10^{18}+d)^{-b}\)，系数 2 位小数。
- 坐标：\(x=\log C\)，\(y=E\%\) **线性、不从 0 起**（按该面板点裁）。不把 \(y\) 改成 log——论文正文说 log-log，我们的 10pp 落差在 log-y 上会消失；拟合仍按上面的 \(\log(E-c)\) OLS，显示用线性 \(y\)。

### 3.5 左 C vs 中 C

同一套 x / 同一套点的集合 / 同一套连线与拟合规则。只换 Y：

- 左：full linear error %
- 中：10-shot error %（多种子则 mean）

---

## 4. N（model size）

Y = **10-shot error %**。

- 每个模型一列。
- **灰淡点** = 该模型在所有评过的 run、所有 ckpt 上的 10-shot（D-scale 4 档 × 8 epoch + 1M e15 × 15 ckpt + 已评的 C-scale last）。面积 = 该点自己的 unique-D，所以一列里能看到小圈和大圈。
- **一个彩色实心** = 上述集合里 error **最低** 的那一个。颜色 = 该模型。面积 = 这个 best 所在档的 D。
- 黑细虚线只连四个彩色 best。
- X = 参数量 log：S+ 21M / B 86M / L 300M / H+ 840M，刻度写模型名。

这是 caption 的 “large dataset **and** amount of compute”：在该模型的全部格子里找 best，**不是**「1M e15 C 曲线上的 best」。后者会丢掉 H+ 在 0.5M 上更好的点（BBBC048 10-shot 旧错：57% vs 真实 ~49%）。

`HS6_EVAL_PROTOCOL.md` 若仍写「N = 1M e15 C 曲线 best」，以本文为准，那条作废。

1-pass **不进 N**。

---

## 5. D（dataset size）

Y = **10-shot error %**。

- X = unique images log，四档 **0.1 / 0.2 / 0.5 / 1M**。
- 每档一列 **灰淡点** = 该档 D-scale e8 run 上，所有模型、所有 8 个 epoch 的 10-shot。面积 = **这一档的 D**（同一列一样大；0.1M 列小，1M 列大）。
- **一个彩色实心** = 该档 (模型 × epoch) 里 error 最低的那个。颜色 = 获胜模型。面积 = 该档 D。
- 黑细虚线只连四个彩色 bottleneck。
- **只使用 D-scale 的 8 epoch**。1M 档 **不混** e15 的 15 ckpt，也 **不混** C-scale e1/e2/e4。
- 1-pass **不进 D**。

缺某个 (模型, 档, epoch) 的 10-shot：该灰点不画；best 在已有点里取。不要用 last 冒充 best-of-8。

---

## 6. 数据白名单 / 黑名单

画进去：

- D-scale：`HS6_Dscale_*_e8_random{10,20,50,100}_seed0_20260820`
- C 1M e15：`HS6_{Splus,B,L,Hplus}_robust_..._e15_seed0_*20260818` 或 `*20260821b`（S+ **必须有 b**）
- C-scale（**只认等比 schedule**）：`HS6_Cscale_*_prop15_*_e{1,2,4}_*` 的 **last only**

不画：

- `HS6_L_scratch_randominit`
- S+ `20260821`（无 b）
- 6M gram / 任何非上述 output
- **旧 C-scale**（目录名含 `wu0` / `wu1` / `wu3`，或没有 `prop15`）：warmup/freeze 按绝对 epoch 钉死，e2 是 50% warmup + 50% 冻最后一层。这些点是 schedule 异常，不是 duration。重训点齐了之后从 C 上拿掉；在那之前若必须对照，边框改灰并在内部笔记标明，**不进 Pareto / 不进黑虚线**。

---

## 7. 上次错在哪（对照检查表）

画完后必须能勾上，否则不准当成成品：

1. N、D 的 Y 是 **10-shot**，不是 full linear，也不是 finetune。
2. N、D 都有 **灰色中间点云**，彩色只是每列一个 best。
3. 0.1M 的圈明显小于 1M；C / N / D 同一套面积映射。
4. C **没有** 中间 ckpt；0.1/0.2/0.5M 各模型各一个实心点；1M 上 last 同色同大。lineA 连 e1–e15 含 e8，lineB 不含 e8。
5. 拟合只有 **黑虚线平滑幂律**（§3.4），没有 Pareto 折线、没有彩色 law。
6. 同色淡线只连 **同一 unique-D 的退火 duration**，不跨 0.1M↔1M。
7. 四个面板在同一张图。
8. 缺训/缺评的 last **没有点**；1M 淡线只连相邻都在的档，不跨缺口。

---

## 8. 脚本

实现必须读本节，而不是记忆上次的 py：

- 主脚本：`scripts/plot_hs6_fig2_paper.py`
- 拟合：同文件 `fit_saturating_power_law`（§3.4）
- 输出：`plot/fig2/scalingvit_fig2match/{stem}_fig2_lineA.{png,svg}` 与 `{stem}_fig2_lineB.{png,svg}`

改视觉常数（颜色、面积、alpha、拟合点集）= 改本文。

---

## 9. C 为什么乱、怎么修（训练，不是画图）

N/D 能看、C 不能看，不是 marker 的问题。三件事叠在一起：

1. **Schedule 没跟 duration 走**（必须重训）。旧 launcher 把 `warmup_epochs` / `freeze_last_layer_epochs` 钉成绝对 epoch：e1=0/0，e2=1/1，e4=3/1，而 `warmup_teacher_temp_epochs` 一律 30。归一化 LR 相对 e15 的偏差：e1 = 100%，e2 = 60%，e4 = 79%。e2 在 Blood/Tissue/Cyclops 上三个模型都比 e1 差，是这条 run 在做另一件事，不是噪声。
2. **10-shot 方差太大**（正当降噪）。3 种子 std 中位 2.15pp；Tissue 整条 C 跨度只有 ~4pp。改成论文的闭式 L2 ridge，种子加到 10–25。禁止挑种子、挑 ckpt。
3. **D 已经饱和**（调参修不了）。0.1M→1M 只有 H+ 有真斜率；S+/B/L 基本平或反向。FLOPs ∝ D×epochs，多给数据不涨分，于是「小 D + 8 epoch」又便宜又不差，Pareto 被 0.1/0.2M 占满。这是结果，不是 bug。所以才要同时出 `allD` 和 `1Monly`。

重训口径（已经锁）：以 e15 为形状锚点，每条 run 走同一条归一化 schedule，**唯一变量是 compute**。

| T | warmup | teacher-temp | freeze | 是否重训 |
|---|---|---|---|---|
| 1 | 0.2（205 step） | 2 | 0.067 | 要 |
| 2 | 0.4 | 4 | 0.133 | 要 |
| 4 | 0.8 | 8 | 0.267 | 要 |
| 8 | 1.6 | 16 | 0.533 | **不**。从 C 链剔除（lineB），留在 D 面板 |
| 15 | 3 | 30 | 1 | **不**。已经是锚点 |

四个数都能被 1025 整除。实现：`dinov3/train/train.py` 的 `_iters()` + `scripts/launch_hs6_cscale_duration_prop.sh` + `scripts/run_hs6_cscale_prop.sh`。

**禁止**用挑数据集 / 挑 ckpt / 挑种子 / 把 \(c\) 调到已有最优之下 来「满足 scaling law」。那是造 law。

### 9.1 代码怎么对齐（本机改 → GitHub → 各机 pull）

细节只在 **本机** 改，`git push origin main`，其它机器 `git pull --ff-only`。不要再 scp 补丁、不要在 xr/hxw 上就地改 `train.py`。

各机 pull 之前如果工作区脏了（包括曾经跑过 `patch_fractional_epoch_schedules.py`）：

```bash
# 若有 bak，先回到 patch 之前，避免和即将 pull 下来的官方 _iters 打架
[ -f dinov3/train/train.py.bak_fracsched ] && cp dinov3/train/train.py.bak_fracsched dinov3/train/train.py
git stash push -u -m "local-before-cscale-pull" -- dinov3/train/train.py
git pull --ff-only
# 确认官方补丁在
grep -n '_iters(cfg.optim\["warmup_epochs"\])' dinov3/train/train.py
```

`scripts/patch_fractional_epoch_schedules.py` 只留给还没 pull 到的旧 checkout，新机器不要再跑它。
