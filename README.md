# DINOv3（当前分支）快速使用说明

## 文档怎么读

| 顺序 | 内容 |
|------|------|
| **1** | 环境配置 |
| **2** | **推荐流程**：多通道显微 → 重打包 → 按通道统计 → 用 `packwds:` + ChannelViT（及可选 LoRA）训练 |
| **3** | **补充**：常用参数、分布式启动、LoRA / 三通道等命令行变体 |
| **4** | **补充数据源**：从 S3 或按通道分目录的本地 shard **直接**训练（`wds:` / `s3wds:` / `cachewds:`） |

配置文件：`dinov3_vit7b16_pretrain_webdataset.yaml`（全量预训练）、`dinov3_vit7b16_pretrain_lora.yaml`（LoRA）、`dinov3_vit7b16_pretrain.yaml`（全量预训练默认无channelvit）。`in_chans`、`enable_channelvit`、`train.dataset_path` 等优先**命令行覆盖**。

**前缀约定：** `train.dataset_path` 必须以合法前缀开头，否则会被当成 ImageNet 风格字符串解析而报错。支持的前缀见 [`dinov3/data/loaders.py`](dinov3/data/loaders.py) 中 `make_dataset`，主要包括：`packwds:`（重打包后的多通道 tar）、`wds:`、`s3wds:`、`cachewds:` 等。

---

## 1) 环境配置

### 1.1 创建环境

优先使用项目里的 `conda.yaml`：

```bash
conda env create -f conda.yaml
conda activate dinov3
```

若使用 `micromamba`：

```bash
micromamba env create -f conda.yaml
micromamba activate dinov3
```

---

## 2) 推荐流程：多通道显微 + ChannelViT（重打包 → 统计 → 训练）

目标：同一物理视野的多通道对齐在**同一条** WebDataset 样本内（[`dinov3/data/repackage`](dinov3/data/repackage) 流水线），再按通道估计 `rgb_mean` / `rgb_std`，最后用 **`packwds:`** 解码 + **ChannelViT** 训练。重打包细节见 **[`dinov3/data/repackage/README.md`](dinov3/data/repackage/README.md)**。

### 2.1 输入数据从哪来

- **已在 NFS / 本地**：按通道分目录的原始布局，直接作为 `preprocess_repack.py` 的 `--input-root`。
- **仅在 S3**：先用 `aws s3 sync` 拉到本机，再重打包；或仅用第 4 节的 `s3wds:` / `cachewds:` **跳过 repackage、直接训**（通道对齐方式不同，见第 4 节说明）。

### 2.2 重打包
对数据进行切割，具体的切割方法详细见 [`dinov3/data/repackage/README.md`](dinov3/data/repackage/README.md) 可以适当调节参数
在**仓库根目录**执行：

```bash
python dinov3/data/repackage/preprocess_repack.py \
  --input-root /mnt/huawei_deepcad/webds_micro_100k_by_channel \
  --output-root /mnt/huawei_deepcad/wds_packed_shards \
  --reference-channel 1 \
  --variance-threshold 40.0 \
  --num-workers 16 \
  --shuffle-buffer-size 10000
```

输出示例：`filtered_mixed_train_w*-{000000..}.tar`（每样本含 `ch1.tif` …、`meta.json` 等）。

### 2.3 按通道统计 mean / std

对 **packed** shard 统计（与解码 uint16→[0,1] 一致），结果粘贴进 YAML 的 `crops.rgb_mean` / `crops.rgb_std`：

```bash
python dinov3/data/repackage/compute_channel_stats.py \
  --shard-pattern "/mnt/huawei_deepcad/wds_packed_shards/filtered_mixed_train_w*-{000000..000999}.tar" \
  --max-channels 8 \
  --max-samples 0
```

- `--max-samples 0`：全量（慢、更准）；可改为 `10000` 做快速近似。

### 2.4 训练：使用 `packwds:`（全量预训练 + ChannelViT + 官方权重）

`packwds:` 会按 `meta.json` / `ch*.tif` 组出固定 `student.in_chans` 维度的张量；**`student.in_chans` / `teacher.in_chans` 须与 `--max-channels`（及 YAML 中张量宽度）一致**。将上一步打印的 `rgb_mean` / `rgb_std` 写入 `dinov3_vit7b16_pretrain_webdataset.yaml` 的 `crops` 段后再启动，或用 Hydra 覆盖列表。

在**仓库根目录**执行（示例 8 通道；请按本机 GPU 设置 `CUDA_VISIBLE_DEVICES` 与 `nproc_per_node`）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_webdataset.yaml \
  --output-dir ./outputs/chvit_packed \
  train.dataset_path="packwds:/mnt/huawei_deepcad/wds_packed_shards/filtered_mixed_train_w*-{000000..000999}.tar" \
  student.in_chans=8 \
  teacher.in_chans=8 \
  student.enable_channelvit=true \
  teacher.enable_channelvit=true \
  student.resume_from_teacher_chkpt=/mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth \
  student.fp8_enabled=false
```

**训练：LoRA + `packwds:`（示例）**

```bash
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 \
  dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_lora.yaml \
  --output-dir ./outputs/lora_packed \
  train.dataset_path="packwds:/mnt/huawei_deepcad/wds_packed_shards/filtered_mixed_train_w*-{000000..000006}.tar" \
  student.in_chans=8 \
  teacher.in_chans=8 \
  student.enable_channelvit=true \
  teacher.enable_channelvit=true \
  student.resume_from_teacher_chkpt=/mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth \
  student.fp8_enabled=false
```

（通道数、shard 范围、是否 ChannelViT 可按数据与实验修改；`crops.rgb_mean` / `rgb_std` 需与通道数匹配，建议沿用 2.3 的输出。）

**训练：nochannelvit + `packwds:`（示例）**
请更改对应yaml中的rgb_mean和rgb_std。同理这里lora请将dinov3_vit7b16_pretrain.yaml改为dinov3_vit7b16_pretrain_lora.yaml
```bash
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 \
  dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
  --output-dir ./outputs/lora_packed \
  train.dataset_path="packwds:/mnt/huawei_deepcad/wds_packed_shards/filtered_mixed_train_w*-{000000..000006}.tar" \
  student.in_chans=3 \
  teacher.in_chans=3 \
  student.resume_from_teacher_chkpt=/mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth \
  student.fp8_enabled=false
```

---

## 3) 补充：参数与启动方式

### 3.1 GPU 与 `torchrun`

- **`CUDA_VISIBLE_DEVICES` 与 `--nproc_per_node`：** 进程可见的 GPU 个数必须等于 `nproc_per_node`。例如 `CUDA_VISIBLE_DEVICES=7` 只暴露 1 张卡时，应使用 `--nproc_per_node=1`。
- 所有示例均假设在**仓库根目录**执行，入口为 `dinov3/train/train.py`。

### 3.2 常用训练相关参数

| 项 | 说明 |
|----|------|
| `train.dataset_path` | 前缀决定数据源；推荐流水线用 `packwds:`；直连 S3/按通道 shard 见第 4 节。 |
| `train.batch_size_per_gpu` | 单卡 batch。当前训练在 **FSDP 数据并行** 下，**全局 batch = `batch_size_per_gpu × world_size`**（`world_size` 为参与训练的 GPU 总数）。DINOv3 原文 ViT-7B 规模预训练常取 **全局 batch 约 4096**（例如 8 卡 × 512/卡、64 卡 × 64/卡 等），需按显存与卡数折算 `batch_size_per_gpu`。 |
| `student.resume_from_teacher_chkpt` | 加载预训练 backbone + checkpoint，初步调试可以不加这部分 |
| `student.fp8_enabled` | 与配置一致；调试可先 `false`。 |
| `student.in_chans` / `teacher.in_chans` | 通道数；数据 `target_channels` 以 `student.in_chans` 为准，建议与 teacher 一致。 |
| `student.enable_channelvit` / `teacher.enable_channelvit` | 多通道显微推荐 `true`。 |
| `crops.rgb_mean` / `crops.rgb_std` | 与通道数一致；packed 数据建议用 2.3 脚本结果。 |
| `compute_precision.hsdp_shards` | `1` 纯 FSDP；`>1` 为 HSDP，需 `world_size % hsdp_shards == 0`。 |
| `train.compile` | 多机时若遇编译问题可设 `train.compile=false`。 |

### 3.3 多机多卡（模板）

各节点 `--master_addr` / `--master_port` 相同，仅 `--node_rank` 不同（`0 … nnodes-1`）。`world_size = nnodes * nproc_per_node`。

**节点 0：**

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --master_addr=10.0.0.1 \
  --master_port=29500 \
  dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_webdataset.yaml \
  --output-dir ./outputs/chvit_packed \
  train.dataset_path="packwds:/mnt/huawei_deepcad/wds_packed_shards/filtered_mixed_train_w*-{000000..000999}.tar" \
  train.compile=false \
  student.in_chans=8 \
  teacher.in_chans=8 \
  student.enable_channelvit=true \
  teacher.enable_channelvit=true \
  student.fp8_enabled=false
```

**节点 1：** 将 `--node_rank=1`，其余同上。

### 3.4 其他命令行变体（非 packed、或未用 S3 直连）

**LoRA + 按通道分目录的本地 `wds:`（单通道 shard，非 `packwds:`）：**

```bash
torchrun --nproc_per_node=4 dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_lora.yaml \
  --output-dir ./outputs/lora_chvit \
  train.dataset_path="wds:/mnt/huawei_deepcad/webds_micro_100k_by_channel/ch1/train-{000000..000128}.tar" \
  student.in_chans=4 \
  teacher.in_chans=4 \
  student.enable_channelvit=true \
  teacher.enable_channelvit=true \
  student.resume_from_teacher_chkpt=/mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
```

**RGB 三通道 + ChannelViT：**

```bash
torchrun --nproc_per_node=4 dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_webdataset.yaml \
  --output-dir ./outputs/rgb3_chvit \
  train.dataset_path="wds:/path/to/shards/train-{000000..000099}.tar" \
  student.in_chans=3 \
  teacher.in_chans=3 \
  student.enable_channelvit=true \
  teacher.enable_channelvit=true
```

**RGB 三通道、不启用 ChannelViT：**

```bash
torchrun --nproc_per_node=4 dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_webdataset.yaml \
  --output-dir ./outputs/rgb3 \
  train.dataset_path="wds:/path/to/shards/train-{000000..000099}.tar" \
  student.in_chans=3 \
  teacher.in_chans=3 \
  student.enable_channelvit=false \
  teacher.enable_channelvit=false
```

更多见 [`dinov3/docs/LORA_CHANNELVIT_USAGE.md`](dinov3/docs/LORA_CHANNELVIT_USAGE.md)。

### 3.5 无 ChannelViT vs 有 ChannelViT（数据与模型）

- **数据：** `dinov3/train/train.py` → `make_dataset(..., target_channels=cfg.student.in_chans)` → `dinov3/data/wds_decoder.py` 等对通道的补齐 / 截断 / 循环填充；`packwds:` 走 `decode_packed_sample`，缺通道零填充到 `target_channels`。
- **模型：** `dinov3/models/vision_transformer.py`：`enable_channelvit=false` 为标准 `PatchEmbed`；`true` 为 `PatchEmbedPerChannel` + `channel_embed` 与 RoPE 多通道分支。

不开启 ChannelViT 时，样本会对齐到 `student.in_chans`（例如 3 通道即常见的「少补齐、多截断」语义）。开启 ChannelViT 时，建议 `student.in_chans` 等于真实通道数。注意`rgb_mean` 和 `rgb_std` 的个数应该和 `in_chans` 相同。

---

## 4) 补充数据源：从 S3 或按通道本地 shard 直接训练

本节适合：**不重跑 repackage**、直接从 **S3** 或 **本机按通道分目录的单通道 tar** 读取。与第 2 节 `packwds:` 的差异在于：样本格式与解码分支不同（单 tar 内多为单通道 `tif`/`npy`，而非 `ch*.tif` + `meta.json` 的 packed 格式）。

| 前缀 | 适用场景 |
|------|----------|
| `wds:` | 本地路径上的 shard（可 `;` 拼接多段 pattern） |
| `s3wds:` | S3 上对象流式读（`pipe:aws s3 cp`），不落盘缓存 |
| `cachewds:` | 先下载到 `train.s3_cache_root`，再本地读 |

均需配合 `dinov3_vit7b16_pretrain_webdataset.yaml`（或 LoRA 配置），并设置 `train.aws_profile` / `train.aws_region`（S3 模式）。

### 4.1 本地 `wds:`

```bash
torchrun --nproc_per_node=4 dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_webdataset.yaml \
  --output-dir ./outputs/debug_ch1 \
  train.dataset_path="wds:/mnt/huawei_deepcad/webds_micro_100k_by_channel/ch1/train-{000000..000128}.tar" \
  student.fp8_enabled=false
```

### 4.2 S3 流式 `s3wds:`

```bash
torchrun --nproc_per_node=4 dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_webdataset.yaml \
  --output-dir ./outputs/debug_ch1 \
  train.dataset_path="s3wds:s3://xuzijing-biofm-ap-southeast-1-100k/webds_micro_100k_by_channel/ch1/train-{000000..000003}.tar" \
  train.aws_profile=sg \
  train.aws_region=ap-southeast-1 \
  student.fp8_enabled=false
```

### 4.3 S3 缓存 `cachewds:`

```bash
mkdir -p "${HOME}/.cache/dinov3_webds"

torchrun --nproc_per_node=4 dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_webdataset.yaml \
  --output-dir ./outputs/debug_ch1 \
  train.dataset_path="cachewds:s3://xuzijing-biofm-ap-southeast-1-100k/webds_micro_100k_by_channel/ch1/train-{000000..000003}.tar" \
  train.s3_cache_root="${HOME}/.cache/dinov3_webds" \
  train.aws_profile=sg \
  train.aws_region=ap-southeast-1 \
  student.fp8_enabled=false
```

`cachewds:` 与 `s3wds:` 的差别不仅是多几个参数：前者会把对象同步到本地再读；后者边训边流式拉取。

### 4.4 DINOv3 原生数据集（ImageNet，无前缀 WebDataset）

```bash
torchrun --nproc_per_node=8 dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
  train.dataset_path="ImageNet:split=TRAIN:root=/path/to/imagenet"
```
