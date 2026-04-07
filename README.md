# DINOv3（当前分支）快速使用说明

本文档包含：
- 环境配置
- 训练命令（单机多卡 / 多机多卡）
- 四种数据加载模式示例（ImageNet / 本地 WebDataset / S3 流式 / S3 缓存）

## 1) 环境配置

### 1.1 创建环境

优先使用项目里的 `conda.yaml`：

```bash
conda env create -f conda.yaml
conda activate dinov3
```

如果你使用的是 `micromamba`：

```bash
micromamba env create -f conda.yaml
micromamba activate dinov3
```


## 2) 训练命令

> 以下示例基于 WebDataset 输入：  
> `train.dataset_path="wds:/path/to/train-{000000..xxx}.tar"`
>
> `ChannelViT`（`student/teacher.in_chans`、`enable_channelvit`）与 LoRA 细节可直接命令行覆盖，无需额外 YAML。

### 2.1 单机多卡（例如 1 机 4 卡）

```bash
torchrun \
  --nproc_per_node=4 \
  dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
  --output-dir ./outputs/debug_ch1 \
  train.dataset_path="wds:/mnt/huawei_deepcad/webds_micro_100k_by_channel/ch1/train-{000000..000128}.tar" \
  student.fp8_enabled=false \
  student.resume_from_teacher_chkpt=/mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
```

说明：
- `--nproc_per_node=4` 表示单机启 4 个进程（通常对应 4 张 GPU）。
- `student.resume_from_teacher_chkpt` 可用于加载官方预训练权重初始化 backbone。
- `student.fp8_enabled=false` 表示关闭 FP8 路径，使用常规混合精度；若硬件与 PyTorch 支持 FP8 且你希望启用，可改为 `true`（需与配置一致）。
- `compute_precision.hsdp_shards=1` 表示纯 FSDP（1D mesh）。当 `hsdp_shards>1` 时启用 HSDP：每组 `hsdp_shards` 张卡内做 FSDP 分片，组间做 DDP 式同步；需满足 `world_size % hsdp_shards == 0`。

### 2.3 命令行开启 ChannelViT（无需 channelvit YAML）

```bash
torchrun --nproc_per_node=8 dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_webdataset.yaml \
  train.dataset_path="wds:/mnt/data/shards/ch1/train-{000000..000099}.tar" \
  student.in_chans=4 \
  teacher.in_chans=4 \
  student.enable_channelvit=true \
  teacher.enable_channelvit=true
```

### 2.4 LoRA + ChannelViT（无需 lora_channelvit YAML）

```bash
CUDA_VISIBLE_DEVICES=0,1,6,7 torchrun --nproc_per_node=4 dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_lora.yaml \
  train.dataset_path="wds:/mnt/huawei_deepcad/webds_micro_100k_by_channel/ch1/train-{000000..000128}.tar" \
  student.in_chans=3 \
  teacher.in_chans=3 \
  student.enable_channelvit=true \
  teacher.enable_channelvit=true
```

### 2.2 多机多卡（例如 2 机，每机 4 卡）

#### 节点 0（主节点，`node_rank=0`）

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --master_addr=10.0.0.1 \
  --master_port=29500 \
  dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
  --output-dir ./outputs/debug_ch1 \
  train.dataset_path="wds:/mnt/huawei_deepcad/webds_micro_100k_by_channel/ch1/train-{000000..000128}.tar" \
  train.compile=false \
  student.fp8_enabled=false \
  student.resume_from_teacher_chkpt=/mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
```

#### 节点 1（`node_rank=1`）

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=1 \
  --master_addr=10.0.0.1 \
  --master_port=29500 \
  dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
  --output-dir ./outputs/debug_ch1 \
  train.dataset_path="wds:/mnt/huawei_deepcad/webds_micro_100k_by_channel/ch1/train-{000000..000128}.tar" \
  train.compile=false \
  student.fp8_enabled=false \
  student.resume_from_teacher_chkpt=/mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
```

说明：
- 所有节点的 `--master_addr` / `--master_port` 必须一致。
- 每台机器只改 `--node_rank`（从 `0` 到 `nnodes-1`）。
- `world_size = nnodes * nproc_per_node`。

---

## 3) 四种数据加载模式（启动命令示例）

模式 2–4 共用同一份 WebDataset 配置 [`dinov3/configs/train/dinov3_vit7b16_pretrain_webdataset.yaml`](dinov3/configs/train/dinov3_vit7b16_pretrain_webdataset.yaml)，只需在命令行改 `train.dataset_path` 前缀；S3 时再带上 `train.aws_profile` / `train.aws_region`，缓存模式再加 `train.s3_cache_root`。

### 模式 1：DINOv3 原生读取（ImageNet 等）

```bash
torchrun --nproc_per_node=8 dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
  train.dataset_path="ImageNet:split=TRAIN:root=/path/to/imagenet"
```

### 模式 2：本地 WebDataset

```bash
torchrun --nproc_per_node=8 dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_webdataset.yaml \
  train.dataset_path="wds:/mnt/data/shards/ch1/train-{000000..000099}.tar"
```

### 模式 3：S3 流式 WebDataset

在**本仓库根目录**执行（需已配置 `aws` CLI 与 profile `sg`，并能访问该 bucket）：

```bash
torchrun --nproc_per_node=4 dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_webdataset.yaml \
  --output-dir ./outputs/debug_ch1 \
  train.dataset_path="s3wds:s3://xuzijing-biofm-ap-southeast-1-100k/webds_micro_100k_by_channel/ch1/train-{000000..000003}.tar" \
  train.aws_profile=sg \
  train.aws_region=ap-southeast-1 \
  student.fp8_enabled=false
```

### 模式 4：S3 缓存 + 本地 WebDataset

`train.s3_cache_root` 请指向**本机可写目录**。下面示例用 `$HOME/.cache/dinov3_webds`，可先 `mkdir` 再跑；同步 shard 时也会按需创建子目录。

在**本仓库根目录**执行：

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


## 4) 常见参数

- `train.dataset_path`：  
  以 `wds:` / `s3wds:` / `cachewds:` 切换 WebDataset 数据源（均使用 `dinov3_vit7b16_pretrain_webdataset.yaml`）；路由逻辑见 `dinov3/data/loaders.py` 中 `make_dataset`。
- `student.in_chans` / `teacher.in_chans`：  
  输入通道数（默认 3）。当前训练代码中，数据解码和 student/teacher backbone 构建都由 `student.in_chans` 驱动；建议命令行同时覆盖 `student.*` 与 `teacher.*` 以保持配置语义一致。
- `compute_precision.hsdp_shards`：  
  `1` 表示纯 FSDP；`>1` 启用 HSDP（需整除 `world_size`）。

## 5) 无 ChannelViT vs 有 ChannelViT（基于当前代码）

这一节按当前仓库实现总结，关键代码路径：
- 数据通道对齐：`dinov3/train/train.py` → `make_dataset(..., target_channels=cfg.student.in_chans)` → `dinov3/data/wds_decoder.py::_ensure_target_channels`
- 模型分支：`dinov3/models/vision_transformer.py` 中 `enable_channelvit` 开关（`PatchEmbed` vs `PatchEmbedPerChannel`）

### 5.1 不开启 ChannelViT（`student.enable_channelvit=false`）

- 走标准 `PatchEmbed`（`Conv2d(in_chans -> embed_dim)`），把输入当作普通多通道图像一次性投影。
- 对 WebDataset 输入，解码阶段会先把样本通道数对齐到 `student.in_chans`：
  - 原图 1 通道：复制到目标通道数；
  - 原图通道数 > 目标：截断前 `target_channels` 个通道；
  - 原图通道数 < 目标（且不为 1）：循环填充到目标通道数。
- 因此你记忆里的“少于 3 补齐、大于 3 截断”在**`student.in_chans=3`** 时成立；若你设 `student.in_chans=4`，规则会变成“对齐到 4”。

### 5.2 开启 ChannelViT（`student.enable_channelvit=true`）

- 走 `PatchEmbedPerChannel`：每个通道独立做 patch embedding，并叠加 `channel_embed`，再展平为 token 序列。
- RoPE 会按通道数复制对齐 token（代码里有 ChannelViT 专门分支）。
- 多通道输入是原生支持的；但输入通道数仍会先在解码阶段对齐到 `student.in_chans`，因此建议将其显式设为你的真实通道数（如 4、6、8）。

### 5.3 实操建议与验证

- 若你要“尽量保留全部通道信息”，请把 `student.in_chans` 设为数据真实通道数；否则在解码阶段就可能被截断/填充。
- 快速验证是否启用 ChannelViT：
  - 开启时日志应出现：`ChannelViT enabled with <N> channels`
  - 不开启时不会出现上述日志，模型走标准 `PatchEmbed`
- 快速验证通道对齐是否生效：临时用 1~2 个 shard 跑一个短任务，观察训练是否稳定起步；若 `student.in_chans` 设得不合理，常见现象是信息丢失（截断）或重复通道（填充）带来效果波动。

