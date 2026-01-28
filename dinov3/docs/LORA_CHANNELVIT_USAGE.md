# LoRA + ChannelViT 联合使用指南

本指南说明如何同时使用 LoRA 和 ChannelViT 功能进行高效的多通道图像微调。

## 概述

- **LoRA (Low-Rank Adaptation)**: 只训练少量参数，大幅降低内存和计算需求
- **ChannelViT**: 支持多通道输入（如生物图像），每个通道独立处理并共享空间位置编码

两者可以完美结合，实现高效的多通道图像微调。

## 配置文件设置

### 1. 使用预配置的联合配置文件

我们提供了一个预配置的配置文件：
```yaml
configs/train/dinov3_vit7b16_pretrain_lora_channelvit.yaml
```

### 2. 手动配置

在你的配置文件中同时设置：

```yaml
MODEL:
  META_ARCHITECTURE: SSLMetaArchLoRA  # 使用 LoRA 架构

# LoRA 配置
lora:
  r: 16                    # LoRA 秩
  alpha: 32.0              # 缩放因子（通常为 2*r）
  dropout: 0.05            # LoRA dropout
  target_modules: null      # null = 默认目标（qkv, proj, mlp）
  enable_q: true           # 对 Q 应用 LoRA
  enable_k: true           # 对 K 应用 LoRA
  enable_v: true           # 对 V 应用 LoRA
  train_bias: false        # 是否训练 bias
  train_heads: true        # 是否训练 DINO/IBOT heads

# ChannelViT 配置（在 student 和 teacher 中）
student:
  in_chans: 4              # 通道数（例如 4 个生物通道）
  enable_channelvit: true  # 启用 ChannelViT

teacher:
  in_chans: 4              # 必须与 student.in_chans 匹配
  enable_channelvit: true  # 必须与 student.enable_channelvit 匹配
```

## 使用方法

### 运行训练

```bash
torchrun --nproc_per_node=8 -m dinov3.train.train \
    --config-file configs/train/dinov3_vit7b16_pretrain_lora_channelvit.yaml \
    --output-dir ./outputs/lora_channelvit_finetune \
    train.dataset_path=/path/to/your/multi_channel/dataset \
    student.resume_from_teacher_chkpt=/path/to/pretrained/checkpoint
```

### 关键参数说明

1. **`lora.r`**: LoRA 秩，控制 LoRA 矩阵的维度
   - 较小值（8-16）：更少的参数，更快的训练
   - 较大值（32-64）：更多的容量，可能更好的性能

2. **`lora.alpha`**: LoRA 缩放因子
   - 通常设为 `2 * r`
   - 控制 LoRA 更新的强度

3. **`student.in_chans`**: 输入通道数
   - 标准 RGB: `3`
   - 生物图像: `4` 或更多
   - 必须与你的数据集通道数匹配

4. **`train.batch_size_per_gpu`**: 批次大小
   - 多通道图像 + LoRA 通常需要较小的批次
   - 建议从 `4` 开始，根据 GPU 内存调整

## 工作原理

### LoRA 应用流程

1. **模型初始化**: 使用 `enable_channelvit=True` 创建 ChannelViT 模型
2. **加载预训练权重**: 从 checkpoint 加载权重
3. **应用 LoRA**: 
   - 将 Linear 层包装为 LoRALinear
   - 冻结原始权重
   - 只训练 LoRA 参数（A 和 B 矩阵）
4. **ChannelViT 特殊处理**:
   - `channel_embed` 参数自动设置为可训练
   - 确保 ChannelViT 功能正常工作

### 可训练参数

使用 LoRA + ChannelViT 时，以下参数是可训练的：

1. **LoRA 参数**: `lora_A` 和 `lora_B` 矩阵
2. **Channel Embedding**: ChannelViT 的 `channel_embed` 参数
3. **DINO/IBOT Heads**: 如果 `lora.train_heads=true`（推荐）
4. **Bias**: 如果 `lora.train_bias=true`

其他所有参数（包括原始 backbone 权重）都被冻结。

## 内存和计算优势

### 参数对比

假设使用 ViT-7B (约 7B 参数)：

- **全量微调**: ~7B 可训练参数
- **LoRA (r=16)**: ~50M 可训练参数（约 0.7%）
- **LoRA + ChannelViT**: ~50M + channel_embed（约 0.7%）

### 内存节省

- **训练内存**: 减少约 70-80%（取决于批次大小）
- **梯度内存**: 大幅减少（只计算 LoRA 参数的梯度）
- **优化器状态**: 只保存 LoRA 参数的优化器状态

## 注意事项

1. **通道数匹配**: `student.in_chans` 和 `teacher.in_chans` 必须匹配
2. **ChannelViT 标志**: `student.enable_channelvit` 和 `teacher.enable_channelvit` 必须匹配
3. **批次大小**: 多通道图像需要更多内存，可能需要减小批次大小
4. **预训练权重**: 如果从标准 RGB 预训练模型开始，ChannelViT 的 `channel_embed` 会随机初始化
5. **Mask 处理**: 如果使用 mask，确保 mask 的维度与 ChannelViT 的 token 序列匹配

## 保存和加载

### 保存 LoRA 权重

```python
# 在训练脚本中
model.save_lora_weights("lora_checkpoint.pth")
```

### 加载 LoRA 权重

```python
# 加载时
model.load_lora_weights("lora_checkpoint.pth", strict=False)
```

注意：LoRA checkpoint 只包含 LoRA 参数和可选的额外状态，不包含完整的模型权重。

## 故障排除

### 问题 1: 内存不足

**解决方案**:
- 减小 `batch_size_per_gpu`
- 减小 `lora.r`
- 使用梯度累积

### 问题 2: ChannelViT 不工作

**检查**:
- `enable_channelvit` 是否在 student 和 teacher 中都设置为 `true`
- `in_chans` 是否匹配
- 输入数据是否为多通道格式 `(B, C, H, W)`

### 问题 3: LoRA 没有应用

**检查**:
- `META_ARCHITECTURE` 是否为 `SSLMetaArchLoRA`
- LoRA 配置是否正确
- 查看日志中的 "Applied LoRA to:" 消息

## 示例工作流

### 完整训练流程

```bash
# 1. 准备多通道数据集
# 确保数据格式为 (B, C, H, W)，其中 C > 3

# 2. 运行 LoRA + ChannelViT 训练
torchrun --nproc_per_node=8 -m dinov3.train.train \
    --config-file configs/train/dinov3_vit7b16_pretrain_lora_channelvit.yaml \
    --output-dir ./outputs/lora_channelvit \
    train.dataset_path=/path/to/dataset \
    student.resume_from_teacher_chkpt=/path/to/pretrained/dinov3_vit7b.pth \
    student.in_chans=4 \
    student.enable_channelvit=true \
    teacher.in_chans=4 \
    teacher.enable_channelvit=true

# 3. 检查训练日志
# 应该看到：
# - "ChannelViT enabled with 4 channels"
# - "Applying LoRA with r=16, alpha=32.0"
# - "ChannelViT channel_embed is set to trainable"
# - "Total LoRA layers added: XXX"
```

## 性能建议

1. **LoRA 秩选择**:
   - 小数据集: `r=8-16`
   - 中等数据集: `r=16-32`
   - 大数据集: `r=32-64`

2. **学习率**:
   - LoRA 通常需要较高的学习率（因为参数少）
   - 建议从 `1e-4` 开始

3. **训练轮数**:
   - LoRA 通常需要更少的训练轮数
   - 建议从 50-100 轮开始

4. **批次大小**:
   - 多通道图像: 从 `batch_size_per_gpu=4` 开始
   - 根据 GPU 内存调整

## 总结

LoRA + ChannelViT 的组合提供了：
- ✅ 高效的多通道图像处理
- ✅ 大幅降低的内存需求
- ✅ 更快的训练速度
- ✅ 灵活的参数控制

这使得在资源受限的环境下进行多通道图像微调成为可能。

