# CSC (Cell Segmentation Challenge) 数据集评估

## 📋 概述

本模块实现了基于 **DINOv3** 的 CSC 细胞分割验证流水线，支持两种评估方式：

1. **Linear Probing**：冻结 DINOv3 主干网络，只训练一个 1x1 卷积层（线性分类器），通过对比 **mIoU** 评估预处理效果
2. **Zero-Shot PCA**：无需训练，直接对 DINOv3 特征进行 PCA 降维可视化，通过颜色分离度评估特征质量

**核心思路**：科学地确定哪种预处理对模型捕捉"生物细节"最有效。

## 🔬 三种预处理方法

| 实验组 | 方法 | 处理逻辑 | 预期表现 |
|--------|------|----------|----------|
| **A组 (minmax)** | 全局 Min-Max | 线性缩放到 0-1 | **一般** - 极亮噪点会将有效信号压缩 |
| **B组 (percentile)** | 百分位截断 | 截断 0.3% 和 99.7% 后归一化 | **中等** - 排除噪点但可能截断高亮区域 |
| **C组 (hybrid)** | 混合方法 | 只截断 99.9% 后 Min-Max | **最优** - 最大程度保留动态范围 |

## 📁 数据集结构

CSC 数据集支持多种图像格式（tif, tiff, png, bmp, jpg）和位深（8-bit, 16-bit）。

数据集目录结构：
```
56-CSC/
├── Training-labeled/
│   └── Training-labeled/
│       ├── images/          # 训练图像（1001 张，多种格式）
│       └── labels/          # 训练标签（1000 个，tiff 格式）
├── Tuning/
│   └── Tuning/
│       ├── images/          # 验证图像（101 张）
│       └── labels/          # 验证标签（101 个，tiff 格式）
└── Testing/
    └── Testing/
        ├── Public/          # 公开测试集（50-51 张图像 + 标签）
        │   ├── images/      # 测试图像
        │   └── labels/      # 测试标签
        └── Hidden/          # 隐藏测试集（400-401 张图像 + 标签）
            ├── images/      # 测试图像
            └── osilab_seg/  # 标签目录（在 osilab_seg/osilab_seg/ 下）
```

**注意**：实际匹配的图像-标签对数可能略少于文件数，因为可能存在格式不匹配的情况。

### 图像和标签匹配

不同分割的图像和标签命名规则：

**Training/Tuning**:
- 图像文件名：`cell_00001.bmp`, `cell_00002.png`, `cell_00003.tif` 等
- 标签文件名：`cell_00001_label.tiff`, `cell_00002_label.tiff` 等

**Testing Public**:
- 图像文件名：`OpenTest_001.png`, `OpenTest_002.tif` 等
- 标签文件名：`OpenTest_001_label.tiff`, `OpenTest_002_label.tiff` 等

**Testing Hidden**:
- 图像文件名：`TestHidden_001.tif`, `TestHidden_002.png` 等
- 标签文件名：`TestHidden_001_label.tiff`, `TestHidden_002_label.tiff` 等
- 标签位置：`Testing/Testing/Hidden/osilab_seg/osilab_seg/` 目录下

**匹配规则**：通过基础名称（去掉后缀和 `_label`）进行匹配

### 标签格式

- **格式**：TIFF 文件
- **类型**：Instance segmentation mask（uint16）
- **含义**：每个像素值为实例 ID（0=背景，1-N=不同的细胞实例）
- **处理**：代码会自动将 instance mask 转换为二值 mask（0=背景，1=前景）用于训练

## 📁 文件结构

```
dinov3/eval/csc/
├── __init__.py              # 模块初始化
├── data_utils.py            # 数据加载工具（支持多种格式）
├── linear_probe.py          # Linear Probing 核心代码
├── zero_shot_pca.py         # Zero-Shot PCA 可视化
└── README.md                # 本文档
```

## 📏 图像尺寸与通道说明

### 尺寸

CSC 图像尺寸**不固定**（例如 1920×2560 等）。

代码会自动将图像尺寸调整为 **patch_size (16)** 的倍数以适配 ViT。

### 通道结构

CSC 图像可以是：
- **单通道**：灰度图像
- **多通道**：RGB 或 RGBA 图像

代码会自动处理：
- 单通道 → 扩展为三通道（复制）
- RGBA → RGB（丢弃 Alpha 通道）
- BGR → RGB（cv2 读取的转换）

### 位深支持

- **8-bit**：uint8（0-255）
- **16-bit**：uint16（0-65535）

预处理函数会根据实际位深自动处理。

## 🚀 快速开始

### 1. 检查数据

确认数据集路径存在：
```bash
ls /mnt/deepcad_nfs/0-large-model-dataset/56-CSC/Training-labeled/Training-labeled/images/ | head
ls /mnt/deepcad_nfs/0-large-model-dataset/56-CSC/Training-labeled/Training-labeled/labels/ | head
```

### 2. Zero-Shot PCA 可视化（无需训练）

快速查看不同预处理方法的特征质量：

```bash
cd /mnt/huawei_deepcad/dinov3
python -m dinov3.eval.csc.zero_shot_pca \
    --checkpoint /mnt/deepcad_nfs/xuzijing/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth \
    --model-size 7b \
    --data-path /mnt/deepcad_nfs/0-large-model-dataset/56-CSC \
    --output-dir dinov3/outputs/csc_pca \
    --split test_public \
    --max-samples 10
```

**参数说明**：
- `--split`: `train`, `tune`, `test`/`test_public`, 或 `test_hidden`（使用哪个数据分割）
  - `test` 或 `test_public`: 使用 Public 测试集
  - `test_hidden`: 使用 Hidden 测试集
- `--max-samples`: 最大处理样本数（默认 20）
- `--kmeans-n-init`: K-Means 初始化次数（默认 10）
- `--kmeans-seed`: 随机种子（默认 0，用于复现）

### 3. Linear Probing 评估

训练线性分类器评估预处理效果：

```bash
conda activate dinov3 && CUDA_VISIBLE_DEVICES=7 python -m dinov3.eval.csc.linear_probe \
    --checkpoint /mnt/deepcad_nfs/xuzijing/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth \
    --model-size 7b \
    --data-path /mnt/deepcad_nfs/0-large-model-dataset/56-CSC \
    --output-dir dinov3/outputs/csc_linear_probe \
    --epochs 10 \
    --batch-size 4 \
    --modes hybrid
```
 percentile hybrid
**参数说明**：
- `--epochs`: 训练轮数（默认 10）
- `--batch-size`: 批次大小（ViT-L: 8, ViT-7B: 2）
- `--lr`: 学习率（默认 1e-3）
- `--modes`: 要测试的预处理模式（默认全部）
- `--use-multi-scale`: 使用多尺度特征（可选）

## ⚙️ 配置参数

### Zero-Shot PCA 参数

```bash
--data-path /mnt/deepcad_nfs/0-large-model-dataset/56-CSC  # 数据集根目录
--output-dir dinov3/outputs/csc_pca                        # 输出目录
--checkpoint <checkpoint_path>                             # DINOv3 checkpoint
--model-size l                                             # 'l' (ViT-L) 或 '7b' (ViT-7B)
--split train                                              # 'train', 'tune', 'test'/'test_public', 或 'test_hidden'
--max-samples 20                                           # 最大样本数
--modes minmax percentile hybrid                           # 预处理模式
--kmeans-n-init 10                                         # K-Means 初始化次数
--kmeans-seed 0                                            # 随机种子
```

### Linear Probing 参数

```bash
--data-path /mnt/deepcad_nfs/0-large-model-dataset/56-CSC  # 数据集根目录
--output-dir dinov3/outputs/csc_linear_probe               # 输出目录
--checkpoint <checkpoint_path>                             # DINOv3 checkpoint
--model-size l                                             # 'l' (ViT-L) 或 '7b' (ViT-7B)
--epochs 10                                                # 训练轮数
--batch-size 8                                             # 批次大小
--lr 1e-3                                                  # 学习率
--num-workers 4                                            # DataLoader workers
--modes minmax percentile hybrid                           # 预处理模式
--use-multi-scale                                          # 使用多尺度特征（可选）
```

## 📊 输出结果

### Zero-Shot PCA 输出

```
outputs/csc_pca/pca_20251225_120000/
├── config.json                 # 实验配置
├── kmeans_miou.json            # K-Means 无监督 mIoU 结果
├── comparison_000.png          # 三种预处理的对比图
├── comparison_001.png          # ...
├── minmax/
│   └── pca_000.png            # A组 PCA 结果
├── percentile/
│   └── pca_000.png            # B组 PCA 结果
└── hybrid/
    └── pca_000.png            # C组 PCA 结果
```

### Linear Probing 输出

```
outputs/csc_linear_probe/20251225_120000/
├── config.json                 # 实验配置
├── summary.json                # 汇总结果
├── minmax/
│   ├── history.json           # 训练历史
│   ├── best_model.pth         # 最佳模型
│   └── visualizations/        # 可视化结果
├── percentile/
│   └── ...
└── hybrid/
    └── ...
```

## 🔧 可用的 Checkpoint

| 模型 | 文件名 | embed_dim | 建议 batch_size |
|------|--------|-----------|-----------------|
| ViT-L | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | 1024 | 8 |
| ViT-7B | `dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth` | 4096 | 2 |

## 🧪 实验原理

### Linear Probing 工作流程

```
多格式图像 → 预处理 (A/B/C) → DINOv3 Backbone (冻结) → Patch 特征 → 1x1 Conv → 分割结果
```

### 预处理方法代码

```python
# A组: Min-Max
normalized = (img - img.min()) / (img.max() - img.min())

# B组: Percentile
p_low, p_high = np.percentile(img, [0.3, 99.7])
normalized = np.clip(img, p_low, p_high)
normalized = (normalized - p_low) / (p_high - p_low)

# C组: Hybrid
p_high = np.percentile(img, 99.9)
normalized = np.clip(img, 0, p_high) / p_high
```

### Zero-Shot PCA 可视化

1. 取 DINOv3 输出的**最后一层 Patch Tokens**
2. 对所有 patch 特征进行 **PCA 降维到 3 维**
3. 将这 3 维特征映射为 **RGB 图像**
4. 使用 **K-Means (K=2)** 进行无监督聚类，计算 mIoU

### K-Means 无监督 mIoU

- 对 patch 特征进行 K-Means 聚类（K=2，前景/背景）
- 尝试两种簇-前景映射，选择最大 IoU
- 对所有样本求平均得到 mIoU

## 📝 预期结果

基于 16-bit 图像的特性，预期：

```
Hybrid (C组) > Percentile (B组) > MinMax (A组)
```

**原因**：Hybrid 方法只剔除最极端的离群值，最大程度保留原始动态范围。

## 🔒 Backbone 冻结说明

Linear Probing 中，backbone 通过以下方式确保不参与反向传播：

```python
# 1. 冻结所有参数
for param in self.backbone.parameters():
    param.requires_grad = False

# 2. 设置为评估模式
self.backbone.eval()

# 3. 前向传播时使用 no_grad
with torch.no_grad():
    features = self.backbone.get_intermediate_layers(x, n=1)
```

## 📊 数据加载特性

### 自动格式检测

代码会自动检测并处理：
- **图像格式**：tif, tiff, png, bmp, jpg, jpeg
- **位深**：8-bit (uint8) 或 16-bit (uint16)
- **通道数**：单通道、RGB、RGBA

### 文件匹配

图像和标签通过基础名称匹配：
- `cell_00001.bmp` ↔ `cell_00001_label.tiff` ✓
- `cell_00002.png` ↔ `cell_00002_label.tiff` ✓

### Instance Mask 转换

CSC 标签是 instance segmentation mask，代码会自动：
1. 读取 TIFF 文件（uint16）
2. 转换为二值 mask：`(mask > 0).astype(int)`（0=背景，1=前景）

## ❓ 常见问题

**Q: 显存不足？**
A: 减小 batch_size 或使用 ViT-L 替代 ViT-7B

**Q: 只测试一种预处理？**
A: `--modes hybrid`

**Q: 使用多尺度特征？**
A: 添加 `--use-multi-scale` 标志

**Q: 如何快速对比不同预处理效果？**
A: 使用 Zero-Shot PCA，无需训练即可直观看到特征质量差异

**Q: 图像格式不支持？**
A: 代码支持 tif, tiff, png, bmp, jpg。如果遇到其他格式，可以转换为上述格式之一。

**Q: 标签文件找不到？**
A: 确保标签文件名格式为 `*_label.tiff` 或 `*_label.tif`，并且基础名称与图像文件匹配。

**Q: Testing 集的标签在哪里？**
A: 
- Public 测试集：标签在 `Testing/Testing/Public/labels/` 目录
- Hidden 测试集：标签在 `Testing/Testing/Hidden/osilab_seg/osilab_seg/` 目录

**Q: 如何使用 Testing 集进行评估？**
A: 使用 `--split test_public` 或 `--split test_hidden` 参数。注意：`test` 和 `test_public` 是等价的。

## 📚 参考

- [NeurIPS 2022 Cell Segmentation Challenge](https://neurips22-cellseg.grand-challenge.org/)
- [CSC Dataset README](../../../../56-CSC/ReadMe.md)（数据集原始说明文档）
- [Cellpose Evaluation](../cellpose/README.md)（相似的评估流程）

