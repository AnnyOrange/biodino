# Cellpose 16-bit 图像预处理验证实验

## 📋 概述

本模块实现了基于 **DINOv3** 的 16-bit Cellpose 细胞分割验证流水线，支持两种评估方式：

1. **Linear Probing**：冻结 DINOv3 主干网络，只训练一个 1x1 卷积层（线性分类器），通过对比 **mIoU** 评估预处理效果
2. **Zero-Shot PCA**：无需训练，直接对 DINOv3 特征进行 PCA 降维可视化，通过颜色分离度评估特征质量

**核心思路**：科学地确定哪种预处理对模型捕捉"生物细节"最有效。

## 🔬 三种预处理方法

| 实验组 | 方法 | 处理逻辑 | 预期表现 |
|--------|------|----------|----------|
| **A组 (minmax)** | 全局 Min-Max | 线性缩放 0-65535 到 0-1 | **最差** - 极亮噪点会将细胞区域压缩到接近 0 |
| **B组 (percentile)** | 百分位截断 | 截断 0.3% 和 99.7% 后归一化 | **中等** - 排除噪点但可能截断高亮细胞核纹理 |
| **C组 (hybrid)** | 混合方法 | 只截断 99.9% 后 Min-Max | **最优** - 最大程度保留 16-bit 动态范围 |

## 📁 文件结构

```
dinov3/eval/cellpose/
├── __init__.py              # 模块初始化
├── linear_probe.py          # Linear Probing 核心代码
├── zero_shot_pca.py         # Zero-Shot PCA 可视化
├── run_experiment.py        # 一键运行脚本
├── test_setup.py            # 环境检查脚本
├── visualize_results.py     # 结果可视化脚本
└── README.md                # 本文档
```

## 📏 图像尺寸与通道说明

### 尺寸

Cellpose 原始图像尺寸为 **512×383** (W×H)。

代码会自动将图像尺寸调整为 **patch_size (16)** 的倍数：
- 原始: 512×383
- 调整后: 512×368

**注意**：不再使用固定的 224×224 尺寸，以保留更多图像细节。

### 通道结构

Cellpose 是 **16-bit 三通道 RGB 图像**，每个通道含义不同：

| 通道 | 颜色 | 内容 | 典型值范围 |
|------|------|------|-----------|
| 通道0 | Red | 细胞核信号 | 0-65535 |
| 通道1 | Green | 细胞质信号 | 0-65535 |
| 通道2 | Blue | 通常为空 | 0 |

### 通道处理方式

**重要**：每个通道**独立进行预处理**，而不是合并为单通道：

```python
# 正确做法：对每个通道独立处理
for c in range(3):
    result[:, :, c] = apply_preprocessing_single_channel(img[:, :, c], mode)

# 错误做法：先合并再处理（会丢失通道间的差异）
# gray = img.mean(axis=2)  # ❌ 不应该这样做
```

## 🚀 快速开始

### 1. 检查环境

```bash
cd /mnt/huawei_deepcad/dinov3
python -m dinov3.eval.cellpose.test_setup
```

### 2. 运行实验

**方法 1：一键运行（推荐）**

修改 `run_experiment.py` 中的 `CONFIG` 参数，然后：

```bash
python -m dinov3.eval.cellpose.run_experiment
```

**方法 2：命令行参数**

```bash
python -m dinov3.eval.cellpose.linear_probe \
    --checkpoint /mnt/deepcad_nfs/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --model-size l \
    --data-path /mnt/deepcad_nfs/0-large-model-dataset/11-Cellpose \
    --output-dir dinov3/outputs/cellpose_linear_probe \
    --epochs 10 \
    --batch-size 8 \
    --modes minmax percentile hybrid
```

### 3. Zero-Shot PCA 可视化（无需训练）

```bash
python -m dinov3.eval.cellpose.zero_shot_pca \
    --checkpoint /mnt/deepcad_nfs/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --model-size l \
    --data-path /mnt/deepcad_nfs/0-large-model-dataset/11-Cellpose \
    --output-dir dinov3/outputs/cellpose_pca \
    --max-samples 20
```

### 4. 可视化 Linear Probing 结果

```bash
python -m dinov3.eval.cellpose.visualize_results dinov3/outputs/cellpose_linear_probe/l_20251225_120000
```

## ⚙️ 配置参数

在 `run_experiment.py` 中修改：

```python
CONFIG = {
    'data_path': '/mnt/deepcad_nfs/0-large-model-dataset/11-Cellpose',
    'output_dir': 'dinov3/outputs/cellpose_linear_probe',
    'checkpoint_dir': '/mnt/deepcad_nfs/xuzijing/checkpoints',
    'model_size': 'l',       # 'l' (ViT-L) 或 '7b' (ViT-7B)
    'epochs': 10,
    'batch_size': 8,         # ViT-L: 8, ViT-7B: 2
    'learning_rate': 1e-3,
    'use_multi_scale': False,
    'modes': ['minmax', 'percentile', 'hybrid'],
}
```

## 📊 输出结果

```
outputs/cellpose_linear_probe/l_20251225_120000/
├── config.json                 # 实验配置
├── summary.json                # 汇总结果
├── training_curves.png         # 训练曲线图 (可视化后生成)
├── comparison.png              # 对比柱状图 (可视化后生成)
├── report.txt                  # 实验报告 (可视化后生成)
├── minmax/
│   ├── history.json
│   ├── best_model.pth
│   └── visualizations/
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
16-bit 图像 → 预处理 (A/B/C) → DINOv3 Backbone (冻结) → Patch 特征 → 1x1 Conv → 分割结果
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

## 📝 预期结果

```
Hybrid (C组) > Percentile (B组) > MinMax (A组)
```

**原因**：16-bit 图像中的极亮噪点会严重影响 Min-Max 归一化，而 Hybrid 方法只剔除最极端的离群值，最大程度保留原始动态范围。

## 🎨 Zero-Shot PCA 可视化

### 原理

1. 取 DINOv3 输出的**最后一层 Patch Tokens**
2. 对所有 patch 特征进行 **PCA 降维到 3 维**
3. 将这 3 维特征映射为 **RGB 图像**

### 评估标准

- ✅ **成功**：细胞主体呈现一种颜色，背景呈现另一种颜色，边缘锐利
- ❌ **失败**：颜色混杂，细胞与背景难以区分

### 输出示例

```
outputs/cellpose_pca/pca_20251225_120000/
├── config.json              # 配置
├── comparison_000.png       # 三种预处理的对比图
├── minmax/
│   └── pca_000.png         # A组 PCA 结果
├── percentile/
│   └── pca_000.png         # B组 PCA 结果
└── hybrid/
    └── pca_000.png         # C组 PCA 结果
```

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

## ❓ 常见问题

**Q: 显存不足？**
A: 减小 batch_size 或使用 ViT-L 替代 ViT-7B

**Q: 只测试一种预处理？**
A: `--modes hybrid`

**Q: 使用多尺度特征？**
A: 添加 `--use-multi-scale` 或设置 `CONFIG['use_multi_scale'] = True`

**Q: 如何快速对比不同预处理效果？**
A: 使用 Zero-Shot PCA，无需训练即可直观看到特征质量差异

