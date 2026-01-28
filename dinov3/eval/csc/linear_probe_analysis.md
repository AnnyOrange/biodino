# CSC Linear Probe 第一个 Epoch 效果最好的原因分析

## 观察到的现象

从训练历史数据可以看到：

```
Epoch 1: test mIoU = 0.4413, train mIoU = 0.6781
Epoch 2: test mIoU = 0.5137, train mIoU = 0.6964  ← 最佳
Epoch 3: test mIoU = 0.4625, train mIoU = 0.7104  ← 开始下降
...
Epoch 10: test mIoU = 0.4963, train mIoU = 0.7276
```

**关键问题**：测试集 mIoU 在 Epoch 2 达到峰值后开始下降，而训练集 mIoU 持续上升。这是典型的**过拟合**现象。

## 主要原因

### 1. **训练集和测试集分布不匹配** ⚠️

当前代码：
```python
train_img_paths, train_mask_paths = get_dataset_paths(args.data_path, 'train')
test_img_paths, test_mask_paths = get_dataset_paths(args.data_path, 'tune')
```

- **训练集**：`Training-labeled` 集合（1001 张图像）
- **测试集**：`Tuning` 集合（101 张图像）

这两个集合可能来自：
- 不同的实验条件
- 不同的成像设备或参数
- 不同的样本来源

**解决方案**：应该使用同一个数据集进行训练/验证分割。

### 2. **数据增强导致的分布差异** ⚠️

```python
train_dataset = CSCLinearProbeDataset(..., augment=True)   # 使用 flip
test_dataset = CSCLinearProbeDataset(..., augment=False)   # 不使用增强
```

训练时使用数据增强（随机翻转），但测试时不使用，会导致：
- 训练时的数据分布与测试时不同
- 模型学习到的特征可能不适用于测试集

### 3. **学习率过大** ⚠️

```python
optimizer = torch.optim.AdamW(
    model.head.parameters(),
    lr=learning_rate,  # 默认 1e-3
    weight_decay=0.01
)
```

初始学习率 `1e-3` 对于线性分类头来说可能过大，导致：
- 模型快速过拟合到训练集
- 泛化能力差

### 4. **模型容量与数据量不匹配** ⚠️

- 训练集：1001 张图像
- 测试集：101 张图像
- 模型：基于 DINOv3-7B 的线性分类头（embed_dim=4096）

相对于数据量，模型可能过于复杂，容易过拟合。

## 建议的解决方案

### 方案 1：使用同一个数据集进行训练/验证分割（推荐）

```python
# 从 train 数据集中划分训练集和验证集
from sklearn.model_selection import train_test_split

all_img_paths, all_mask_paths = get_dataset_paths(args.data_path, 'train')
train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
    train_test_split(all_img_paths, all_mask_paths, test_size=0.2, random_state=42)
```

### 方案 2：降低学习率

```python
# 尝试更小的学习率
optimizer = torch.optim.AdamW(
    model.head.parameters(),
    lr=1e-4,  # 从 1e-3 降低到 1e-4
    weight_decay=0.01
)
```

### 方案 3：增加正则化

```python
optimizer = torch.optim.AdamW(
    model.head.parameters(),
    lr=1e-3,
    weight_decay=0.1  # 增加权重衰减
)

# 或者在训练时使用 Dropout
class DINOv3LinearSegmenter(nn.Module):
    def __init__(self, ...):
        ...
        self.head = nn.Sequential(
            nn.Conv2d(...),
            nn.Dropout(0.1),  # 添加 Dropout
        )
```

### 方案 4：使用早停（Early Stopping）

```python
# 在 run_experiment 中添加早停
patience = 3
best_miou = 0.0
patience_counter = 0

for epoch in range(1, epochs + 1):
    ...
    if test_metrics['mIoU'] > best_miou:
        best_miou = test_metrics['mIoU']
        patience_counter = 0
        # 保存模型
    else:
        patience_counter += 1
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
```

### 方案 5：减少数据增强的强度

```python
# 只使用轻微的数据增强
if self.augment:
    # 只在水平方向翻转，减少数据分布变化
    if np.random.rand() > 0.5:
        img = np.flip(img, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
    # 移除垂直翻转
```

## 当前代码的问题总结

1. ✅ **训练集和测试集来自不同数据源** → 分布不匹配
2. ✅ **数据增强只在训练时使用** → 训练/测试分布不一致
3. ✅ **学习率可能过大** → 快速过拟合
4. ✅ **没有早停机制** → 持续训练导致过拟合
5. ✅ **权重衰减可能不够** → 正则化不足

## 推荐的修改优先级

1. **高优先级**：使用同一个数据集进行训练/验证分割
2. **高优先级**：添加早停机制
3. **中优先级**：降低学习率（1e-3 → 1e-4）
4. **中优先级**：增加权重衰减（0.01 → 0.1）
5. **低优先级**：减少数据增强的强度

