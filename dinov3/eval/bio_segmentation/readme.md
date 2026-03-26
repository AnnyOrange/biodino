# Bio-Segmentation Evaluation Pipeline

评估流程分三个阶段：**数据解压 → 特征预提取（Linear Probe 专用）→ 运行评估**。

---

## 架构说明

| 评估方法 | 特征提取策略 | 官方对应 Config |
|---------|------------|----------------|
| Linear Probe | `backbone_out_layers: LAST`（最后 1 层，n=1） | `config-ade20k-linear-training.yaml` |
| Mask2Former | `backbone_out_layers: FOUR_EVEN_INTERVALS`（4 层均匀间隔，由 `DINOv3_Adapter` 内部处理，**不走 cache**） | `config-ade20k-m2f-inference.yaml` |

Mask2Former 在训练时在线提取特征，**不需要先跑 feature_extractor.py**。

---

## 各数据集默认 img_size

| 数据集 | 默认 img_size | 说明 |
|--------|:-----------:|------|
| CoNIC | 256 | 原生 patch 256×256，无需 resize |
| PanNuke | 256 | 原生 patch 256×256，无需 resize |
| TissueNet | 256 | 荧光 patch，原生较小 |
| LIVECell | 512 | TIF 图像 ~520×696，对齐官方 crop_size=512 |
| MoNuSeg | 512 | 原图 1000×1000，Linear Probe 用 512；M2F 用 slide 推理 |
| BBBC038 | 512 | 原图尺寸不一，统一为 512 |

---

## Step 0：数据解压

```bash
# BBBC038 (DSB2018)
python -m dinov3.eval.bio_segmentation.scripts.extract_datasets \
    --src-dir /data1/xuzijing/dataset/BBBC038 \
    --dst-dir /data1/xuzijing/dataset \
    --datasets bbbc038 \
    --overwrite

# CoNIC
python -m dinov3.eval.bio_segmentation.scripts.extract_datasets \
    --src-dir /data1/xuzijing/dataset/CoNIC \
    --dst-dir /data1/xuzijing/dataset \
    --datasets conic

# LIVECell — images/ is already extracted inside LIVECell_dataset_2021/
# The script will detect the existing images/ and print the correct --data-root.
# Point --src-dir to the inner folder so the script finds images.zip next to it.
python -m dinov3.eval.bio_segmentation.scripts.extract_datasets \
    --src-dir /data1/xuzijing/dataset/LIVECell/LIVECell_dataset_2021 \
    --dst-dir /data1/xuzijing/dataset \
    --datasets livecell

# MoNuSeg
python -m dinov3.eval.bio_segmentation.scripts.extract_datasets \
    --src-dir /data1/xuzijing/dataset/monuseg \
    --dst-dir /data1/xuzijing/dataset \
    --datasets monuseg

# PanNuke
python -m dinov3.eval.bio_segmentation.scripts.extract_datasets \
    --src-dir /data1/xuzijing/dataset/PanNuke \
    --dst-dir /data1/xuzijing/dataset \
    --datasets pannuke

# TissueNet
python -m dinov3.eval.bio_segmentation.scripts.extract_datasets \
    --src-dir /data1/xuzijing/dataset/TissueNet \
    --dst-dir /data1/xuzijing/dataset \
    --datasets tissuenet
```

---

## Step 1：特征预提取（Linear Probe 专用）

使用 `backbone_out_layers: LAST`（最后 1 层）对齐官方 linear segmentation 配置。

生成的缓存文件命名格式：`{dataset}_{split}_{model}_{layers}_{size}.npz`

### ViT-L（推荐优先跑）

```bash
CKPT_L=/data1/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

for DATASET in bbbc038 conic livecell monuseg pannuke tissuenet; do
    for SPLIT in train val test; do
        python -m dinov3.eval.bio_segmentation.feature_extractor \
            --dataset    $DATASET \
            --data-root  /data1/xuzijing/dataset/${DATASET}/extracted \
            --checkpoint $CKPT_L \
            --output-dir ./cache/${DATASET} \
            --model-size l \
            --split      $SPLIT
            # 默认: --layers last1, --img-size 自动选取每数据集规范尺寸
    done
done
```

### ViT-7B

```bash
CKPT_7B=/data1/xuzijing/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth

for DATASET in bbbc038 conic livecell monuseg pannuke tissuenet; do
    for SPLIT in train val test; do
        python -m dinov3.eval.bio_segmentation.feature_extractor \
            --dataset    $DATASET \
            --data-root  /data1/xuzijing/dataset/${DATASET}/extracted \
            --checkpoint $CKPT_7B \
            --output-dir ./cache/${DATASET} \
            --model-size 7b \
            --split      $SPLIT \
            --batch-size 4   # 7B 显存大，减小 batch
    done
done
```
# LIVECell: data-root points to the outer LIVECell/ folder
# (NOT .../livecell/extracted — LIVECell uses its own nested directory layout)
for SPLIT in train val test; do
    python -m dinov3.eval.bio_segmentation.feature_extractor \
        --dataset    livecell \
        --data-root  /data1/xuzijing/dataset/LIVECell \
        --checkpoint $CKPT_7B \
        --output-dir ./cache/livecell \
        --model-size 7b \
        --split      $SPLIT \
        --batch-size 4
done

> **缓存文件示例**（MoNuSeg, ViT-L）:
> ```
> ./cache/monuseg/monuseg_train_l_last1_s512.npz
> ./cache/monuseg/monuseg_val_l_last1_s512.npz
> ./cache/monuseg/monuseg_test_l_last1_s512.npz
> ```

---

## Step 2a：Linear Probe（使用缓存特征）

所有数据集统一格式，替换 `DATASET` 和对应 cache 路径即可。

### MoNuSeg

```bash
python -m dinov3.eval.bio_segmentation.linear_probe \
    --dataset    monuseg \
    --use-cached-features \
    --train-cache ./cache/monuseg/monuseg_train_l_last1_s512.npz \
    --val-cache   ./cache/monuseg/monuseg_val_l_last1_s512.npz \
    --test-cache  ./cache/monuseg/monuseg_test_l_last1_s512.npz \
    --output-dir  ./outputs/linear_probe/monuseg_vitl \
    --epochs 50 --batch-size 64 --lr 1e-3
```

### BBBC038

```bash
python -m dinov3.eval.bio_segmentation.linear_probe \
    --dataset    bbbc038 \
    --use-cached-features \
    --train-cache ./cache/bbbc038/bbbc038_train_l_last1_s512.npz \
    --val-cache   ./cache/bbbc038/bbbc038_val_l_last1_s512.npz \
    --test-cache  ./cache/bbbc038/bbbc038_test_l_last1_s512.npz \
    --output-dir  ./outputs/linear_probe/bbbc038_vitl \
    --epochs 50 --batch-size 64 --lr 1e-3
```

### CoNIC（7类，img_size=256）

```bash
python -m dinov3.eval.bio_segmentation.linear_probe \
    --dataset    conic \
    --use-cached-features \
    --train-cache ./cache/conic/conic_train_l_last1_s256.npz \
    --val-cache   ./cache/conic/conic_val_l_last1_s256.npz \
    --test-cache  ./cache/conic/conic_test_l_last1_s256.npz \
    --output-dir  ./outputs/linear_probe/conic_vitl \
    --epochs 50 --batch-size 64 --lr 1e-3
```

### LIVECell

```bash
python -m dinov3.eval.bio_segmentation.linear_probe \
    --dataset    livecell \
    --use-cached-features \
    --train-cache ./cache/livecell/livecell_train_l_last1_s512.npz \
    --val-cache   ./cache/livecell/livecell_val_l_last1_s512.npz \
    --test-cache  ./cache/livecell/livecell_test_l_last1_s512.npz \
    --output-dir  ./outputs/linear_probe/livecell_vitl \
    --epochs 50 --batch-size 64 --lr 1e-3
```

### PanNuke（6类，img_size=256）

```bash
python -m dinov3.eval.bio_segmentation.linear_probe \
    --dataset    pannuke \
    --use-cached-features \
    --train-cache ./cache/pannuke/pannuke_train_l_last1_s256.npz \
    --val-cache   ./cache/pannuke/pannuke_val_l_last1_s256.npz \
    --test-cache  ./cache/pannuke/pannuke_test_l_last1_s256.npz \
    --output-dir  ./outputs/linear_probe/pannuke_vitl \
    --epochs 50 --batch-size 64 --lr 1e-3
```

### TissueNet

```bash
python -m dinov3.eval.bio_segmentation.linear_probe \
    --dataset    tissuenet \
    --use-cached-features \
    --train-cache ./cache/tissuenet/tissuenet_train_l_last1_s256.npz \
    --val-cache   ./cache/tissuenet/tissuenet_val_l_last1_s256.npz \
    --test-cache  ./cache/tissuenet/tissuenet_test_l_last1_s256.npz \
    --output-dir  ./outputs/linear_probe/tissuenet_vitl \
    --epochs 50 --batch-size 64 --lr 1e-3
```

> 对 ViT-7B，替换 cache 路径中的 `_l_` 为 `_7b_` 即可。

---

## Step 2b：Mask2Former（在线训练，不需要 cache）

Mask2Former 内部通过 `DINOv3_Adapter` 使用 `FOUR_EVEN_INTERVALS`（4 层均匀间隔），
推理时对大图采用滑窗（`--inference-mode slide`），对小 patch 使用全图推理（`whole`）。

### MoNuSeg（大图 1000×1000 → slide 推理）

```bash
CUDA_VISIBLE_DEVICES=0 python -m dinov3.eval.bio_segmentation.mask2former \
    --dataset    monuseg \
    --data-root  /data1/xuzijing/dataset/monuseg/extracted \
    --checkpoint /data1/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --output-dir ./outputs/mask2former/monuseg_vitl \
    --model-size l \
    --inference-mode slide \
    --crop-size  512 \
    --stride     341 \
    --epochs 50 --batch-size 2 --lr 1e-4 --adapter-lr 1e-5
```

### BBBC038（大图 → slide 推理）

```bash
CUDA_VISIBLE_DEVICES=0 python -m dinov3.eval.bio_segmentation.mask2former \
    --dataset    bbbc038 \
    --data-root  /data1/xuzijing/dataset/bbbc038/extracted \
    --checkpoint /data1/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --output-dir ./outputs/mask2former/bbbc038_vitl \
    --model-size l \
    --inference-mode slide \
    --crop-size  512 --stride 341 \
    --epochs 50 --batch-size 2 --lr 1e-4 --adapter-lr 1e-5
```

### LIVECell

```bash
CUDA_VISIBLE_DEVICES=0 python -m dinov3.eval.bio_segmentation.mask2former \
    --dataset    livecell \
    --data-root  /data1/xuzijing/dataset/LIVECell \
    --checkpoint /data1/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --output-dir ./outputs/mask2former/livecell_vitl \
    --model-size l \
    --inference-mode whole \
    --crop-size  512 \
    --epochs 50 --batch-size 2 --lr 1e-4 --adapter-lr 1e-5
```

### CoNIC（小 patch 256×256 → whole 推理）

```bash
CUDA_VISIBLE_DEVICES=0 python -m dinov3.eval.bio_segmentation.mask2former \
    --dataset    conic \
    --data-root  /data1/xuzijing/dataset/conic/extracted \
    --checkpoint /data1/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --output-dir ./outputs/mask2former/conic_vitl \
    --model-size l \
    --inference-mode whole \
    --crop-size  256 \
    --epochs 50 --batch-size 4 --lr 1e-4 --adapter-lr 1e-5
```

### PanNuke（小 patch 256×256 → whole 推理）

```bash
CUDA_VISIBLE_DEVICES=0 python -m dinov3.eval.bio_segmentation.mask2former \
    --dataset    pannuke \
    --data-root  /data1/xuzijing/dataset/pannuke/extracted \
    --checkpoint /data1/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --output-dir ./outputs/mask2former/pannuke_vitl \
    --model-size l \
    --inference-mode whole \
    --crop-size  256 \
    --epochs 50 --batch-size 4 --lr 1e-4 --adapter-lr 1e-5
```

### TissueNet

```bash
CUDA_VISIBLE_DEVICES=0 python -m dinov3.eval.bio_segmentation.mask2former \
    --dataset    tissuenet \
    --data-root  /data1/xuzijing/dataset/tissuenet/extracted \
    --checkpoint /data1/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --output-dir ./outputs/mask2former/tissuenet_vitl \
    --model-size l \
    --inference-mode whole \
    --crop-size  256 \
    --epochs 50 --batch-size 4 --lr 1e-4 --adapter-lr 1e-5
```

---

## 指标说明

| 类型 | 指标 | 含义 |
|------|------|------|
| Semantic | mIoU | 每类 Intersection-over-Union 均值 |
| Semantic | mDice | 每类 Dice 系数均值 |
| Semantic | mPrecision / mRecall | 像素级精确率/召回率均值 |
| Instance | AJI | Aggregated Jaccard Index（细胞计数偏差敏感） |
| Instance | AP50 | IoU@0.5 的平均精确率 |
| Instance | AP (COCO) | COCO-style AP（IoU=0.5:0.95） |
| Instance | bPQ | 二值 Panoptic Quality |
| Instance | mPQ | 多类 Panoptic Quality（CoNIC/PanNuke） |

---

## 注意事项

- MoNuSeg / BBBC038 没有官方 val split → 自动从 train 随机抽取（MoNuSeg 20%，BBBC038 15%），索引保存到 `{data_root}/monuseg_val_indices.npy`
- PanNuke：Fold1+2 = train，Fold3 = val = test（3-fold 交叉验证标准做法）
- CoNIC：自动 80/10/10 随机划分，索引保存到 `conic_split_indices.npz`
- LIVECell / TissueNet：官方提供 train/val/test split
