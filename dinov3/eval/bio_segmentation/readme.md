# Bio-Segmentation Evaluation Pipeline

评估流程分三个阶段：**数据解压 → 特征预提取（Linear Probe 专用）→ 运行评估**。

`feature_extractor` / `linear_probe`（online）/ `zero_shot_pca` 通过 **`load_dinov3_backbone`** 调用 **`build_model_for_eval`**：用 **`--train-config`** 指向的训练 YAML（与 `ssl_default_config` 合并）建出与训练一致的 **teacher backbone**，再加载 **`--checkpoint`**。

---

## Backbone 与 `--checkpoint`（`model_utils` + `build_model_for_eval`）

- **`--train-config`（必填，上述脚本）**：与训练该权重时相同的结构配置（尤其 **`student.arch`、`patch_size`、`n_storage_tokens`、`mask_k_bias`、`in_chans`、`enable_channelvit`、`ffn_layer` / `ffn_ratio`** 等）。**不要**再用 `--model-size` 字符串猜结构。
- **`--checkpoint`** 推荐优先使用：
  1. **训练 DCP 目录**：`.../ckpt/<iter>/`（与 `build_model_for_eval` 的 DCP 路径一致；多卡加载时需 **`torchrun`**，与训练相同），或  
  2. **合并权重、带 `teacher` 键**：例如导出得到的 **`teacher_checkpoint_trainstyle.pth`**（`init_model_from_checkpoint_for_evals(..., "teacher")`）。
- 亦支持 **`{"model": ...}`** / **`{"state_dict": ...}`** 或顶层即为 state dict 的 `.pth`；脚本会按文件内容自动选择顶层键。
- **DCP 加载**依赖分布式与 FSDP 包装；单机单卡跑 **合并 `.pth`** 时无需 DCP。

### 可选：从训练 checkpoint 导出合并 `.pth`（辅助）

若需要离线拷贝、或在不启动 DCP 的环境下使用，仍可用 **`export_backbone_from_train_ckpt`** 导出（**`torchrun`**，config 与训练一致）：

```bash
CUDA_VISIBLE_DEVICES=5 python dinov3/eval/bio_segmentation/export_backbone_from_train_ckpt.py \
  --config-file dinov3/configs/train/microscopy_continual_vitl16.yaml \
  --output-dir /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vitL16_2xa100_OEP1025_ep15_b1024_ckpt_1025 \
  --ckpt-iter 15374 \
  student.in_chans=3 \
  teacher.in_chans=3 \
  student.enable_channelvit=false \
  teacher.enable_channelvit=false
```

- **`--ckpt-iter`**：`latest` 或迭代号（如 `299`）。
- **`--export-student`**：额外写出 `student_backbone_evalstyle.pth`。
- 输出：**`<output_dir>/eval/export_<iter>/`**，其中 **`teacher_checkpoint_trainstyle.pth`** 最贴近训练侧，适合作为 **`--checkpoint`**；**`teacher_backbone_evalstyle.pth`** 仅含 backbone，亦可使用。

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

生成的缓存文件命名格式：`{dataset}_{split}_{train_config_stem}_{layers}_{size}.npz`  
其中 **`{train_config_stem}`** 为 **`--train-config`** 文件名的 stem（例如 `microscopy_continual_vitl16`）。

### ViT-L（推荐优先跑）

```bash
CKPT_L=/mnt/huawei_deepcad/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
CKPT_L=/mnt/huawei_deepcad/dinov3/outputs/bio_continue_vitL16_2xa100_OEP1025_ep15_b1024_ckpt_1025/eval/export_15374/teacher_backbone_evalstyle.pth
CKPT_L=/mnt/huawei_deepcad/dinov3/outputs/bio_continue_vitL16_OEP1025_ep15_b1024_1025/ckpt/2049/checkpoint.pth
TRAIN_CFG=dinov3/configs/train/microscopy_continual_vitl16.yaml
TRAIN_CFG=dinov3/configs/train/microscopy_continual_vitb16.yaml
CKPT_B=/mnt/huawei_deepcad/dinov3/outputs/bio_continue_1025_a100_grad_acc_2_base/ckpt/3074/checkpoint.pth
CKPT_B=/mnt/huawei_deepcad/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
conda activate dinov3
cd /mnt/huawei_deepcad/dinov3
monuseg pannuke tissuenet
for DATASET in bbbc038 conic; do
    for SPLIT in train val test; do
        CUDA_VISIBLE_DEVICES=2 python -m dinov3.eval.bio_segmentation.feature_extractor \
            --dataset    $DATASET \
            --data-root  /mnt/huawei_deepcad/benchmark/segmentation/${DATASET}/extracted \
            --checkpoint $CKPT_L \
            --train-config $TRAIN_CFG \
            --output-dir ./cache/${DATASET}/base_L_new \
            --split      $SPLIT
            # 默认: --layers last1, --img-size 自动选取每数据集规范尺寸
    done
done
```

其它规模：换 **`$CKPT_*`** 与 **`$TRAIN_CFG`**（及 yaml 内 **`student.arch` 等**）与真实权重一致即可。

### ViT-7B

```bash
CKPT_7B=/data1/xuzijing/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
TRAIN_CFG=dinov3/configs/train/dinov3_vit7b16_pretrain.yaml

for DATASET in bbbc038 conic livecell monuseg pannuke tissuenet; do
    for SPLIT in train val test; do
        python -m dinov3.eval.bio_segmentation.feature_extractor \
            --dataset    $DATASET \
            --data-root  /data1/xuzijing/dataset/${DATASET}/extracted \
            --checkpoint $CKPT_7B \
            --train-config $TRAIN_CFG \
            --output-dir ./cache/${DATASET} \
            --split      $SPLIT \
            --batch-size 4   # 7B 显存大，减小 batch
    done
done
```

### ViT-7B：LIVECell 单独示例

LIVECell 的 **`--data-root`** 指向外层 **`LIVECell/`**（不是 `.../livecell/extracted`，目录布局与其它数据集不同）。

```bash
CKPT_7B=/data1/xuzijing/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
TRAIN_CFG=dinov3/configs/train/dinov3_vit7b16_pretrain.yaml

for SPLIT in train val test; do
    python -m dinov3.eval.bio_segmentation.feature_extractor \
        --dataset    livecell \
        --data-root  /data1/xuzijing/dataset/LIVECell \
        --checkpoint $CKPT_7B \
        --train-config $TRAIN_CFG \
        --output-dir ./cache/livecell \
        --split      $SPLIT \
        --batch-size 4
done
```

> **缓存文件示例**（MoNuSeg，`TRAIN_CFG=microscopy_continual_vitl16.yaml`）:
> ```
> ./cache/monuseg/monuseg_train_microscopy_continual_vitl16_last1_s512.npz
> ./cache/monuseg/monuseg_val_microscopy_continual_vitl16_last1_s512.npz
> ./cache/monuseg/monuseg_test_microscopy_continual_vitl16_last1_s512.npz
> ```

---

## Step 2a：Linear Probe（使用缓存特征）

所有数据集统一格式，替换 `DATASET` 和对应 cache 路径即可。缓存文件名中的 **config stem** 段须与生成缓存时 **`feature_extractor` 的 `--train-config`** 一致；**online 模式**下需 **`--train-config`**、**`--checkpoint`**（DCP 目录、`teacher_checkpoint_trainstyle.pth` 或兼容的 `.pth`）。
monuseg pannuke tissuenet bbbc038 conic
for DATASET in tissuenet conic; do
    CUDA_VISIBLE_DEVICES=7 python -m dinov3.eval.bio_segmentation.linear_probe \
    --dataset    $DATASET \
    --use-cached-features \
    --train-cache ./cache/$DATASET/base_s/$DATASET\_train_microscopy_continual_vits16_last1_s256.npz \
    --val-cache   ./cache/$DATASET/base_s/$DATASET\_val_microscopy_continual_vits16_last1_s256.npz \
    --test-cache  ./cache/$DATASET/base_s/$DATASET\_test_microscopy_continual_vits16_last1_s256.npz \
    --output-dir  ./outputs/linear_probe/$DATASET\_vits \
    --epochs 50 --batch-size 64 --lr 1e-3
done

### MoNuSeg

```bash
CUDA_VISIBLE_DEVICES=6 python -m dinov3.eval.bio_segmentation.linear_probe \
    --dataset    monuseg \
    --use-cached-features \
    --train-cache ./cache/monuseg/monuseg_train_microscopy_continual_vitl16_last1_s512.npz \
    --val-cache   ./cache/monuseg/monuseg_val_microscopy_continual_vitl16_last1_s512.npz \
    --test-cache  ./cache/monuseg/monuseg_test_microscopy_continual_vitl16_last1_s512.npz \
    --output-dir  ./outputs/linear_probe/monuseg_vitl \
    --epochs 50 --batch-size 64 --lr 1e-3
```

### BBBC038

```bash
python -m dinov3.eval.bio_segmentation.linear_probe \
    --dataset    bbbc038 \
    --use-cached-features \
    --train-cache ./cache/bbbc038/base_l/bbbc038_train_microscopy_continual_vitl16_last1_s512.npz \
    --val-cache   ./cache/bbbc038/base_l/bbbc038_val_microscopy_continual_vitl16_last1_s512.npz \
    --test-cache  ./cache/bbbc038/base_l/bbbc038_test_microscopy_continual_vitl16_last1_s512.npz \
    --output-dir  ./outputs/linear_probe/bbbc038_vitl \
    --epochs 50 --batch-size 64 --lr 1e-3
```

### CoNIC（7类，img_size=256）

```bash
python -m dinov3.eval.bio_segmentation.linear_probe \
    --dataset    conic \
    --use-cached-features \
    --train-cache ./cache/conic/conic_train_microscopy_continual_vitl16_last1_s256.npz \
    --val-cache   ./cache/conic/conic_val_microscopy_continual_vitl16_last1_s256.npz \
    --test-cache  ./cache/conic/conic_test_microscopy_continual_vitl16_last1_s256.npz \
    --output-dir  ./outputs/linear_probe/conic_vitl \
    --epochs 50 --batch-size 64 --lr 1e-3
```

### LIVECell

```bash
python -m dinov3.eval.bio_segmentation.linear_probe \
    --dataset    livecell \
    --use-cached-features \
    --train-cache ./cache/livecell/livecell_train_microscopy_continual_vitl16_last1_s512.npz \
    --val-cache   ./cache/livecell/livecell_val_microscopy_continual_vitl16_last1_s512.npz \
    --test-cache  ./cache/livecell/livecell_test_microscopy_continual_vitl16_last1_s512.npz \
    --output-dir  ./outputs/linear_probe/livecell_vitl \
    --epochs 50 --batch-size 64 --lr 1e-3
```

### PanNuke（6类，img_size=256）

```bash
python -m dinov3.eval.bio_segmentation.linear_probe \
    --dataset    pannuke \
    --use-cached-features \
    --train-cache ./cache/pannuke/pannuke_train_microscopy_continual_vitl16_last1_s256.npz \
    --val-cache   ./cache/pannuke/pannuke_val_microscopy_continual_vitl16_last1_s256.npz \
    --test-cache  ./cache/pannuke/pannuke_test_microscopy_continual_vitl16_last1_s256.npz \
    --output-dir  ./outputs/linear_probe/pannuke_vitl \
    --epochs 50 --batch-size 64 --lr 1e-3
```

### TissueNet

```bash
python -m dinov3.eval.bio_segmentation.linear_probe \
    --dataset    tissuenet \
    --use-cached-features \
    --train-cache ./cache/tissuenet/tissuenet_train_microscopy_continual_vitl16_last1_s256.npz \
    --val-cache   ./cache/tissuenet/tissuenet_val_microscopy_continual_vitl16_last1_s256.npz \
    --test-cache  ./cache/tissuenet/tissuenet_test_microscopy_continual_vitl16_last1_s256.npz \
    --output-dir  ./outputs/linear_probe/tissuenet_vitl \
    --epochs 50 --batch-size 64 --lr 1e-3
```

> 换 backbone 或训练 config 时，把 cache 文件名里的 **config stem** 段改成与 **`--train-config`** 的文件名 stem 一致（例如 `microscopy_continual_vitl16` ↔ `dinov3_vit7b16_pretrain`）。

---

## Step 2b：Mask2Former（在线训练，不需要 cache）

> **说明**：下列命令使用模块路径 **`dinov3.eval.bio_segmentation.mask2former`**。若当前分支**未包含**该入口，请以你环境中的 Mask2Former 训练脚本为准；加载 backbone 时应传入与训练一致的 **`--train-config`**，以及 **`--checkpoint`**（DCP 目录、`teacher_checkpoint_trainstyle.pth` 或兼容的合并权重）。

Mask2Former 内部通过 `DINOv3_Adapter` 使用 `FOUR_EVEN_INTERVALS`（4 层均匀间隔），
推理时对大图采用滑窗（`--inference-mode slide`），对小 patch 使用全图推理（`whole`）。
下面的 `batch-size` / `num-workers` 以当前 32GB GPU 为基准；若使用 `torchrun`，`batch-size` 表示**单卡 batch**。
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/home/lxy/miniconda3/envs/dinov3/lib/python3.11/site-packages/torch/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
### MoNuSeg（大图 1000×1000 → slide 推理）

```bash
CUDA_VISIBLE_DEVICES=7 python -m dinov3.eval.bio_segmentation.mask2former \
    --dataset    monuseg \
    --data-root  /mnt/huawei_deepcad/benchmark/segmentation/monuseg/extracted \
    --checkpoint /mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth \
    --output-dir ./outputs/mask2former/monuseg_vit7b \
    --train-config dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
    --inference-mode slide \
    --crop-size  512 \
    --stride     341 \
    --num-workers 4 \
    --epochs 20 --batch-size 2 --lr 1e-4 --adapter-lr 1e-5
```

### BBBC038（大图 → slide 推理）

```bash
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 -m dinov3.eval.bio_segmentation.mask2former \
    --dataset    bbbc038 \
    --data-root  /data/dataset/segmentation/bbbc038/extracted \
    --checkpoint /data/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth\
    --output-dir ./outputs/mask2former/bbbc038_vit7b \
    --train-config dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
    --inference-mode slide \
    --crop-size  512 --stride 341 \
    --num-workers 8 \
    --epochs 50 --batch-size 4 --lr 1e-4 --adapter-lr 1e-5
```

### LIVECell

```bash
CUDA_VISIBLE_DEVICES=4 python -m dinov3.eval.bio_segmentation.mask2former \
    --dataset    livecell \
    --data-root  /mnt/huawei_deepcad/benchmark/segmentation/LIVECell \
    --checkpoint /mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth \
    --output-dir ./outputs/mask2former/livecell_vit7b \
    --train-config dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
    --inference-mode whole \
    --crop-size  512 \
    --num-workers 8 \
    --epochs 50 --batch-size 2 --lr 1e-4 --adapter-lr 1e-5
```

### CoNIC（小 patch 256×256 → whole 推理）

```bash
CUDA_VISIBLE_DEVICES=5 python -m dinov3.eval.bio_segmentation.mask2former \
    --dataset    conic \
    --data-root  /mnt/huawei_deepcad/benchmark/segmentation/conic/extracted \
    --checkpoint /mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth \
    --output-dir ./outputs/mask2former/conic_vit7b_ddp \
    --train-config dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
    --inference-mode whole \
    --crop-size  256 \
    --num-workers 4 \
    --epochs 50 --batch-size 8 --lr 1e-4 --adapter-lr 1e-5
```

### PanNuke（小 patch 256×256 → whole 推理）

```bash
CUDA_VISIBLE_DEVICES=5 python -m dinov3.eval.bio_segmentation.mask2former \
    --dataset    pannuke \
    --data-root  /mnt/huawei_deepcad/benchmark/segmentation/pannuke/extracted \
    --checkpoint /mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth \
    --output-dir ./outputs/mask2former/pannuke_vit7b \
    --train-config dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
    --inference-mode whole \
    --crop-size  256 \
    --num-workers 8 \
    --epochs 50 --batch-size 8 --lr 1e-4 --adapter-lr 1e-5
```

### TissueNet

```bash
CUDA_VISIBLE_DEVICES=6 python -m dinov3.eval.bio_segmentation.mask2former \
    --dataset    tissuenet \
    --data-root  /mnt/huawei_deepcad/benchmark/segmentation/tissuenet/extracted \
    --checkpoint /mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth \
    --output-dir ./outputs/mask2former/tissuenet_vit7b \
    --train-config dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
    --inference-mode whole \
    --crop-size  256 \
    --num-workers 8 \
    --epochs 50 --batch-size 8 --lr 1e-4 --adapter-lr 1e-5
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

- **自训权重 → 本 pipeline**：优先将 **`--checkpoint`** 指向 **DCP 目录**或 **`teacher_checkpoint_trainstyle.pth`**，并传入与训练一致的 **`--train-config`**；**`export_backbone_from_train_ckpt`** 为可选辅助。结构字段（含 **`student.n_storage_tokens` / `student.mask_k_bias`**）须与训练 yaml 一致。
- MoNuSeg / BBBC038 没有官方 val split → 自动从 train 随机抽取（MoNuSeg 20%，BBBC038 15%），索引保存到 `{data_root}/monuseg_val_indices.npy`
- PanNuke：Fold1+2 = train，Fold3 = val = test（3-fold 交叉验证标准做法）
- CoNIC：自动 80/10/10 随机划分，索引保存到 `conic_split_indices.npz`
- LIVECell / TissueNet：官方提供 train/val/test split
