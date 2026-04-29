# Bio-Segmentation Linear Probe（单命令版）

你原来的流程是三步：

1. 数据解压  
2. 特征提取（`feature_extractor`）  
3. Linear Probe 训练（`linear_probe`）

现在可以用一个统一入口直接跑完。

---

## 运行前检查（建议先做）

```bash
conda run -n dinov3 python -c "import omegaconf; print('omegaconf ok')"
nvidia-smi
```

- 第一条用于确认依赖在 `dinov3` 环境里；
- 第二条用于确认当前机器能看到 GPU。

如果你当前 shell 已经 `conda activate dinov3`，下面命令可直接用 `python -m ...`；  
如果不想切环境，直接用 `conda run -n dinov3 python -m ...`。

---

## 一行命令（推荐）

```bash
conda run -n dinov3 python -m dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline \
  --datasets bbbc038 conic monuseg pannuke tissuenet \
  --checkpoints-dir /mnt/huawei_deepcad/dinov3/outputs/bio_continue_1025_a100_grad_acc_2_base/ckpt \
  --checkpoint-iters latest \
  --train-config dinov3/configs/train/microscopy_continual_vitb16.yaml \
  --data-root-base /mnt/huawei_deepcad/benchmark/segmentation \
  --gpu 0
```

我已在 `2026-04-28` 做过 dry-run 验证，命令链路可正常展开（`latest` / 指定多个 iter 都可用）。
同日做过真实启动验证：流程可进入数据集构建与 checkpoint 加载；当前会话因为无可用 CUDA 设备而停止。

上面这条命令会自动执行：

- 对每个数据集跑 `train/val/test` 三个 split 的特征缓存；
- 自动按 `cfg_stem` + `img_size` 拼出 cache 文件名；
- 用缓存特征跑 linear probe；
- 按 checkpoint 迭代号分别写输出目录。

---

## 你的 checkpoint 目录结构可直接用

例如你给的目录：

```bash
/mnt/huawei_deepcad/dinov3/outputs/bio_continue_1025_a100_grad_acc_2_base/ckpt
├── 1024/
├── 2049/
├── 3074/
...
└── 15374/
```

脚本会自动识别 `<iter>/checkpoint.pth`。

---

## `--checkpoint-iters` 用法

- 只跑最新一个：`--checkpoint-iters latest`
- 跑全部：`--checkpoint-iters all`
- 跑指定迭代：`--checkpoint-iters 3074 6149 9224`
- 支持逗号：`--checkpoint-iters 3074,6149,9224`

例如：

```bash
conda run -n dinov3 python -m dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline \
  --datasets conic \
  --checkpoints-dir /mnt/huawei_deepcad/dinov3/outputs/bio_continue_1025_a100_grad_acc_2_base/ckpt \
  --checkpoint-iters 3074,6149,9224 \
  --train-config dinov3/configs/train/microscopy_continual_vitb16.yaml \
  --data-root-base /mnt/huawei_deepcad/benchmark/segmentation \
  --gpu 0
```

---

## 如果希望把“解压”也合并进同一条命令

加上 `--extract-src-dir`（可选再加 `--extract-dst-dir`）：

```bash
conda run -n dinov3 python -m dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline \
  --datasets bbbc038 conic monuseg pannuke tissuenet \
  --extract-src-dir /path/to/raw_archives_or_dataset_folder \
  --extract-dst-dir /mnt/huawei_deepcad/benchmark/segmentation \
  --checkpoints-dir /mnt/huawei_deepcad/dinov3/outputs/bio_continue_1025_a100_grad_acc_2_base/ckpt \
  --checkpoint-iters latest \
  --train-config dinov3/configs/train/microscopy_continual_vitb16.yaml \
  --data-root-base /mnt/huawei_deepcad/benchmark/segmentation \
  --gpu 0
```

说明：`extract_datasets` 使用单个 `--src-dir` 扫描数据。若你六个数据集分散在多个源目录，建议先手动整理到同一 source root，或分开解压后再跑单命令训练。

---

## 主要参数

- `--datasets`：要跑的数据集（支持 `bbbc038 conic livecell monuseg pannuke tissuenet`）
- `--checkpoints-dir`：checkpoint 根目录（必须是 `<iter>/checkpoint.pth` 结构）
- `--train-config`：训练 YAML，必须和 checkpoint 架构一致
- `--data-root-base`：数据根目录  
  - 非 livecell：`<base>/<dataset>/extracted`
  - livecell：`<base>/LIVECell`
- `--feature-img-size`：默认 `0`（按数据集默认尺寸：`conic/pannuke/tissuenet=256`，其余 `512`）
- `--layers`：不传则默认 `last1`，传入例如 `--layers 4 11 17 23` 时会自动改 cache 命名 tag
- `--probe-epochs` / `--probe-batch-size` / `--probe-lr`：linear probe 训练超参
- `--cache-root` / `--output-root` / `--run-name`：缓存与结果输出路径组织
- `--dry-run`：只打印将执行的命令，不真正运行

---

## 先验证再正式跑（推荐）

先 dry-run：

```bash
conda run -n dinov3 python -m dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline \
  --datasets bbbc038 conic monuseg pannuke tissuenet \
  --checkpoints-dir /mnt/huawei_deepcad/dinov3/outputs/bio_continue_1025_a100_grad_acc_2_base/ckpt \
  --checkpoint-iters latest \
  --train-config dinov3/configs/train/microscopy_continual_vitb16.yaml \
  --data-root-base /mnt/huawei_deepcad/benchmark/segmentation \
  --gpu 0 \
  --dry-run
```

确认没问题后，把最后一行 `--dry-run` 去掉即可正式开跑。

---

## 输出目录

默认会生成：

```text
cache/linear_probe_pipeline/<run_name>/<dataset>/<iter>/*.npz
outputs/linear_probe_pipeline/<run_name>/<dataset>/<iter>/results.json
```

其中 `<run_name>` 默认为 `--train-config` 的 stem（例如 `microscopy_continual_vitb16`）。

---

## 入口脚本

统一入口在：

```text
dinov3/eval/bio_segmentation/scripts/run_linear_probe_pipeline.py
```
