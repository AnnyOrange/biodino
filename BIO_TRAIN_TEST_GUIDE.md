# BioDINO Train And Test Guide

This document describes how to create the environment, train RGB ViT-7B with
the provided shell script, and run all BioDINO benchmark tests with the eval
shell script.

The commands assume the repository is located at:

```bash
/mnt/huawei_deepcad/dinov3
```

and benchmark data is located at:

```bash
/mnt/huawei_deepcad/benchmark
```

## 1. Create Environment

Install Miniforge/Mambaforge or any mamba-compatible conda distribution first.
Then create the environment:

```bash
mamba create -n dinov3 python=3.11 -y
conda activate dinov3
```

Install the repository and Python dependencies:

```bash
cd /mnt/huawei_deepcad/dinov3
pip install -e .
pip install -r requirement.txt
```

## 2. Training Script

Training is launched by:

```bash
scripts/run_vit7b_rgb_multinode.sh
```

This script is for RGB ViT-7B training. It explicitly uses:

```text
student.in_chans=3
teacher.in_chans=3
student.enable_channelvit=false
teacher.enable_channelvit=false
```


### 2.1 Single-Node Training

Example for one node with 8 GPUs:

```bash
cd /mnt/huawei_deepcad/dinov3
conda activate dinov3

DATASET_PATH='packwds:/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*-{000000..000999}.tar' \
OUTPUT_DIR='/mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_1node' \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=8 \
GRAD_ACCUM_STEPS=1 \
bash scripts/run_vit7b_rgb_multinode.sh
```

`OUTPUT_DIR` is the experiment output folder. It will contain:

```text
OUTPUT_DIR/
  config.yaml
  ckpt/
    <iter>/
      checkpoint.pth
  nan_logs/
  ...
```

If memory is safe, try increasing:

```bash
BATCH_SIZE_PER_GPU=16
```

If out of memory, reduce:

```bash
BATCH_SIZE_PER_GPU=4
```

The effective global batch size is:

```text
global_batch = NNODES * NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS
```

### 2.2 Two-Node Training

Assume two nodes, each with 8 GPUs. The same shared filesystem path should be
visible on both nodes.

On node 0:

```bash
cd /mnt/huawei_deepcad/dinov3
conda activate dinov3

DATASET_PATH='packwds:/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*-{000000..000999}.tar' \
OUTPUT_DIR='/mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes' \
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR='<NODE0_IP>' \
MASTER_PORT=29500 \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=8 \
GRAD_ACCUM_STEPS=1 \
bash scripts/run_vit7b_rgb_multinode.sh
```

On node 1:

```bash
cd /mnt/huawei_deepcad/dinov3
conda activate dinov3

DATASET_PATH='packwds:/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*-{000000..000999}.tar' \
OUTPUT_DIR='/mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes' \
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR='<NODE0_IP>' \
MASTER_PORT=29500 \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=8 \
GRAD_ACCUM_STEPS=1 \
bash scripts/run_vit7b_rgb_multinode.sh
```

Only `NODE_RANK` changes between nodes. `MASTER_ADDR` should be node 0's IP on
all nodes.

### 2.3 Training Dry Run

Use dry run to print the command without starting training:

```bash
DRY_RUN=1 \
DATASET_PATH='packwds:/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*-{000000..000999}.tar' \
OUTPUT_DIR='/mnt/huawei_deepcad/dinov3/outputs/debug_vit7b_rgb' \
NPROC_PER_NODE=8 \
bash scripts/run_vit7b_rgb_multinode.sh
```

### 2.4 Short Training Smoke Test

Run a short job first to verify data loading, NCCL, model initialization, and
checkpoint saving:

```bash
DATASET_PATH='packwds:/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*-{000000..000999}.tar' \
OUTPUT_DIR='/mnt/huawei_deepcad/dinov3/outputs/debug_vit7b_rgb_smoke' \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=4 \
GRAD_ACCUM_STEPS=1 \
OFFICIAL_EPOCH_LENGTH=20 \
CHECKPOINT_PERIOD=20 \
SAVECKP_FREQ=1 \
bash scripts/run_vit7b_rgb_multinode.sh
```

## 3. Testing / Evaluation Script

All bio benchmark evals are launched by:

```bash
scripts/run_bio_benchmark_all.sh
```

Basic usage:

```bash
bash scripts/run_bio_benchmark_all.sh <checkpoints_dir> <train_config> <output_dir> [benchmark_root]
```

Example:

```bash
cd /mnt/huawei_deepcad/dinov3
conda activate dinov3

bash scripts/run_bio_benchmark_all.sh \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/ckpt \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/config.yaml \
  /mnt/huawei_deepcad/dinov3/outputs/bio_eval_vit7b_rgb_b200_2nodes \
  /mnt/huawei_deepcad/benchmark
```

By default this evaluates:

```text
segmentation:    bbbc038 conic monuseg pannuke tissuenet
classification:  bloodmnist bbbc048 cyclops midog25
regression:      bbbc013
detection:       livecell
```

and writes:

```text
output_dir/
  command_manifest.json
  bio_segmentation.md
  bio_classification.md
  bio_regression.md
  bio_detection.md
  bio_segmentation/
  bio_classification/
  bio_regression/
  bio_detection/
```

### 3.1 Evaluate All Checkpoints

Default is `CHECKPOINT_ITERS=all`, so this evaluates every:

```text
<checkpoints_dir>/<iter>/checkpoint.pth
```

Command:

```bash
CHECKPOINT_ITERS=all \
bash scripts/run_bio_benchmark_all.sh \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/ckpt \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/config.yaml \
  /mnt/huawei_deepcad/dinov3/outputs/bio_eval_vit7b_rgb_b200_2nodes \
  /mnt/huawei_deepcad/benchmark
```

### 3.2 Evaluate Latest Checkpoint Only

```bash
CHECKPOINT_ITERS=latest \
bash scripts/run_bio_benchmark_all.sh \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/ckpt \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/config.yaml \
  /mnt/huawei_deepcad/dinov3/outputs/bio_eval_vit7b_rgb_b200_2nodes_latest \
  /mnt/huawei_deepcad/benchmark
```

### 3.3 Evaluate Selected Checkpoints

```bash
CHECKPOINT_ITERS="7174 8199 10249" \
bash scripts/run_bio_benchmark_all.sh \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/ckpt \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/config.yaml \
  /mnt/huawei_deepcad/dinov3/outputs/bio_eval_vit7b_rgb_b200_2nodes_selected \
  /mnt/huawei_deepcad/benchmark
```

### 3.4 Use Specific GPUs

If `GPUS` is not set, the eval script uses all visible GPUs. To use specific
GPUs:

```bash
GPUS="0 2 3" \
bash scripts/run_bio_benchmark_all.sh \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/ckpt \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/config.yaml \
  /mnt/huawei_deepcad/dinov3/outputs/bio_eval_vit7b_rgb_b200_2nodes \
  /mnt/huawei_deepcad/benchmark
```

One job per GPU is recommended:

```bash
JOBS_PER_GPU=1
```

### 3.5 Eval Dry Run

Print all generated eval jobs without running them:

```bash
DRY_RUN=1 \
CHECKPOINT_ITERS=latest \
bash scripts/run_bio_benchmark_all.sh \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/ckpt \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/config.yaml \
  /mnt/huawei_deepcad/dinov3/outputs/bio_eval_dryrun \
  /mnt/huawei_deepcad/benchmark
```

### 3.6 Eval Smoke Test

Run a tiny end-to-end check. This is not for final numbers:

```bash
SMOKE=1 \
CHECKPOINT_ITERS=latest \
GPUS="0" \
bash scripts/run_bio_benchmark_all.sh \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/ckpt \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/config.yaml \
  /mnt/huawei_deepcad/dinov3/outputs/bio_eval_smoke \
  /mnt/huawei_deepcad/benchmark
```

## 4. Common Eval Options

The eval shell script can be controlled by environment variables:

```bash
CHECKPOINT_ITERS=all
GPUS="0 1 2 3"
JOBS_PER_GPU=1
TASKS="segmentation classification regression detection"
LAYER_PRESET=last1
NUM_WORKERS=4

SEGMENTATION_DATASETS="bbbc038 conic monuseg pannuke tissuenet"
CLASSIFICATION_DATASETS="bloodmnist bbbc048 cyclops midog25"
REGRESSION_DATASETS="bbbc013"
DETECTION_DATASETS="livecell"

SEG_FEATURE_BATCH_SIZE=32
SEG_PROBE_EPOCHS=50
CLS_EPOCHS=10
CLS_BATCH_SIZE=256
REG_BATCH_SIZE=128
DET_EPOCHS=5
DET_BATCH_SIZE=8
```

Example: run only classification and regression:

```bash
TASKS="classification regression" \
CHECKPOINT_ITERS=latest \
bash scripts/run_bio_benchmark_all.sh \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/ckpt \
  /mnt/huawei_deepcad/dinov3/outputs/bio_continue_vit7b_rgb_b200_2nodes/config.yaml \
  /mnt/huawei_deepcad/dinov3/outputs/bio_eval_cls_reg \
  /mnt/huawei_deepcad/benchmark
```


