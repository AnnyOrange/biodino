#!/usr/bin/env bash
# Launch the matched HS6-L e11->e15 SKDT and shuffled-control pair on 3090-qi.
set -euo pipefail

REPO=${REPO:-/mnt/huawei_deepcad/dinov3}
PYTHON_BIN=${PYTHON_BIN:-/home/bbnc/anaconda3/envs/dinov3/bin/python}
BASELINE_RUN=${BASELINE_RUN:-$REPO/outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_1m_he_mix085_015_4xa100_deepcad_20260903}
RESUME_CHECKPOINT_DIR=${RESUME_CHECKPOINT_DIR:-$BASELINE_RUN/ckpt/11274}
LAUNCHER=$REPO/scripts/launch_hs6_l_scout_transport_4xa100_deepcad_20260826.sh
STAMP=${STAMP:-20260907}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
GPU_GROUP_TRUE=${GPU_GROUP_TRUE:-0,1,2,3}
GPU_GROUP_SHUFFLE=${GPU_GROUP_SHUFFLE:-4,5,6,7}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-64}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
EXPECTED_GLOBAL_BATCH_SIZE=${EXPECTED_GLOBAL_BATCH_SIZE:-1024}

common=(
  REPO="$REPO"
  PYTHON_BIN="$PYTHON_BIN"
  RESUME_CHECKPOINT_DIR="$RESUME_CHECKPOINT_DIR"
  WAIT_FOR_GPUS=0
  NPROC_PER_NODE="$NPROC_PER_NODE"
  BATCH_SIZE_PER_GPU="$BATCH_SIZE_PER_GPU"
  GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS"
  EXPECTED_GLOBAL_BATCH_SIZE="$EXPECTED_GLOBAL_BATCH_SIZE"
  OFFICIAL_EPOCH_LENGTH=1025
  EPOCHS=15
  CHECKPOINT_PERIOD=1025
  CHECKPOINT_MAX_TO_KEEP=16
  SEED=0
  LOSS_WEIGHT=0.5
)

true_output=$REPO/outputs/01_training_runs/HS6_L_e11to15_SKDT_stable_w0.5_bs${BATCH_SIZE_PER_GPU}_gb${EXPECTED_GLOBAL_BATCH_SIZE}_seed0_${NPROC_PER_NODE}x3090qi_${STAMP}
shuffle_output=$REPO/outputs/01_training_runs/HS6_L_e11to15_SKDT_shuffled_stable_w0.5_bs${BATCH_SIZE_PER_GPU}_gb${EXPECTED_GLOBAL_BATCH_SIZE}_seed0_${NPROC_PER_NODE}x3090qi_${STAMP}
log_root=$REPO/outputs/auto_train_logs
mkdir -p "$log_root"

env "${common[@]}" \
  GPU_GROUP="$GPU_GROUP_TRUE" MASTER_PORT=31911 TARGET_MODE=stable_delta \
  RUN_TAG=$(basename "$true_output") OUTPUT_DIR="$true_output" \
  "$LAUNCHER" >"$log_root/$(basename "$true_output").log" 2>&1 &
true_pid=$!

env "${common[@]}" \
  GPU_GROUP="$GPU_GROUP_SHUFFLE" MASTER_PORT=31912 TARGET_MODE=shuffled_stable_delta \
  RUN_TAG=$(basename "$shuffle_output") OUTPUT_DIR="$shuffle_output" \
  "$LAUNCHER" >"$log_root/$(basename "$shuffle_output").log" 2>&1 &
shuffle_pid=$!

printf 'SKDT pid=%s output=%s\n' "$true_pid" "$true_output"
printf 'shuffled pid=%s output=%s\n' "$shuffle_pid" "$shuffle_output"
