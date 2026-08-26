#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/huawei_deepcad/dinov3}"
TRAIN_RUN="${TRAIN_RUN:-$REPO/outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_6m_mix1m03_10tv107_8x5090zxr_20260826}"
INPUT_ROOT="${INPUT_ROOT:-$REPO/outputs/02_eval_inputs/hs6_l_6m_full_1m_20260826}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/outputs/02_eval_runs/hs6_l_6m_full_1m_3090fleet_20260826}"
PYTHON_BIN="${PYTHON_BIN:-/home/inspur/anaconda3/envs/dinov3/bin/python}"
GPU="${GPU:-0}"
WORKER_NAME="${WORKER_NAME:-$(hostname)-gpu${GPU}}"
LOG_ROOT="$OUTPUT_ROOT/_state/worker_logs"

mkdir -p "$LOG_ROOT"
exec "$PYTHON_BIN" -u "$REPO/scripts/run_hs6_l_6m_full_eval_fleet_worker.py" \
  --repo "$REPO" \
  --train-run "$TRAIN_RUN" \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --python-bin "$PYTHON_BIN" \
  --gpu "$GPU" \
  --worker "$WORKER_NAME" \
  >>"$LOG_ROOT/${WORKER_NAME}.log" 2>&1
