#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$HOME/biodino}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/fixed_pass_random}"
RUN_BASE="${RUN_BASE:-/mnt/data/biodino_fixed_pass}"
RUN_ROOT="$RUN_BASE/outputs/01_training_runs"
REPORT_ROOT="$RUN_BASE/outputs/00_reports/splus_bl_fixed_pass_data_scaling_20260810"
LOG_ROOT="$RUN_BASE/outputs/auto_train_logs/l_fixed_pass_20260811"
STATE_ROOT="$RUN_BASE/outputs/00_reports/l_fixed_pass_20260811"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/data/weights}"
PYTHON_BIN="${PYTHON_BIN:-/home/xzj/miniconda3/envs/dinov3/bin/python}"
TRAINER="$REPO/scripts/run_bl_fixed_pass_data_scaling_train_20260810.sh"
MAX_TO_KEEP="${MAX_TO_KEEP:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"

mkdir -p "$RUN_ROOT" "$REPORT_ROOT" "$LOG_ROOT" "$STATE_ROOT"
exec 9>"$STATE_ROOT/launcher.lock"
if ! flock -n 9; then
  echo "L fixed-pass launcher is already active" >&2
  exit 2
fi

for label in 10 20 50; do
  [[ -f "$DATA_ROOT/ratio_${label}.0/.transfer_complete" ]] || {
    echo "Missing completed random${label} dataset" >&2
    exit 2
  }
done
[[ -s "$WEIGHTS_DIR/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" ]]
[[ -x "$PYTHON_BIN" && -x "$TRAINER" ]]

run_label() {
  local label="$1" gpus="$2" port="$3"
  local log="$LOG_ROOT/L_random${label}.log"
  local status="$STATE_ROOT/L_random${label}"
  date -u '+%Y-%m-%dT%H:%M:%SZ' >"${status}.started"
  if env \
    REPO="$REPO" \
    RUN_ROOT="$RUN_ROOT" \
    REPORT_ROOT="$REPORT_ROOT" \
    WEIGHTS_DIR="$WEIGHTS_DIR" \
    RANDOM_DATA_ROOT="$DATA_ROOT" \
    FULL_DATA_ROOT="/mnt/data/microscopy-100k-patched" \
    PYTHON_BIN="$PYTHON_BIN" \
    LABELS="$label" \
    GPU_GROUP="$gpus" \
    MASTER_PORT="$port" \
    MAX_TO_KEEP="$MAX_TO_KEEP" \
    NUM_WORKERS="$NUM_WORKERS" \
    RUN_DATE=20260810 \
    DRY_RUN=0 \
    bash "$TRAINER" L >"$log" 2>&1; then
    date -u '+%Y-%m-%dT%H:%M:%SZ' >"${status}.done"
    return 0
  fi
  date -u '+%Y-%m-%dT%H:%M:%SZ' >"${status}.failed"
  return 1
}

run_label 10 0,1,2,3 30110 &
pid10=$!
run_label 20 4,5,6,7 30120 &
pid20=$!

rc=0
pid50=""
if wait "$pid10"; then
  run_label 50 0,1,2,3 30150 &
  pid50=$!
else
  rc=1
fi
wait "$pid20" || rc=1
if [[ -n "$pid50" ]]; then
  wait "$pid50" || rc=1
fi

if [[ "$rc" -eq 0 ]]; then
  date -u '+%Y-%m-%dT%H:%M:%SZ' >"$STATE_ROOT/all_training.done"
fi
exit "$rc"
