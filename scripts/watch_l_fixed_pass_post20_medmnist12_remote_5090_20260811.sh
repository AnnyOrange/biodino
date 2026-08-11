#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$HOME/biodino}"
RUN_BASE="${RUN_BASE:-/mnt/data/biodino_fixed_pass}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/data/benchmark}"
PYTHON_BIN="${PYTHON_BIN:-/home/xzj/miniconda3/envs/dinov3/bin/python}"
WEIGHT="${WEIGHT:-/mnt/data/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_IDLE_MAX_MIB="${GPU_IDLE_MAX_MIB:-1024}"

RAW_ROOT="$RUN_BASE/outputs/01_training_runs"
POINT_ROOT="$RAW_ROOT/SBL_splus_datafp_alpha075_20260810/L"
EVAL_ROOT="$RUN_BASE/outputs/02_eval_runs/L_datafp_alpha075_medmnist12_20260811"
TRAIN_STATE="$RUN_BASE/outputs/00_reports/l_fixed_pass_20260811"
STATE_ROOT="$RUN_BASE/outputs/00_reports/l_fixed_pass_medmnist12_20260811"
LOG_ROOT="$RUN_BASE/outputs/auto_eval_logs/l_fixed_pass_medmnist12_20260811"
INTERPOLATOR="$REPO/scripts/interpolate_teacher_checkpoints.py"
VALIDATOR="$REPO/scripts/validate_partial_bio_eval.py"
DATA_ROOT="$BENCHMARK_ROOT/Classification/MedMNIST"

DATASETS=(
  bloodmnist pathmnist tissuemnist breastmnist
  organamnist organcmnist organsmnist dermamnist
  octmnist pneumoniamnist retinamnist chestmnist
)

mkdir -p "$POINT_ROOT" "$EVAL_ROOT" "$STATE_ROOT" "$LOG_ROOT"
exec 9>"$STATE_ROOT/watcher.lock"
if ! flock -n 9; then
  echo "L fixed-pass MedMNIST-12 watcher is already active" >&2
  exit 2
fi

log() {
  printf '[%s] [L-data-fp-medmnist12] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

on_exit() {
  local rc=$?
  if [[ "$rc" -eq 0 ]]; then
    date -u '+%Y-%m-%dT%H:%M:%SZ' >"$STATE_ROOT/all_evaluations.done"
    rm -f "$STATE_ROOT/all_evaluations.failed"
  else
    printf '%s rc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$rc" >"$STATE_ROOT/all_evaluations.failed"
  fi
}
trap on_exit EXIT

[[ -x "$PYTHON_BIN" && -s "$WEIGHT" && -f "$INTERPOLATOR" && -f "$VALIDATOR" ]]
[[ -f "$DATA_ROOT/.transfer_complete" ]]
for dataset in "${DATASETS[@]}"; do
  [[ -s "$DATA_ROOT/$dataset.npz" ]]
done

while [[ ! -f "$TRAIN_STATE/L_random20.done" ]]; do
  if [[ -f "$TRAIN_STATE/L_random20.failed" ]]; then
    log "ERROR: random20 training failed; refusing to evaluate incomplete checkpoints"
    exit 3
  fi
  log "waiting for random20 training to complete"
  sleep "$POLL_SECONDS"
done

while true; do
  mapfile -t used_mib < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  blocked=""
  for gpu in 4 5 6 7; do
    used="${used_mib[$gpu]:-999999}"
    if (( used > GPU_IDLE_MAX_MIB )); then
      blocked+=" gpu${gpu}=${used}MiB"
    fi
  done
  [[ -z "$blocked" ]] && break
  log "waiting for GPUs 4-7 to become idle:$blocked"
  sleep "$POLL_SECONDS"
done

interpolate_point() {
  local label="$1" passes="$2" oel raw checkpoint point
  case "$label" in
    10) oel=103 ;;
    20) oel=205 ;;
    *) return 2 ;;
  esac
  raw="$RAW_ROOT/L_s6recipe_sigreg005_datafp_random${label}_e15_gb1024_seed0_4x5090_20260810"
  checkpoint=$((passes * oel - 1))
  point="$POINT_ROOT/random_${label}/pass_${passes}"
  if [[ ! -s "$point/ckpt/75/checkpoint.pth" ]]; then
    [[ -s "$raw/ckpt/$checkpoint/checkpoint.pth" && -s "$raw/config.yaml" ]]
    log "interpolating random${label} pass${passes} at alpha=0.75"
    "$PYTHON_BIN" "$INTERPOLATOR" \
      --official-checkpoint "$WEIGHT" \
      --bio-checkpoint "$raw/ckpt/$checkpoint/checkpoint.pth" \
      --bio-config "$raw/config.yaml" \
      --output-root "$point" \
      --alphas 0.75
  fi
  [[ -s "$point/ckpt/75/checkpoint.pth" && -s "$point/config.yaml" && -s "$point/interpolation_manifest.json" ]]
}

evaluate_point() {
  local label="$1" passes="$2" point output marker log_file
  point="$POINT_ROOT/random_${label}/pass_${passes}"
  output="$EVAL_ROOT/random_${label}/pass_${passes}"
  marker="$output/_online_status/ckpt_75_medmnist12.done"
  log_file="$LOG_ROOT/random${label}_pass${passes}.log"
  if [[ -f "$marker" ]]; then
    log "random${label} pass${passes} already complete"
    return 0
  fi
  mkdir -p "$output"
  log "evaluating random${label} pass${passes}: GPUs 4-7, three tests per GPU"
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    CHECKPOINT_ITERS=75 \
    GPUS="4 5 6 7" \
    JOBS_PER_GPU=3 \
    MAX_CONCURRENT_JOBS=12 \
    MAX_CPU_JOBS=12 \
    CONCURRENT_TASK_GROUPS=0 \
    TASKS=classification \
    CLASSIFICATION_DATASETS="${DATASETS[*]}" \
    FROZEN_DATASETS_PER_JOB=1 \
    FROZEN_BATCH_SIZE=16 \
    FROZEN_CHANNEL_POLICY=auto \
    AUTOCAST_DTYPE=bf16 \
    NUM_WORKERS=2 \
    EVAL_BLAS_THREADS=1 \
    DRY_RUN=0 \
    bash "$REPO/scripts/run_bio_benchmark_all.sh" \
      "$point/ckpt" "$point/config.yaml" "$output" "$BENCHMARK_ROOT" \
      >"$log_file" 2>&1
  "$PYTHON_BIN" "$VALIDATOR" \
    --eval-root "$output" \
    --checkpoint 75 \
    --panel-name medmnist12 \
    --datasets "${DATASETS[@]}" \
    >>"$log_file" 2>&1
  [[ -f "$marker" ]]
  log "completed random${label} pass${passes}"
}

for label in 10 20; do
  for passes in 8 15; do
    interpolate_point "$label" "$passes"
    evaluate_point "$label" "$passes"
  done
done

log "all four MedMNIST-12 evaluation points are complete"
