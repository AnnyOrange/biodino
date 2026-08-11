#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "B" && "$1" != "L" ) ]]; then
  echo "Usage: $0 <B|L>" >&2
  exit 2
fi

MODEL="$1"
REPO="${REPO:-/mnt/huawei_deepcad/dinov3}"
RUN_ROOT="${RUN_ROOT:-$REPO/outputs/01_training_runs}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/huawei_deepcad/weights}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO/docs/scaling_law/bio_sweet_spot/scripts/12_train_rgb_bioaug_robust.sh}"
RANDOM_DATA_ROOT="${RANDOM_DATA_ROOT:-/mnt/huawei_deepcad/deduplication/random}"
FULL_DATA_ROOT="${FULL_DATA_ROOT:-/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle}"
GPU_GROUP="${GPU_GROUP:-0,1,2,3}"
LABEL_STRING="${LABELS:-10 20 50 100}"
MASTER_PORT="${MASTER_PORT:-30110}"
DRY_RUN="${DRY_RUN:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_TO_KEEP="${MAX_TO_KEEP:-20}"
RUN_DATE="${RUN_DATE:-20260810}"

case "$MODEL" in
  B)
    SIZE=b
    LR=0.0001
    BATCH_SIZE_PER_GPU=32
    GRAD_ACCUM_STEPS=8
    PYTHON_BIN="${PYTHON_BIN:-/home/bbnc/anaconda3/envs/dinov3/bin/python}"
    RUN_TAG="4x3090"
    ;;
  L)
    SIZE=l
    LR=0.00005
    BATCH_SIZE_PER_GPU=16
    GRAD_ACCUM_STEPS=16
    PYTHON_BIN="${PYTHON_BIN:-/home/lxy/miniconda3/envs/dinov3/bin/python}"
    RUN_TAG="4x5090"
    ;;
esac

labels=(10 20 50 100)
samples=(104877 209754 524385 1048771)
shards=(35 69 172 326)
epoch_lengths=(103 205 513 1025)

log() {
  printf '[%s] [%s-data-fp] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$MODEL" "$*"
}

contains_label() {
  [[ " $LABEL_STRING " == *" $1 "* ]]
}

dataset_for_label() {
  local label="$1" shard_count="$2"
  if [[ "$label" == "100" ]]; then
    local root="$FULL_DATA_ROOT"
    local count
    count="$(find "$root" -maxdepth 1 -type f -name '*.tar' | wc -l)"
    [[ "$count" -eq "$shard_count" ]] || {
      echo "ERROR: full pool has $count shards; expected $shard_count" >&2
      return 2
    }
    printf '%s\n' "$root/filtered_mixed_train_w*-{000000..000999}.tar"
    return
  fi

  local root="$RANDOM_DATA_ROOT/ratio_${label}.0"
  local count last_shard
  count="$(find "$root" -maxdepth 1 -type f -name '*.tar' | wc -l)"
  [[ "$count" -eq "$shard_count" ]] || {
    echo "ERROR: ratio $label has $count shards; expected $shard_count" >&2
    return 2
  }
  last_shard=$((shard_count - 1))
  printf '%s\n' "$root/filtered_mixed_train_w00-{000000..$(printf '%06d' "$last_shard")}.tar"
}

cd "$REPO"
[[ -x "$PYTHON_BIN" && -f "$TRAIN_SCRIPT" ]] || {
  echo "ERROR: missing Python environment or training launcher" >&2
  exit 2
}

read -r -a requested_labels <<<"$LABEL_STRING"
for label in "${requested_labels[@]}"; do
  case "$label" in 10|20|50|100) ;; *) echo "ERROR: invalid label $label" >&2; exit 2 ;; esac
done

manifest_root="$REPO/outputs/00_reports/splus_bl_fixed_pass_data_scaling_20260810"
mkdir -p "$manifest_root"
manifest="$manifest_root/${MODEL}_training_manifest_${RUN_TAG}.csv"
if [[ ! -f "$manifest" ]]; then
  echo 'model,pool,samples,oel,train_dir,checkpoint_8pass,checkpoint_15pass,seed,global_batch,sigreg' >"$manifest"
fi

for i in "${!labels[@]}"; do
  label="${labels[$i]}"
  contains_label "$label" || continue
  oel="${epoch_lengths[$i]}"
  checkpoint_8=$((8 * oel - 1))
  checkpoint_15=$((15 * oel - 1))

  if [[ "$MODEL" == "L" && "$label" == "100" ]]; then
    output="$RUN_ROOT/L_s6recipe_sigreg005_gb1024_lr5e5_wu2_e15_seed0_4x5090_20260804"
  else
    output="$RUN_ROOT/${MODEL}_s6recipe_sigreg005_datafp_random${label}_e15_gb1024_seed0_${RUN_TAG}_${RUN_DATE}"
  fi

  case "$label" in
    10) pool=0.1M ;;
    20) pool=0.2M ;;
    50) pool=0.5M ;;
    100) pool=1.0M ;;
  esac
  if ! rg -q "^${MODEL},${pool}," "$manifest" 2>/dev/null; then
    echo "$MODEL,$pool,${samples[$i]},$oel,$output,$checkpoint_8,$checkpoint_15,0,1024,0.05" >>"$manifest"
  fi

  if [[ -s "$output/ckpt/$checkpoint_15/checkpoint.pth" ]]; then
    log "random$label already complete at checkpoint $checkpoint_15"
    continue
  fi

  dataset_glob="$(dataset_for_label "$label" "${shards[$i]}")"
  no_resume=1
  [[ -s "$output/config.yaml" ]] && no_resume=0
  log "starting random$label on GPUs [$GPU_GROUP], OEL=$oel, targets=$checkpoint_8/$checkpoint_15"

  CUDA_VISIBLE_DEVICES="$GPU_GROUP" \
  OMP_NUM_THREADS=4 \
  PYTHON_BIN="$PYTHON_BIN" \
  MODEL_SIZE="$SIZE" \
  DATASET_GLOB="$dataset_glob" \
  WEIGHTS_DIR="$WEIGHTS_DIR" \
  OUTPUT_DIR="$output" \
  NPROC_PER_NODE=4 \
  BATCH_SIZE_PER_GPU="$BATCH_SIZE_PER_GPU" \
  GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
  GLOBAL_BATCH_SIZE=1024 \
  EPOCHS=15 \
  LR="$LR" \
  MIN_LR=0.000001 \
  WARMUP_EPOCHS=2 \
  TEACHER_WARMUP_EPOCHS=5 \
  OFFICIAL_EPOCH_LENGTH="$oel" \
  CHECKPOINT_PERIOD="$oel" \
  MAX_TO_KEEP="$MAX_TO_KEEP" \
  SIGREG_ENABLED=true \
  SIGREG_WEIGHT=0.05 \
  TRAIN_SEED=0 \
  NUM_WORKERS="$NUM_WORKERS" \
  MASTER_PORT="$MASTER_PORT" \
  NO_RESUME="$no_resume" \
  DRY_RUN="$DRY_RUN" \
  bash "$TRAIN_SCRIPT"

  if [[ "$DRY_RUN" == "0" ]]; then
    [[ -s "$output/ckpt/$checkpoint_8/checkpoint.pth" ]]
    [[ -s "$output/ckpt/$checkpoint_15/checkpoint.pth" ]]
    log "completed random$label"
  fi
done
