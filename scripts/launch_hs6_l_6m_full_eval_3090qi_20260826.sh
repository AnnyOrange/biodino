#!/usr/bin/env bash
# Online full downstream matrix for the HS6 ViT-L 6M mixed-data run.
set -euo pipefail

REPO=${REPO:-/mnt/huawei_deepcad/dinov3}
TRAIN_DIR=${TRAIN_DIR:-$REPO/outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_6m_mix1m03_10tv107_8x5090zxr_20260826}
INPUT_ROOT=${INPUT_ROOT:-$REPO/outputs/02_eval_inputs/hs6_l_6m_full_1m_20260826}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/outputs/02_eval_runs/hs6_l_6m_full_1m_3090fleet_20260826}
PYTHON_BIN=${PYTHON_BIN:-/home/bbnc/anaconda3/envs/dinov3/bin/python}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-/mnt/huawei_deepcad/benchmark}
# GPU 0 is reserved for the external-FM gap-fill queue on 3090-qi.
GPUS=${GPUS:-1 2 3 4 5 6 7}
OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH:-5899}
EPOCHS=${EPOCHS:-15}
TEACHER_SNAPSHOT_PERIOD=${TEACHER_SNAPSHOT_PERIOD:-488}
EVAL_PERIOD=${EVAL_PERIOD:-976}
POLL_SECONDS=${POLL_SECONDS:-30}

log() {
  printf '[%s] [hs6-L-6m-full] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -d "$BENCHMARK_ROOT" ]] || { echo "ERROR: missing benchmark root: $BENCHMARK_ROOT" >&2; exit 2; }
[[ $((EVAL_PERIOD % TEACHER_SNAPSHOT_PERIOD)) -eq 0 ]] || {
  echo "ERROR: EVAL_PERIOD must be a multiple of TEACHER_SNAPSHOT_PERIOD" >&2
  exit 2
}

total_updates=$((OFFICIAL_EPOCH_LENGTH * EPOCHS))
expected=()
for ((updates=EVAL_PERIOD; updates<=total_updates; updates+=EVAL_PERIOD)); do
  expected+=("$((updates - 1))")
done
EXPECTED_CKPTS="${expected[*]}"

mkdir -p "$INPUT_ROOT" "$OUTPUT_DIR"
for checkpoint in "${expected[@]}"; do
  adapter_dir="$INPUT_ROOT/$checkpoint"
  source="$TRAIN_DIR/eval/training_$checkpoint/teacher_checkpoint.pth"
  mkdir -p "$adapter_dir"
  if [[ ! -e "$adapter_dir/checkpoint.pth" && ! -L "$adapter_dir/checkpoint.pth" ]]; then
    ln -s "$source" "$adapter_dir/checkpoint.pth"
  fi
done

manifest="$INPUT_ROOT/checkpoint_curve.tsv"
if [[ ! -f "$manifest" ]]; then
  printf 'checkpoint_id\timage_visits\tepoch_float\tkind\tsource\n' > "$manifest"
  for checkpoint in "${expected[@]}"; do
    updates=$((checkpoint + 1))
    "$PYTHON_BIN" - "$checkpoint" "$updates" "$OFFICIAL_EPOCH_LENGTH" "$TRAIN_DIR" >> "$manifest" <<'PY'
import sys

checkpoint, updates, epoch_length, train_dir = sys.argv[1:]
source = f"{train_dir}/eval/training_{checkpoint}/teacher_checkpoint.pth"
print(f"{checkpoint}\t{updates * 1024}\t{int(updates) / int(epoch_length):.8f}\tteacher\t{source}")
PY
  done
fi

while [[ ! -s "$TRAIN_DIR/config.yaml" ]]; do
  log "waiting for training config: $TRAIN_DIR/config.yaml"
  sleep "$POLL_SECONDS"
done

log "watching ${#expected[@]} full-matrix points at ~1M-image spacing"
log "tasks=classification regression retrieval detection segmentation ood"
log "train=$TRAIN_DIR"
log "output=$OUTPUT_DIR"
log "gpus=$GPUS"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TRAIN_CONFIG="$TRAIN_DIR/config.yaml"
export CKPT_DIR="$INPUT_ROOT"
export EXPECTED_CKPTS
export ALLOW_MISSING_TRAINING=1
export PRUNE_COMPLETED_CACHE=1
export PYTHON_BIN BENCHMARK_ROOT GPUS POLL_SECONDS
export TASKS="classification regression retrieval detection segmentation ood"
export CLASSIFICATION_DATASETS="bloodmnist pathmnist tissuemnist breastmnist organamnist organcmnist organsmnist dermamnist octmnist pneumoniamnist retinamnist chestmnist bbbc048-cellcycle cyclops-protein-loc midog25-atypical pcam nct-crc-he lc25000 chammi-allen-task1 chammi-allen-task2 chammi-cp-task1 chammi-cp-task2 chammi-cp-task3 chammi-hpa-task1 chammi-hpa-task2"
export REGRESSION_DATASETS="bbbc013 bbbc005 conic-cell-count livecell-cell-count"
export RETRIEVAL_DATASETS="lc25000 nct-crc-he-100 nct-crc-he-1k crc-val-he-7k hpa-subcellular rxrx1-cross"
export DETECTION_DATASETS="livecell bbbc038 conic"
export SEGMENTATION_DATASETS="bbbc038 conic monuseg pannuke tissuenet livecell multimodal_cellseg cellpose"
export FROZEN_CHANNEL_POLICY=auto
export FROZEN_CHANNEL_TTA_SAMPLES=8
export SEGMENTATION_CHANNEL_POLICY=auto
export SEGMENTATION_CHANNEL_TTA_SAMPLES=8
export SEGMENTATION_PROTOCOL=best
export FROZEN_BATCH_SIZE=16
export SEG_FEATURE_BATCH_SIZE=8
export SEG_PROBE_BATCH_SIZE=8
export DET_BATCH_SIZE=8
export OOD_DEVICE=cuda
export OOD_BATCH_SIZE=16
export FROZEN_DATASETS_PER_JOB=1
export SEGMENTATION_DATASETS_PER_JOB=1
export JOBS_PER_GPU=2
export MAX_CONCURRENT_JOBS=16
export MAX_CPU_JOBS=16
export NUM_WORKERS=2
export SEG_FEATURE_NUM_WORKERS=2
export SEG_PROBE_NUM_WORKERS=2
export OOD_NUM_WORKERS=2
export DRY_RUN=0

exec bash docs/scaling_law/bio_sweet_spot/scripts/25_watch_b_online_full_eval.sh \
  "$TRAIN_DIR" "$OUTPUT_DIR" hs6-L-6m-full
