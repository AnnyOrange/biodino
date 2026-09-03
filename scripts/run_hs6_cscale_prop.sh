#!/usr/bin/env bash
# C-duration e1 -> e2 -> e4 serial, using the e15-anchored proportional
# schedule (launch_hs6_cscale_duration_prop.sh).
#
# Supersedes run_hs6_cscale_{splus_xr_e4,b_hxw_e124}.sh, whose absolute
# warmup/freeze epoch counts made each duration traverse a different schedule
# (e2 ran at 50% warmup with the last layer frozen for half the run).
#
# Per-model LR / config / weights are pinned here so every host trains the
# same thing; host-specific paths come from env.
#
#   MODEL=Splus|B|L|Hplus  (required)
#   REPO PYTHON_BIN WEIGHTS DATA_ROOT OUT_ROOT HOST_TAG  (host-specific)
#   EPOCH_LIST  (default "1 2 4")
set -euo pipefail

MODEL=${MODEL:?set MODEL=Splus|B|L|Hplus}
REPO=${REPO:?set REPO}
PYTHON_BIN=${PYTHON_BIN:?set PYTHON_BIN}
WEIGHTS=${WEIGHTS:?set WEIGHTS}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT}
OUT_ROOT=${OUT_ROOT:-$REPO/outputs/01_training_runs}
HOST_TAG=${HOST_TAG:?set HOST_TAG, e.g. 8x5090xr}
DATE_TAG=${DATE_TAG:-$(date '+%Y%m%d')}
EPOCH_LIST=${EPOCH_LIST:-"1 2 4"}
SHARD_COUNT_EXPECTED=${SHARD_COUNT_EXPECTED:-326}

case "$MODEL" in
  Splus) LR=0.0002;   LR_TAG=lr2e4;    CONFIG_FILE=dinov3/configs/train/microscopy_continual_vits16.yaml ;;
  B)     LR=0.00015;  LR_TAG=lr1p5e4;  CONFIG_FILE=dinov3/configs/train/microscopy_continual_vitb16_robust.yaml ;;
  L)     LR=0.0001;   LR_TAG=lr1e4;    CONFIG_FILE=dinov3/configs/train/microscopy_continual_vitl16.yaml ;;
  Hplus) LR=0.00005;  LR_TAG=lr5e5;    CONFIG_FILE=dinov3/configs/train/microscopy_continual_vith16plus.yaml ;;
  *) echo "ERROR: unknown MODEL=$MODEL" >&2; exit 2 ;;
esac

DATASET_PATH=${DATASET_PATH:-"packwds_robust:${DATA_ROOT}/filtered_mixed_train_w*-{000000..000999}.tar::pct=1,99"}
LOG=${LOG:-$OUT_ROOT/../auto_eval_logs/hs6_cscale_prop_${MODEL,,}_${HOST_TAG}_${DATE_TAG}.log}

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

log() { printf '[%s] [cscale-prop-%s-%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$MODEL" "$HOST_TAG" "$*"; }

LAUNCH=${LAUNCH:-$REPO/scripts/launch_hs6_cscale_duration_prop.sh}
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing python $PYTHON_BIN" >&2; exit 2; }
[[ -f "$LAUNCH" ]] || { echo "ERROR: missing $LAUNCH" >&2; exit 2; }
[[ -s "$WEIGHTS" ]] || { echo "ERROR: missing weights $WEIGHTS" >&2; exit 2; }

shard_count=$(find "$DATA_ROOT" -maxdepth 1 -type f -name 'filtered_mixed_train_w*.tar' | wc -l)
[[ "$shard_count" -eq "$SHARD_COUNT_EXPECTED" ]] || {
  echo "ERROR: expected $SHARD_COUNT_EXPECTED 1M shards, found $shard_count" >&2; exit 2
}

export MODEL REPO PYTHON_BIN WEIGHTS DATASET_PATH CONFIG_FILE LR
export GPU_GROUP=${GPU_GROUP:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-64}
export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
export MASTER_PORT=${MASTER_PORT:-31975}
export WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

log "start $MODEL C-scale (proportional schedule) shards=$shard_count epochs='$EPOCH_LIST'"
for epochs in $EPOCH_LIST; do
  export EPOCHS=$epochs
  export OUTPUT_DIR="$OUT_ROOT/HS6_Cscale_${MODEL}_robust_biosafe256_gb1024_${LR_TAG}_prop15_nosig_e${epochs}_random100_seed0_${HOST_TAG}_${DATE_TAG}"
  log "launch $MODEL e$epochs -> $OUTPUT_DIR"
  bash "$LAUNCH"
done
log "all $MODEL C-scale proportional runs finished"
