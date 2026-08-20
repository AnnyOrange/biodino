#!/usr/bin/env bash
set -euo pipefail

# H-S0 RGB ViT-L + NRI on eight 5090s at 5090-hxw-xzj.
# Score the finished run against historical H-S0 L ck 15374, not a new control.

REPO=${REPO:-/home/xzj/biodino}
PYTHON_BIN=${PYTHON_BIN:-/home/xzj/miniconda3/envs/dinov3/bin/python}
DATA_ROOT=${DATA_ROOT:-/mnt/data/microscopy-100k-patched}
DATASET_GLOB=${DATASET_GLOB:-$DATA_ROOT/filtered_mixed_train_w*-*.tar}
if [[ -z "${WEIGHTS:-}" ]]; then
  if [[ -f /mnt/data/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth ]]; then
    WEIGHTS=/mnt/data/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  else
    WEIGHTS=/data/xuzijing/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  fi
fi
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/outputs/01_training_runs/nri_hs0_rgb_l_e15_gb1024_8x5090hxw}
GPU_GROUP=${GPU_GROUP:-0,1,2,3,4,5,6,7}
MASTER_PORT=${MASTER_PORT:-31951}
SUBMIT=${SUBMIT:-1}
DRY_RUN=${DRY_RUN:-0}

NPROC_PER_NODE=8
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-16}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
GLOBAL_BATCH_SIZE=1024
OFFICIAL_EPOCH_LENGTH=1025
EPOCHS=15
WARMUP_EPOCHS=3
FREEZE_LAST_LAYER_EPOCHS=1

log() {
  printf '[%s] [nri-hs0-rgb-8x5090hxw] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

IFS=',' read -r -a gpu_ids <<<"$GPU_GROUP"
if [[ ${#gpu_ids[@]} -ne $NPROC_PER_NODE ]]; then
  echo "ERROR: GPU_GROUP must contain exactly $NPROC_PER_NODE ids; got $GPU_GROUP" >&2
  exit 2
fi

effective_batch=$((NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))
[[ "$effective_batch" -eq "$GLOBAL_BATCH_SIZE" ]] || {
  echo "ERROR: effective global batch $effective_batch != $GLOBAL_BATCH_SIZE" >&2
  exit 2
}

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$WEIGHTS" ]] || { echo "ERROR: missing weights: $WEIGHTS" >&2; exit 2; }
[[ -f "$REPO/dinov3/configs/train/microscopy_continual_vitl16_nri.yaml" ]] || {
  echo "ERROR: missing NRI RGB config" >&2
  exit 2
}
count_shards() {
  local glob="$1"
  local -a matches=()
  shopt -s nullglob
  matches=($glob)
  shopt -u nullglob
  echo "${#matches[@]}"
}
shard_count=$(count_shards "$DATASET_GLOB")
[[ "$shard_count" -gt 0 ]] || {
  echo "ERROR: no shards match DATASET_GLOB=$DATASET_GLOB" >&2
  exit 2
}

if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "ERROR: refusing to overwrite non-empty output: $OUTPUT_DIR" >&2
  exit 2
fi

log "output=$OUTPUT_DIR"
log "gpus=$GPU_GROUP recipe=H-S0 RGB NRI crop=224/96 lr=1e-4 wu=$WARMUP_EPOCHS tw=30"
log "batch=$NPROC_PER_NODE x $BATCH_SIZE_PER_GPU x accum $GRAD_ACCUM_STEPS = $effective_batch"
log "schedule=$EPOCHS x $OFFICIAL_EPOCH_LENGTH updates shards=$shard_count"
log "weights=$WEIGHTS"

[[ "$DRY_RUN" == 0 ]] || exit 0

mkdir -p "$OUTPUT_DIR"
cd "$REPO"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_GROUP"
ulimit -n 65536 2>/dev/null || true

cmd=(
  "$PYTHON_BIN" -m torch.distributed.run
  --nnodes=1
  --node_rank=0
  --nproc_per_node="$NPROC_PER_NODE"
  --master_addr=127.0.0.1
  --master_port="$MASTER_PORT"
  dinov3/train/train.py
  --config-file dinov3/configs/train/microscopy_continual_vitl16_nri.yaml
  --output-dir "$OUTPUT_DIR"
  --no-resume
  "train.dataset_path=packwds:$DATASET_GLOB"
  train.batch_size_per_gpu="$BATCH_SIZE_PER_GPU"
  train.num_workers=2
  train.seed=0
  train.OFFICIAL_EPOCH_LENGTH="$OFFICIAL_EPOCH_LENGTH"
  train.cache_dataset=false
  train.compile=false
  train.prefetch_factor=1
  train.pin_memory=false
  train.checkpointing=false
  student.in_chans=3
  teacher.in_chans=3
  student.enable_channelvit=false
  teacher.enable_channelvit=false
  student.stem_type=null
  teacher.stem_type=null
  student.norm_layer=layernorm
  "student.resume_from_teacher_chkpt=$WEIGHTS"
  optim.epochs="$EPOCHS"
  optim.scaling_rule=fixed
  optim.lr=0.0001
  optim.min_lr=0.000001
  optim.warmup_epochs="$WARMUP_EPOCHS"
  optim.freeze_last_layer_epochs="$FREEZE_LAST_LAYER_EPOCHS"
  optim.gradient_accumulation_steps="$GRAD_ACCUM_STEPS"
  teacher.warmup_teacher_temp_epochs=30
  crops.global_crops_size=224
  crops.local_crops_size=96
  crops.augmentation_policy=dinov3
  crops.float_input=false
  'crops.rgb_mean=[0.511375,0.598449,0.683452]'
  'crops.rgb_std=[0.340017,0.306132,0.284308]'
  sigreg.enabled=false
  channel_subset.enabled=false
  nested_channel_innovation.enabled=false
  nested_resolution_innovation.enabled=true
  gram.use_loss=false
  gram.compute_stats=false
  evaluation.eval_period_iterations=0
  checkpointing.period="$OFFICIAL_EPOCH_LENGTH"
  checkpointing.max_to_keep=3
  checkpointing.sharded=false
)

if [[ "$SUBMIT" == 1 ]]; then
  log "submitting with nohup"
  nohup "${cmd[@]}" > "$OUTPUT_DIR/train.log" 2>&1 &
  echo $! > "$OUTPUT_DIR/train.pid"
  log "pid=$(cat "$OUTPUT_DIR/train.pid") log=$OUTPUT_DIR/train.log"
  exit 0
fi

log "running in foreground"
exec "${cmd[@]}"
