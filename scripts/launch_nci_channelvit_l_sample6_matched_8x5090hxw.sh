#!/usr/bin/env bash
set -euo pipefail

# Matched ChannelViT-L sample6 continue on eight 5090s at 5090-hxw-xzj.
# GPUs 0-3: control (NCI off). GPUs 4-7: masked_shared NCI w=0.5.
# 4 x 16 x accum 16 = global batch 1024 per arm. Do not score against
# Residual-MC or RGB H-S0 / H-S6.

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
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO/outputs/01_training_runs/nci_chvit_l_sample6_matched_e15_gb1024_8x5090hxw}
GPU_CONTROL=${GPU_CONTROL:-0,1,2,3}
GPU_NCI=${GPU_NCI:-4,5,6,7}
MASTER_PORT_CONTROL=${MASTER_PORT_CONTROL:-31961}
MASTER_PORT_NCI=${MASTER_PORT_NCI:-31962}
SUBMIT=${SUBMIT:-1}
DRY_RUN=${DRY_RUN:-0}

NPROC_PER_ARM=4
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-16}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16}
GLOBAL_BATCH_SIZE=1024
OFFICIAL_EPOCH_LENGTH=1025
EPOCHS=15
SAMPLE_CHANNELS=6

log() {
  printf '[%s] [nci-chvit-l-sample6-8x5090hxw] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

check_gpu_group() {
  local group="$1" expected="$2" label="$3"
  local -a ids
  IFS=',' read -r -a ids <<<"$group"
  if [[ ${#ids[@]} -ne $expected ]]; then
    echo "ERROR: $label must contain exactly $expected ids; got $group" >&2
    exit 2
  fi
}

count_shards() {
  local glob="$1"
  local -a matches=()
  shopt -s nullglob
  matches=($glob)
  shopt -u nullglob
  echo "${#matches[@]}"
}

check_gpu_group "$GPU_CONTROL" "$NPROC_PER_ARM" GPU_CONTROL
check_gpu_group "$GPU_NCI" "$NPROC_PER_ARM" GPU_NCI

effective_batch=$((NPROC_PER_ARM * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))
[[ "$effective_batch" -eq "$GLOBAL_BATCH_SIZE" ]] || {
  echo "ERROR: effective global batch $effective_batch != $GLOBAL_BATCH_SIZE" >&2
  exit 2
}

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$WEIGHTS" ]] || { echo "ERROR: missing weights: $WEIGHTS" >&2; exit 2; }
[[ -f "$REPO/dinov3/configs/train/microscopy_continual_vitl16_nci_control.yaml" ]] || {
  echo "ERROR: missing NCI control config" >&2
  exit 2
}
[[ -f "$REPO/dinov3/configs/train/microscopy_continual_vitl16_nci.yaml" ]] || {
  echo "ERROR: missing NCI config" >&2
  exit 2
}
shard_count=$(count_shards "$DATASET_GLOB")
[[ "$shard_count" -gt 0 ]] || {
  echo "ERROR: no shards match DATASET_GLOB=$DATASET_GLOB" >&2
  exit 2
}

control_dir="$OUTPUT_ROOT/control"
nci_dir="$OUTPUT_ROOT/nci_masked_shared_w05"
for out in "$control_dir" "$nci_dir"; do
  if [[ -d "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "ERROR: refusing to overwrite non-empty output: $out" >&2
    exit 2
  fi
done

dataset_path="packwds_chvit_robust:${DATASET_GLOB}::sample_channels=${SAMPLE_CHANNELS},min_channels=1,pct=1,99"

log "output_root=$OUTPUT_ROOT"
log "control_gpus=$GPU_CONTROL nci_gpus=$GPU_NCI"
log "carrier=ChannelViT-L sample${SAMPLE_CHANNELS} protocol=masked_shared w=0.5"
log "batch=$NPROC_PER_ARM x $BATCH_SIZE_PER_GPU x accum $GRAD_ACCUM_STEPS = $effective_batch per arm"
log "schedule=$EPOCHS x $OFFICIAL_EPOCH_LENGTH updates shards=$shard_count"
log "weights=$WEIGHTS"

build_cmd() {
  local config_file="$1" output_dir="$2" nproc="$3" master_port="$4" nci_enabled="$5"
  cmd=(
    "$PYTHON_BIN" -m torch.distributed.run
    --nnodes=1
    --node_rank=0
    --nproc_per_node="$nproc"
    --master_addr=127.0.0.1
    --master_port="$master_port"
    dinov3/train/train.py
    --config-file "$config_file"
    --output-dir "$output_dir"
    --no-resume
    "train.dataset_path=$dataset_path"
    train.batch_size_per_gpu="$BATCH_SIZE_PER_GPU"
    train.num_workers=2
    train.seed=0
    train.OFFICIAL_EPOCH_LENGTH="$OFFICIAL_EPOCH_LENGTH"
    train.cache_dataset=false
    train.compile=false
    train.prefetch_factor=1
    train.pin_memory=false
    train.checkpointing=false
    student.in_chans=8
    teacher.in_chans=8
    student.enable_channelvit=true
    teacher.enable_channelvit=true
    student.stem_type=null
    teacher.stem_type=null
    student.norm_layer=layernorm
    "student.resume_from_teacher_chkpt=$WEIGHTS"
    optim.epochs="$EPOCHS"
    optim.scaling_rule=fixed
    optim.lr=0.0001
    optim.min_lr=0.000001
    optim.warmup_epochs=3
    optim.freeze_last_layer_epochs=1
    optim.gradient_accumulation_steps="$GRAD_ACCUM_STEPS"
    teacher.warmup_teacher_temp_epochs=30
    crops.global_crops_size=224
    crops.local_crops_size=96
    crops.augmentation_policy=dinov3
    crops.float_input=false
    'crops.rgb_mean=[0.511375,0.598449,0.683452,0.119112,0.105159,0.082019,0.419583,0.307523]'
    'crops.rgb_std=[0.340017,0.306132,0.284308,0.187627,0.161061,0.151326,0.157172,0.025357]'
    sigreg.enabled=false
    channel_subset.enabled=false
    nested_channel_innovation.enabled="$nci_enabled"
    nested_channel_innovation.observation_protocol=masked_shared
    nested_channel_innovation.loss_weight=0.5
    nested_resolution_innovation.enabled=false
    gram.use_loss=false
    gram.compute_stats=false
    evaluation.eval_period_iterations=0
    checkpointing.period="$OFFICIAL_EPOCH_LENGTH"
    checkpointing.max_to_keep=3
    checkpointing.sharded=false
  )
}

run_arm() {
  local output_dir="$1" gpu_group="$2"
  shift 2
  mkdir -p "$output_dir"
  (
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
    export CUDA_VISIBLE_DEVICES="$gpu_group"
    ulimit -n 65536 2>/dev/null || true
    if [[ "$SUBMIT" == 1 ]]; then
      nohup "$@" > "$output_dir/train.log" 2>&1 &
    else
      "$@" > "$output_dir/train.log" 2>&1 &
    fi
    echo $! > "$output_dir/train.pid"
  )
}

[[ "$DRY_RUN" == 0 ]] || exit 0

mkdir -p "$control_dir" "$nci_dir"

build_cmd dinov3/configs/train/microscopy_continual_vitl16_nci_control.yaml \
  "$control_dir" "$NPROC_PER_ARM" "$MASTER_PORT_CONTROL" false
run_arm "$control_dir" "$GPU_CONTROL" "${cmd[@]}"
log "control_pid=$(cat "$control_dir/train.pid") log=$control_dir/train.log"

build_cmd dinov3/configs/train/microscopy_continual_vitl16_nci.yaml \
  "$nci_dir" "$NPROC_PER_ARM" "$MASTER_PORT_NCI" true
run_arm "$nci_dir" "$GPU_NCI" "${cmd[@]}"
log "nci_pid=$(cat "$nci_dir/train.pid") log=$nci_dir/train.log"

if [[ "$SUBMIT" != 1 ]]; then
  wait "$(cat "$control_dir/train.pid")"
  wait "$(cat "$nci_dir/train.pid")"
fi
