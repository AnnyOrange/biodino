#!/usr/bin/env bash
# Full-budget HS6 ViT-L on 4xA100 with anchor-relative relation transport from
# the label-free HS6 S+ scout. The launcher also supports short smoke runs.
set -euo pipefail

REPO=${REPO:-/mnt/huawei_deepcad/dinov3}
PYTHON_BIN=${PYTHON_BIN:-/home/deepcad/anaconda3/envs/dinov3/bin/python}
LARGE_ANCHOR=${LARGE_ANCHOR:-/mnt/huawei_deepcad/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}
SCOUT_ANCHOR=${SCOUT_ANCHOR:-/mnt/huawei_deepcad/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth}
SCOUT_DIR=${SCOUT_DIR:-$REPO/outputs/01_training_runs/HS6_Splus_robust_biosafe256_gb1024_lr2e4_wu3_tw30_nosig_e15_seed0_8x5090xr_20260821b}
SCOUT_CHECKPOINT=${SCOUT_CHECKPOINT:-$SCOUT_DIR/ckpt/12299/checkpoint.pth}
SCOUT_CONFIG=${SCOUT_CONFIG:-$SCOUT_DIR/config.yaml}

TARGET_MODE=${TARGET_MODE:-stable_delta}
LOSS_WEIGHT=${LOSS_WEIGHT:-0.5}
GPU_GROUP=${GPU_GROUP:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-31843}
POLL_SECONDS=${POLL_SECONDS:-60}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1}
DRY_RUN=${DRY_RUN:-0}

BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-64}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
EXPECTED_GLOBAL_BATCH_SIZE=${EXPECTED_GLOBAL_BATCH_SIZE:-1024}
OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH:-1025}
EPOCHS=${EPOCHS:-15}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-3}
TEACHER_WARMUP_EPOCHS=${TEACHER_WARMUP_EPOCHS:-30}
FREEZE_LAST_LAYER_EPOCHS=${FREEZE_LAST_LAYER_EPOCHS:-1}
CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-1025}
CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-16}
EVAL_PERIOD=${EVAL_PERIOD:-0}
SEED=${SEED:-0}

DATASET_PATH=${DATASET_PATH:-'packwds_robust:/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*-*.tar::pct=1,99'}
RGB_MEAN=${RGB_MEAN:-'[0.514666,0.488834,0.498267]'}
RGB_STD=${RGB_STD:-'[0.338707,0.339202,0.336091]'}
RUN_TAG=${RUN_TAG:-HS6_L_scout_${TARGET_MODE}_w${LOSS_WEIGHT}_splus12ep_bs${BATCH_SIZE_PER_GPU}_gb${EXPECTED_GLOBAL_BATCH_SIZE}_e${EPOCHS}_seed${SEED}_4xa100_deepcad_20260826}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/outputs/01_training_runs/$RUN_TAG}

log() {
  printf '[%s] [hs6-L-scout] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

IFS=',' read -r -a gpu_ids <<<"$GPU_GROUP"
if [[ ${#gpu_ids[@]} -ne $NPROC_PER_NODE ]]; then
  echo "ERROR: GPU_GROUP has ${#gpu_ids[@]} ids but NPROC_PER_NODE=$NPROC_PER_NODE" >&2
  exit 2
fi
case "$TARGET_MODE" in
  delta|stable_delta) ;;
  *) echo "ERROR: full training only accepts causal target modes delta or stable_delta" >&2; exit 2 ;;
esac

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing Python: $PYTHON_BIN" >&2; exit 2; }
for path in "$LARGE_ANCHOR" "$SCOUT_ANCHOR" "$SCOUT_CHECKPOINT" "$SCOUT_CONFIG"; do
  [[ -s "$path" ]] || { echo "ERROR: missing required file: $path" >&2; exit 2; }
done
[[ -f "$REPO/dinov3/configs/train/microscopy_continual_vitl16.yaml" ]] || {
  echo "ERROR: missing ViT-L training config" >&2
  exit 2
}
[[ -d /mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle ]] || {
  echo "ERROR: missing frozen HS6 1M WDS" >&2
  exit 2
}

if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "ERROR: refusing to overwrite non-empty output: $OUTPUT_DIR" >&2
  exit 2
fi

effective_batch=$((NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))
[[ "$effective_batch" -eq "$EXPECTED_GLOBAL_BATCH_SIZE" ]] || {
  echo "ERROR: effective global batch $effective_batch != expected $EXPECTED_GLOBAL_BATCH_SIZE" >&2
  exit 2
}

cmd=(
  "$PYTHON_BIN" -m torch.distributed.run
  --nnodes=1
  --node_rank=0
  --nproc_per_node="$NPROC_PER_NODE"
  --master_addr=127.0.0.1
  --master_port="$MASTER_PORT"
  dinov3/train/train.py
  --config-file dinov3/configs/train/microscopy_continual_vitl16.yaml
  --output-dir "$OUTPUT_DIR"
  --no-resume
  compute_precision.distributed_mode=ddp
  "train.dataset_path=$DATASET_PATH"
  train.batch_size_per_gpu="$BATCH_SIZE_PER_GPU"
  train.num_workers=2
  train.seed="$SEED"
  train.OFFICIAL_EPOCH_LENGTH="$OFFICIAL_EPOCH_LENGTH"
  train.wds_deterministic_resampling=false
  train.cache_dataset=false
  train.compile=false
  train.wds_shuffle_buffer=50
  train.prefetch_factor=1
  train.pin_memory=false
  train.checkpointing=true
  train.checkpointing_full=false
  student.in_chans=3
  teacher.in_chans=3
  student.enable_channelvit=false
  teacher.enable_channelvit=false
  student.stem_type=null
  teacher.stem_type=null
  student.norm_layer=layernormbf16
  student.pos_embed_rope_rescale_coords=2
  student.pos_embed_rope_dtype=fp32
  "student.resume_from_teacher_chkpt=$LARGE_ANCHOR"
  optim.epochs="$EPOCHS"
  optim.scaling_rule=fixed
  optim.lr=0.0001
  optim.min_lr=0.000001
  optim.warmup_epochs="$WARMUP_EPOCHS"
  optim.freeze_last_layer_epochs="$FREEZE_LAST_LAYER_EPOCHS"
  optim.gradient_accumulation_steps="$GRAD_ACCUM_STEPS"
  teacher.warmup_teacher_temp_epochs="$TEACHER_WARMUP_EPOCHS"
  crops.global_crops_size=256
  crops.local_crops_size=112
  crops.augmentation_policy=bio_safe
  crops.horizontal_flips=true
  crops.float_input=false
  "crops.rgb_mean=$RGB_MEAN"
  "crops.rgb_std=$RGB_STD"
  sigreg.enabled=false
  channel_subset.enabled=false
  gram.use_loss=false
  gram.compute_stats=false
  scout_kernel_transport.enabled=true
  scout_kernel_transport.loss_weight="$LOSS_WEIGHT"
  scout_kernel_transport.current_feature_protocol=mask_matched
  scout_kernel_transport.directional_damping=0.1
  scout_kernel_transport.displacement_budget_ratio=0.0
  scout_kernel_transport.target_mode="$TARGET_MODE"
  scout_kernel_transport.stable_relative_eigenvalue=0.05
  scout_kernel_transport.stable_min_eigenvalue=0.000001
  "scout_kernel_transport.scout_config_path=$SCOUT_CONFIG"
  "scout_kernel_transport.scout_anchor_checkpoint=$SCOUT_ANCHOR"
  "scout_kernel_transport.scout_adapted_checkpoint=$SCOUT_CHECKPOINT"
  evaluation.eval_period_iterations="$EVAL_PERIOD"
  checkpointing.period="$CHECKPOINT_PERIOD"
  checkpointing.max_to_keep="$CHECKPOINT_MAX_TO_KEEP"
  checkpointing.sharded=false
)

log "output=$OUTPUT_DIR"
log "target=$TARGET_MODE weight=$LOSS_WEIGHT scout=$SCOUT_CHECKPOINT"
log "HS6 baseline held fixed: robust pct=1,99, bio_safe 256/112, LR=1e-4, wu=$WARMUP_EPOCHS, tw=$TEACHER_WARMUP_EPOCHS, no SIGReg"
log "schedule=$EPOCHS x $OFFICIAL_EPOCH_LENGTH; final_ckpt=$((EPOCHS * OFFICIAL_EPOCH_LENGTH - 1))"
log "batch=$NPROC_PER_NODE x $BATCH_SIZE_PER_GPU x accum $GRAD_ACCUM_STEPS = $effective_batch"
printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU_GROUP"
printf ' %q' "${cmd[@]}"
printf '\n'

[[ "$DRY_RUN" == 0 ]] || exit 0

if [[ "$WAIT_FOR_GPUS" == 1 ]]; then
  while true; do
    busy=()
    for gpu in "${gpu_ids[@]}"; do
      pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | paste -sd, -)
      [[ -z "$pids" ]] || busy+=("gpu${gpu}:${pids}")
    done
    [[ ${#busy[@]} -ne 0 ]] || break
    log "waiting for selected GPUs: ${busy[*]}"
    sleep "$POLL_SECONDS"
  done
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
cd "$REPO"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
ulimit -n 65536 2>/dev/null || true

log "launching training"
export CUDA_VISIBLE_DEVICES="$GPU_GROUP"
exec "${cmd[@]}"
