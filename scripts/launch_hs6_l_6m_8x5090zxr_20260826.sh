#!/usr/bin/env bash
# HS6 ViT-L on the 6M mix (0.3 frozen 1M + 0.7 10t_v1), 8x5090-zxr.
# Recipe locked to HS6 L: packwds_robust pct=1,99, bio_safe 256/112,
# GB=1024, LR=1e-4, warmup=3, teacher_warmup=30, no SIGReg, 15 epochs.
# Per-GPU batch 16 is the proven 8x5090 L HS6 setting (bs32 OOM'd).
set -euo pipefail

REPO=${REPO:-/mnt/huawei_deepcad/dinov3}
PYTHON_BIN=${PYTHON_BIN:-/home/lxy/miniconda3/envs/dinov3/bin/python}
WEIGHTS=${WEIGHTS:-/mnt/huawei_deepcad/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_6m_mix1m03_10tv107_8x5090zxr_20260826}
GPU_GROUP=${GPU_GROUP:-0,1,2,3,4,5,6,7}
MASTER_PORT=${MASTER_PORT:-31827}
POLL_SECONDS=${POLL_SECONDS:-60}
DRY_RUN=${DRY_RUN:-0}

NPROC_PER_NODE=8
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-16}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
GLOBAL_BATCH_SIZE=1024
OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH:-5899}
EPOCHS=${EPOCHS:-15}
# Keep resumable optimizer checkpoints every ~1M image visits.  The lighter
# teacher snapshots below are retained every ~0.5M for online evaluation.
CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-977}
CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-8}
EVAL_PERIOD=${EVAL_PERIOD:-488}

DATASET_PATH=${DATASET_PATH:-'mixwds_robust:0.3=/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*.tar||0.7=/mnt/huawei_blm/deepcad_10t_v1/wds_patched_shuffle/filtered_mixed_train*.tar::pct=1,99'}
RGB_MEAN=${RGB_MEAN:-'[0.4353042207404451,0.4297971553424909,0.4330304733388387]'}
RGB_STD=${RGB_STD:-'[0.3518810967670564,0.3483934408968676,0.34717070725260313]'}

log() {
  printf '[%s] [hs6-L-6m-8x5090zxr] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

IFS=',' read -r -a gpu_ids <<<"$GPU_GROUP"
if [[ ${#gpu_ids[@]} -ne $NPROC_PER_NODE ]]; then
  echo "ERROR: GPU_GROUP must contain exactly $NPROC_PER_NODE GPU ids; got $GPU_GROUP" >&2
  exit 2
fi

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$WEIGHTS" ]] || { echo "ERROR: missing initialization weights: $WEIGHTS" >&2; exit 2; }
[[ -f "$REPO/dinov3/configs/train/microscopy_continual_vitl16.yaml" ]] || {
  echo "ERROR: missing ViT-L training config" >&2
  exit 2
}
[[ -d /mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle ]] || {
  echo "ERROR: missing frozen 1M WDS" >&2
  exit 2
}
[[ -d /mnt/huawei_blm/deepcad_10t_v1/wds_patched_shuffle ]] || {
  echo "ERROR: missing 10t_v1 WDS" >&2
  exit 2
}

old_shards=$(find /mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle -maxdepth 1 -type f -name 'filtered_mixed_train_w*.tar' | wc -l)
new_shards=$(find /mnt/huawei_blm/deepcad_10t_v1/wds_patched_shuffle -maxdepth 1 -type f -name 'filtered_mixed_train*.tar' | wc -l)
[[ "$old_shards" -gt 0 && "$new_shards" -gt 0 ]] || {
  echo "ERROR: shard counts old=$old_shards new=$new_shards" >&2
  exit 2
}

if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "ERROR: refusing to overwrite non-empty output: $OUTPUT_DIR" >&2
  exit 2
fi

effective_batch=$((NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))
[[ "$effective_batch" -eq "$GLOBAL_BATCH_SIZE" ]] || {
  echo "ERROR: effective global batch $effective_batch != $GLOBAL_BATCH_SIZE" >&2
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
  "train.dataset_path=$DATASET_PATH"
  train.batch_size_per_gpu="$BATCH_SIZE_PER_GPU"
  train.num_workers=2
  train.seed=0
  train.OFFICIAL_EPOCH_LENGTH="$OFFICIAL_EPOCH_LENGTH"
  train.cache_dataset=false
  train.compile=false
  train.wds_shuffle_buffer=50
  train.prefetch_factor=1
  train.pin_memory=false
  train.checkpointing=false
  student.in_chans=3
  teacher.in_chans=3
  student.enable_channelvit=false
  teacher.enable_channelvit=false
  student.stem_type=null
  teacher.stem_type=null
  student.norm_layer=layernormbf16
  student.pos_embed_rope_rescale_coords=2
  student.pos_embed_rope_dtype=fp32
  "student.resume_from_teacher_chkpt=$WEIGHTS"
  optim.epochs="$EPOCHS"
  optim.scaling_rule=fixed
  optim.lr=0.0001
  optim.min_lr=0.000001
  optim.warmup_epochs=3
  optim.freeze_last_layer_epochs=1
  optim.gradient_accumulation_steps="$GRAD_ACCUM_STEPS"
  teacher.warmup_teacher_temp_epochs=30
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
  evaluation.eval_period_iterations="$EVAL_PERIOD"
  checkpointing.period="$CHECKPOINT_PERIOD"
  checkpointing.max_to_keep="$CHECKPOINT_MAX_TO_KEEP"
  checkpointing.sharded=false
)

log "dataset=$DATASET_PATH"
log "old_shards=$old_shards new_shards=$new_shards"
log "output=$OUTPUT_DIR"
log "HS6: mixwds_robust 0.3/0.7, bio_safe 256/112, LR=1e-4, wu=3, tw=30, nosig"
log "schedule=$EPOCHS epochs x $OFFICIAL_EPOCH_LENGTH updates; final_ckpt=$((EPOCHS * OFFICIAL_EPOCH_LENGTH - 1))"
log "batch=$NPROC_PER_NODE GPUs x $BATCH_SIZE_PER_GPU/GPU x accum $GRAD_ACCUM_STEPS = $effective_batch"
log "snapshots=teacher every $EVAL_PERIOD updates (~$((EVAL_PERIOD * effective_batch)) images); resumable every $CHECKPOINT_PERIOD updates (~$((CHECKPOINT_PERIOD * effective_batch)) images), keep=$CHECKPOINT_MAX_TO_KEEP"
log "mean/std robust mix from mix_1m_0.3_plus_10t_v1_0.7_wds_rgb.json"
printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU_GROUP"
printf ' %q' "${cmd[@]}"
printf '\n'

[[ "$DRY_RUN" == 0 ]] || exit 0

while true; do
  busy=()
  for gpu in "${gpu_ids[@]}"; do
    pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | paste -sd, -)
    [[ -z "$pids" ]] || busy+=("gpu${gpu}:${pids}")
  done
  if [[ ${#busy[@]} -eq 0 ]]; then
    break
  fi
  log "waiting for selected GPUs: ${busy[*]}"
  sleep "$POLL_SECONDS"
done

mkdir -p "$(dirname "$OUTPUT_DIR")"
cd "$REPO"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
ulimit -n 65536 2>/dev/null || true

log "selected GPUs are idle; launching training"
export CUDA_VISIBLE_DEVICES="$GPU_GROUP"
exec "${cmd[@]}"
