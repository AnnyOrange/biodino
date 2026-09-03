#!/usr/bin/env bash
# One from-scratch HS6 cosine run that *ends* at EPOCHS (C-duration point).
#
# Difference vs launch_hs6_cscale_duration.sh: every schedule is a fixed
# FRACTION of the run, anchored on the e15 reference shape, so that all
# durations traverse the same normalized schedule and compute is the only
# variable. The old launcher pinned warmup/freeze to absolute epoch counts,
# which made e1..e15 traverse LR trajectories differing by up to a full peak
# LR and put e2 at 50% warmup + 50% frozen last layer.
#
#   e15 reference:  warmup 3/15 = 0.2   temp 30/15 = 2x   freeze 1/15
#   duration T:     warmup 0.2*T        temp 2*T          freeze T/15
#
# Fractional epochs require the int() casts in build_schedulers (train.py).
#
# Required env: MODEL REPO PYTHON_BIN WEIGHTS DATASET_PATH CONFIG_FILE
#               GPU_GROUP NPROC_PER_NODE BATCH_SIZE_PER_GPU GRAD_ACCUM_STEPS
#               EPOCHS OUTPUT_DIR LR
set -euo pipefail

MODEL=${MODEL:?set MODEL=Splus|B|L|Hplus}
REPO=${REPO:?}
PYTHON_BIN=${PYTHON_BIN:?}
WEIGHTS=${WEIGHTS:?}
DATASET_PATH=${DATASET_PATH:?}
CONFIG_FILE=${CONFIG_FILE:?}
GPU_GROUP=${GPU_GROUP:?}
NPROC_PER_NODE=${NPROC_PER_NODE:?}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:?}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:?}
EPOCHS=${EPOCHS:?}
OUTPUT_DIR=${OUTPUT_DIR:?}
LR=${LR:?}

MASTER_PORT=${MASTER_PORT:-31971}
POLL_SECONDS=${POLL_SECONDS:-30}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1}
DRY_RUN=${DRY_RUN:-0}
NUM_WORKERS=${NUM_WORKERS:-2}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH:-1025}
EVAL_PERIOD=${EVAL_PERIOD:-0}
CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-2}

# e15-anchored proportional schedule.
REF_EPOCHS=${REF_EPOCHS:-15}
REF_WARMUP=${REF_WARMUP:-3}
REF_TEACHER_WARMUP=${REF_TEACHER_WARMUP:-30}
REF_FREEZE=${REF_FREEZE:-1}

case "$EPOCHS" in
  1 | 2 | 4 | 8 | 15) ;;
  *) echo "ERROR: EPOCHS must be 1|2|4|8|15, got $EPOCHS" >&2; exit 2 ;;
esac

frac() { awk -v e="$EPOCHS" -v r="$REF_EPOCHS" -v v="$1" 'BEGIN{printf "%.10g", v*e/r}'; }
WARMUP_EPOCHS=${WARMUP_EPOCHS:-$(frac "$REF_WARMUP")}
TEACHER_WARMUP_EPOCHS=${TEACHER_WARMUP_EPOCHS:-$(frac "$REF_TEACHER_WARMUP")}
FREEZE_LAST_LAYER_EPOCHS=${FREEZE_LAST_LAYER_EPOCHS:-$(frac "$REF_FREEZE")}

# Guard the invariants the old launcher violated.
awk -v w="$WARMUP_EPOCHS" -v e="$EPOCHS" 'BEGIN{exit !(w < e)}' || {
  echo "ERROR: warmup_epochs=$WARMUP_EPOCHS must be < EPOCHS=$EPOCHS" >&2; exit 2
}
awk -v t="$TEACHER_WARMUP_EPOCHS" -v e="$EPOCHS" -v r="$REF_TEACHER_WARMUP" -v re="$REF_EPOCHS" \
  'BEGIN{exit !(t/e > 0.999*r/re && t/e < 1.001*r/re)}' || {
  echo "ERROR: teacher temp ramp $TEACHER_WARMUP_EPOCHS not proportional to EPOCHS=$EPOCHS" >&2; exit 2
}

CHECKPOINT_PERIOD=$((EPOCHS * OFFICIAL_EPOCH_LENGTH))
RGB_MEAN=${RGB_MEAN:-'[0.514666,0.488834,0.498267]'}
RGB_STD=${RGB_STD:-'[0.338707,0.339202,0.336091]'}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

log() {
  printf '[%s] [hs6-cscale-prop-%s-e%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$MODEL" "$EPOCHS" "$*"
}

IFS=',' read -r -a gpu_ids <<<"$GPU_GROUP"
if [[ ${#gpu_ids[@]} -ne $NPROC_PER_NODE ]]; then
  echo "ERROR: GPU_GROUP has ${#gpu_ids[@]} ids but NPROC_PER_NODE=$NPROC_PER_NODE" >&2
  exit 2
fi

effective_batch=$((NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))
[[ "$effective_batch" -eq "$GLOBAL_BATCH_SIZE" ]] || {
  echo "ERROR: effective global batch $effective_batch != $GLOBAL_BATCH_SIZE" >&2
  exit 2
}

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -s "$WEIGHTS" ]] || { echo "ERROR: missing init weights: $WEIGHTS" >&2; exit 2; }
[[ -f "$REPO/$CONFIG_FILE" ]] || { echo "ERROR: missing config $REPO/$CONFIG_FILE" >&2; exit 2; }
[[ -f "$REPO/dinov3/train/train.py" ]] || { echo "ERROR: missing repo: $REPO" >&2; exit 2; }

# Fractional warmup/freeze silently corrupt the schedule without the int()
# casts in build_schedulers; refuse to launch against an unpatched repo.
grep -q '_iters(cfg.optim\["warmup_epochs"\])' "$REPO/dinov3/train/train.py" || {
  echo "ERROR: $REPO/dinov3/train/train.py lacks the fractional-epoch _iters() casts" >&2
  exit 2
}

if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "ERROR: refusing to overwrite non-empty output: $OUTPUT_DIR" >&2
  exit 2
fi

cmd=(
  "$PYTHON_BIN" -m torch.distributed.run
  --nnodes=1
  --node_rank=0
  --nproc_per_node="$NPROC_PER_NODE"
  --master_addr=127.0.0.1
  --master_port="$MASTER_PORT"
  dinov3/train/train.py
  --config-file "$CONFIG_FILE"
  --output-dir "$OUTPUT_DIR"
  --no-resume
  "train.dataset_path=$DATASET_PATH"
  train.batch_size_per_gpu="$BATCH_SIZE_PER_GPU"
  train.num_workers="$NUM_WORKERS"
  train.seed=0
  train.OFFICIAL_EPOCH_LENGTH="$OFFICIAL_EPOCH_LENGTH"
  train.cache_dataset=false
  train.compile=false
  train.wds_shuffle_buffer=50
  train.prefetch_factor="$PREFETCH_FACTOR"
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
  optim.lr="$LR"
  optim.min_lr=0.000001
  optim.warmup_epochs="$WARMUP_EPOCHS"
  optim.freeze_last_layer_epochs="$FREEZE_LAST_LAYER_EPOCHS"
  optim.gradient_accumulation_steps="$GRAD_ACCUM_STEPS"
  optim.expected_effective_batch_size="$GLOBAL_BATCH_SIZE"
  optim.lr_reference_effective_batch_size="$GLOBAL_BATCH_SIZE"
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
  evaluation.eval_period_iterations="$EVAL_PERIOD"
  checkpointing.period="$CHECKPOINT_PERIOD"
  checkpointing.max_to_keep="$CHECKPOINT_MAX_TO_KEEP"
  checkpointing.sharded=false
)

log "init=$WEIGHTS"
log "dataset=$DATASET_PATH"
log "output=$OUTPUT_DIR"
log "gpus=$GPU_GROUP  batch=${NPROC_PER_NODE}x${BATCH_SIZE_PER_GPU}xaccum${GRAD_ACCUM_STEPS}=$effective_batch"
log "schedule=e${EPOCHS} x ${OFFICIAL_EPOCH_LENGTH}  wu=${WARMUP_EPOCHS} freeze_last=${FREEZE_LAST_LAYER_EPOCHS} tw=${TEACHER_WARMUP_EPOCHS}"
log "shape anchored on e${REF_EPOCHS}: warmup $(awk -v w="$WARMUP_EPOCHS" -v e="$EPOCHS" 'BEGIN{printf "%.1f", 100*w/e}')% freeze $(awk -v f="$FREEZE_LAST_LAYER_EPOCHS" -v e="$EPOCHS" 'BEGIN{printf "%.1f", 100*f/e}')% of run"
log "ckpt_period=$CHECKPOINT_PERIOD (last iter $((CHECKPOINT_PERIOD - 1))) lr=$LR"
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
    if [[ ${#busy[@]} -eq 0 ]]; then
      break
    fi
    log "waiting for selected GPUs: ${busy[*]}"
    sleep "$POLL_SECONDS"
  done
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU_GROUP"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_P2P_DISABLE
export NCCL_IB_DISABLE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
ulimit -n 65536 2>/dev/null || true

log "selected GPUs are idle; launching C-scale $MODEL e$EPOCHS"
"${cmd[@]}"
log "finished $MODEL e$EPOCHS → $OUTPUT_DIR"
