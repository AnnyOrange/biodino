#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Train one B/L/H+/7B run with the locked S+ microscopy recipe.

The default is DRY_RUN=1. Required environment variables:
  MODEL_SIZE=b|l|hplus|7b
  DATASET_GLOB=/path/to/filtered_mixed_train_w*-{000000..000999}.tar
  WEIGHTS_DIR=/path/to/dinov3_official_weights

Important optional variables:
  INIT_WEIGHTS             Override the model-specific file in WEIGHTS_DIR.
  OUTPUT_DIR               Default: timestamped directory under outputs/01_training_runs.
  LR                       Model-specific center point by default.
  WARMUP_EPOCHS            Default: 2.
  EPOCHS                   Default: 15.
  GLOBAL_BATCH_SIZE        Default: 1024.
  BATCH_SIZE_PER_GPU       Defaults on 8x H100/B200: B=128, L=64, H+=32, 7B=4.
  GRAD_ACCUM_STEPS         Auto-computed to keep global batch 1024.
  SIGREG_ENABLED           Default: true. Set false for the no-SIGReg control.
  SIGREG_WEIGHT            Default: 0.05.
  TRAIN_SEED               Default: 0.
  SHARDED_CHECKPOINT       Defaults to true for 7B and false otherwise.
  NPROC_PER_NODE           Default: detected GPU count, otherwise 8.
  NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT
  SMOKE=1                  One short memory/numerics check (20 iterations).
  SMOKE_UPDATES            Override the smoke length; default: 20.
  NO_RESUME=1              Start clean instead of resuming OUTPUT_DIR.
  DRY_RUN=0                Launch training; default is 1.

Examples:
  MODEL_SIZE=b DATASET_GLOB='/path/to/shards/*.tar' WEIGHTS_DIR=/path/to/weights \
    LR=0.00015 WARMUP_EPOCHS=2 DRY_RUN=1 \
    bash docs/scaling_law/bio_sweet_spot/scripts/12_train_rgb_bioaug_robust.sh

  MODEL_SIZE=7b DATASET_GLOB='/path/to/shards/*.tar' WEIGHTS_DIR=/path/to/weights \
    SMOKE=1 DRY_RUN=0 \
    bash docs/scaling_law/bio_sweet_spot/scripts/12_train_rgb_bioaug_robust.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_SIZE="${MODEL_SIZE:-}"
DATASET_GLOB="${DATASET_GLOB:-}"
WEIGHTS_DIR="${WEIGHTS_DIR:-}"
if [[ -z "$MODEL_SIZE" || -z "$DATASET_GLOB" || ( -z "$WEIGHTS_DIR" && -z "${INIT_WEIGHTS:-}" ) ]]; then
  usage >&2
  echo "ERROR: MODEL_SIZE, DATASET_GLOB, and WEIGHTS_DIR (or INIT_WEIGHTS) are required." >&2
  exit 2
fi

case "$MODEL_SIZE" in
  b)
    MODEL_TAG=vitb16
    CONFIG_FILE="${CONFIG_FILE:-dinov3/configs/train/microscopy_continual_vitb16.yaml}"
    INIT_FILENAME=dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
    DEFAULT_BATCH_SIZE=128
    DEFAULT_LR=0.00010
    DEFAULT_ACTIVATION_CHECKPOINTING=false
    DEFAULT_SHARDED_CHECKPOINT=false
    DEFAULT_MAX_TO_KEEP=30
    SCHEDULE_VERSION=legacy
    ;;
  l)
    MODEL_TAG=vitl16
    CONFIG_FILE="${CONFIG_FILE:-dinov3/configs/train/microscopy_continual_vitl16.yaml}"
    INIT_FILENAME=dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
    DEFAULT_BATCH_SIZE=64
    DEFAULT_LR=0.00005
    DEFAULT_ACTIVATION_CHECKPOINTING=false
    DEFAULT_SHARDED_CHECKPOINT=false
    DEFAULT_MAX_TO_KEEP=30
    SCHEDULE_VERSION=legacy
    ;;
  hplus|h+|h)
    MODEL_SIZE=hplus
    MODEL_TAG=vith16plus
    CONFIG_FILE="${CONFIG_FILE:-dinov3/configs/train/microscopy_continual_vith16plus.yaml}"
    INIT_FILENAME=dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
    DEFAULT_BATCH_SIZE=32
    DEFAULT_LR=0.000025
    DEFAULT_ACTIVATION_CHECKPOINTING=false
    DEFAULT_SHARDED_CHECKPOINT=false
    DEFAULT_MAX_TO_KEEP=30
    SCHEDULE_VERSION=legacy
    ;;
  7b|vit7b)
    MODEL_SIZE=7b
    MODEL_TAG=vit7b16
    CONFIG_FILE="${CONFIG_FILE:-dinov3/configs/train/dinov3_vit7b16_pretrain.yaml}"
    INIT_FILENAME=dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
    DEFAULT_BATCH_SIZE=4
    DEFAULT_LR=0.00001
    DEFAULT_ACTIVATION_CHECKPOINTING=true
    DEFAULT_SHARDED_CHECKPOINT=true
    DEFAULT_MAX_TO_KEEP=3
    SCHEDULE_VERSION=v2
    ;;
  *)
    echo "ERROR: MODEL_SIZE must be b, l, hplus, or 7b; got '$MODEL_SIZE'." >&2
    exit 2
    ;;
esac

INIT_WEIGHTS="${INIT_WEIGHTS:-${WEIGHTS_DIR%/}/${INIT_FILENAME}}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-$DEFAULT_ACTIVATION_CHECKPOINTING}"
LR="${LR:-$DEFAULT_LR}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
TEACHER_WARMUP_EPOCHS="${TEACHER_WARMUP_EPOCHS:-5}"
EPOCHS="${EPOCHS:-15}"
MIN_LR="${MIN_LR:-0.000001}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-$DEFAULT_BATCH_SIZE}"
NUM_WORKERS="${NUM_WORKERS:-10}"
OFFICIAL_EPOCH_LENGTH="${OFFICIAL_EPOCH_LENGTH:-1025}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-1025}"
MAX_TO_KEEP="${MAX_TO_KEEP:-$DEFAULT_MAX_TO_KEEP}"
FREEZE_LAST_LAYER_EPOCHS="${FREEZE_LAST_LAYER_EPOCHS:-1}"
SIGREG_ENABLED="${SIGREG_ENABLED:-true}"
SIGREG_WEIGHT="${SIGREG_WEIGHT:-0.05}"
TRAIN_SEED="${TRAIN_SEED:-0}"
SHARDED_CHECKPOINT="${SHARDED_CHECKPOINT:-$DEFAULT_SHARDED_CHECKPOINT}"
DRY_RUN="${DRY_RUN:-1}"
SMOKE="${SMOKE:-0}"
SMOKE_UPDATES="${SMOKE_UPDATES:-20}"
if ! [[ "$SMOKE_UPDATES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SMOKE_UPDATES must be a positive integer; got '$SMOKE_UPDATES'." >&2
  exit 2
fi
if [[ "$SIGREG_ENABLED" != "true" && "$SIGREG_ENABLED" != "false" ]]; then
  echo "ERROR: SIGREG_ENABLED must be true or false; got '$SIGREG_ENABLED'." >&2
  exit 2
fi

if [[ "$SMOKE" == "1" ]]; then
  EPOCHS=1
  WARMUP_EPOCHS=1
  TEACHER_WARMUP_EPOCHS=1
  OFFICIAL_EPOCH_LENGTH="$SMOKE_UPDATES"
  CHECKPOINT_PERIOD="$SMOKE_UPDATES"
  MAX_TO_KEEP=2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "ERROR: activate the dinov3 environment or set PYTHON_BIN." >&2
  exit 2
fi

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l)"
  else
    NPROC_PER_NODE=8
  fi
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29620}"
TOTAL_GPUS=$(( NNODES * NPROC_PER_NODE ))
PER_MICROBATCH=$(( TOTAL_GPUS * BATCH_SIZE_PER_GPU ))
if [[ -z "${GRAD_ACCUM_STEPS:-}" ]]; then
  if (( GLOBAL_BATCH_SIZE % PER_MICROBATCH != 0 )); then
    echo "ERROR: global batch $GLOBAL_BATCH_SIZE is not divisible by world x batch/GPU ($PER_MICROBATCH)." >&2
    echo "Set BATCH_SIZE_PER_GPU or GRAD_ACCUM_STEPS explicitly." >&2
    exit 2
  fi
  GRAD_ACCUM_STEPS=$(( GLOBAL_BATCH_SIZE / PER_MICROBATCH ))
fi
if (( GRAD_ACCUM_STEPS < 1 )); then
  echo "ERROR: GRAD_ACCUM_STEPS must be at least 1." >&2
  exit 2
fi
EFFECTIVE_GLOBAL_BATCH=$(( PER_MICROBATCH * GRAD_ACCUM_STEPS ))
if (( EFFECTIVE_GLOBAL_BATCH != GLOBAL_BATCH_SIZE )); then
  echo "ERROR: effective global batch is $EFFECTIVE_GLOBAL_BATCH, expected $GLOBAL_BATCH_SIZE." >&2
  exit 2
fi

RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
LR_TAG="${LR//./p}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/01_training_runs/${MODEL_TAG}_rgb_bioaug_robust_lr${LR_TAG}_wu${WARMUP_EPOCHS}_${RUN_STAMP}}"
DATASET_PATH="packwds_robust:${DATASET_GLOB}::pct=1,99"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: config not found: $CONFIG_FILE" >&2
  exit 2
fi
if [[ "$DRY_RUN" == "0" && ! -f "$INIT_WEIGHTS" ]]; then
  echo "ERROR: initialization checkpoint not found: $INIT_WEIGHTS" >&2
  exit 2
fi
if [[ "$DRY_RUN" == "0" ]] && ! "$PYTHON" -c 'import omegaconf, torch' >/dev/null 2>&1; then
  echo "ERROR: $PYTHON is not a usable dinov3 environment; activate it or set PYTHON_BIN." >&2
  exit 2
fi
if [[ "$DRY_RUN" == "0" && "${NO_RESUME:-0}" == "1" && -d "$OUTPUT_DIR" ]]; then
  if [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "ERROR: refusing a clean start in non-empty OUTPUT_DIR: $OUTPUT_DIR" >&2
    echo "Choose a new directory, or set NO_RESUME=0 to resume this run." >&2
    exit 2
  fi
fi

export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
ulimit -n "${OPEN_FILES_LIMIT:-65536}" 2>/dev/null || true

train_flags=()
if [[ "${NO_RESUME:-0}" == "1" ]]; then
  train_flags+=(--no-resume)
fi

cmd=(
  "$PYTHON" -m torch.distributed.run
  --nnodes="$NNODES"
  --node_rank="$NODE_RANK"
  --nproc_per_node="$NPROC_PER_NODE"
  --master_addr="$MASTER_ADDR"
  --master_port="$MASTER_PORT"
  dinov3/train/train.py
  --config-file "$CONFIG_FILE"
  --output-dir "$OUTPUT_DIR"
  "${train_flags[@]}"
  train.dataset_path="$DATASET_PATH"
  train.batch_size_per_gpu="$BATCH_SIZE_PER_GPU"
  train.num_workers="$NUM_WORKERS"
  train.seed="$TRAIN_SEED"
  train.OFFICIAL_EPOCH_LENGTH="$OFFICIAL_EPOCH_LENGTH"
  train.cache_dataset=false
  train.compile=false
  train.wds_shuffle_buffer=50
  train.prefetch_factor=1
  train.pin_memory=false
  train.checkpointing="$ACTIVATION_CHECKPOINTING"
  student.in_chans=3
  teacher.in_chans=3
  student.enable_channelvit=false
  teacher.enable_channelvit=false
  student.stem_type=null
  teacher.stem_type=null
  student.norm_layer=layernormbf16
  student.pos_embed_rope_rescale_coords=2
  student.pos_embed_rope_dtype=fp32
  student.resume_from_teacher_chkpt="$INIT_WEIGHTS"
  optim.epochs="$EPOCHS"
  optim.scaling_rule=fixed
  optim.gradient_accumulation_steps="$GRAD_ACCUM_STEPS"
  crops.global_crops_size=256
  crops.local_crops_size=112
  crops.augmentation_policy=bio_safe
  crops.horizontal_flips=true
  crops.float_input=false
  crops.rgb_mean="[0.514666,0.488834,0.498267]"
  crops.rgb_std="[0.338707,0.339202,0.336091]"
  sigreg.enabled="$SIGREG_ENABLED"
  sigreg.mode=bottleneck
  sigreg.loss_weight="$SIGREG_WEIGHT"
  sigreg.num_slices=1024
  sigreg.range_max=5.0
  sigreg.n_knots=17
  sigreg.koleo_too=false
  channel_subset.enabled=false
  gram.use_loss=false
  gram.compute_stats=false
  checkpointing.period="$CHECKPOINT_PERIOD"
  checkpointing.max_to_keep="$MAX_TO_KEEP"
  checkpointing.sharded="$SHARDED_CHECKPOINT"
)

if [[ "$SCHEDULE_VERSION" == "v2" ]]; then
  cmd+=(
    schedules.lr.start=0
    schedules.lr.peak="$LR"
    schedules.lr.end="$MIN_LR"
    schedules.lr.warmup_epochs="$WARMUP_EPOCHS"
    schedules.lr.freeze_last_layer_epochs="$FREEZE_LAST_LAYER_EPOCHS"
    schedules.teacher_temp.warmup_epochs="$TEACHER_WARMUP_EPOCHS"
  )
else
  cmd+=(
    optim.lr="$LR"
    optim.min_lr="$MIN_LR"
    optim.warmup_epochs="$WARMUP_EPOCHS"
    optim.freeze_last_layer_epochs="$FREEZE_LAST_LAYER_EPOCHS"
    teacher.warmup_teacher_temp_epochs="$TEACHER_WARMUP_EPOCHS"
  )
fi

if [[ "$SIGREG_ENABLED" == "true" ]]; then
  OBJECTIVE_LABEL="DINO+iBOT+SIGReg(${SIGREG_WEIGHT})"
else
  OBJECTIVE_LABEL="DINO+iBOT (SIGReg disabled)"
fi

cat <<INFO
[train-rgb] model=${MODEL_SIZE} tag=${MODEL_TAG} schedule=${SCHEDULE_VERSION}
[train-rgb] output=${OUTPUT_DIR}
[train-rgb] init=${INIT_WEIGHTS}
[train-rgb] input=RGB packwds_robust pct=1,99; aug=bio_safe; crop=256/112; float_input=false
[train-rgb] objective=${OBJECTIVE_LABEL}; SIGReg KoLeo companion/channel_subset/gram=off
[train-rgb] lr=${LR} min_lr=${MIN_LR} warmup=${WARMUP_EPOCHS} teacher_warmup=${TEACHER_WARMUP_EPOCHS}
[train-rgb] seed=${TRAIN_SEED}
[train-rgb] epochs=${EPOCHS} official_epoch_length=${OFFICIAL_EPOCH_LENGTH} checkpoint_period=${CHECKPOINT_PERIOD}
[train-rgb] checkpoint_sharded=${SHARDED_CHECKPOINT} max_to_keep=${MAX_TO_KEEP}
[train-rgb] batch=${TOTAL_GPUS} GPUs x ${BATCH_SIZE_PER_GPU}/GPU x accum ${GRAD_ACCUM_STEPS} = ${EFFECTIVE_GLOBAL_BATCH}
[train-rgb] activation_checkpointing=${ACTIVATION_CHECKPOINTING} smoke=${SMOKE} dry_run=${DRY_RUN}
INFO
printf ' %q' "${cmd[@]}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

exec "${cmd[@]}"
