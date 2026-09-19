#!/usr/bin/env bash
# Direct launcher for HS6 ViT-7B continual training on the 1TB or formal 5TB mix.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_hs6_vit7b.sh 1tb
  bash scripts/train_hs6_vit7b.sh 5tb

The 5TB mode is the formal 30% 1TB replay + 70% 4TB/5t_v1 recipe. It does not
include the experimental 5% H&E branch.

Common overrides:
  OUTPUT_DIR=/path/to/run
  INIT_WEIGHTS=/path/to/dinov3_vit7b16_checkpoint.pth
  PYTHON_BIN=/path/to/python
  NNODES=1 NODE_RANK=0 NPROC_PER_NODE=8
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
  BATCH_SIZE_PER_GPU=8 GLOBAL_BATCH_SIZE=1024
  GRAD_ACCUM_STEPS=16
  LR=3e-5 EPOCHS=15
  NUM_WORKERS=2 PREFETCH_FACTOR=1
  FP8_ENABLED=true TRAIN_COMPILE=false
  EVAL_PERIOD=0 CHECKPOINT_PERIOD=<mode default> CHECKPOINT_MAX_TO_KEEP=2
  RESUME=1                 Resume from OUTPUT_DIR/ckpt when present.
  DRY_RUN=1                Validate and print the command without launching.
  SKIP_DATA_CHECK=1        Skip checks for the default dataset roots.

For multi-node training, run the same command on every node and change only
NODE_RANK. OUTPUT_DIR must be on shared storage.
EOF
}

MODE="${1:-${MODE:-}}"
case "$MODE" in
  -h|--help)
    usage
    exit 0
    ;;
  1tb|5tb)
    ;;
  *)
    usage >&2
    echo "ERROR: mode must be '1tb' or '5tb'." >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
CONFIG_FILE="${CONFIG_FILE:-$REPO/dinov3/configs/train/microscopy_continual_vit7b16_hs6.yaml}"
INIT_WEIGHTS="${INIT_WEIGHTS-/mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth}"

ONE_TB_ROOT="${ONE_TB_ROOT:-/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle}"
FOUR_TB_ROOT="${FOUR_TB_ROOT:-/mnt/huawei_blm/deepcad_5t_v1/wds_patched_shuffle}"

case "$MODE" in
  1tb)
    DEFAULT_DATASET_PATH="packwds_robust:$ONE_TB_ROOT/filtered_mixed_train_w*-*.tar::pct=1,99"
    RGB_MEAN="${RGB_MEAN:-[0.514666,0.488834,0.498267]}"
    RGB_STD="${RGB_STD:-[0.338707,0.339202,0.336091]}"
    OFFICIAL_EPOCH_LENGTH="${OFFICIAL_EPOCH_LENGTH:-1025}"
    DEFAULT_CHECKPOINT_PERIOD=1025
    ;;
  5tb)
    DEFAULT_DATASET_PATH="mixwds_robust:0.3=$ONE_TB_ROOT/filtered_mixed_train_w*.tar||0.7=$FOUR_TB_ROOT/filtered_mixed_train*.tar::pct=1,99"
    RGB_MEAN="${RGB_MEAN:-[0.5126699404721016,0.5020022506395592,0.5064769301636908]}"
    RGB_STD="${RGB_STD:-[0.3497517202150124,0.34941518705400204,0.34802097842537794]}"
    OFFICIAL_EPOCH_LENGTH="${OFFICIAL_EPOCH_LENGTH:-4098}"
    # Preserve the formal 5TB run's roughly 0.5M-image recovery cadence.
    DEFAULT_CHECKPOINT_PERIOD=488
    ;;
esac

DATASET_PATH="${DATASET_PATH:-$DEFAULT_DATASET_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/01_training_runs/HS6_7B_robust_biosafe256_gb1024_lr3e5_wu3_tw30_nosig_e15_${MODE}}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x /home/lxy/miniconda3/envs/dinov3/bin/python ]]; then
  PYTHON=/home/lxy/miniconda3/envs/dinov3/bin/python
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "ERROR: no Python executable found; set PYTHON_BIN." >&2
  exit 2
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    gpu_list="${CUDA_VISIBLE_DEVICES// /}"
    IFS=',' read -r -a visible_gpus <<<"$gpu_list"
    NPROC_PER_NODE="${#visible_gpus[@]}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l)"
  else
    echo "ERROR: cannot determine NPROC_PER_NODE; set it explicitly." >&2
    exit 2
  fi
fi

BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))
PER_MICRO_STEP=$((TOTAL_GPUS * BATCH_SIZE_PER_GPU))

if [[ -n "${GRAD_ACCUM_STEPS:-}" ]]; then
  ACCUM_SOURCE=manual
else
  if (( PER_MICRO_STEP <= 0 || GLOBAL_BATCH_SIZE % PER_MICRO_STEP != 0 )); then
    echo "ERROR: GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be divisible by" >&2
    echo "       NNODES*NPROC_PER_NODE*BATCH_SIZE_PER_GPU=$PER_MICRO_STEP." >&2
    echo "       Set GRAD_ACCUM_STEPS explicitly only if a different effective batch is intentional." >&2
    exit 2
  fi
  GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / PER_MICRO_STEP))
  ACCUM_SOURCE=auto
fi

EFFECTIVE_GLOBAL_BATCH=$((PER_MICRO_STEP * GRAD_ACCUM_STEPS))
if (( EFFECTIVE_GLOBAL_BATCH != GLOBAL_BATCH_SIZE )); then
  echo "ERROR: effective global batch is $EFFECTIVE_GLOBAL_BATCH, expected $GLOBAL_BATCH_SIZE." >&2
  exit 2
fi

LR="${LR:-3e-5}"
EPOCHS="${EPOCHS:-15}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
FP8_ENABLED="${FP8_ENABLED:-true}"
TRAIN_COMPILE="${TRAIN_COMPILE:-false}"
EVAL_PERIOD="${EVAL_PERIOD:-0}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-$DEFAULT_CHECKPOINT_PERIOD}"
CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-2}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DATA_CHECK="${SKIP_DATA_CHECK:-0}"

[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: missing config: $CONFIG_FILE" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "ERROR: Python is not executable: $PYTHON" >&2; exit 2; }
[[ -z "$INIT_WEIGHTS" || -s "$INIT_WEIGHTS" ]] || {
  echo "ERROR: missing or empty initialization checkpoint: $INIT_WEIGHTS" >&2
  exit 2
}
if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == "127.0.0.1" || "$MASTER_ADDR" == "localhost" ]]; then
  echo "ERROR: set MASTER_ADDR to node 0's reachable address for multi-node training." >&2
  exit 2
fi

if [[ "$SKIP_DATA_CHECK" != 1 && "$DATASET_PATH" == "$DEFAULT_DATASET_PATH" ]]; then
  [[ -d "$ONE_TB_ROOT" ]] || { echo "ERROR: missing 1TB root: $ONE_TB_ROOT" >&2; exit 2; }
  if [[ "$MODE" == 5tb ]]; then
    [[ -d "$FOUR_TB_ROOT" ]] || { echo "ERROR: missing 4TB root: $FOUR_TB_ROOT" >&2; exit 2; }
  fi
fi

if [[ "$DRY_RUN" != 1 && "$NODE_RANK" == 0 && "$RESUME" != 1 && -d "$OUTPUT_DIR" ]]; then
  if find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit | read -r _; then
    echo "ERROR: refusing to start a new run in non-empty OUTPUT_DIR: $OUTPUT_DIR" >&2
    echo "       Set RESUME=1 to resume or choose another OUTPUT_DIR." >&2
    exit 2
  fi
fi

resume_args=()
if [[ "$RESUME" != 1 ]]; then
  resume_args+=(--no-resume)
fi

cmd=(
  "$PYTHON" -m torch.distributed.run
  --nnodes="$NNODES"
  --node_rank="$NODE_RANK"
  --nproc_per_node="$NPROC_PER_NODE"
  --master_addr="$MASTER_ADDR"
  --master_port="$MASTER_PORT"
  "$REPO/dinov3/train/train.py"
  --config-file "$CONFIG_FILE"
  --output-dir "$OUTPUT_DIR"
  "${resume_args[@]}"
  "train.dataset_path=$DATASET_PATH"
  train.batch_size_per_gpu="$BATCH_SIZE_PER_GPU"
  train.num_workers="$NUM_WORKERS"
  train.prefetch_factor="$PREFETCH_FACTOR"
  train.OFFICIAL_EPOCH_LENGTH="$OFFICIAL_EPOCH_LENGTH"
  train.compile="$TRAIN_COMPILE"
  train.sharded_eval_checkpoint=true
  student.resume_from_teacher_chkpt="$INIT_WEIGHTS"
  student.fp8_enabled="$FP8_ENABLED"
  optim.epochs="$EPOCHS"
  optim.lr="$LR"
  optim.scaling_rule=fixed
  optim.gradient_accumulation_steps="$GRAD_ACCUM_STEPS"
  "crops.rgb_mean=$RGB_MEAN"
  "crops.rgb_std=$RGB_STD"
  evaluation.eval_period_iterations="$EVAL_PERIOD"
  checkpointing.period="$CHECKPOINT_PERIOD"
  checkpointing.max_to_keep="$CHECKPOINT_MAX_TO_KEEP"
  checkpointing.sharded=true
)

echo "[hs6-7b] mode=$MODE node=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT"
echo "[hs6-7b] dataset=$DATASET_PATH"
echo "[hs6-7b] mean=$RGB_MEAN std=$RGB_STD"
echo "[hs6-7b] output=$OUTPUT_DIR init=${INIT_WEIGHTS:-<random>}"
echo "[hs6-7b] batch=$TOTAL_GPUS GPUs x $BATCH_SIZE_PER_GPU/GPU x accum $GRAD_ACCUM_STEPS = $EFFECTIVE_GLOBAL_BATCH ($ACCUM_SOURCE)"
echo "[hs6-7b] schedule=$EPOCHS epochs x $OFFICIAL_EPOCH_LENGTH updates; lr=$LR; fp8=$FP8_ENABLED"
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" == 1 ]]; then
  exit 0
fi

cd "$REPO"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

exec "${cmd[@]}"
