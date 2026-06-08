#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run DINOv3 ViT-7B RGB training on single-node or multi-node GPUs.

Usage:
  DATASET_PATH='packwds:/path/to/rgb_shards/train-{000000..000999}.tar' \
  OUTPUT_DIR='outputs/bio_continue_vit7b_rgb' \
  NNODES=2 NODE_RANK=0 MASTER_ADDR=<NODE0_IP> \
  bash scripts/run_vit7b_rgb_multinode.sh

Run the same command on every node, changing only NODE_RANK.

Important env vars:
  DATASET_PATH         DINOv3 dataset string. Default: packed microscopy RGB shards
                       (packwds:.../webds_micro_100k_by_channel_patched_shuffle/...).
  OUTPUT_DIR           Required. Training output dir.
  NNODES               Number of nodes. Default: 1
  NODE_RANK            Rank of this node. Default: 0
  MASTER_ADDR          Master node IP/hostname. Default: 127.0.0.1
  MASTER_PORT          Master port. Default: 29500
  NPROC_PER_NODE       GPUs per node. Default: visible GPU count from nvidia-smi, fallback 8
  CUDA_VISIBLE_DEVICES GPUs visible to this node. Default: leave unchanged
  CONFIG_FILE          Default: dinov3/configs/train/dinov3_vit7b16_pretrain.yaml
  INIT_WEIGHTS         Continue-training init → student.resume_from_teacher_chkpt.
                       Default: /mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
                       Set INIT_WEIGHTS='' to train from scratch (random init).
  PYTHON_BIN           Python executable. Default: current python; else set PYTHON_BIN explicitly.
  GLOBAL_BATCH_SIZE    Target global batch. Default: 4096 (DINOv3 ViT-7B standard).
  BATCH_SIZE_PER_GPU   Per-GPU micro-batch (memory knob). Default: 8; try 16 if memory is healthy.
  GRAD_ACCUM_STEPS     Default: auto = ceil(GLOBAL_BATCH_SIZE / (NNODES*NPROC_PER_NODE*BATCH_SIZE_PER_GPU)).
                       Set it explicitly to override (then GLOBAL_BATCH_SIZE is ignored).
  NUM_WORKERS          Default: 10
  OFFICIAL_EPOCH_LENGTH Default: 1000
  SAVECKP_FREQ         Default: 20
  CHECKPOINT_PERIOD    Default: 1000
  NO_RESUME=1          Add --no-resume.
  DRY_RUN=1            Print command only.

This script keeps the model in normal RGB mode:
  student.in_chans=3
  teacher.in_chans=3
  student.enable_channelvit=false
  teacher.enable_channelvit=false
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  usage >&2
  echo "ERROR: OUTPUT_DIR is required." >&2
  exit 2
fi
# Single-quote default so the {000000..000999} brace range is preserved literally
# (a `}` inside ${VAR:-...} would prematurely close the parameter expansion).
if [[ -z "${DATASET_PATH:-}" ]]; then
  DATASET_PATH='packwds:/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*-{000000..000999}.tar'
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "ERROR: no python found; activate the dinov3 env or set PYTHON_BIN=/path/to/python" >&2
  exit 2
fi

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l)"
  else
    NPROC_PER_NODE=8
  fi
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONFIG_FILE="${CONFIG_FILE:-dinov3/configs/train/dinov3_vit7b16_pretrain.yaml}"
# Use `-` (not `:-`) so INIT_WEIGHTS='' is honored as "train from scratch".
INIT_WEIGHTS="${INIT_WEIGHTS-/mnt/huawei_deepcad/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-8}"
# DINOv3 ViT-7B pretraining uses a global batch of ~4096. Default to that and
# auto-derive GRAD_ACCUM_STEPS from the world size so the global batch stays 4096
# regardless of node/GPU count. The LR auto-scales to the effective global batch
# (dinov3/configs/config.py), so matching 4096 keeps the schedule like DINOv3.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4096}"
TOTAL_GPUS=$(( NNODES * NPROC_PER_NODE ))
PER_OPT_STEP=$(( TOTAL_GPUS * BATCH_SIZE_PER_GPU ))
if [[ -n "${GRAD_ACCUM_STEPS:-}" ]]; then
  ACCUM_SOURCE="manual"        # explicit override wins; GLOBAL_BATCH_SIZE ignored
elif (( PER_OPT_STEP > 0 )); then
  GRAD_ACCUM_STEPS=$(( (GLOBAL_BATCH_SIZE + PER_OPT_STEP - 1) / PER_OPT_STEP ))  # ceil
  if (( GRAD_ACCUM_STEPS < 1 )); then GRAD_ACCUM_STEPS=1; fi
  ACCUM_SOURCE="auto"
else
  GRAD_ACCUM_STEPS=1
  ACCUM_SOURCE="auto"
fi
EFFECTIVE_GLOBAL_BATCH=$(( TOTAL_GPUS * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS ))
NUM_WORKERS="${NUM_WORKERS:-10}"
OFFICIAL_EPOCH_LENGTH="${OFFICIAL_EPOCH_LENGTH:-1000}"
SAVECKP_FREQ="${SAVECKP_FREQ:-20}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-1000}"

export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

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
  train.dataset_path="$DATASET_PATH"
  train.batch_size_per_gpu="$BATCH_SIZE_PER_GPU"
  optim.gradient_accumulation_steps="$GRAD_ACCUM_STEPS"
  train.num_workers="$NUM_WORKERS"
  train.OFFICIAL_EPOCH_LENGTH="$OFFICIAL_EPOCH_LENGTH"
  train.saveckp_freq="$SAVECKP_FREQ"
  checkpointing.period="$CHECKPOINT_PERIOD"
  student.in_chans=3
  teacher.in_chans=3
  student.enable_channelvit=false
  teacher.enable_channelvit=false
  student.resume_from_teacher_chkpt="$INIT_WEIGHTS"
  crops.rgb_mean="[0.511375,0.598449,0.683452]"
  crops.rgb_std="[0.340017,0.306132,0.284308]"
)

if [[ "${NO_RESUME:-0}" == "1" ]]; then
  cmd+=(--no-resume)
fi

echo "[run_vit7b_rgb_multinode] node ${NODE_RANK}/${NNODES}, gpus=${NPROC_PER_NODE}, master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[run_vit7b_rgb_multinode] output=${OUTPUT_DIR}"
echo "[run_vit7b_rgb_multinode] dataset=${DATASET_PATH}"
echo "[run_vit7b_rgb_multinode] init=${INIT_WEIGHTS:-<from scratch>}"
echo "[run_vit7b_rgb_multinode] batch: world=${TOTAL_GPUS} x bs/gpu=${BATCH_SIZE_PER_GPU} x accum=${GRAD_ACCUM_STEPS} = global ${EFFECTIVE_GLOBAL_BATCH} (target ${GLOBAL_BATCH_SIZE}, ${ACCUM_SOURCE})"
if [[ "${ACCUM_SOURCE}" == "auto" ]] && (( EFFECTIVE_GLOBAL_BATCH != GLOBAL_BATCH_SIZE )); then
  echo "[run_vit7b_rgb_multinode] WARN: global batch ${EFFECTIVE_GLOBAL_BATCH} != target ${GLOBAL_BATCH_SIZE}; pick BATCH_SIZE_PER_GPU so ${GLOBAL_BATCH_SIZE} divides (world*bs) evenly." >&2
fi
printf ' %q' "${cmd[@]}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec "${cmd[@]}"
