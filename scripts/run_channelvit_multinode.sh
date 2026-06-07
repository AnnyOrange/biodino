#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run optional ChannelViT multi-channel training on single-node or multi-node GPUs.

Usage:
  DATASET_PATH='packwds_chvit:/path/to/packed_shards/filtered_mixed_train_w*-{000000..000999}.tar::sample_channels=6' \
  OUTPUT_DIR='outputs/bio_continue_channelvit_sample6_fixedinit' \
  bash scripts/run_channelvit_multinode.sh

Run the same command on every node, changing only NODE_RANK.

Important env vars:
  DATASET_PATH          Required. Should use packwds_chvit:...::sample_channels=N
  OUTPUT_DIR           Required. Training output dir.
  CONFIG_FILE          Default: dinov3/configs/train/bio_channelvit_sample6_fixedinit.yaml
  INIT_WEIGHTS          Default: /mnt/huawei_deepcad/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
  SAMPLE_CHANNELS       Default: 6. Controls dataset_path suffix only if DATASET_PATH is not provided.
  IN_CHANS              Default: 8. Channel embedding capacity in the model config.
  NNODES               Number of nodes. Default: 1
  NODE_RANK            Rank of this node. Default: 0
  MASTER_ADDR          Master node IP/hostname. Default: 127.0.0.1
  MASTER_PORT          Master port. Default: 29500
  NPROC_PER_NODE       GPUs per node. Default: visible GPU count from nvidia-smi, fallback 8
  BATCH_SIZE_PER_GPU   Default: 16
  GRAD_ACCUM_STEPS     Default: 16
  NUM_WORKERS          Default: 6
  NO_RESUME=1          Add --no-resume.
  DRY_RUN=1            Print command only.

This is an optional multi-channel ablation, not the default RGB recipe.
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

SAMPLE_CHANNELS="${SAMPLE_CHANNELS:-6}"
if [[ -z "${DATASET_PATH:-}" ]]; then
  DATASET_PATH="packwds_chvit:/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*-{000000..000999}.tar::sample_channels=${SAMPLE_CHANNELS}"
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
CONFIG_FILE="${CONFIG_FILE:-dinov3/configs/train/bio_channelvit_sample6_fixedinit.yaml}"
INIT_WEIGHTS="${INIT_WEIGHTS:-/mnt/huawei_deepcad/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"
IN_CHANS="${IN_CHANS:-8}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
NUM_WORKERS="${NUM_WORKERS:-6}"
OFFICIAL_EPOCH_LENGTH="${OFFICIAL_EPOCH_LENGTH:-1025}"
SAVECKP_FREQ="${SAVECKP_FREQ:-20}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-1000}"

export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
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
  student.resume_from_teacher_chkpt="$INIT_WEIGHTS"
  student.in_chans="$IN_CHANS"
  teacher.in_chans="$IN_CHANS"
  student.enable_channelvit=true
  teacher.enable_channelvit=true
)

if [[ "${NO_RESUME:-0}" == "1" ]]; then
  cmd+=(--no-resume)
fi

echo "[run_channelvit_multinode] node ${NODE_RANK}/${NNODES}, gpus=${NPROC_PER_NODE}, master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[run_channelvit_multinode] output=${OUTPUT_DIR}"
echo "[run_channelvit_multinode] dataset=${DATASET_PATH}"
echo "[run_channelvit_multinode] init=${INIT_WEIGHTS}"
printf ' %q' "${cmd[@]}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec "${cmd[@]}"
