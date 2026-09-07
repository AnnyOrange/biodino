#!/usr/bin/env bash
# Run one pure-SSL acquisition-tangent arm on exactly four RTX 5090 GPUs.
set -euo pipefail

REPO=${REPO:-/home/xzj/biodino}
PYTHON_BIN=${PYTHON_BIN:-/home/xzj/miniconda3/envs/dinov3/bin/python}
CONFIG=${CONFIG:-$REPO/dinov3/configs/train/microscopy_continual_vitl16_intervention_firewall.yaml}
MATURE_CKPT=${MATURE_CKPT:-/home/xzj/hs6_mature_readout_20260831/ckpt/15374/checkpoint.pth}
RUN_ROOT=${RUN_ROOT:-/mnt/data/xzj/intervention_firewall_20260902/runs}
ARM=${ARM:-true}
GPU_GROUP=${GPU_GROUP:-0,1,2,3}
MASTER_PORT=${MASTER_PORT:-32921}
MAX_UPDATES=${MAX_UPDATES:-64}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-64}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
NUM_WORKERS=${NUM_WORKERS:-2}
SEED=${SEED:-0}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-acq_tangent_cls}
DRY_RUN=${DRY_RUN:-0}

case "$ARM" in
  baseline)
    ACQ_ENABLED=false
    ACQ_MODE=gradient_projection
    ;;
  true)
    ACQ_ENABLED=true
    ACQ_MODE=gradient_projection
    ;;
  shuffled)
    ACQ_ENABLED=true
    ACQ_MODE=shuffled_gradient_projection
    ;;
  *)
    echo "ERROR: ARM must be baseline, true, or shuffled; got $ARM" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a gpu_ids <<<"$GPU_GROUP"
[[ ${#gpu_ids[@]} -eq 4 ]] || { echo "ERROR: exactly four GPUs are required" >&2; exit 2; }
effective_batch=$((4 * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))
[[ "$effective_batch" -eq 1024 ]] || {
  echo "ERROR: effective batch $effective_batch != HS6 batch 1024" >&2
  exit 2
}
[[ -x "$PYTHON_BIN" && -s "$CONFIG" && -s "$MATURE_CKPT" ]] || {
  echo "ERROR: missing Python, config, or mature checkpoint" >&2
  exit 2
}

run_name="${EXPERIMENT_TAG}_${ARM}_seed${SEED}_u${MAX_UPDATES}"
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_ROOT/$run_name}
[[ ! -e "$OUTPUT_DIR" ]] || { echo "ERROR: refusing to overwrite $OUTPUT_DIR" >&2; exit 2; }
DATASET_PATH=${DATASET_PATH:-'packwds_robust:/mnt/data/microscopy-100k-patched/filtered_mixed_train_w00-{000000..000020}.tar::pct=1,99'}
cmd=(
  "$PYTHON_BIN" -m torch.distributed.run
  --nnodes=1 --node_rank=0 --nproc_per_node=4
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT"
  dinov3/train/train.py
  --config-file "$CONFIG"
  --output-dir "$OUTPUT_DIR"
  --no-resume
  "train.dataset_path=$DATASET_PATH"
  train.batch_size_per_gpu="$BATCH_SIZE_PER_GPU"
  train.num_workers="$NUM_WORKERS"
  train.seed="$SEED"
  train.max_updates="$MAX_UPDATES"
  train.OFFICIAL_EPOCH_LENGTH=64
  optim.gradient_accumulation_steps="$GRAD_ACCUM_STEPS"
  "student.resume_from_teacher_chkpt=$MATURE_CKPT"
  acquisition_orbit_deflation.enabled="$ACQ_ENABLED"
  acquisition_orbit_deflation.mode="$ACQ_MODE"
  acquisition_orbit_deflation.projection_scope=cls
  acquisition_orbit_deflation.projection_strength=1.0
  nested_channel_innovation.enabled=false
  conditional_morphology_graph.enabled=false
  nested_resolution_innovation.enabled=false
  scout_kernel_transport.enabled=false
  expert_consensus_residual.enabled=false
  global_bridge_transport.enabled=false
  evaluation.eval_period_iterations="$MAX_UPDATES"
  checkpointing.period="$MAX_UPDATES"
)

printf '[intervention-firewall] arm=%s mode=%s GPUs=%s batch=4x%sx%s=%s updates=%s output=%s\n' \
  "$ARM" "$ACQ_MODE" "$GPU_GROUP" "$BATCH_SIZE_PER_GPU" "$GRAD_ACCUM_STEPS" \
  "$effective_batch" "$MAX_UPDATES" "$OUTPUT_DIR"
printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU_GROUP"
printf ' %q' "${cmd[@]}"
printf '\n'
[[ "$DRY_RUN" == 0 ]] || exit 0

mkdir -p "$RUN_ROOT"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU_GROUP"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
ulimit -n 65536 2>/dev/null || true
exec "${cmd[@]}"
