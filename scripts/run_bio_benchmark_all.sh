#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run all BioDINO benchmark evals for a checkpoint folder.

Usage:
  bash scripts/run_bio_benchmark_all.sh <checkpoints_dir> <train_config> <output_dir> [benchmark_root]

Example:
  bash scripts/run_bio_benchmark_all.sh \
    outputs/bio_continue_true_channelvit_sample6_fixedinit/ckpt \
    outputs/bio_continue_true_channelvit_sample6_fixedinit/config.yaml \
    outputs/bio_eval_true_channelvit_sample6_fixedinit_all

Optional env vars:
  PYTHON_BIN=/path/to/python        Python in the dinov3 env. Default: current python (else set PYTHON_BIN).
  CHECKPOINT_ITERS="all"            all | latest | "7174 8199"; default: all.
  GPUS="0 1 2 3"                   If unset, bio_benchmark uses all currently visible GPUs.
  JOBS_PER_GPU=1                   Parallel jobs per GPU.
  TASKS="segmentation classification regression detection retrieval"
  LAYER_PRESET=last1               last1 | even4 | last4 | layerwise.
  NUM_WORKERS=4
  PROBE_BACKEND=sklearn            Frozen cls/reg/multilabel/retrieval probe backend.
                                   sklearn reproduces benchmark_results_*.md; torch is the legacy ablation.
  FROZEN_BATCH_SIZE=64             Feature-extraction batch size for the frozen probes.
  SEED=0                           Split seed for the frozen probes.
  *_DATASETS                       CLASSIFICATION_/REGRESSION_/RETRIEVAL_/SEGMENTATION_/DETECTION_DATASETS.
  SEG_FEATURE_BATCH_SIZE=32
  SEG_PROBE_EPOCHS=50
  DET_EPOCHS=5
  DET_BATCH_SIZE=8
  DRY_RUN=1                        Print generated jobs but do not run.
  SMOKE=1                          Tiny end-to-end check, not final numbers.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

CHECKPOINTS_DIR="$1"
TRAIN_CONFIG="$2"
OUTPUT_DIR="$3"
BENCHMARK_ROOT="${4:-/mnt/huawei_deepcad/benchmark}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "ERROR: no python found; activate the dinov3 env or set PYTHON_BIN=/path/to/python" >&2
  exit 2
fi

CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-all}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
TASKS="${TASKS:-segmentation classification regression detection retrieval}"
LAYER_PRESET="${LAYER_PRESET:-last1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

SEGMENTATION_DATASETS="${SEGMENTATION_DATASETS:-bbbc038 conic monuseg pannuke tissuenet}"
# Names must match the dinov3.eval.bio_frozen_eval registry (chestmnist = multilabel).
CLASSIFICATION_DATASETS="${CLASSIFICATION_DATASETS:-bloodmnist bbbc048-cellcycle cyclops-protein-loc midog25-atypical chestmnist}"
REGRESSION_DATASETS="${REGRESSION_DATASETS:-bbbc013}"
RETRIEVAL_DATASETS="${RETRIEVAL_DATASETS:-lc25000 nct-crc-he-1k crc-val-he-7k}"
DETECTION_DATASETS="${DETECTION_DATASETS:-livecell}"

# Frozen cls/reg/multilabel/retrieval probes (sklearn reproduces the reported md).
PROBE_BACKEND="${PROBE_BACKEND:-sklearn}"
FROZEN_BATCH_SIZE="${FROZEN_BATCH_SIZE:-64}"
SEED="${SEED:-0}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
SEG_FEATURE_BATCH_SIZE="${SEG_FEATURE_BATCH_SIZE:-32}"
SEG_PROBE_EPOCHS="${SEG_PROBE_EPOCHS:-50}"
DET_EPOCHS="${DET_EPOCHS:-5}"
DET_BATCH_SIZE="${DET_BATCH_SIZE:-8}"

cmd=(
  "$PYTHON" -m dinov3.eval.bio_benchmark
  --checkpoints-dir "$CHECKPOINTS_DIR"
  --checkpoint-iters $CHECKPOINT_ITERS
  --train-config "$TRAIN_CONFIG"
  --benchmark-root "$BENCHMARK_ROOT"
  --output-dir "$OUTPUT_DIR"
  --tasks $TASKS
  --jobs-per-gpu "$JOBS_PER_GPU"
  --num-workers "$NUM_WORKERS"
  --segmentation-datasets $SEGMENTATION_DATASETS
  --classification-datasets $CLASSIFICATION_DATASETS
  --regression-datasets $REGRESSION_DATASETS
  --retrieval-datasets $RETRIEVAL_DATASETS
  --detection-datasets $DETECTION_DATASETS
  --probe-backend "$PROBE_BACKEND"
  --frozen-batch-size "$FROZEN_BATCH_SIZE"
  --seed "$SEED"
  --train-fraction "$TRAIN_FRACTION"
  --layer-preset "$LAYER_PRESET"
  --seg-feature-batch-size "$SEG_FEATURE_BATCH_SIZE"
  --seg-probe-epochs "$SEG_PROBE_EPOCHS"
  --det-epochs "$DET_EPOCHS"
  --det-batch-size "$DET_BATCH_SIZE"
)

if [[ -n "${GPUS:-}" ]]; then
  cmd+=(--gpus $GPUS)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  cmd+=(--smoke)
fi

echo "[run_bio_benchmark_all] Python: $PYTHON"
echo "[run_bio_benchmark_all] Checkpoints: $CHECKPOINTS_DIR"
echo "[run_bio_benchmark_all] Train config: $TRAIN_CONFIG"
echo "[run_bio_benchmark_all] Benchmark root: $BENCHMARK_ROOT"
echo "[run_bio_benchmark_all] Output: $OUTPUT_DIR"
echo "[run_bio_benchmark_all] Command:"
printf ' %q' "${cmd[@]}"
echo

exec "${cmd[@]}"
