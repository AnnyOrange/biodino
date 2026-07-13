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
  MAX_CONCURRENT_JOBS=0            Global subprocess cap; 0 derives from GPUs x jobs-per-GPU.
  MAX_CPU_JOBS=0                   CPU-heavy subprocess cap; 0 matches MAX_CONCURRENT_JOBS.
  CONCURRENT_TASK_GROUPS=0         1 mixes core and dense jobs; default runs them in phases.
  EVAL_HARDWARE_PROFILE=auto       auto | conservative | balanced | throughput.
  FROZEN_DATASETS_PER_JOB=1        Group datasets to reuse one loaded checkpoint.
  TASKS="segmentation classification regression detection retrieval ood"
  SEGMENTATION_PROTOCOL=best        best | manual. best uses the dataset-specific final protocol.
  SEGMENTATION_MULTICHANNEL=0       1 passes --multichannel to the segmentation pipeline
                                   (meaningful for dualroute + TissueNet true channels).
  SEGMENTATION_CHANNEL_POLICY=auto  auto | native | first3 | compact3 | zerofill3 | mean3 | sample3_tta.
  SEGMENTATION_CHANNEL_TTA_SAMPLES=8
  SEGMENTATION_CHANNEL_POLICY_SEED=0
  LAYER_PRESET=last1               manual protocol only: last1 | even4 | last4 | layerwise.
  NUM_WORKERS=4
  FROZEN_BATCH_SIZE=64             Feature-extraction batch size for the frozen probes.
  FROZEN_CHANNEL_POLICY=auto        auto | native | first3 | compact3 | zerofill3 | mean3 | sample3_tta.
  FROZEN_CHANNEL_TTA_SAMPLES=8
  FROZEN_CHANNEL_POLICY_SEED=0
  AUTOCAST_DTYPE=bf16              bf16 | fp16 | fp32 for frozen feature extraction.
  CLASSIFICATION_RESOLUTION_PROTOCOL=best
                                   best | manual. best uses the 2026-06-23 5-dataset ablation table.
  CLASSIFICATION_IMAGE_SIZE=224    Manual/fallback final square crop size for classification/multilabel.
  CLASSIFICATION_RESIZE_SIZE=0     Optional pre-crop resize size; 0 uses ImageNet eval ratio.
  TRAIN_FRACTION=0.8               Fallback internal frozen-probe train fraction.
  SEED=0                           Split seed for fallback internal frozen probes.
  *_DATASETS                       CLASSIFICATION_/REGRESSION_/RETRIEVAL_/SEGMENTATION_/DETECTION_DATASETS.
  SEG_FEATURE_BATCH_SIZE=32
  SEG_FEATURE_NUM_WORKERS=4
  SEG_PROBE_EPOCHS=50
  SEG_PROBE_BATCH_SIZE=32
  SEG_PROBE_NUM_WORKERS=4
  DET_EPOCHS=5
  DET_BATCH_SIZE=8
  DETECTION_CHANNEL_POLICY=auto     Channel policy label for dense detection.
  OOD_TASKS="xray cryo"              OOD modalities to run when TASKS contains ood.
  OOD_DEVICE=cuda:0                  Device for the sequential OOD runner.
  OOD_BATCH_SIZE=64
  OOD_NUM_WORKERS=4
  OOD_CHANNEL_POLICY=auto           Reuses frozen-probe channel handling.
  OOD_ID_DATASETS="bloodmnist bbbc048 cyclops"
  OOD_ID_MAX_SAMPLES=3000
  OOD_XRAY_SLICES_PER_VOLUME=8
  OOD_CRYO_MAX_PARTICLES_PER_PROJECT=20000
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
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-0}"
MAX_CPU_JOBS="${MAX_CPU_JOBS:-0}"
FROZEN_DATASETS_PER_JOB="${FROZEN_DATASETS_PER_JOB:-1}"
CONCURRENT_TASK_GROUPS="${CONCURRENT_TASK_GROUPS:-0}"
EVAL_HARDWARE_PROFILE="${EVAL_HARDWARE_PROFILE:-auto}"
EVAL_BLAS_THREADS="${EVAL_BLAS_THREADS:-1}"
TASKS="${TASKS:-segmentation classification regression detection retrieval ood}"
SEGMENTATION_PROTOCOL="${SEGMENTATION_PROTOCOL:-best}"
SEGMENTATION_MULTICHANNEL="${SEGMENTATION_MULTICHANNEL:-0}"
SEGMENTATION_CHANNEL_POLICY="${SEGMENTATION_CHANNEL_POLICY:-auto}"
SEGMENTATION_CHANNEL_TTA_SAMPLES="${SEGMENTATION_CHANNEL_TTA_SAMPLES:-8}"
SEGMENTATION_CHANNEL_POLICY_SEED="${SEGMENTATION_CHANNEL_POLICY_SEED:-0}"
LAYER_PRESET="${LAYER_PRESET:-last1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

detect_min_gpu_memory_mb() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '
    NR == 1 { min = $1 }
    $1 < min { min = $1 }
    END { if (NR > 0) print int(min) }
  '
}

if [[ -z "${FROZEN_BATCH_SIZE:-}" ]]; then
  profile="$EVAL_HARDWARE_PROFILE"
  if [[ "$profile" == "auto" ]]; then
    gpu_mem_mb="$(detect_min_gpu_memory_mb || true)"
    if [[ "${gpu_mem_mb:-0}" -ge 38000 ]]; then
      profile="throughput"
    elif [[ "${gpu_mem_mb:-0}" -ge 30000 ]]; then
      profile="balanced"
    else
      profile="conservative"
    fi
  fi
  case "$profile" in
    conservative) FROZEN_BATCH_SIZE=64 ;;
    balanced) FROZEN_BATCH_SIZE=160 ;;
    throughput) FROZEN_BATCH_SIZE=192 ;;
    *) echo "ERROR: unknown EVAL_HARDWARE_PROFILE='$EVAL_HARDWARE_PROFILE'" >&2; exit 2 ;;
  esac
  RESOLVED_HARDWARE_PROFILE="$profile"
else
  RESOLVED_HARDWARE_PROFILE="explicit"
fi

# Numerical libraries otherwise create one full-CPU thread pool per eval
# subprocess, which quickly exhausts RLIMIT_NPROC under multi-GPU sweeps.
export OPENBLAS_NUM_THREADS="$EVAL_BLAS_THREADS"
export OMP_NUM_THREADS="$EVAL_BLAS_THREADS"
export MKL_NUM_THREADS="$EVAL_BLAS_THREADS"
export NUMEXPR_NUM_THREADS="$EVAL_BLAS_THREADS"
export VECLIB_MAXIMUM_THREADS="$EVAL_BLAS_THREADS"

# Full supported default suite. Override *_DATASETS for cheap checkpoint sweeps.
DEFAULT_SEGMENTATION_DATASETS="bbbc038 conic monuseg pannuke tissuenet livecell multimodal_cellseg cellpose"
DEFAULT_CLASSIFICATION_DATASETS="bloodmnist pathmnist tissuemnist breastmnist organamnist organcmnist organsmnist dermamnist octmnist pneumoniamnist retinamnist chestmnist bbbc048-cellcycle cyclops-protein-loc midog25-atypical pcam nct-crc-he lc25000 chammi-allen-task1 chammi-allen-task2 chammi-cp-task1 chammi-cp-task2 chammi-cp-task3 chammi-hpa-task1 chammi-hpa-task2"
DEFAULT_REGRESSION_DATASETS="bbbc013 bbbc005"
DEFAULT_RETRIEVAL_DATASETS="lc25000 nct-crc-he-100 nct-crc-he-1k crc-val-he-7k"
DEFAULT_DETECTION_DATASETS="livecell"

SEGMENTATION_DATASETS="${SEGMENTATION_DATASETS:-$DEFAULT_SEGMENTATION_DATASETS}"
# Names must match the dinov3.eval.bio_frozen_eval registry (chestmnist = multilabel).
CLASSIFICATION_DATASETS="${CLASSIFICATION_DATASETS:-$DEFAULT_CLASSIFICATION_DATASETS}"
REGRESSION_DATASETS="${REGRESSION_DATASETS:-$DEFAULT_REGRESSION_DATASETS}"
RETRIEVAL_DATASETS="${RETRIEVAL_DATASETS:-$DEFAULT_RETRIEVAL_DATASETS}"
DETECTION_DATASETS="${DETECTION_DATASETS:-$DEFAULT_DETECTION_DATASETS}"

# Frozen cls/reg/multilabel/retrieval probes use the canonical sklearn protocol.
FROZEN_BATCH_SIZE="${FROZEN_BATCH_SIZE:-64}"
FROZEN_CHANNEL_POLICY="${FROZEN_CHANNEL_POLICY:-auto}"
FROZEN_CHANNEL_TTA_SAMPLES="${FROZEN_CHANNEL_TTA_SAMPLES:-8}"
FROZEN_CHANNEL_POLICY_SEED="${FROZEN_CHANNEL_POLICY_SEED:-0}"
AUTOCAST_DTYPE="${AUTOCAST_DTYPE:-bf16}"
CLASSIFICATION_RESOLUTION_PROTOCOL="${CLASSIFICATION_RESOLUTION_PROTOCOL:-best}"
CLASSIFICATION_IMAGE_SIZE="${CLASSIFICATION_IMAGE_SIZE:-224}"
CLASSIFICATION_RESIZE_SIZE="${CLASSIFICATION_RESIZE_SIZE:-0}"
SEED="${SEED:-0}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
SEG_FEATURE_BATCH_SIZE="${SEG_FEATURE_BATCH_SIZE:-32}"
SEG_FEATURE_NUM_WORKERS="${SEG_FEATURE_NUM_WORKERS:-4}"
SEG_PROBE_EPOCHS="${SEG_PROBE_EPOCHS:-50}"
SEG_PROBE_BATCH_SIZE="${SEG_PROBE_BATCH_SIZE:-32}"
SEG_PROBE_NUM_WORKERS="${SEG_PROBE_NUM_WORKERS:-4}"
DET_EPOCHS="${DET_EPOCHS:-5}"
DET_BATCH_SIZE="${DET_BATCH_SIZE:-8}"
DETECTION_CHANNEL_POLICY="${DETECTION_CHANNEL_POLICY:-auto}"

BIO_TASKS=()
RUN_OOD=0
for task in $TASKS; do
  if [[ "$task" == "ood" ]]; then
    RUN_OOD=1
  else
    BIO_TASKS+=("$task")
  fi
done

cmd=()
if [[ "${#BIO_TASKS[@]}" -gt 0 ]]; then
  cmd=(
  "$PYTHON" -m dinov3.eval.bio_benchmark
  --checkpoints-dir "$CHECKPOINTS_DIR"
  --checkpoint-iters $CHECKPOINT_ITERS
  --train-config "$TRAIN_CONFIG"
  --benchmark-root "$BENCHMARK_ROOT"
  --output-dir "$OUTPUT_DIR"
  --tasks "${BIO_TASKS[@]}"
  --jobs-per-gpu "$JOBS_PER_GPU"
  --max-concurrent-jobs "$MAX_CONCURRENT_JOBS"
  --max-cpu-jobs "$MAX_CPU_JOBS"
  --frozen-datasets-per-job "$FROZEN_DATASETS_PER_JOB"
  --num-workers "$NUM_WORKERS"
  --segmentation-datasets $SEGMENTATION_DATASETS
  --classification-datasets $CLASSIFICATION_DATASETS
  --regression-datasets $REGRESSION_DATASETS
  --retrieval-datasets $RETRIEVAL_DATASETS
  --detection-datasets $DETECTION_DATASETS
  --frozen-batch-size "$FROZEN_BATCH_SIZE"
  --frozen-channel-policy "$FROZEN_CHANNEL_POLICY"
  --frozen-channel-tta-samples "$FROZEN_CHANNEL_TTA_SAMPLES"
  --frozen-channel-policy-seed "$FROZEN_CHANNEL_POLICY_SEED"
  --autocast-dtype "$AUTOCAST_DTYPE"
  --classification-resolution-protocol "$CLASSIFICATION_RESOLUTION_PROTOCOL"
  --classification-image-size "$CLASSIFICATION_IMAGE_SIZE"
  --classification-resize-size "$CLASSIFICATION_RESIZE_SIZE"
  --seed "$SEED"
  --train-fraction "$TRAIN_FRACTION"
  --segmentation-protocol "$SEGMENTATION_PROTOCOL"
  --segmentation-channel-policy "$SEGMENTATION_CHANNEL_POLICY"
  --segmentation-channel-tta-samples "$SEGMENTATION_CHANNEL_TTA_SAMPLES"
  --segmentation-channel-policy-seed "$SEGMENTATION_CHANNEL_POLICY_SEED"
  --layer-preset "$LAYER_PRESET"
  --seg-feature-batch-size "$SEG_FEATURE_BATCH_SIZE"
  --seg-feature-num-workers "$SEG_FEATURE_NUM_WORKERS"
  --seg-probe-epochs "$SEG_PROBE_EPOCHS"
  --seg-probe-batch-size "$SEG_PROBE_BATCH_SIZE"
  --seg-probe-num-workers "$SEG_PROBE_NUM_WORKERS"
  --det-epochs "$DET_EPOCHS"
  --det-batch-size "$DET_BATCH_SIZE"
  --detection-channel-policy "$DETECTION_CHANNEL_POLICY"
  )
fi

if [[ "${#cmd[@]}" -gt 0 && -n "${GPUS:-}" ]]; then
  cmd+=(--gpus $GPUS)
fi

if [[ "${#cmd[@]}" -gt 0 && "$CONCURRENT_TASK_GROUPS" == "1" ]]; then
  cmd+=(--concurrent-task-groups)
fi

if [[ "${#cmd[@]}" -gt 0 && "$SEGMENTATION_MULTICHANNEL" == "1" ]]; then
  cmd+=(--segmentation-multichannel)
fi

if [[ "${#cmd[@]}" -gt 0 && "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ "${#cmd[@]}" -gt 0 && "${SMOKE:-0}" == "1" ]]; then
  cmd+=(--smoke)
fi

select_checkpoint_iters() {
  local tokens="$1"
  local discovered=()
  local child name
  shopt -s nullglob
  for child in "$CHECKPOINTS_DIR"/*; do
    [[ -d "$child" ]] || continue
    name="$(basename "$child")"
    [[ "$name" =~ ^[0-9]+$ ]] || continue
    if [[ -f "$child/checkpoint.pth" || -f "$child/.metadata" ]]; then
      discovered+=("$name")
    fi
  done
  shopt -u nullglob
  if [[ "${#discovered[@]}" -eq 0 ]]; then
    echo "ERROR: no checkpoint.pth or DCP .metadata found under $CHECKPOINTS_DIR" >&2
    return 2
  fi
  mapfile -t discovered < <(printf '%s\n' "${discovered[@]}" | sort -n)
  tokens="${tokens//,/ }"
  local selected=()
  for token in $tokens; do
    case "$token" in
      all)
        selected+=("${discovered[@]}")
        ;;
      latest)
        selected+=("${discovered[-1]}")
        ;;
      *)
        selected+=("$token")
        ;;
    esac
  done
  printf '%s\n' "${selected[@]}" | awk '!seen[$0]++' | sort -n
}

run_ood_eval() {
  local ood_tasks="${OOD_TASKS:-xray cryo}"
  local ood_root="${OOD_ROOT:-${BENCHMARK_ROOT%/}/ood}"
  local ood_output="${OOD_OUTPUT_DIR:-${OUTPUT_DIR%/}/ood}"
  local ood_device="${OOD_DEVICE:-cuda:0}"
  local ood_batch_size="${OOD_BATCH_SIZE:-64}"
  local ood_num_workers="${OOD_NUM_WORKERS:-4}"
  local ood_id_datasets="${OOD_ID_DATASETS:-bloodmnist bbbc048 cyclops}"
  local ood_id_max_samples="${OOD_ID_MAX_SAMPLES:-3000}"
  local ood_xray_input_mode="${OOD_XRAY_INPUT_MODE:-three_slices}"
  local ood_xray_slices_per_volume="${OOD_XRAY_SLICES_PER_VOLUME:-8}"
  local ood_cryo_max_particles_per_project="${OOD_CRYO_MAX_PARTICLES_PER_PROJECT:-20000}"
  local ood_model_name="${OOD_MODEL_NAME:-$(basename "$(dirname "$CHECKPOINTS_DIR")")}"
  local ood_autocast_dtype="${OOD_AUTOCAST_DTYPE:-$AUTOCAST_DTYPE}"
  local ood_n_last_blocks="${OOD_N_LAST_BLOCKS:-1}"
  local ood_channel_policy="${OOD_CHANNEL_POLICY:-$FROZEN_CHANNEL_POLICY}"
  local ood_resize_size="${OOD_RESIZE_SIZE:-256}"
  local ood_crop_size="${OOD_CROP_SIZE:-224}"
  local ckpt ood_task
  mapfile -t OOD_CHECKPOINT_ITERS < <(select_checkpoint_iters "$CHECKPOINT_ITERS")
  if [[ "${#OOD_CHECKPOINT_ITERS[@]}" -eq 0 ]]; then
    echo "ERROR: no OOD checkpoint iterations selected from CHECKPOINT_ITERS='$CHECKPOINT_ITERS'" >&2
    return 2
  fi
  mkdir -p "$ood_output"
  for ckpt in "${OOD_CHECKPOINT_ITERS[@]}"; do
    for ood_task in $ood_tasks; do
    local ood_cmd=(
      "$PYTHON" -m dinov3.eval.eval_ood.dinov3_runner
      --model-name "$ood_model_name"
      --ckpt-root "$CHECKPOINTS_DIR"
      --ckpt-iter "$ckpt"
      --train-config "$TRAIN_CONFIG"
      --output-dir "${ood_output%/}/$ood_task"
      --ood-root "$ood_root"
      --benchmark-root "$BENCHMARK_ROOT"
      --tasks "$ood_task"
      --device "$ood_device"
      --batch-size "$ood_batch_size"
      --num-workers "$ood_num_workers"
      --seed "$SEED"
      --n-last-blocks "$ood_n_last_blocks"
      --channel-policy "$ood_channel_policy"
      --channel-tta-samples "$FROZEN_CHANNEL_TTA_SAMPLES"
      --channel-policy-seed "$FROZEN_CHANNEL_POLICY_SEED"
      --autocast-dtype "$ood_autocast_dtype"
      --resize-size "$ood_resize_size"
      --crop-size "$ood_crop_size"
      --xray-input-mode "$ood_xray_input_mode"
      --xray-slices-per-volume "$ood_xray_slices_per_volume"
      --cryo-max-particles-per-project "$ood_cryo_max_particles_per_project"
      --id-max-samples "$ood_id_max_samples"
      --id-datasets $ood_id_datasets
    )
    if [[ -n "${OOD_XRAY_MAX_VOLUMES:-}" ]]; then
      ood_cmd+=(--xray-max-volumes "$OOD_XRAY_MAX_VOLUMES")
    fi
    if [[ -n "${OOD_CRYO_MAX_PROJECTS:-}" ]]; then
      ood_cmd+=(--cryo-max-projects "$OOD_CRYO_MAX_PROJECTS")
    fi
    if [[ -n "${OOD_CRYO_MAX_PER_CLASS:-}" ]]; then
      ood_cmd+=(--cryo-max-per-class "$OOD_CRYO_MAX_PER_CLASS")
    fi
    if [[ "${OOD_CRYO_INVERT:-0}" == "1" ]]; then
      ood_cmd+=(--cryo-invert)
    fi
    if [[ "${OOD_OVERWRITE_FEATURES:-0}" == "1" ]]; then
      ood_cmd+=(--overwrite-features)
    fi
    echo "[run_bio_benchmark_all] OOD command:"
    printf ' %q' "${ood_cmd[@]}"
    echo
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      "${ood_cmd[@]}"
    fi
    done
  done
}

echo "[run_bio_benchmark_all] Python: $PYTHON"
echo "[run_bio_benchmark_all] Checkpoints: $CHECKPOINTS_DIR"
echo "[run_bio_benchmark_all] Train config: $TRAIN_CONFIG"
echo "[run_bio_benchmark_all] Benchmark root: $BENCHMARK_ROOT"
echo "[run_bio_benchmark_all] Output: $OUTPUT_DIR"
echo "[run_bio_benchmark_all] Segmentation protocol: $SEGMENTATION_PROTOCOL"
echo "[run_bio_benchmark_all] Segmentation multichannel: $SEGMENTATION_MULTICHANNEL"
echo "[run_bio_benchmark_all] Segmentation channel policy: $SEGMENTATION_CHANNEL_POLICY"
echo "[run_bio_benchmark_all] Frozen channel policy: $FROZEN_CHANNEL_POLICY"
echo "[run_bio_benchmark_all] Frozen batch size: $FROZEN_BATCH_SIZE (profile=$RESOLVED_HARDWARE_PROFILE)"
echo "[run_bio_benchmark_all] Resource caps: max_jobs=$MAX_CONCURRENT_JOBS max_cpu_jobs=$MAX_CPU_JOBS concurrent_groups=$CONCURRENT_TASK_GROUPS"
echo "[run_bio_benchmark_all] Frozen datasets per model load: $FROZEN_DATASETS_PER_JOB"
echo "[run_bio_benchmark_all] Numerical-library threads per subprocess: $EVAL_BLAS_THREADS"
echo "[run_bio_benchmark_all] Classification resolution protocol: $CLASSIFICATION_RESOLUTION_PROTOCOL"
echo "[run_bio_benchmark_all] Tasks: $TASKS"
echo "[run_bio_benchmark_all] Bio tasks: ${BIO_TASKS[*]:-(none)}"
echo "[run_bio_benchmark_all] OOD enabled: $RUN_OOD"

if [[ "${#cmd[@]}" -gt 0 ]]; then
  echo "[run_bio_benchmark_all] Bio command:"
  printf ' %q' "${cmd[@]}"
  echo
  "${cmd[@]}"
fi

if [[ "$RUN_OOD" == "1" ]]; then
  run_ood_eval
fi
