#!/usr/bin/env bash
# Evaluate the causal 1024-step Scout target ablation on the four locked
# retrieval/clustering datasets, one arm per DeepCAD A100.
set -euo pipefail

REPO=${REPO:-/mnt/huawei_deepcad/dinov3}
PYTHON_BIN=${PYTHON_BIN:-/home/deepcad/anaconda3/envs/dinov3/bin/python}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-/mnt/huawei_deepcad/benchmark}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO/outputs/02_eval_runs/scout_u1024_ret4_geometry_deepcad_20260826}
DRY_RUN=${DRY_RUN:-0}

ARMS=(
  "full_full|$REPO/outputs/01_training_runs/scout_scale_u1024_b20_20260819_r1/full_full"
  "stable_delta|$REPO/outputs/01_training_runs/scout_scale_u1024_b20_20260819_r1/stable_mask_w20"
  "raw_delta|$REPO/outputs/01_training_runs/scout_target_controls_u1024_b20_20260819_r1/delta_mask_w20"
  "shuffled_stable|$REPO/outputs/01_training_runs/scout_target_controls_u1024_b20_20260819_r1/stable_shuffled_mask_w20"
)

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -d "$BENCHMARK_ROOT/Retrieval_Clustering" ]] || {
  echo "ERROR: missing retrieval benchmark root: $BENCHMARK_ROOT" >&2
  exit 2
}

mkdir -p "$OUTPUT_ROOT/logs"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

pids=()
for gpu in "${!ARMS[@]}"; do
  IFS='|' read -r label train_dir <<<"${ARMS[$gpu]}"
  checkpoint="$train_dir/ckpt/1023/checkpoint.pth"
  config="$train_dir/config.yaml"
  output="$OUTPUT_ROOT/$label"
  log="$OUTPUT_ROOT/logs/$label.log"
  [[ -s "$checkpoint" ]] || { echo "ERROR: missing checkpoint: $checkpoint" >&2; exit 2; }
  [[ -s "$config" ]] || { echo "ERROR: missing config: $config" >&2; exit 2; }

  cmd=(
    "$PYTHON_BIN" -m dinov3.eval.bio_frozen_eval.run_retrieval_clustering
    --checkpoint "$checkpoint"
    --train-config "$config"
    --benchmark-root "$BENCHMARK_ROOT"
    --datasets lc25000 nct-crc-he-100 nct-crc-he-1k crc-val-he-7k
    --output-dir "$output"
    --model-name "$label"
    --device cuda
    --batch-size 64
    --num-workers 4
    --metric-device cpu
    --n-last-blocks 1
  )
  printf 'CUDA_VISIBLE_DEVICES=%q' "$gpu"
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" == 0 ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log" 2>&1 &
    pids+=("$!")
  fi
done

[[ "$DRY_RUN" == 0 ]] || exit 0
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
