#!/usr/bin/env bash
# Evaluate matched IFT arms on bidirectional HPA-Cyclops organelle retrieval.
set -euo pipefail

REPO=${REPO:-/home/xzj/biodino}
PYTHON_BIN=${PYTHON_BIN:-/home/xzj/miniconda3/envs/dinov3/bin/python}
RUN_ROOT=${RUN_ROOT:-/mnt/data/xzj/intervention_factorized_topology_20260903/runs}
REPORT_ROOT=${REPORT_ROOT:-/mnt/data/xzj/intervention_factorized_topology_20260903/eval_cross_species}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-/mnt/data/benchmark}
MANIFEST=${MANIFEST:-/mnt/data/xzj/expert_consensus_20260831/protocols/hpa_cyclops_shared_organelle_v1.csv}
GPU_GROUP=${GPU_GROUP:-4,5,6,7}
MAX_UPDATES=${MAX_UPDATES:-64}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-4}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-ift_v1}

IFS=',' read -r -a gpus <<<"$GPU_GROUP"
[[ ${#gpus[@]} -eq 4 ]] || { echo "ERROR: exactly four GPUs are required" >&2; exit 2; }
[[ -s "$MANIFEST" ]] || { echo "ERROR: missing cross-species manifest $MANIFEST" >&2; exit 2; }
mkdir -p "$REPORT_ROOT/logs"

specs=()
for readout in nlb2_avg nlb2_cls; do
  for arm in baseline true shuffled; do
    specs+=("$arm|$readout")
  done
done

run_spec() {
  local spec=$1 gpu=$2 arm readout run_name checkpoint config output log pooling
  IFS='|' read -r arm readout <<<"$spec"
  run_name="${EXPERIMENT_TAG}_${arm}_seed${SEED}_u${MAX_UPDATES}"
  checkpoint="$RUN_ROOT/$run_name/eval/training_$((MAX_UPDATES - 1))/teacher_checkpoint.pth"
  config="$RUN_ROOT/$run_name/config.yaml"
  output="$REPORT_ROOT/$arm/$readout"
  log="$REPORT_ROOT/logs/${arm}_${readout}.log"
  if [[ -s "$output/result.json" ]]; then
    printf '[ift-cross] skip complete arm=%s readout=%s\n' "$arm" "$readout"
    return 0
  fi
  [[ -s "$checkpoint" && -s "$config" ]] || {
    echo "ERROR: missing checkpoint/config for $arm: $checkpoint" >&2
    return 2
  }
  pooling=${readout##*_}
  cmd=(
    "$PYTHON_BIN" "$REPO/scripts/eval_hpa_cyclops_cross_species.py"
    --checkpoint "$checkpoint"
    --train-config "$config"
    --benchmark-root "$BENCHMARK_ROOT"
    --manifest "$MANIFEST"
    --output-dir "$output"
    --model-name "${EXPERIMENT_TAG}_${arm}_seed${SEED}_u${MAX_UPDATES}_${readout}"
    --device cuda --metric-device cpu
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
    --n-last-blocks 2
  )
  [[ "$pooling" == avg ]] || cmd+=(--no-avgpool)
  printf '[ift-cross] arm=%s readout=%s gpu=%s\n' "$arm" "$readout" "$gpu"
  (
    cd "$REPO"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
    "${cmd[@]}"
  ) >"$log" 2>&1
}

status=0
for ((wave_start=0; wave_start<${#specs[@]}; wave_start+=4)); do
  pids=()
  labels=()
  for offset in 0 1 2 3; do
    index=$((wave_start + offset))
    ((index < ${#specs[@]})) || continue
    run_spec "${specs[$index]}" "${gpus[$offset]}" &
    pids+=("$!")
    labels+=("${specs[$index]}")
  done
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      printf '[ift-cross] complete %s\n' "${labels[$index]}"
    else
      printf '[ift-cross] FAILED %s\n' "${labels[$index]}" >&2
      status=1
    fi
  done
done
[[ "$status" == 0 ]] || exit "$status"

"$PYTHON_BIN" "$REPO/scripts/analyze_cross_species_organelle_screen.py" \
  --input-root "$REPORT_ROOT" --output "$REPORT_ROOT/screen_summary.json"
"$PYTHON_BIN" "$REPO/scripts/analyze_cross_species_paired_bootstrap.py" \
  --input-root "$REPORT_ROOT" --output "$REPORT_ROOT/paired_bootstrap.json" \
  --bootstrap-samples 10000 --seed "$SEED" --device cpu
touch "$REPORT_ROOT/eval.complete"
