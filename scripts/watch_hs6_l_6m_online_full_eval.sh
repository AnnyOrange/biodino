#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 TRAIN_RUN OUTPUT_ROOT TAG [EXPECTED_CHECKPOINTS]" >&2
  exit 2
fi

TRAIN_RUN="$1"
OUTPUT_ROOT="$2"
TAG="$3"
EXPECTED_CHECKPOINTS="${4:-90}"
REPO="${REPO:-/mnt/huawei_deepcad/dinov3}"
PYTHON_BIN="${PYTHON_BIN:-/home/bbnc/venvs/external_fm/bin/python}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
POLL_SECONDS="${POLL_SECONDS:-30}"
INPUT_ROOT="${INPUT_ROOT:-$REPO/outputs/02_eval_inputs/${TAG}}"
WORKER="$REPO/scripts/run_hs6_l_6m_full_eval_fleet_worker.py"
LOG_ROOT="$OUTPUT_ROOT/_state/worker_logs"

test -x "$PYTHON_BIN"
test -f "$WORKER"
test -f "$TRAIN_RUN/config.yaml"
mkdir -p "$LOG_ROOT"

pids=()
for gpu in $GPUS; do
  worker_name="${TAG}-$(hostname)-gpu${gpu}"
  "$PYTHON_BIN" -u "$WORKER" \
    --repo "$REPO" \
    --train-run "$TRAIN_RUN" \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --python-bin "$PYTHON_BIN" \
    --gpu "$gpu" \
    --worker "$worker_name" \
    --poll-seconds "$POLL_SECONDS" \
    >>"$LOG_ROOT/${worker_name}.log" 2>&1 &
  pids+=("$!")
done

echo "[$(date -u +%FT%TZ)] launched ${#pids[@]} workers; expected checkpoints=$EXPECTED_CHECKPOINTS"
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
