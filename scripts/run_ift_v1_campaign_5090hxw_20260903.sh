#!/usr/bin/env bash
# Run the matched baseline/true/shuffled IFT screen and audit it.
set -euo pipefail

REPO=${REPO:-/home/xzj/biodino}
RUNNER=${RUNNER:-$REPO/scripts/run_intervention_factorized_topology_arm_5090hxw_20260903.sh}
PYTHON_BIN=${PYTHON_BIN:-/home/xzj/miniconda3/envs/dinov3/bin/python}
WORK_ROOT=${WORK_ROOT:-/mnt/data/xzj/intervention_factorized_topology_20260903}
RUN_ROOT=${RUN_ROOT:-$WORK_ROOT/runs}
LOG_ROOT=${LOG_ROOT:-$WORK_ROOT/logs}
REPORT_ROOT=${REPORT_ROOT:-$WORK_ROOT/reports/ift_v1}
MAX_UPDATES=${MAX_UPDATES:-64}
SEED=${SEED:-0}

[[ ! -e "$WORK_ROOT/training.complete" ]] || {
  echo "ERROR: refusing to rerun a completed campaign at $WORK_ROOT" >&2
  exit 2
}
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$REPORT_ROOT"

run_arm() {
  local arm=$1 gpu_group=$2 master_port=$3
  ARM="$arm" GPU_GROUP="$gpu_group" MASTER_PORT="$master_port" \
    MAX_UPDATES="$MAX_UPDATES" SEED="$SEED" RUN_ROOT="$RUN_ROOT" \
    bash "$RUNNER" >"$LOG_ROOT/${arm}_seed${SEED}_u${MAX_UPDATES}.log" 2>&1
}

run_arm baseline 0,1,2,3 33301 &
baseline_pid=$!
run_arm true 4,5,6,7 33302 &
true_pid=$!
printf '%s\n' "$baseline_pid" >"$WORK_ROOT/baseline.pid"
printf '%s\n' "$true_pid" >"$WORK_ROOT/true.pid"

status=0
wait "$baseline_pid" || status=1
wait "$true_pid" || status=1
[[ "$status" == 0 ]] || { echo "ERROR: baseline or true arm failed" >&2; exit 1; }

run_arm shuffled 0,1,2,3 33303 &
shuffled_pid=$!
printf '%s\n' "$shuffled_pid" >"$WORK_ROOT/shuffled.pid"
wait "$shuffled_pid"

"$PYTHON_BIN" "$REPO/scripts/audit_intervention_factorized_training.py" \
  --run-root "$RUN_ROOT" --experiment-tag ift_v1 \
  --seed "$SEED" --updates "$MAX_UPDATES" \
  --output "$REPORT_ROOT/training_audit.json"
touch "$WORK_ROOT/training.complete"
