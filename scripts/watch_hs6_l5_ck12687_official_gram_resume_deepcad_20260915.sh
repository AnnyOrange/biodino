#!/usr/bin/env bash
# Resume the official-Gram curve from its first segment's full ck15127.
set -euo pipefail

REPO=${REPO:-/mnt/huawei_deepcad/dinov3}
RUN=$REPO/outputs/01_training_runs/HS6_L5_ck12687_official_gram_a12687_b32_gb1024_noac_4xdeepcad_u2440_contract_v2_20260915
LAUNCHER=$REPO/scripts/launch_hs6_l5_ck12687_official_gram_deepcad_20260915.sh
CURRENT_SESSION=${CURRENT_SESSION:-hs6_l5_ck12687_offgram_v2}
RESUME_SESSION=${RESUME_SESSION:-hs6_l5_ck12687_offgram_long}
POLL_SECONDS=${POLL_SECONDS:-60}
LOG=$REPO/outputs/auto_train_logs/hs6_l5_ck12687_official_gram_resume_to60999_20260915.log

log() {
  printf '[%s] [hs6-L5-Gram-resume-watch] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG"
}

mkdir -p "$(dirname "$LOG")"
log "waiting for complete full optimizer checkpoint ck15127"
while [[ ! -s "$RUN/ckpt/15127/checkpoint.pth" ]]; do
  if ! tmux has-session -t "$CURRENT_SESSION" 2>/dev/null; then
    log "ERROR initial training session ended before ck15127 was complete"
    exit 2
  fi
  sleep "$POLL_SECONDS"
done

# Do not race the final checkpoint writer or the original torchrun teardown.
while tmux has-session -t "$CURRENT_SESSION" 2>/dev/null; do
  sleep 10
done
sleep 30

[[ $(stat -c %s "$RUN/ckpt/15127/checkpoint.pth") -gt 6000000000 ]] || {
  log "ERROR ck15127 is too small to be a complete full checkpoint"
  exit 2
}
[[ -s "$RUN/eval/training_15127/teacher_checkpoint.pth" ]] || {
  log "ERROR ck15127 teacher checkpoint is missing"
  exit 2
}
[[ $(wc -l < "$RUN/raw_loss_metrics.jsonl") -eq 2440 ]] || {
  log "ERROR initial segment does not contain exactly 2440 optimizer updates"
  exit 2
}
tmux has-session -t "$RESUME_SESSION" 2>/dev/null && {
  log "ERROR resume session already exists: $RESUME_SESSION"
  exit 2
}

log "launching full optimizer resume from ck15127 through ck60999"
tmux new-session -d -s "$RESUME_SESSION" \
  "cd '$REPO' && MODE=resume MASTER_PORT=32191 ACTIVATION_CHECKPOINTING=false ACTIVATION_CHECKPOINTING_FULL=false bash '$LAUNCHER' >> '$LOG' 2>&1"
log "RESUME_LAUNCHED session=$RESUME_SESSION"
