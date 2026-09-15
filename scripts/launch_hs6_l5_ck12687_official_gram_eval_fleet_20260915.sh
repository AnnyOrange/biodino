#!/usr/bin/env bash
# Start shared online full-registry workers on 3090-qi and single-card 3090 hosts.
set -euo pipefail

REPO=${REPO:-/mnt/huawei_deepcad/dinov3}
EVAL_REPO=${EVAL_REPO:-$REPO/outputs/02_eval_runtime/dinov3_a029eef0_full_registry.GOjODo}
TRAIN_RUN=$REPO/outputs/01_training_runs/HS6_L5_ck12687_official_gram_a12687_b32_gb1024_noac_4xdeepcad_u2440_contract_v2_20260915
INPUT_ROOT=$REPO/outputs/02_eval_inputs/hs6_l5_ck12687_official_gram_curve_20260915
OUTPUT_ROOT=$REPO/outputs/02_eval_runs/hs6_l5_ck12687_official_gram_curve_fullregistry_3090fleet_20260915
BASELINE_ROOT=$REPO/outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908
WORKER=$REPO/scripts/run_hs6_l_6m_full_eval_fleet_worker.py
LANES=classification_a,classification_b,classification_c,classification_d,regression,retrieval,detection,segmentation_a,segmentation_b,segmentation_c,segmentation_d

launch_worker() {
  local host=$1 python=$2 worker=$3
  ssh -o BatchMode=yes "$host" \
    "cd '$REPO' && setsid -f env FROZEN_BATCH_SIZE=64 '$python' -u '$WORKER' \
      --repo '$EVAL_REPO' \
      --train-run '$TRAIN_RUN' \
      --snapshot-root '$TRAIN_RUN/eval' \
      --snapshot-dir-prefix training_ \
      --snapshot-filename teacher_checkpoint.pth \
      --input-root '$INPUT_ROOT' \
      --output-root '$OUTPUT_ROOT' \
      --benchmark-root /mnt/huawei_deepcad/benchmark \
      --python-bin '$python' \
      --gpu 0 \
      --worker '$worker' \
      --official-epoch-length 4098 \
      --effective-global-batch 1024 \
      --epochs 15 \
      --full-eval-period 488 \
      --min-local-checkpoint-id 13175 \
      --expected-checkpoints 99 \
      --include-lanes '$LANES' \
      --jobs-per-gpu 3 \
      --poll-seconds 60 \
      --ready-age-seconds 60 \
      --max-attempts 5 \
      --allow-busy-gpu \
      > '$OUTPUT_ROOT/logs/$worker.driver.log' 2>&1 < /dev/null"
}

mkdir -p "$INPUT_ROOT" "$OUTPUT_ROOT/logs"
if [[ ! -e "$OUTPUT_ROOT/point_12687" ]]; then
  ln -s "$BASELINE_ROOT/point_12687" "$OUTPUT_ROOT/point_12687"
fi

# Four currently unclaimed cards on the 8x3090 host. Existing workers remain on 2/5/6/7.
for gpu in ${QI_GPUS:-0 1 3 4}; do
  ssh -o BatchMode=yes 3090-qi \
    "cd '$REPO' && setsid -f env FROZEN_BATCH_SIZE=64 /home/bbnc/anaconda3/envs/dinov3/bin/python -u '$WORKER' \
      --repo '$EVAL_REPO' --train-run '$TRAIN_RUN' --snapshot-root '$TRAIN_RUN/eval' \
      --snapshot-dir-prefix training_ --snapshot-filename teacher_checkpoint.pth \
      --input-root '$INPUT_ROOT' --output-root '$OUTPUT_ROOT' \
      --benchmark-root /mnt/huawei_deepcad/benchmark \
      --python-bin /home/bbnc/anaconda3/envs/dinov3/bin/python --gpu '$gpu' \
      --worker '3090qi-gpu${gpu}-gram12687' --official-epoch-length 4098 \
      --effective-global-batch 1024 --epochs 15 --full-eval-period 488 \
      --min-local-checkpoint-id 13175 --expected-checkpoints 99 \
      --include-lanes '$LANES' --jobs-per-gpu 3 --poll-seconds 60 \
      --ready-age-seconds 60 --max-attempts 5 --allow-busy-gpu \
      > '$OUTPUT_ROOT/logs/3090qi-gpu${gpu}-gram12687.driver.log' 2>&1 < /dev/null"
done

launch_worker cpu1 /home/inspur/anaconda3/envs/dinov3/bin/python cpu1-gpu0-gram12687
launch_worker cpu2 /home/inspur/anaconda3/envs/dinov3/bin/python cpu2-gpu0-gram12687
launch_worker cpu9 /home/inspur/miniconda3/envs/dinov3/bin/python cpu9-gpu0-gram12687
launch_worker cpu10 /home/inspur/anaconda3/envs/dinov3/bin/python cpu10-gpu0-gram12687
launch_worker cpu11 /home/inspur/anaconda3/envs/dinov3/bin/python cpu11-gpu0-gram12687
launch_worker cpu15 /home/inspur/anaconda3/envs/dinov3/bin/python cpu15-gpu0-gram12687

printf 'VALID_EVAL_FLEET_LAUNCHED output=%s\n' "$OUTPUT_ROOT"
