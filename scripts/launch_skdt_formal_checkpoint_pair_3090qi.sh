#!/usr/bin/env bash
# Launch one matched/shuffled checkpoint under the approved formal-v1 matrix.
set -euo pipefail

CHECKPOINT_ITER=${CHECKPOINT_ITER:?Set CHECKPOINT_ITER, for example 14349}
RUNTIME_REPO=${RUNTIME_REPO:-/mnt/huawei_deepcad/dinov3/outputs/02_eval_runtime/bio_eval_formal_retry_584af01}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/skdt_formal_v1_3090qi_checkpoint_${CHECKPOINT_ITER}}
PYTHON_BIN=${PYTHON_BIN:-/home/bbnc/anaconda3/envs/dinov3/bin/python}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-/mnt/huawei_deepcad/benchmark}

classification=(
  bloodmnist pathmnist tissuemnist breastmnist organamnist organcmnist organsmnist
  dermamnist octmnist pneumoniamnist retinamnist chestmnist bbbc048-cellcycle
  cyclops-protein-loc midog25-atypical pcam nct-crc-he chammi-allen-task1
  chammi-allen-task2 chammi-cp-task1 chammi-cp-task2 chammi-cp-task3
  chammi-hpa-task1 chammi-hpa-task2
)
regression=(bbbc005 bbbc013 conic-cell-count livecell-cell-count)
retrieval=(nct-crc-he-1k crc-val-he-7k hpa-subcellular rxrx1-cross)
segmentation=(bbbc038 cellpose conic livecell monuseg pannuke tissuenet multimodal_cellseg)
detection=(livecell bbbc038 conic)

launch_arm() {
  local arm=$1
  local train_dir=$2
  shift 2
  local gpus=("$@")
  local arm_output=$OUTPUT_ROOT/$arm
  mkdir -p "$arm_output"
  (
    cd "$RUNTIME_REPO"
    nohup "$PYTHON_BIN" -m dinov3.eval.bio_benchmark \
      --checkpoints-dir "$train_dir/ckpt" \
      --checkpoint-iters "$CHECKPOINT_ITER" \
      --train-config "$train_dir/config.yaml" \
      --benchmark-root "$BENCHMARK_ROOT" \
      --output-dir "$arm_output" \
      --tasks classification regression retrieval segmentation detection \
      --classification-datasets "${classification[@]}" \
      --regression-datasets "${regression[@]}" \
      --retrieval-datasets "${retrieval[@]}" \
      --segmentation-datasets "${segmentation[@]}" \
      --detection-datasets "${detection[@]}" \
      --gpus "${gpus[@]}" \
      --jobs-per-gpu 1 \
      --max-concurrent-jobs "${#gpus[@]}" \
      --max-cpu-jobs "${#gpus[@]}" \
      --frozen-batch-size 64 \
      --frozen-datasets-per-job 1 \
      --frozen-n-last-blocks 1 \
      --segmentation-datasets-per-job 1 \
      --segmentation-protocol best \
      --segmentation-split-protocol formal-v1 \
      --seg-feature-batch-size 32 \
      --seg-probe-batch-size 32 \
      --seg-probe-epochs 50 \
      --det-batch-size 8 \
      --det-epochs 5 \
      --conic-split-protocol official-baseline-fold0-nested-v1 \
      --num-workers 2 \
      --seg-feature-num-workers 2 \
      --seg-probe-num-workers 2 \
      --classification-resolution-protocol best \
      --regression-resolution-protocol best \
      --autocast-dtype bf16 \
      --seed 0 \
      --run-name skdt_formal_v1 \
      >"$arm_output/launcher.log" 2>&1 </dev/null &
    echo "$!" >"$arm_output/launcher.pid"
  )
}

repo_root=/mnt/huawei_deepcad/dinov3
launch_arm matched \
  "$repo_root/outputs/01_training_runs/HS6_L_e11to15_SKDT_stable_w0.5_bs64_gb1024_seed0_2x3090qi_20260908" \
  2 5
launch_arm shuffled \
  "$repo_root/outputs/01_training_runs/HS6_L_e11to15_SKDT_shuffled_stable_w0.5_bs64_gb1024_seed0_2x3090qi_20260908" \
  6 7

printf 'matched pid=%s\n' "$(cat "$OUTPUT_ROOT/matched/launcher.pid")"
printf 'shuffled pid=%s\n' "$(cat "$OUTPUT_ROOT/shuffled/launcher.pid")"
