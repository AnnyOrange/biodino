#!/usr/bin/env bash
# Official-structure Gram continuation from the surviving ck12687 teacher weights.
set -euo pipefail

REPO=${REPO:-/mnt/huawei_deepcad/dinov3}
PYTHON_BIN=${PYTHON_BIN:-/home/deepcad/anaconda3/envs/dinov3/bin/python}
MODE=${MODE:-formal} # smoke | formal
GPU_GROUP=${GPU_GROUP:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-32187}

BASE_CHECKPOINT_ID=12687
START_ITERATION=12688
ANCHOR_CHECKPOINT_ID=12687
BASE_RUN=$REPO/outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907
ANCHOR_CKPT=$BASE_RUN/eval/training_$ANCHOR_CHECKPOINT_ID/teacher_checkpoint.pth
ANCHOR_SHA256=c14adbb36543b952231b41a5bc5e5dfd6a1204b4bdb8cbecdd8193999dd8456a
OFFICIAL_GRAM_AST_SHA256=9c1f3d4a34329e8e9467d44439520b5b558cb7216bb73d596f0aeab154a199f0
OFFICIAL_DINOV3_COMMIT=6876159a11b4df116f30f667f8c9888617df0751

BATCH_SIZE_PER_GPU=32
GRAD_ACCUM_STEPS=8
NUM_WORKERS=${NUM_WORKERS:-1}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-false}
ACTIVATION_CHECKPOINTING_FULL=${ACTIVATION_CHECKPOINTING_FULL:-false}
ACTIVATION_CHECKPOINTING_BLOCKS=${ACTIVATION_CHECKPOINTING_BLOCKS:-0}
OFFICIAL_EPOCH_LENGTH=4098
EPOCHS=15
CHECKPOINT_PERIOD=488
EVAL_PERIOD=488
RUN_UPDATES=2440

DATASET_PATH='mixwds_robust:0.3=/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*.tar||0.7=/mnt/huawei_blm/deepcad_5t_v1/wds_patched_shuffle/filtered_mixed_train*.tar::pct=1,99'
RGB_MEAN='[0.5126699404721016,0.5020022506395592,0.5064769301636908]'
RGB_STD='[0.3497517202150124,0.34941518705400204,0.34802097842537794]'

case "$MODE" in
  smoke)
    RUN_UPDATES=1
    CHECKPOINT_PERIOD=1
    EVAL_PERIOD=1
    OUTPUT_DIR=${OUTPUT_DIR:-$REPO/outputs/01_training_runs/HS6_L5_ck12687_official_gram_a12687_b32_4xdeepcad_u1_smoke_20260915}
    ;;
  formal)
    OUTPUT_DIR=${OUTPUT_DIR:-$REPO/outputs/01_training_runs/HS6_L5_ck12687_official_gram_a12687_b32_gb1024_noac_4xdeepcad_u2440_contract_v2_20260915}
    ;;
  *)
    echo "ERROR: MODE must be smoke or formal, got $MODE" >&2
    exit 2
    ;;
esac

log() {
  printf '[%s] [hs6-L5-ck12687-official-gram] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

[[ -x "$PYTHON_BIN" ]] || { log "ERROR missing Python: $PYTHON_BIN"; exit 2; }
[[ -s "$ANCHOR_CKPT" ]] || { log "ERROR missing ck12687 anchor: $ANCHOR_CKPT"; exit 2; }
actual_anchor_sha=$(sha256sum "$ANCHOR_CKPT" | awk '{print $1}')
[[ "$actual_anchor_sha" == "$ANCHOR_SHA256" ]] || {
  log "ERROR ck12687 SHA mismatch: expected=$ANCHOR_SHA256 actual=$actual_anchor_sha"
  exit 2
}

# Compare the local GramLoss syntax tree to Meta's official implementation.
actual_gram_ast_sha=$(
  cd "$REPO"
  "$PYTHON_BIN" - <<'PY'
import ast
import hashlib
from pathlib import Path

tree = ast.parse(Path("dinov3/loss/gram_loss.py").read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GramLoss")
print(hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest())
PY
)
[[ "$actual_gram_ast_sha" == "$OFFICIAL_GRAM_AST_SHA256" ]] || {
  log "ERROR local GramLoss is not Meta official AST: expected=$OFFICIAL_GRAM_AST_SHA256 actual=$actual_gram_ast_sha"
  exit 2
}

IFS=',' read -r -a gpu_ids <<<"$GPU_GROUP"
[[ ${#gpu_ids[@]} -eq $NPROC_PER_NODE ]] || {
  log "ERROR GPU_GROUP=$GPU_GROUP does not contain $NPROC_PER_NODE GPUs"
  exit 2
}
if [[ -e "$OUTPUT_DIR" ]]; then
  log "ERROR refusing to overwrite existing output: $OUTPUT_DIR"
  exit 2
fi

MAX_UPDATES=$((START_ITERATION + RUN_UPDATES))
ENDPOINT_CHECKPOINT=$((MAX_UPDATES - 1))
EFFECTIVE_GLOBAL_BATCH=$((NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))

cmd=(
  "$PYTHON_BIN" -m torch.distributed.run
  --nnodes=1 --node_rank=0 --nproc_per_node="$NPROC_PER_NODE"
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT"
  dinov3/train/train.py
  --no-resume
  --config-file dinov3/configs/train/microscopy_continual_vitl16.yaml
  --output-dir "$OUTPUT_DIR"
  --seed 0
  "train.dataset_path=$DATASET_PATH"
  train.batch_size_per_gpu="$BATCH_SIZE_PER_GPU"
  train.num_workers="$NUM_WORKERS"
  train.seed=0
  train.OFFICIAL_EPOCH_LENGTH="$OFFICIAL_EPOCH_LENGTH"
  train.start_iteration_override="$START_ITERATION"
  train.max_updates="$MAX_UPDATES"
  train.cache_dataset=false
  train.compile=false
  train.wds_shuffle_buffer=50
  train.wds_deterministic_resampling=true
  train.prefetch_factor=1
  train.pin_memory=false
  train.checkpointing="$ACTIVATION_CHECKPOINTING"
  train.checkpointing_full="$ACTIVATION_CHECKPOINTING_FULL"
  train.checkpointing_blocks="$ACTIVATION_CHECKPOINTING_BLOCKS"
  student.in_chans=3
  teacher.in_chans=3
  student.enable_channelvit=false
  teacher.enable_channelvit=false
  student.stem_type=null
  teacher.stem_type=null
  student.norm_layer=layernormbf16
  student.pos_embed_rope_rescale_coords=2
  student.pos_embed_rope_dtype=fp32
  student.resume_from_teacher_chkpt="$ANCHOR_CKPT"
  optim.epochs="$EPOCHS"
  optim.scaling_rule=fixed
  optim.lr=0.0001
  optim.min_lr=0.000001
  optim.warmup_epochs=3
  optim.freeze_last_layer_epochs=1
  optim.gradient_accumulation_steps="$GRAD_ACCUM_STEPS"
  teacher.warmup_teacher_temp_epochs=30
  crops.global_crops_size=256
  crops.local_crops_size=112
  crops.gram_teacher_crops_size=512
  crops.gram_teacher_no_distortions=true
  crops.localcrops_subset_of_globalcrops=false
  crops.share_color_jitter=false
  crops.paired_global_geometry=false
  crops.augmentation_policy=bio_safe
  crops.horizontal_flips=false
  crops.float_input=false
  "crops.rgb_mean=$RGB_MEAN"
  "crops.rgb_std=$RGB_STD"
  sigreg.enabled=false
  channel_subset.enabled=false
  gram.use_loss=true
  gram.require_official_fixed_anchor_contract=true
  gram.compute_stats=false
  gram.loss_weight=2.0
  gram.inter_image_loss_weight=0.0
  gram.global_relation_loss_weight=0.0
  gram.global_relation_ckpt=null
  gram.ema_teacher=false
  gram.ckpt="$ANCHOR_CKPT"
  gram.it_load_ema_teacher=-1
  gram.rep_update=true
  gram.update_frequency=10000
  gram.it_first_update=1010000
  gram.max_updates=3
  gram.tokens_used=all
  gram.normalized=true
  gram.img_level=true
  gram.remove_neg=false
  gram.remove_only_teacher_neg=false
  gram.loss_weight_schedule=null
  gram.global_teacher_resize_method=bicubic
  gram.global_teacher_resize_antialias=false
  evaluation.eval_period_iterations="$EVAL_PERIOD"
  checkpointing.period="$CHECKPOINT_PERIOD"
  checkpointing.max_to_keep=null
  checkpointing.keep_every=99999999999999999
  checkpointing.sharded=false
)

log "mode=$MODE base=ck$BASE_CHECKPOINT_ID logical_start=$START_ITERATION endpoint=ck$ENDPOINT_CHECKPOINT"
log "optimizer_state=fresh_from_teacher model_and_gram_anchor=$ANCHOR_CKPT sha256=$ANCHOR_SHA256"
log "official_gram=meta_commit:$OFFICIAL_DINOV3_COMMIT AST:$actual_gram_ast_sha normalized=true img_level=true tokens=all remove_neg=false weight=2"
log "geometry=student256/clean_teacher512 resize=bicubic antialias=false extensions=disabled"
log "batch=${NPROC_PER_NODE}x${BATCH_SIZE_PER_GPU}xaccum${GRAD_ACCUM_STEPS}=$EFFECTIVE_GLOBAL_BATCH checkpoints=every${CHECKPOINT_PERIOD}:keep_all"
log "activation_checkpointing=$ACTIVATION_CHECKPOINTING full=$ACTIVATION_CHECKPOINTING_FULL blocks=$ACTIVATION_CHECKPOINTING_BLOCKS"
log "output=$OUTPUT_DIR"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU_GROUP"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
ulimit -n 65536 2>/dev/null || true
exec "${cmd[@]}"
