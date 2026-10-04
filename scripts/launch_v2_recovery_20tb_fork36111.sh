#!/usr/bin/env bash
# Adaptive-v2 recovery arm on the 20TB route2 corpus: restart-free fork of the 20TB no-GRAM baseline
# (HS6_L_..._20tb_route2_mix009_021_0604_0096_8x5090zxr_20260924) at its full-state checkpoint 36111
# (model + AdamW moments, iteration), with the anchor-recoverability loss added (fork built by
# scripts/fork_vanilla_checkpoint_for_recovery.py).  Everything else is the baseline recipe verbatim:
# same data mix / shuffle buffer / crops / schedule (61 x 4098 updates), FSDP (fp32 masters), 8 ranks x 16 x acc8 = 1024
# (per-micro-step batch 128 like the baseline), every 488-update optimizer checkpoint kept.
#
#   usage: bash scripts/launch_v2_recovery_20tb_fork36111.sh <arm> <gpus e.g. 0,1,2,3,4,5,6,7> <master_port>
#   arms : cls        anchor = frozen EMA teacher @36111, CLS-only global stream
#          cls_slow   CLS-only, anchor = EMA(teacher) momentum 0.9995 (tau ~2k updates)
#          cls_slow2  CLS-only, anchor = EMA(teacher) momentum 0.9998 (tau ~5k updates)
#          global     anchor frozen, CLS + patch-mean stream (5TB arm `global`)
#   env  : REPO PYTHON_BIN FORK OUT_ROOT R0_WEIGHT R9_WEIGHT NUM_WORKERS *_OVERRIDE (RW / AM / GLOBAL_TOKENS)
set -euo pipefail
ARM=${1:?arm}; GPUS=${2:?gpus}; PORT=${3:?port}
REPO=${REPO:-/mnt/huawei_deepcad/dinov3}
PYTHON_BIN=${PYTHON_BIN:-/home/lxy/miniconda3/envs/dinov3/bin/python}
OUT_ROOT=${OUT_ROOT:-$REPO/outputs/01_training_runs/hs6_l_20tb_v2_recovery_fork36111_20261004}
FORK=${FORK:-$OUT_ROOT/fork}
FORK_IT=${FORK_IT:-36111}
OUT=$OUT_ROOT/$ARM
ANCHOR=$FORK/anchor/teacher_checkpoint.pth
NUM_WORKERS=${NUM_WORKERS:-2}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-1}
WDS_SHUFFLE_BUFFER=${WDS_SHUFFLE_BUFFER:-3000}
BATCH_SIZE_PER_GPU=16; GRAD_ACCUM_STEPS=8; GLOBAL_BATCH_SIZE=1024
OFFICIAL_EPOCH_LENGTH=4098; EPOCHS=61; CHECKPOINT_PERIOD=488; EVAL_PERIOD=488; CHECKPOINT_KEEP_EVERY=16592
POLL_SECONDS=${POLL_SECONDS:-60}

GLOBAL_TOKENS=cls; RW=1.0; AM=0.0; LW=0.0
case "$ARM" in
  cls) ;;
  cls_slow) AM=0.9995;;
  cls_slow2) AM=0.9998;;
  global) GLOBAL_TOKENS=cls_patchmean;;
  *) echo "unknown arm $ARM"; exit 1;;
esac
GLOBAL_TOKENS=${GLOBAL_TOKENS_OVERRIDE:-$GLOBAL_TOKENS}; RW=${RW_OVERRIDE:-$RW}; AM=${AM_OVERRIDE:-$AM}

# ---- data: identical to the baseline launcher (scripts/launch_hs6_l_20tb_route2_8x5090zxr_20260924.sh) ----
ONE_TB=${ONE_TB:-/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle}
FOUR_TB=${FOUR_TB:-/mnt/huawei_blm/deepcad_5t_v1/wds_patched_shuffle}
ROUTE2=${ROUTE2:-/mnt/deepcad_nfs/deepcad_100t/final-data/20TB_takeover_20260922_2350/route2_strict_pathology_20260923/route2_15tb_no_old5_micro_tars}
R0_WEIGHT=${R0_WEIGHT:-0.5581}; R9_WEIGHT=${R9_WEIGHT:-0.1419}
DATASET_PATH="mixwds_robust:0.09=$ONE_TB/filtered_mixed_train_w*.tar||0.21=$FOUR_TB/filtered_mixed_train*.tar||$R0_WEIGHT=$ROUTE2/filtered_projection_20TB_nested-r0*.tar||$R9_WEIGHT=$ROUTE2/filtered_projection_20TB_nested-r9*.tar::pct=1,99"
RGB_MEAN='[0.43700109438633017,0.4152422629198214,0.4177610059272783]'
RGB_STD='[0.34138578574532263,0.34111549859535956,0.3432925245219299]'
count_tars() { find "$1" -maxdepth 1 -type f -name "$2" | wc -l; }
n1=$(count_tars "$ONE_TB" 'filtered_mixed_train_w*.tar'); n4=$(count_tars "$FOUR_TB" 'filtered_mixed_train*.tar')
nm=$(count_tars "$ROUTE2" 'filtered_projection_20TB_nested-r0*.tar'); np=$(count_tars "$ROUTE2" 'filtered_projection_20TB_nested-r9*.tar')
[[ "$n1" -eq 326 && "$n4" -eq 1251 && "$nm" -eq 3671 && "$np" -eq 283 ]] || { echo "ERROR: unexpected shards 1TB=$n1/326 4TB=$n4/1251 r0=$nm/3671 r9=$np/283" >&2; exit 2; }

IFS=',' read -r -a gpu_ids <<<"$GPUS"
NGPU=${#gpu_ids[@]}
[[ $((NGPU * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS)) -eq $GLOBAL_BATCH_SIZE ]] || { echo "ERROR: need 8 gpus (8 x 16 x acc8 = 1024); got $NGPU" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing python $PYTHON_BIN" >&2; exit 2; }
[[ -f "$FORK/ckpt/$FORK_IT/checkpoint.pth" ]] || { echo "ERROR: missing forked checkpoint $FORK/ckpt/$FORK_IT/checkpoint.pth" >&2; exit 2; }
[[ -f "$ANCHOR" ]] || { echo "ERROR: missing anchor $ANCHOR" >&2; exit 2; }
mkdir -p "$OUT/ckpt/$FORK_IT"
[[ -f "$OUT/ckpt/$FORK_IT/checkpoint.pth" ]] || ln "$FORK/ckpt/$FORK_IT/checkpoint.pth" "$OUT/ckpt/$FORK_IT/checkpoint.pth" 2>/dev/null || cp "$FORK/ckpt/$FORK_IT/checkpoint.pth" "$OUT/ckpt/$FORK_IT/checkpoint.pth"
latest=$(find "$OUT/ckpt" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | awk '/^[0-9]+$/' | sort -n | tail -1)

log() { printf '[%s] [20tb-v2-%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$ARM" "$*"; }
log "arm=$ARM gpus=$GPUS port=$PORT resume_from=ck$latest recovery: tokens=$GLOBAL_TOKENS weight=$RW anchor_momentum=$AM"

while true; do
  busy=()
  for gpu in "${gpu_ids[@]}"; do
    pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | paste -sd, -)
    [[ -z "$pids" ]] || busy+=("gpu${gpu}:${pids}")
  done
  [[ ${#busy[@]} -eq 0 ]] && break
  log "waiting for selected GPUs: ${busy[*]}"; sleep "$POLL_SECONDS"
done

cd "$REPO"
cat > "$OUT/launch_manifest_$(date -u +%Y%m%dT%H%M%SZ).json" <<JSON
{"time":"$(date -u +%FT%TZ)","host":"$(hostname)","arm":"$ARM","gpus":"$GPUS","port":$PORT,"resume_from":$latest,
 "fork_source":"20TB route2 no-GRAM baseline full state ck$FORK_IT (8x5090zxr_20260924)","optimizer":"continuous (AdamW moments copied)",
 "anchor":"EMA teacher @$FORK_IT (in-run)","recovery":{"mode":"fixed","local_weight":$LW,"global_weight":1.0,"loss_weight":$RW,"global_tokens":"$GLOBAL_TOKENS","anchor_momentum":$AM},
 "layout":"fsdp $NGPU ranks x bs$BATCH_SIZE_PER_GPU x acc$GRAD_ACCUM_STEPS = $GLOBAL_BATCH_SIZE","git_head":"$(git -C $REPO rev-parse --short HEAD 2>/dev/null || echo unknown)"}
JSON
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=${NCCL_DEBUG:-WARN} NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MALLOC_MMAP_THRESHOLD_=131072 MALLOC_ARENA_MAX=4
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
ulimit -n 65536 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="$GPUS"
log "launching"
exec "$PYTHON_BIN" -m torch.distributed.run --nnodes=1 --node_rank=0 --nproc_per_node="$NGPU" --master_addr=127.0.0.1 --master_port="$PORT" \
  dinov3/train/train.py --config-file dinov3/configs/train/microscopy_continual_vitl16.yaml --output-dir "$OUT" --seed 0 \
  compute_precision.distributed_mode=fsdp compute_precision.param_dtype=bf16 \
  "train.dataset_path=$DATASET_PATH" train.batch_size_per_gpu=$BATCH_SIZE_PER_GPU train.num_workers=$NUM_WORKERS train.seed=0 \
  train.OFFICIAL_EPOCH_LENGTH=$OFFICIAL_EPOCH_LENGTH train.start_iteration_override=null train.max_updates=null \
  train.cache_dataset=false train.compile=false train.wds_shuffle_buffer=$WDS_SHUFFLE_BUFFER train.prefetch_factor=$PREFETCH_FACTOR train.pin_memory=false \
  train.checkpointing=false train.checkpointing_full=false train.checkpointing_blocks=0 \
  student.in_chans=3 teacher.in_chans=3 student.enable_channelvit=false teacher.enable_channelvit=false \
  student.stem_type=null teacher.stem_type=null student.norm_layer=layernormbf16 student.pos_embed_rope_rescale_coords=2 student.pos_embed_rope_dtype=fp32 \
  "student.resume_from_teacher_chkpt=$ANCHOR" \
  optim.epochs=$EPOCHS optim.scaling_rule=fixed optim.lr=0.0001 optim.min_lr=0.000001 optim.warmup_epochs=3 optim.freeze_last_layer_epochs=1 \
  optim.gradient_accumulation_steps=$GRAD_ACCUM_STEPS teacher.warmup_teacher_temp_epochs=30 \
  crops.global_crops_size=256 crops.local_crops_size=112 crops.gram_teacher_crops_size=null crops.gram_teacher_no_distortions=false \
  crops.localcrops_subset_of_globalcrops=false crops.share_color_jitter=false crops.paired_global_geometry=false \
  crops.augmentation_policy=bio_safe crops.horizontal_flips=true crops.float_input=false "crops.rgb_mean=$RGB_MEAN" "crops.rgb_std=$RGB_STD" \
  sigreg.enabled=false channel_subset.enabled=false \
  gram.use_loss=true gram.require_official_fixed_anchor_contract=false gram.compute_stats=false gram.loss_weight=2.0 gram.inter_image_loss_weight=0.0 \
  gram.global_relation_loss_weight=0.0 gram.global_relation_ckpt=null gram.ema_teacher=false "gram.ckpt=$ANCHOR" gram.it_load_ema_teacher=-1 \
  gram.rep_update=true gram.update_frequency=10000 gram.it_first_update=1010000 gram.max_updates=3 gram.tokens_used=all gram.normalized=true gram.img_level=true \
  gram.remove_neg=false gram.remove_only_teacher_neg=false gram.loss_weight_schedule=null gram.global_teacher_resize_method=bicubic gram.global_teacher_resize_antialias=false \
  recovery.enabled=true recovery.mode=fixed recovery.loss_weight=$RW recovery.global_weight=1.0 recovery.local_weight=$LW \
  recovery.global_tokens=$GLOBAL_TOKENS recovery.anchor_momentum=$AM \
  evaluation.eval_period_iterations=$EVAL_PERIOD checkpointing.period=$CHECKPOINT_PERIOD checkpointing.max_to_keep=null checkpointing.keep_every=$CHECKPOINT_KEEP_EVERY checkpointing.sharded=false
