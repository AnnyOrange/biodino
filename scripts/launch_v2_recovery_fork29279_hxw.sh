#!/usr/bin/env bash
# Launch one Adaptive-v2 arm on 5090-hxw-xzj, forked WITHOUT optimizer restart from the original 5TB no-GRAM
# full-state checkpoint at 29279 (see scripts/fork_vanilla_checkpoint_for_recovery.py).
#   usage: bash scripts/launch_v2_recovery_fork29279_hxw.sh <arm: global|global_local> <gpus e.g. 0,1> <master_port> [max_updates]
# Arms differ ONLY in recovery.local_weight (0 = global stream only, 1 = global + 16-patch local stream).
# Everything else mirrors the 20260927 Adaptive continuation launch, except: recovery.mode=fixed (no adaptive gate),
# anchor = EMA teacher at the fork step (in-run anchor), DDP 2 ranks x bs64 x acc8 = global 1024 on 32GB 5090s.
set -euo pipefail
ARM=${1:?arm}; GPUS=${2:?gpus}; PORT=${3:?port}; MAXU=${4:-35136}
case "$ARM" in global) LW=0.0;; global_local) LW=1.0;; *) echo "arm must be global|global_local"; exit 1;; esac
REPO=${REPO:-$HOME/biodino}
ROOT=${ROOT:-$REPO/outputs/01_training_runs/hs6_l5_v2_recovery_fork29279_20260930}
FORK=$ROOT/fork; OUT=$ROOT/$ARM
ANCHOR=$FORK/anchor/teacher_checkpoint.pth
DATA="mixwds_robust:0.3=/data/microscopy-100k-patched/filtered_mixed_train_w*.tar||0.7=$HOME/storage/merged/4TB/wds_patched_shuffle/filtered_mixed_train*.tar::pct=1,99"
PY=${PY:-$HOME/miniconda3/envs/dinov3/bin/python}
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l); [ "$NGPU" -eq 2 ] || { echo "expect 2 gpus per arm (bs64 x 2 x acc8 = 1024)"; exit 1; }
[ -f "$FORK/ckpt/29279/checkpoint.pth" ] || { echo "missing forked checkpoint $FORK/ckpt/29279/checkpoint.pth"; exit 1; }
[ -f "$ANCHOR" ] || { echo "missing anchor $ANCHOR"; exit 1; }
mkdir -p "$OUT/ckpt/29279"
[ -f "$OUT/ckpt/29279/checkpoint.pth" ] || ln "$FORK/ckpt/29279/checkpoint.pth" "$OUT/ckpt/29279/checkpoint.pth" 2>/dev/null || cp "$FORK/ckpt/29279/checkpoint.pth" "$OUT/ckpt/29279/checkpoint.pth"
cd "$REPO"
cat > "$OUT/launch_manifest.json" <<JSON
{"time":"$(date -u +%FT%TZ)","host":"$(hostname)","arm":"$ARM","gpus":"$GPUS","port":$PORT,"max_updates":$MAXU,
 "fork_source":"original 5TB no-GRAM full state ck29279 (sha256 4f67ef63...)","optimizer":"continuous (AdamW step 29280)",
 "anchor":"EMA teacher @29279 (in-run)","recovery":{"mode":"fixed","local_weight":$LW,"global_weight":1.0,"loss_weight":1.0},
 "layout":"ddp 2 ranks x bs64 x acc8 = 1024","git_head":"$(cat $REPO/GIT_HEAD 2>/dev/null || echo unknown)"}
JSON
CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$REPO OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup $PY -m torch.distributed.run --nproc_per_node=2 --master_port=$PORT dinov3/train/train.py \
  --config-file dinov3/configs/train/microscopy_continual_vitl16.yaml --output-dir "$OUT" --seed 0 \
  compute_precision.distributed_mode=ddp compute_precision.param_dtype=bf16 \
  "train.dataset_path=$DATA" train.batch_size_per_gpu=64 train.num_workers=4 train.seed=0 train.OFFICIAL_EPOCH_LENGTH=4098 \
  train.start_iteration_override=null train.max_updates=$MAXU train.cache_dataset=false train.compile=false \
  train.wds_shuffle_buffer=50 train.wds_deterministic_resampling=true train.prefetch_factor=2 train.pin_memory=true \
  train.checkpointing=true train.checkpointing_full=true train.checkpointing_blocks=24 \
  student.in_chans=3 teacher.in_chans=3 student.enable_channelvit=false teacher.enable_channelvit=false \
  student.stem_type=null teacher.stem_type=null student.norm_layer=layernormbf16 student.pos_embed_rope_rescale_coords=2 student.pos_embed_rope_dtype=fp32 \
  "student.resume_from_teacher_chkpt=$ANCHOR" \
  optim.epochs=15 optim.scaling_rule=fixed optim.lr=0.0001 optim.min_lr=1e-06 optim.warmup_epochs=3 optim.freeze_last_layer_epochs=1 optim.gradient_accumulation_steps=8 \
  teacher.warmup_teacher_temp_epochs=30 \
  crops.global_crops_size=256 crops.local_crops_size=112 crops.gram_teacher_crops_size=512 crops.gram_teacher_no_distortions=true \
  crops.localcrops_subset_of_globalcrops=false crops.share_color_jitter=false crops.paired_global_geometry=false crops.augmentation_policy=bio_safe crops.horizontal_flips=false crops.float_input=false \
  "crops.rgb_mean=[0.5126699404721016,0.5020022506395592,0.5064769301636908]" "crops.rgb_std=[0.3497517202150124,0.34941518705400204,0.34802097842537794]" \
  sigreg.enabled=false channel_subset.enabled=false \
  gram.use_loss=true gram.require_official_fixed_anchor_contract=false gram.compute_stats=false gram.loss_weight=2.0 gram.inter_image_loss_weight=0.0 \
  gram.global_relation_loss_weight=0.0 gram.global_relation_ckpt=null gram.ema_teacher=false "gram.ckpt=$ANCHOR" gram.it_load_ema_teacher=-1 \
  gram.rep_update=true gram.update_frequency=10000 gram.it_first_update=1010000 gram.max_updates=3 gram.tokens_used=all gram.normalized=true gram.img_level=true \
  gram.remove_neg=false gram.remove_only_teacher_neg=false gram.loss_weight_schedule=null gram.global_teacher_resize_method=bicubic gram.global_teacher_resize_antialias=false \
  recovery.enabled=true recovery.mode=fixed recovery.loss_weight=1.0 recovery.global_weight=1.0 recovery.local_weight=$LW \
  evaluation.eval_period_iterations=488 checkpointing.period=488 checkpointing.max_to_keep=100 checkpointing.keep_every=99999999999999999 checkpointing.sharded=false \
  > "$OUT/console.log" 2>&1 &
echo "launched $ARM on GPUs $GPUS (pid $!) -> $OUT/console.log"
