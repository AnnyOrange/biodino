#!/usr/bin/env python3
"""Explicit DDP launch with matched reset/stream/schedule and a durable manifest."""
import argparse, json, os, subprocess
from pathlib import Path

REPO=Path('/mnt/huawei_deepcad/dinov3')
SOURCE=Path('/mnt/huawei_deepcad/dinov3_selective_retention_snapshot_20260923')
BASE=REPO/'outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907'
ROOT=REPO/'outputs/01_training_runs/hs6_l5_selective_retention_20260923'

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['vanilla','gram','fixed','adaptive','adaptive_strong'],required=True)
 p.add_argument('--gpus',required=True);p.add_argument('--batch',type=int,default=64)
 p.add_argument('--steps',type=int,default=2440);p.add_argument('--period',type=int,default=488)
 p.add_argument('--tag',default='formal');p.add_argument('--checkpoint',type=int,default=12687)
 p.add_argument('--port',type=int,default=32630);p.add_argument('--checkpoint-blocks',type=int,default=0)
 p.add_argument('--resume',action='store_true')
 a=p.parse_args();n=len(a.gpus.split(','));assert 1024%(n*a.batch)==0
 out=ROOT/f'{a.arm}_{a.tag}';out.mkdir(parents=True,exist_ok=True)
 anchor=BASE/f'eval/training_{a.checkpoint}/teacher_checkpoint.pth'
 start=a.checkpoint+1;recover=a.arm in ('fixed','adaptive','adaptive_strong')
 opt={
 'compute_precision.distributed_mode':'ddp','compute_precision.param_dtype':'bf16',
 'train.dataset_path':'mixwds_robust:0.3=/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*.tar||0.7=/mnt/huawei_blm/deepcad_5t_v1/wds_patched_shuffle/filtered_mixed_train*.tar::pct=1,99',
 'train.batch_size_per_gpu':a.batch,'train.num_workers':4,'train.seed':0,
 'train.OFFICIAL_EPOCH_LENGTH':4098,'train.start_iteration_override':start,'train.max_updates':start+a.steps,
 'train.cache_dataset':False,'train.compile':False,'train.wds_shuffle_buffer':50,
 'train.wds_deterministic_resampling':True,'train.prefetch_factor':2,'train.pin_memory':True,
 'train.checkpointing':a.checkpoint_blocks>0,'train.checkpointing_full':True,'train.checkpointing_blocks':a.checkpoint_blocks,
 'student.in_chans':3,'teacher.in_chans':3,'student.enable_channelvit':False,'teacher.enable_channelvit':False,
 'student.stem_type':None,'teacher.stem_type':None,'student.norm_layer':'layernormbf16',
 'student.pos_embed_rope_rescale_coords':2,'student.pos_embed_rope_dtype':'fp32','student.resume_from_teacher_chkpt':str(anchor),
 'optim.epochs':15,'optim.scaling_rule':'fixed','optim.lr':.0001,'optim.min_lr':.000001,
 'optim.warmup_epochs':3,'optim.freeze_last_layer_epochs':1,'optim.gradient_accumulation_steps':1024//(n*a.batch),
 'teacher.warmup_teacher_temp_epochs':30,'crops.global_crops_size':256,'crops.local_crops_size':112,
 'crops.gram_teacher_crops_size':512,'crops.gram_teacher_no_distortions':True,
 'crops.localcrops_subset_of_globalcrops':False,'crops.share_color_jitter':False,'crops.paired_global_geometry':False,
 'crops.augmentation_policy':'bio_safe','crops.horizontal_flips':False,'crops.float_input':False,
 'crops.rgb_mean':[.5126699404721016,.5020022506395592,.5064769301636908],
 'crops.rgb_std':[.3497517202150124,.34941518705400204,.34802097842537794],
 'sigreg.enabled':False,'channel_subset.enabled':False,'gram.use_loss':a.arm!='vanilla',
 'gram.require_official_fixed_anchor_contract':a.arm=='gram','gram.compute_stats':False,'gram.loss_weight':2.0,
 'gram.inter_image_loss_weight':0.,'gram.global_relation_loss_weight':0.,'gram.global_relation_ckpt':None,
 'gram.ema_teacher':False,'gram.ckpt':str(anchor),'gram.it_load_ema_teacher':-1,
 'gram.rep_update':True,'gram.update_frequency':10000,'gram.it_first_update':1010000,'gram.max_updates':3,
 'gram.tokens_used':'all','gram.normalized':True,'gram.img_level':True,'gram.remove_neg':False,
 'gram.remove_only_teacher_neg':False,'gram.loss_weight_schedule':None,
 'gram.global_teacher_resize_method':'bicubic','gram.global_teacher_resize_antialias':False,
 'recovery.enabled':recover,'recovery.mode':'fixed' if a.arm=='fixed' else 'adaptive',
 'recovery.loss_weight':5. if a.arm=='adaptive_strong' else 1.,
 'evaluation.eval_period_iterations':a.period,'checkpointing.period':a.period,
 'checkpointing.max_to_keep':2,'checkpointing.keep_every':99999999999999999,'checkpointing.sharded':False,
 }
 py='/home/lxy/miniconda3/envs/dinov3/bin/python'
 cmd=[py,'-m','torch.distributed.run',f'--nproc_per_node={n}',f'--master_port={a.port}',
      'dinov3/train/train.py','--no-resume','--config-file','dinov3/configs/train/microscopy_continual_vitl16.yaml',
      '--output-dir',str(out),'--seed','0']
 def value(v):
  if isinstance(v,(bool,type(None),list)):return json.dumps(v,separators=(',',':'))
  return str(v)
 if a.resume:
  cmd.remove('--no-resume');opt['train.start_iteration_override']=None
 cmd.extend(f'{k}={value(v)}' for k,v in opt.items())
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpus,PYTHONPATH=str(SOURCE),OMP_NUM_THREADS='2',
    OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 manifest={'arm':a.arm,'source':str(SOURCE),'command':cmd,'gpus':a.gpus,'actual_batch_per_rank':a.batch,
           'effective_batch':1024,'start':start,'end':start+a.steps-1,'optimizer':'fresh_from_same_teacher',
           'distributed_mode':'ddp','selection':'fixed budget; no downstream test feedback'}
 manifest['resume']=a.resume
 (out/('resume_manifest.json' if a.resume else 'launch_manifest.json')).write_text(json.dumps(manifest,indent=2))
 with (out/'console.log').open('a',buffering=1) as log:
  proc=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
  (out/'trainer.pid').write_text(str(proc.pid));rc=proc.wait()
 (out/'exit.json').write_text(json.dumps({'returncode':rc}));raise SystemExit(rc)

if __name__=='__main__':main()
