"""One fixed50epoch frozen HoVerNet fit, both complete official DSB scoring sets."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
PY=os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable!=PY:os.execv(PY,[PY,'-u',__file__,*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),'/mnt/huawei_deepcad/benchmark_model']
os.environ['DINOV3_CODE_ROOT']=str(ROOT)
import numpy as np,torch
from torch.utils.data import ConcatDataset
from dinov3.eval.bio_segmentation.instance_seg import train as trainer
from dinov3.eval.bio_segmentation.instance_seg.model import DINOHoVerNet
from dinov3.eval.bio_segmentation.datasets.bbbc038_official import BBBC038OfficialDataset,dsb_mean_precision,resolve_path
from dinov3.eval.bio_segmentation.metrics import accumulate_instance_metrics
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
MANIFEST=ROOT/'outputs/02_eval_inputs/formal_v4/bbbc038_official_manifest.json'
def sha(p):return hashlib.sha256(resolve_path(p).read_bytes()).hexdigest()
def dump(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def main():
    from unified_paper_gpu_admission import reserve_for_current_command
    reserve_for_current_command()
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--seed',type=int,choices=range(5),required=True);a=p.parse_args()
    spec=next(m for m in json.loads((OUT/'config.json').read_text())['models']if m['model']==a.model)
    out=OUT/'results'/a.model/'bbbc038_official'/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'component_result.json').exists():return
    torch.set_num_threads(2)
    manifest=json.loads(MANIFEST.read_text());manifest_sha=sha(MANIFEST)
    assert {s:len(r)for s,r in manifest['splits'].items()}==dict(train=670,stage1=65,stage2=106)
    for rows in manifest['splits'].values():
        for r in rows:
            assert sha(r['image'])==r['image_sha256']and sha(r['instances'])==r['instances_sha256']
    stage_datasets={s:BBBC038OfficialDataset(MANIFEST,s,do_normalize=True)for s in ['stage1','stage2']}
    dataset=ConcatDataset(list(stage_datasets.values()));dataset.records=[r for ds in stage_datasets.values()for r in ds.records]
    train_ds=BBBC038OfficialDataset(MANIFEST,'train',do_normalize=False)
    train_ds.preload()
    trainer._build_instance_dataset=lambda args,split,do_normalize=True:train_ds if split=='train'else dataset
    original_builder=trainer.build_dino_hovernet;models=[]
    def build(**kwargs):
        if spec['kind']=='biodino':
            from omegaconf import OmegaConf
            arch=OmegaConf.load(spec['train_config']).student.arch
            kwargs['layers']=[{'vit_small':11,'vit_base':11,'vit_large':23,'vit_huge2':31}[arch]]
            model=original_builder(**kwargs)
        else:
            from run_fm_companion_rules import FMSpatialBackbone
            from run_new_fm_v4_detection_20261016 import NewSpatialBackbone,MODELS
            from run_unified_paper_ctc import new_spatial_backbone_class
            cls=new_spatial_backbone_class(NewSpatialBackbone)if a.model in MODELS else FMSpatialBackbone
            backbone=cls(a.model)
            model=DINOHoVerNet(backbone,layers=[0],freeze_backbone=True,num_types=0,feature_size=32,embed_proj=384).cuda()
            model._extract=lambda images:[backbone.get_intermediate_layers(images,n=1,reshape=True)[0].clone()]
        assert not any(p.requires_grad for p in model.backbone.parameters())
        digest=hashlib.sha256()
        for name,tensor in model.decoder.state_dict().items():digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
        dump(out/'head_initialization.json',dict(seed=a.seed,sha256=digest.hexdigest(),trainable_encoder_parameters=0))
        models.append(model);return model
    trainer.build_dino_hovernet=build
    records=[]
    @torch.inference_mode()
    def evaluate(model,base_ds,device,num_types,crop_size,stride,patch_size,max_images=None,tta=False,fg_thresh=.5,energy_thresh=.4):
        assert max_images is None and len(base_ds)==171
        model.eval();predictions=[];truths=[];pred_dir=out/'predictions';pred_dir.mkdir(exist_ok=True)
        for index,r in enumerate(base_ds.records):
            image,semantic,truth=base_ds[index]
            logits=trainer.sliding_window_predict(model,image.to(device),crop_size=crop_size,stride=stride,patch_size=patch_size,num_types=num_types,tta=tta)
            pred,_=trainer.postprocess(logits['np'],logits['hv'],logits['tp'],fg_thresh=fg_thresh,energy_thresh=energy_thresh)
            truth=truth.numpy().astype(np.int32);pred=pred.astype(np.int32)
            path=pred_dir/f"{r['sample_id']}.npy";np.save(path,pred)
            metric=accumulate_instance_metrics([pred],[truth]);metric['dsb_mean_precision']=dsb_mean_precision(pred,truth)
            stage='stage1'if index<65 else 'stage2'
            records.append(dict(model=a.model,checkpoint=spec['checkpoint'],dataset='bbbc038-official',task='instance_segmentation',fold=stage,seed=a.seed,sample_id=r['sample_id'],group_id=r['group_id'],truth_path=r['instances'],truth_sha256=r['instances_sha256'],prediction_path=str(path),prediction_sha256=sha(path),metrics=metric))
            predictions.append(pred);truths.append(truth)
            if (index+1)%10==0:print('OFFICIAL TEST',a.model,a.seed,index+1,171,flush=True)
        dump(out/'sample_predictions.json',records)
        result=accumulate_instance_metrics(predictions,truths);result['dsb_mean_precision']=float(np.mean([r['metrics']['dsb_mean_precision']for r in records]));return result
    trainer.evaluate=evaluate
    args=trainer.parse_args(['--dataset','bbbc038','--data-root',str(MANIFEST),'--checkpoint',spec['checkpoint'],'--train-config',spec.get('train_config','not_applicable'),'--output-dir',str(out),'--layers','0','--freeze-backbone','--epochs','50','--batch-size','8','--seed',str(a.seed),'--num-workers','2','--selection-mode','final_epoch'])
    invocation=dict(model=a.model,checkpoint=spec['checkpoint'],dataset='bbbc038-official',seed=a.seed,protocol=manifest['protocol'],manifest=str(MANIFEST),manifest_sha256=manifest_sha,
        recipe=vars(args),seed_semantics='Training randomness: decoder initialization, minibatch shuffle, crop and augmentation; fixed train/test memberships',
        feature_policy='One final frozen feature map; same HoVerNet decoder. FM published normalization/channels at required224 input, bilinear common stride16 decoder grid, as native CTC adapter.',
        selection='Predeclared final epoch50; no validation or test hyperparameter selection; all670 official training images',group_limitation=manifest['grouping'])
    dump(out/'invocation.json',invocation)
    print('START OFFICIAL DSB',a.model,a.seed,'50epochs B8 train670 test65+106',flush=True)
    result=trainer.run(args)
    assert len(records)==171 and 'test'in result and result['_meta']['global_steps']==50*(670//8)
    stages={}
    for stage in ['stage1','stage2']:
        selected=[r for r in records if r['fold']==stage];stages[stage]=dict(n_test=len(selected),metrics={k:float(np.mean([r['metrics'][k]for r in selected]))for k in selected[0]['metrics']})
    dump(out/'component_result.json',{**invocation,'task':'instance_segmentation','stages':stages,'prediction_path':str(out/'sample_predictions.json'),'prediction_sha256':sha(out/'sample_predictions.json'),'head_sha256':sha(out/'best_head.pth'),'initialization':str(out/'head_initialization.json'),'training_result':str(out/'results.json'),'completed_epochs':50,'completed_at':time.time()})
    print('COMPLETE OFFICIAL DSB',a.model,a.seed,stages,flush=True)
if __name__=='__main__':main()
