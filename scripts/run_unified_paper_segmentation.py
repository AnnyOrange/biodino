#!/usr/bin/env python3
"""Extend already audited frozen BioDINO dense caches to missing downstream seeds."""
import argparse,json,os,sys
from pathlib import Path
PY=os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable!=PY:os.execv(PY,[PY,'-u',__file__,*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/'outputs/02_eval_runs/unified_paper'
import torch
from dinov3.eval.bio_segmentation.linear_probe import run_cached_linear_probe,DATASET_CONFIGS
from dinov3.eval.bio_frozen_eval.external_fm_protocol import validate_best_head
from unified_paper_segmentation_artifacts import recover_predictions,attach_source_manifest

def main():
 from unified_paper_gpu_admission import reserve_for_current_command
 reserve_for_current_command()
 p=argparse.ArgumentParser();p.add_argument('--cell-index',type=int,required=True);p.add_argument('--inventory-file',type=Path,default=OUT/'biodino_segmentation_inventory.json');p.add_argument('--budget',type=int,choices=[20,50],required=True);p.add_argument('--seed',type=int,choices=range(5),required=True);p.add_argument('--recover-existing',action='store_true');a=p.parse_args()
 cell=json.loads(a.inventory_file.read_text())['cells'][a.cell_index]
 if cell['dataset']=='multimodal_cellseg' and cell['split']!='neurips22-author-public52-v2':raise ValueError('Withdrawn cross-set label pairing; fresh author-public52 split required')
 if a.recover_existing:
  if cell.get('requires_fresh_training'):raise ValueError('Original cache unavailable and retained head failed reproduction; shared-protocol fresh fits queued separately')
  if not Path(cell['caches']['test']).is_file():raise ValueError('Test cache unavailable for retained-head recovery')
 elif not all(Path(p).is_file() for p in cell['caches'].values()):raise ValueError('Frozen split caches unavailable')
 torch.set_num_threads(2);out=OUT/'results'/cell['model']/cell.get('result_family','segmentation')/cell['dataset']/cell['split']/f'E{a.budget}'/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'component_result.json').exists():return
 if 'class_weight_mode' in cell:weight=cell['class_weight_mode']
 else:
  inv=json.loads(Path(cell['invocation']).read_text());cmd=inv['command'];weight=cmd[cmd.index('--probe-class-weight-mode')+1]
 (out/'invocation_manifest.json').write_text(json.dumps({'cell':cell,'budget':a.budget,'seed':a.seed,'seed_semantics':'head initialization, dropout and shuffle; fixed cached members','class_weight_mode':weight},indent=2)+'\n')
 if a.recover_existing:r=recover_predictions(cell,a.budget,a.seed,out)
 elif (out/'results.json').exists() and (out/'test_semantic_predictions.npz').exists():
  r=json.loads((out/'results.json').read_text());meta=r['_meta']
  assert meta['seed']==a.seed and meta['probe_epochs']==a.budget and meta['probe_batch_size']==32
  assert meta['class_weight_mode']==weight and meta['optimizer']=='AdamW' and meta['learning_rate']==1e-3 and meta['weight_decay']==1e-4
  validate_best_head(meta)
  print('REUSE completed fit and test predictions; finish source manifest attachment only',flush=True)
 else:r=run_cached_linear_probe(train_cache=cell['caches']['train'],val_cache=cell['caches']['val'],test_cache=cell['caches']['test'],output_dir=str(out),**DATASET_CONFIGS[cell['dataset']],epochs=a.budget,lr=1e-3,batch_size=32,weight_decay=1e-4,dropout=.1,num_workers=0,eval_every=1,seed=a.seed,class_weight_mode=weight,save_predictions=True)
 torch.cuda.empty_cache()
 from unified_paper_gpu_admission import release_current_reservation
 release_current_reservation()
 validate_best_head(r['_meta']);r.update(model=cell['model'],dataset=cell['dataset'],task='segmentation',budget=a.budget,seed=a.seed,split=cell['split'],source_invocation=cell['invocation'],prediction_path=str(out/'test_semantic_predictions.npz'))
 r.update(attach_source_manifest(cell,out));r['execution_mode']='retained_head_prediction_recovery' if a.recover_existing else 'downstream_training'
 tmp=out/'component_result.tmp';tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(out/'component_result.json');print('COMPLETE',cell['model'],cell['dataset'],a.budget,a.seed,r['test']['mDice'],flush=True)
if __name__=='__main__':main()
