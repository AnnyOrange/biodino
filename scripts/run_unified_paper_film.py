#!/usr/bin/env python3
"""FILM fixed biological-group split; five actual random linear-head fits."""
import argparse,hashlib,json,os,sys,time,socket
from pathlib import Path
PY=os.environ.get('UNIFIED_PAPER_PYTHON','/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python')
if sys.executable!=PY:os.execv(PY,[PY,'-u',__file__,*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),'/mnt/huawei_deepcad/benchmark_model']
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,accuracy_score,f1_score
from dinov3.eval.bio_frozen_eval.film import FILMAgeDataset,aggregate_group_features
from run_unified_paper_global import make_encoder
OUT=ROOT/'outputs/02_eval_runs/unified_paper';INPUT=ROOT/'outputs/02_eval_inputs/formal_v4/film_fixed_manifest.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def prepare():
 source=ROOT/'outputs/02_eval_runs/unprotocolized_continuation_20260917/data_repair/film_manifest.json'
 m=json.loads(source.read_text());fold=m['repetitions'][0]['folds'][0];inner=fold['inner_folds'][0]
 split={'train':inner['train_indices'],'val':inner['val_indices'],'test':fold['test_indices']}
 groups={k:{m['records'][i]['group_id'] for i in ix} for k,ix in split.items()}
 assert not(groups['train']&groups['test'] or groups['train']&groups['val'] or groups['val']&groups['test'])
 for k,ix in split.items():assert len({m['records'][i]['target'] for i in ix})==4
 canonical_source='/mnt/huawei_deepcad/dinov3/'+str(source.relative_to(ROOT))
 cfg={'protocol':'film-fixed-biological-groups-linear5-v1','source_manifest':canonical_source,'source_manifest_sha256':sha(source),'records':m['records'],'splits':split,
 'split_origin':'Reuse existing seed0 outer fold0 test and inner fold0 train/validation, fixed once before model extraction; no new per-seed split.',
 'split_counts':{k:len(v) for k,v in groups.items()},'preprocessing':'Shared raw126-band stack p1/p99, spectral mean replicated RGB, resize256/crop224; published encoder normalization',
 'feature':'Frozen final CLS + patch mean (published no-CLS fallback); no 4-even selection',
 'head':'Train-only StandardScaler; torch Linear(D,4); class-balanced CE; full-batch Adam lr0.001 wd0; 500 epochs; validation balanced_accuracy every25epochs, tie earliest. Seeds0..4 change head initialization only.',
 'epochs':500,'seeds':list(range(5)),'primary_metric':'balanced_accuracy','official':False,'reason':'Later fixed-test seed supplement supersedes earlier repeated outer-split development; source task is project-defined age classification, not source-author assay.',
 'limitations':m['limitations']}
 if INPUT.exists():assert json.loads(INPUT.read_text())==cfg
 else:dump(INPUT,cfg)
 if not (OUT/'film_config.json').exists():dump(OUT/'film_config.json',cfg)
 return cfg

def main():
 p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--prepare',action='store_true');a=p.parse_args();cfg=prepare()
 if a.prepare:print(cfg['split_counts']);return
 from unified_paper_gpu_admission import reserve_for_current_command
 reserve_for_current_command();torch.set_num_threads(2)
 spec=next(m for m in json.loads((OUT/'config.json').read_text())['models'] if m['model']==a.model)
 root=OUT/'results'/a.model/'film_age';root.mkdir(parents=True,exist_ok=True);target=root/'component_result.json'
 if target.exists():return
 data_root=Path(os.environ.get('UNIFIED_PAPER_BENCHMARK_ROOT','/mnt/huawei_deepcad/benchmark'))/'ood/ood_classification/datasets/FILM'
 packed_images=None
 pack=Path(os.environ.get('UNIFIED_PAPER_FILM_PACK',str(OUT/'prepared_inputs/film_rgb')))
 if (pack/'manifest.json').is_file():
  transport=json.loads((pack/'manifest.json').read_text());assert transport['canonical_manifest_sha256']==sha(INPUT)
  assert transport['source_sha256']==[r['sha256'] for r in cfg['records']]
  assert sha(pack/'inputs.npz')==transport['archive_sha256']
  with np.load(pack/'inputs.npz',allow_pickle=False) as arrays:packed_images=arrays['images']
  assert packed_images.dtype==np.float32 and packed_images.shape==(len(cfg['records']),1,200,200)
  assert [hashlib.sha256(x.tobytes()).hexdigest() for x in packed_images]==transport['plane_sha256']
 else:
  for r in cfg['records']:assert sha(data_root/r['path'])==r['sha256']
 cache=root/'features.npy';meta=root/'features.json'
 if not cache.exists():
  encoder=make_encoder(spec,'film_age',224,256);ds=FILMAgeDataset(data_root,cfg) if packed_images is None else packed_images
  chunks=[]
  for start in range(0,len(ds),4):
   images=[ds[i][0].mean(0,keepdim=True).expand(3,-1,-1).contiguous() if packed_images is None else torch.from_numpy(packed_images[i]).expand(3,-1,-1).contiguous() for i in range(start,min(start+4,len(ds)))]
   chunks.append(encoder.encode_images(images));print('ENCODE',a.model,start+len(images),len(ds),flush=True)
  x=np.concatenate(chunks).astype(np.float32);np.save(cache,x);dump(meta,{'manifest_sha256':sha(INPUT),'checkpoint':spec['checkpoint'],'model':a.model,'shape':list(x.shape)})
  del encoder;torch.cuda.empty_cache()
 else:
  assert json.loads(meta.read_text())['manifest_sha256']==sha(INPUT);x=np.load(cache)
 from unified_paper_gpu_admission import release_current_reservation
 release_current_reservation()
 data={k:aggregate_group_features(x,cfg['records'],ix) for k,ix in cfg['splits'].items()}
 scaler=StandardScaler().fit(data['train'][0]);tx=torch.tensor(scaler.transform(data['train'][0]),dtype=torch.float32);ty=torch.tensor(data['train'][1],dtype=torch.long)
 vx=torch.tensor(scaler.transform(data['val'][0]),dtype=torch.float32);ex=torch.tensor(scaler.transform(data['test'][0]),dtype=torch.float32)
 weights=torch.tensor(len(ty)/(4*np.bincount(ty.numpy(),minlength=4)),dtype=torch.float32);results=[]
 for seed in cfg['seeds']:
  dst=root/f'seed{seed}/component_result.json'
  if dst.exists():results.append(json.loads(dst.read_text()));continue
  torch.manual_seed(seed);head=torch.nn.Linear(tx.shape[1],4);opt=torch.optim.Adam(head.parameters(),lr=.001);best=-1.;beststate=None;history=[]
  for epoch in range(1,501):
   loss=torch.nn.functional.cross_entropy(head(tx),ty,weight=weights);assert torch.isfinite(loss)
   opt.zero_grad();loss.backward();opt.step()
   if epoch%25==0:
    with torch.no_grad():val=float(balanced_accuracy_score(data['val'][1],head(vx).argmax(1).numpy()))
    history.append({'epoch':epoch,'train_loss':float(loss.detach()),'validation_balanced_accuracy':val})
    if val>best:best=val;bestepoch=epoch;beststate={k:v.detach().clone() for k,v in head.state_dict().items()}
  head.load_state_dict(beststate)
  with torch.no_grad():scores=torch.softmax(head(ex),1).numpy()
  truth=data['test'][1];pred=scores.argmax(1);metric={'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),'accuracy':float(accuracy_score(truth,pred)),'macro_f1':float(f1_score(truth,pred,average='macro',zero_division=0))}
  prediction=dst.parent/'predictions.json';dump(prediction,{'sample_id':data['test'][2],'group_id':data['test'][2],'y_true':truth.tolist(),'y_pred':pred.tolist(),'probabilities':scores.tolist()})
  dump(dst.parent/'history.json',history);torch.save({'head':beststate,'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_},dst.parent/'head.pth')
  result={'model':a.model,'checkpoint':spec['checkpoint'],'dataset':'film_age','task':'classification','seed':seed,'metrics':metric,'selected_epoch':bestepoch,'validation_score':best,'prediction_path':str(prediction),'protocol':cfg['protocol'],'manifest':str(INPUT),'manifest_sha256':sha(INPUT),'n_train':len(ty),'n_val':len(vx),'n_test':len(ex),'host':socket.gethostname(),'pid':os.getpid()};dump(dst,result);results.append(result);print('REAL FIT COMPLETE',a.model,seed,metric,flush=True)
 dump(target,{'model':a.model,'checkpoint':spec['checkpoint'],'dataset':'film_age','protocol':cfg['protocol'],'manifest':str(INPUT),'manifest_sha256':sha(INPUT),'metrics':{m:float(np.mean([r['metrics'][m] for r in results])) for m in results[0]['metrics']},'sd':{m:float(np.std([r['metrics'][m] for r in results],ddof=1)) for m in results[0]['metrics']},'seeds':results,'completed_seeds':len(results),'required_seeds':5,'host':socket.gethostname(),'pid':os.getpid()})
if __name__=='__main__':main()
