"""Validate the entire NuCLS target geometry, true-mask loss and native oracle."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from dinov3.eval.bio_detection.nucls import NuCLSDataset,genuine_mask_loss,mask_api,native_metrics
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def main():
 torch.set_num_threads(2);source=ROOT/'outputs/02_eval_inputs/formal_v4/nucls_native_manifest.json';manifest=json.loads(source.read_text())
 allset=NuCLSDataset(manifest,1,'train');allset.records=manifest['records'];mask_count=0
 for i in range(len(allset)):
  image,t=allset[i];assert tuple(image.shape[-2:])==tuple(t['masks'].shape[-2:])
  assert len(t['boxes'])==len(t['masks'])==len(t['labels']) and torch.all(t['boxes'][:,2:]>t['boxes'][:,:2])
  available=t['masks'].flatten(1).amax(1)<255
  assert all(bool(available[j])==o['genuine_mask'] for j,o in enumerate(allset.records[i]['objects']))
  assert torch.all(t['masks'][available].flatten(1).sum(1)>0)
  mask_count+=int(available.sum())
 boxes=torch.tensor([[1.,1.,7.,7.],[1.,1.,7.,7.]])
 masks=torch.zeros((2,8,8),dtype=torch.uint8);masks[0,2:6,2:6]=1;masks[1]=255
 logits=torch.zeros((2,9,4,4),requires_grad=True)
 loss=genuine_mask_loss(logits,[boxes],[masks],[torch.tensor([1,2])],[torch.tensor([0,1])]);loss.backward()
 assert torch.isfinite(loss) and logits.grad[0].abs().sum()>0 and logits.grad[1].abs().sum()==0
 all_missing=torch.zeros((1,9,4,4),requires_grad=True)
 missing_loss=genuine_mask_loss(all_missing,[boxes[:1]],[masks[1:]],[torch.tensor([2])],[torch.tensor([0])]);missing_loss.backward()
 assert missing_loss==0 and all_missing.grad.abs().sum()==0
 empty=torch.zeros((0,9,4,4),requires_grad=True)
 empty_loss=genuine_mask_loss(empty,[boxes[:0]],[masks[:0]],[torch.zeros(0,dtype=torch.long)],[torch.zeros(0,dtype=torch.long)])
 assert empty_loss==0
 ds=NuCLSDataset(manifest,1,'val');predictions=[];api=mask_api()
 for i in range(len(ds)):
  _,t=ds[i];keep=t['labels']!=8;pred={'boxes':t['boxes'][keep],'labels':t['labels'][keep],'scores':torch.ones(int(keep.sum())),'mask_rle':[]}
  for m in t['masks'][keep]:
   # No true mask exists for rectangle-only annotations; zero is a prediction.
   a=m.numpy();a=np.zeros_like(a) if a.max()==255 else a
   enc=api.encode(np.asfortranarray(a));enc['counts']=enc['counts'].decode('ascii');pred['mask_rle'].append(enc)
  predictions.append(pred)
 metrics,rows=native_metrics(ds,predictions)
 for key in ['bbox_ap_50_95','bbox_ap_50','bbox_ap_75','matched_mask_dice_pooled','matched_mask_dice_mean','matched_classification_accuracy','detection_f1_iou50']:assert abs(metrics[key]-1)<1e-10,(key,metrics)
 result={'status':'passed','images_validated':len(allset),'genuine_masks_validated':mask_count,'box_only_mask_gradient':0,'empty_target_loss':0,'oracle_images':len(ds),'oracle_metrics':metrics,'benchmark_result':False}
 out=OUT/'validation/nucls_native_adapter.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
 geom=OUT/'nucls_native_geometry.json';d=json.loads(geom.read_text());d.update(status='validated_ready_for_real_model_pilot',validation=str(out));geom.write_text(json.dumps(d,indent=2)+'\n');print(result,flush=True)
if __name__=='__main__':main()
