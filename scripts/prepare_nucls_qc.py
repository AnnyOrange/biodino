#!/usr/bin/env python3
"""Validate the complete corrected cohort and index publisher folds without re-splitting."""
import collections,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
DATA=Path('/mnt/huawei_deepcad/benchmark/ood/ood_detection/datasets/NuCLS');QC=DATA/'raw/QC/QC'

def main():
 records=json.loads((OUT/'nucls_qc_download_manifest.json').read_text())
 missing=[r['local_path'] for r in records if not Path(r['local_path']).is_file()]
 if missing:raise RuntimeError(f'Published cohort download incomplete: {len(missing)}/{len(records)} files')
 rgb={p.stem:p for p in (QC/'rgb').glob('*.png')};masks={p.stem:p for p in (QC/'mask').glob('*.png')};tables={p.stem:p for p in (QC/'csv').glob('*.csv')}
 if not (set(rgb)==set(masks)==set(tables) and len(rgb)==1744):raise ValueError('QC 1744 FOV image/mask/annotation identity mismatch')
 rows=[];main_classes=collections.Counter();raw_classes=collections.Counter();types=collections.Counter()
 for name in sorted(rgb):
  slide=name.split('_id-')[0];patient='-'.join(slide.split('-')[:3]);hospital=slide.split('-')[1]
  with Image.open(rgb[name]) as im:shape=(im.height,im.width);im.verify()
  mask=np.array(Image.open(masks[name]));assert mask.ndim==3 and mask.shape[2]==3,(name,mask.shape,shape)
  # Publisher convention is a product, not packed little-endian channels.
  instance=mask[:,:,1].astype(np.uint16)*mask[:,:,2].astype(np.uint16)
  ann=list(csv.DictReader(tables[name].open()));main_classes.update(a['main_classification'] for a in ann);raw_classes.update(a['raw_classification'] for a in ann);types.update(a['type'] for a in ann)
  clipped=0
  for a in ann:
   x1,y1,x2,y2=[float(a[k]) for k in ['xmin','ymin','xmax','ymax']]
   if not np.isfinite([x1,y1,x2,y2]).all() or x1>x2 or y1>y2:raise ValueError(f'Invalid annotation geometry: {name}')
   clipped+=int(not (0<=x1<=x2<=shape[1] and 0<=y1<=y2<=shape[0]))
  rows.append({'sample_id':name,'group_id':patient,'slide_id':slide,'hospital':hospital,'image_path':str(rgb[name]),'mask_path':str(masks[name]),'annotation_csv':str(tables[name]),'height':shape[0],'width':shape[1],'mask_height':mask.shape[0],'mask_width':mask.shape[1],'image_mask_shape_match':mask.shape[:2]==shape,'n_boxes_needing_clipping':clipped,'n_annotations':len(ann),'n_polylines':sum(a['type']=='polyline' for a in ann),'n_rectangles':sum(a['type']=='rectangle' for a in ann),'mask_instance_ids':int(np.count_nonzero(np.unique(instance))),'sha256':{kind:hashlib.sha256(p.read_bytes()).hexdigest() for kind,p in [('image',rgb[name]),('mask',masks[name]),('annotations',tables[name])]}})
 folds=[];all_slides={r['slide_id'] for r in rows}
 for fold in range(1,6):
  membership={split:set(x['slide_name'] for x in csv.DictReader((QC/'train_test_splits'/f'fold_{fold}_{split}.csv').open())) for split in ['train','test']}
  if membership['train']&membership['test']:raise ValueError('Publisher slide overlap')
  if not all_slides<=membership['train']|membership['test']:raise ValueError('Unassigned FOV slide')
  ix={split:[i for i,r in enumerate(rows) if r['slide_id'] in slides] for split,slides in membership.items()}
  train_groups={rows[i]['group_id'] for i in ix['train']};test_groups={rows[i]['group_id'] for i in ix['test']}
  if train_groups&test_groups:raise ValueError('Publisher patient overlap')
  folds.append({'fold':fold,'role':'selection_only' if fold==1 else 'evaluation','indices':ix,'n_train':len(ix['train']),'n_test':len(ix['test']),'n_train_patients':len(train_groups),'n_test_patients':len(test_groups),'test_hospitals':sorted({rows[i]['hospital'] for i in ix['test']}),'source_csv':{split:str(QC/'train_test_splits'/f'fold_{fold}_{split}.csv') for split in membership}})
 output=ROOT/'outputs/02_eval_inputs/formal_v4/nucls_qc_manifest.json';payload={'dataset':'NuCLS corrected single-rater QC','source':'https://sites.google.com/view/nucls/data-format','schema_version':1,'scope':'native nuclei detection / instance segmentation; rectangle annotations are not genuine segmentation targets','mask_encoding':'class=channel0; instance=uint16(channel1)*uint16(channel2); FOV class253; ambiguous class99','class_codes':str(DATA/'metadata/nucleus_GTcodes.csv'),'records':rows,'folds':folds,'selection_rule':'Existing project rule: publisher fold1 development/selection; folds2-5 evaluation; five training seeds within each fixed evaluation fold','statistics_group':'patient/slide, not FOV patches','download_manifest':str(OUT/'nucls_qc_download_manifest.json')}
 tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(output)
 summary={'status':'data_and_publisher_folds_complete','files':len(records),'fovs':len(rows),'slides':len(all_slides),'patients':len({r['group_id'] for r in rows}),'image_mask_dimension_mismatches':sum(not r['image_mask_shape_match'] for r in rows),'maximum_dimension_difference':max(max(abs(r['height']-r['mask_height']),abs(r['width']-r['mask_width'])) for r in rows),'boxes_needing_clipping':sum(r['n_boxes_needing_clipping'] for r in rows),'annotations':dict(main_classes),'raw_classes':dict(raw_classes),'annotation_types':dict(types),'folds':[{k:v for k,v in f.items() if k not in ['indices','source_csv']} for f in folds],'manifest':str(output),'manifest_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'remaining':'Shared native detection/instance head and evaluator integration; no model test run yet'}
 (OUT/'nucls_qc_cohort_validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
 if (DATA/'download_status.json').exists():
  old=json.loads((DATA/'download_status.json').read_text());(OUT/'corrections/nucls_original_download_status.json').write_text(json.dumps(old,indent=2))
  old.update(download_status='complete_corrected_QC',source_method='publisher_embedded_folder_full_inventory_via_process_local_proxy',num_images=1744,num_annotation_files=1744,can_use_for_detection=True,failure_reason='',manual_download_instruction='',notes='Full corrected cohort and publisher folds validated; native evaluation adapter pending',cohort_validation=str(OUT/'nucls_qc_cohort_validation.json'))
  (DATA/'download_status.json').write_text(json.dumps(old,indent=2)+'\n')
if __name__=='__main__':main()
