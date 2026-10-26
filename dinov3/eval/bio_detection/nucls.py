"""NuCLS QC publisher folds, vector annotations and genuine-mask-only supervision."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

CLASSES=['tumor_nonMitotic','nonTILnonMQ_stromal','lymphocyte','macrophage','plasma_cell','tumor_mitotic','other_nucleus','AMBIGUOUS']

def prepare_manifest(source):
    """Use RGB/CSV coordinates; preserve vectors instead of resizing mixed masks.

    The published class253 field supplies the annotated FOV rectangle. The
    author's Cropper excludes boxes with either cropped side <=8 pixels. The
    source mixed raster is retained, but rectangle regions never become genuine
    segmentation targets. Every published image/fold member remains represented.
    """
    original=json.loads(Path(source).read_text());records=[];quarantine=[]
    for row in original['records']:
        mask=np.asarray(Image.open(row['mask_path']))
        yy,xx=np.where(mask[:,:,0]==253)
        if not len(xx):raise ValueError(f'Missing annotated FOV: {row["sample_id"]}')
        x0,y0=max(0,int(xx.min())),max(0,int(yy.min()))
        x1,y1=min(row['width'],int(xx.max())+1),min(row['height'],int(yy.max())+1)
        width,height=x1-x0,y1-y0
        if min(width,height)<=8:raise ValueError('Invalid NuCLS annotated field')
        objects=[]
        with open(row['annotation_csv']) as stream:annotations=list(csv.DictReader(stream))
        for index,a in enumerate(annotations):
            label=CLASSES.index(a['main_classification'])+1
            raw=np.array([float(a[k]) for k in ['xmin','ymin','xmax','ymax']])
            box=raw-np.array([x0,y0,x0,y0]);box[[0,2]]=box[[0,2]].clip(0,width);box[[1,3]]=box[[1,3]].clip(0,height)
            if min(box[2:]-box[:2])<=8:
                quarantine.append({'sample_id':row['sample_id'],'annotation_index':index,'type':a['type'],'source_box':raw.tolist(),'cropped_box':box.tolist(),'reason':'Author Cropper min_boxside8; outside FOV, point or small clipped object'})
                continue
            polygon=[]
            if a['type']=='polyline':
                xs=np.fromstring(a['coords_x'],sep=',');ys=np.fromstring(a['coords_y'],sep=',')
                if len(xs)!=len(ys) or len(xs)<3 or not np.isfinite([xs,ys]).all():raise ValueError('Invalid published polygon')
                polygon=np.stack((xs-x0,ys-y0),axis=1).tolist()
            objects.append({'annotation_index':index,'label':label,'class_name':a['main_classification'],'source_type':a['type'],
                'box':box.tolist(),'polygon':polygon,'genuine_mask':bool(polygon) and label!=8,'ambiguous':label==8})
        records.append({**row,'path':row['image_path'],'source_width':row['width'],'source_height':row['height'],
            'width':width,'height':height,'crop_xyxy':[x0,y0,x1,y1],'objects':objects,'boxes':[a['box'] for a in objects],'labels':[a['label'] for a in objects]})
    return {'dataset':'NuCLS corrected QC native hybrid','classes':CLASSES,'records':records,'folds':original['folds'],
        'source_manifest':str(source),'source_manifest_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'quarantined_annotations':quarantine,'geometry':'RGB/CSV coordinate system; class253 annotated-FOV bounds clipped to RGB; vector polygons rasterized independently to avoid rectangle overlays. No resizing of mismatched mixed raster masks.',
        'mask_supervision':'Only unambiguous published polylines; rectangle/point/ambiguous masks unavailable',
        'grouping':'TCGA patient; official institution/slide fold memberships unchanged',
        'limitation':'Shared project native head comparison, not reproduction of the NuCLS paper model; vector masks differ from overlap-occluded mixed raster masks',
        'crop_rule_source':'NuCLS/nucls_model/torchvision_detection_utils/transforms.py Cropper(min_boxside=8)'}

class NuCLSDataset(torch.utils.data.Dataset):
    def __init__(self,manifest,fold,split):
        f=next(x for x in manifest['folds'] if x['fold']==fold)
        indices=[] if split=='val' and fold!=1 else f['indices']['test' if split in ['val','test'] else 'train']
        self.records=[manifest['records'][i] for i in indices];self.classes=CLASSES
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        from torchvision.transforms.functional import pil_to_tensor
        row=self.records[index]
        from .native import resolve_source_image
        with Image.open(resolve_source_image(row['path'])) as image:pixels=pil_to_tensor(image.convert('RGB').crop(row['crop_xyxy'])).float()/255
        boxes=torch.tensor(row['boxes'],dtype=torch.float32).reshape(-1,4)
        masks=[]
        for obj in row['objects']:
            # Sentinel denotes unavailable segmentation supervision, not a mask.
            mask=np.zeros((row['height'],row['width']),np.uint8) if obj['genuine_mask'] else np.full((row['height'],row['width']),255,np.uint8)
            if obj['genuine_mask']:cv2.fillPoly(mask,[np.rint(obj['polygon']).astype(np.int32)],1)
            masks.append(mask)
        mask_array=np.stack(masks) if masks else np.zeros((0,row['height'],row['width']),np.uint8)
        target={'boxes':boxes,'labels':torch.tensor(row['labels'],dtype=torch.long),'image_id':torch.tensor(index),
            'area':(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]),'iscrowd':torch.tensor([int(a['ambiguous']) for a in row['objects']]),
            'masks':torch.from_numpy(mask_array)}
        return pixels,target

def genuine_mask_loss(mask_logits,proposals,gt_masks,gt_labels,matched_idxs):
    """Torchvision mask loss on the subset with real mask annotations only."""
    from torchvision.models.detection.roi_heads import project_masks_on_boxes
    keep=[];targets=[];labels=[];size=mask_logits.shape[-1]
    for boxes,masks,classes,matched in zip(proposals,gt_masks,gt_labels,matched_idxs):
        valid=masks.flatten(1).amax(1)<255 if len(masks) else torch.zeros(0,dtype=torch.bool,device=masks.device)
        use=valid[matched];keep.append(use)
        targets.append(project_masks_on_boxes(masks,boxes[use],matched[use],size))
        labels.append(classes[matched[use]])
    if not keep or not any(x.any() for x in keep):return mask_logits.sum()*0
    logits=mask_logits[torch.cat(keep)];labels=torch.cat(labels);targets=torch.cat(targets)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits[torch.arange(len(labels),device=labels.device),labels],targets)

def mask_api():
    import sys
    pinned=Path(__file__).resolve().parents[3]/'outputs/02_eval_runtime/pycocotools_2_0_10'
    if str(pinned) not in sys.path:sys.path.insert(0,str(pinned))
    from pycocotools import mask
    return mask

def compact_prediction(pred):
    """Keep every scored non-ambiguous box and lossless thresholded mask RLE."""
    api=mask_api();keep=pred['labels']!=8
    result={k:pred[k][keep].detach().cpu() for k in ['boxes','labels','scores']}
    result['mask_rle']=[]
    for mask in pred['masks'][keep,0].detach().cpu().numpy():
        encoded=api.encode(np.asfortranarray(mask>.5,dtype=np.uint8))
        encoded['counts']=encoded['counts'].decode('ascii');result['mask_rle'].append(encoded)
    return result

def native_metrics(dataset,predictions):
    """COCO box AP and genuine-mask overlap on IoU>=.5 Hungarian matches.

    Follows the publisher's detection/matching structure. Includes zero-overlap
    matched masks and corrects the ambiguous-target mask indexing; both choices
    are explicit project corrections, not an exact reproduction of paper scores.
    """
    import contextlib,io
    from scipy.optimize import linear_sum_assignment
    api=mask_api()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if len(dataset)!=len(predictions):raise ValueError('Missing NuCLS evaluation images')
    images=[];annotations=[];detections=[];sample_metrics=[]
    for index,(row,pred) in enumerate(zip(dataset.records,predictions)):
        images.append({'id':index,'height':row['height'],'width':row['width']})
        for obj in row['objects']:
            x,y,xx,yy=obj['box']
            annotations.append({'id':len(annotations)+1,'image_id':index,'category_id':obj['label'],
                'bbox':[x,y,xx-x,yy-y],'area':(xx-x)*(yy-y),'iscrowd':int(obj['ambiguous'])})
        for box,label,score in zip(pred['boxes'].tolist(),pred['labels'].tolist(),pred['scores'].tolist()):
            x,y,xx,yy=box
            if not np.isfinite([x,y,xx,yy,score]).all() or xx<=x or yy<=y or not 1<=label<8:raise ValueError('Invalid NuCLS prediction')
            detections.append({'image_id':index,'category_id':label,'bbox':[x,y,xx-x,yy-y],'score':score})
        objects=[o for o in row['objects'] if not o['ambiguous']]
        # Publisher matching uses integer box coordinates and is class agnostic.
        a=np.array([o['box'] for o in objects],dtype=np.int32).reshape(-1,4)
        b=pred['boxes'].numpy().astype(np.int32).reshape(-1,4)
        conf=np.zeros((7,7),dtype=np.int64);intersect=area_sum=matched_masks=0;dice_sum=0.;matches=0
        if len(a) and len(b):
            intersection=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
            union=(a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-intersection
            iou=intersection/np.maximum(union,1)
            # Forbidden matches must not steal an above-threshold match.
            cost=1-iou;cost[iou<.5]=1e6
            ii,jj=linear_sum_assignment(cost)
            for i,j in zip(ii,jj):
                if iou[i,j]<.5:continue
                matches+=1;obj=objects[i];conf[obj['label']-1,int(pred['labels'][j])-1]+=1
                if not obj['genuine_mask']:continue
                truth=np.zeros((row['height'],row['width']),np.uint8);cv2.fillPoly(truth,[np.rint(obj['polygon']).astype(np.int32)],1)
                prediction=api.decode(pred['mask_rle'][j]).astype(bool)
                overlap=int(np.logical_and(truth,prediction).sum());total=int(truth.sum()+prediction.sum())
                intersect+=overlap;area_sum+=total;matched_masks+=1;dice_sum+=2*overlap/total if total else 0.
        sample_metrics.append({'sample_id':row['sample_id'],'group_id':row['group_id'],'true_nuclei':len(a),'predicted_nuclei':len(b),'matched_nuclei':matches,
            'mask_intersection':intersect,'mask_area_sum':area_sum,'matched_genuine_masks':matched_masks,'mask_dice_sum':dice_sum,'classification_confusion':conf.tolist()})
    with contextlib.redirect_stdout(io.StringIO()):
        truth=COCO();truth.dataset={'info':{'description':'NuCLS shared frozen hybrid head'},'licenses':[],
            'images':images,'annotations':annotations,'categories':[{'id':i+1,'name':name} for i,name in enumerate(CLASSES)]};truth.createIndex()
        if detections:result=truth.loadRes(detections)
        else:
            result=COCO();result.dataset={**truth.dataset,'annotations':[]};result.createIndex()
        evaluator=COCOeval(truth,result,'bbox');evaluator.params.maxDets=[1,10,300];evaluator.params.catIds=list(range(1,8))
        evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    precision=evaluator.eval['precision'][:,:,:,0,-1];valid=precision[precision>-1]
    confusion=sum((np.asarray(x['classification_confusion']) for x in sample_metrics),np.zeros((7,7),np.int64))
    area=sum(x['mask_area_sum'] for x in sample_metrics);overlap=sum(x['mask_intersection'] for x in sample_metrics);nmask=sum(x['matched_genuine_masks'] for x in sample_metrics)
    ntrue=sum(x['true_nuclei'] for x in sample_metrics);npred=sum(x['predicted_nuclei'] for x in sample_metrics);nmatch=int(confusion.sum())
    metrics={'bbox_ap_50_95':float(valid.mean()) if len(valid) else None,'bbox_ap_50':float(evaluator.stats[1]),'bbox_ap_75':float(evaluator.stats[2]),
        'matched_mask_dice_pooled':2*overlap/area if area else None,'matched_mask_dice_mean':sum(x['mask_dice_sum'] for x in sample_metrics)/nmask if nmask else None,
        'matched_classification_accuracy':float(np.trace(confusion)/nmatch) if nmatch else None,
        'detection_f1_iou50':2*nmatch/(ntrue+npred) if ntrue+npred else None,
        'evaluated_images':len(dataset),'matched_genuine_masks':nmask,'matched_nuclei':nmatch,'true_nuclei':ntrue,'predicted_nuclei':npred,'max_detections':300}
    return metrics,sample_metrics

def build_hybrid_detector(backbone,layers):
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection import roi_heads
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign
    from .native import FrozenDINOForRCNN
    class HybridMaskRCNN(MaskRCNN):
        def forward(self,images,targets=None):
            previous=roi_heads.maskrcnn_loss;roi_heads.maskrcnn_loss=genuine_mask_loss
            try:return super().forward(images,targets)
            finally:roi_heads.maskrcnn_loss=previous
    return HybridMaskRCNN(FrozenDINOForRCNN(backbone,layers),num_classes=len(CLASSES)+1,min_size=512,max_size=1024,
        rpn_anchor_generator=AnchorGenerator(((16,32,64,128,256),),((.5,1.,2.),)),
        box_roi_pool=MultiScaleRoIAlign(['0'],7,2),mask_roi_pool=MultiScaleRoIAlign(['0'],14,2),
        box_detections_per_img=300,rpn_pre_nms_top_n_train=2000,rpn_post_nms_top_n_train=1000)
