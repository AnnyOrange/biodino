"""Versioned FM14 recipes and conservative, batch-tolerant result reuse."""
from __future__ import annotations

import math

PROTOCOL_ID = 'fm14-rules-alignment-20260917-v2'
MODELS = ('dinov2','mae','siglip2','pe','bioclip','cytoself','jump_cp',
          'cytoimagenet','uni','conch','phikon2','virchow2','gigapath','hoptimus0')
CONV_MODELS = frozenset(('cytoself','cytoimagenet'))
SEGMENTATION = {
    'cellpose': (512,'pad','last','none'),
    'conic': (256,'stretch','even4','sqrt_inverse'),
    'livecell': (512,'pad','even4','none'),
    'monuseg': (768,'pad','last','none'),
    'pannuke': (256,'stretch','even4','none'),
    'tissuenet': (256,'stretch','last','none'),
    'multimodal_cellseg': (512,'pad','last','none'),
    'bbbc038': (512,'pad','even4','none'),
}
PANNUKE_PROTOCOLS = (
    'pannuke-fold1-train-fold2-val-fold3-test',
    'pannuke-fold2-train-fold1-val-fold3-test',
    'pannuke-fold3-train-fold2-val-fold1-test',
)
BATCH_FIELDS = frozenset(('batch_size','feature_batch_size','encoder_batch_size','probe_batch_size'))


def even4_indices(depth):
    if depth < 4: raise ValueError('Four distinct transformer blocks are required')
    known = {12:(2,5,8,11),24:(4,11,17,23),32:(7,15,23,31),40:(9,19,29,39)}
    if depth in known:return list(known[depth])
    first = depth//6
    return [math.ceil(first+(depth-1-first)*i/3) for i in range(4)]


def resolve_dense_layers(requested, depth=None):
    if requested not in ('last','even4'):raise ValueError('Unknown dense layer request')
    supported = depth is not None and depth >= 4
    resolved = 'even4' if requested=='even4' and supported else 'last'
    return dict(requested_layers=requested,resolved_layers=resolved,
        layer_indices=even4_indices(depth) if resolved=='even4' else [depth-1] if depth else [],
        fallback_reason='No supported four-block spatial readout' if requested!=resolved else '')


def dense_recipe(model,dataset,view='primary-last',depth=None,split_protocol=None,observation=False):
    if model not in MODELS:raise ValueError('FM14 model required')
    if dataset not in SEGMENTATION:raise ValueError('Unregistered segmentation dataset')
    if dataset=='bbbc038' and not observation:raise ValueError('BBBC038 requires a separate observation output')
    if view not in ('primary-last','dataset-best'):raise ValueError('Unknown comparison view')
    size,resize,requested,weight=SEGMENTATION[dataset]
    if dataset=='pannuke':
        if split_protocol not in PANNUKE_PROTOCOLS:raise ValueError('PanNuke requires an explicit official fold rotation')
    else:
        expected='official-baseline-fold0-nested-v1' if dataset=='conic' else 'formal-static-v1'
        if split_protocol is not None and split_protocol!=expected:raise ValueError('Legacy/custom split is not allowed')
        split_protocol=expected
    layers=resolve_dense_layers('last' if view=='primary-last' else requested,depth)
    return dict(protocol_id=PROTOCOL_ID,model=model,task='segmentation',dataset=dataset,
        scope='OBSERVATIONAL' if observation else 'FORMAL_COMPONENT',comparison_view=view,
        image_size=size,resize_mode=resize,split_protocol=split_protocol,class_weight_mode=weight,
        feature_batch_size=32,probe_batch_size=32,autocast_dtype='bf16',num_workers=2,
        probe_budgets=[20,50],seeds=[0,1,2],probe_eval_every=1,
        head='Dropout2d+BatchNorm2d+Conv1x1',optimizer='AdamW',lr=1e-3,weight_decay=1e-4,dropout=.1,
        scheduler='CosineAnnealingLR',selection_metric='val-mIoU',selection_tie='earliest-epoch',
        test_evaluations=1,max_feature_side=0,**layers)


def reuse_decision(expected,actual):
    """Missing evidence is a review, never a batch-only automatic rerun."""
    differences={};unknown=[];batch_differences={}
    for key,value in expected.items():
        if key in BATCH_FIELDS:
            if actual.get(key) not in (None,'') and actual[key]!=value:
                batch_differences[key]=dict(expected=value,actual=actual[key])
            continue
        if actual.get(key) in (None,''):unknown.append(key)
        elif actual[key]!=value:differences[key]=dict(expected=value,actual=actual[key])
    status = 'RETEST_NON_BATCH_DIFFERENCE' if differences else 'REVIEW_EVIDENCE' if unknown else (
        'REUSE_BATCH_TOLERATED' if batch_differences else 'REUSE_MATCHED')
    return dict(status=status,differences=differences,unknown_dimensions=unknown,
                batch_differences=batch_differences,batch_only_retest=False)


def validate_best_head(meta):
    epochs=meta['probe_epochs'];history=meta['validation_history']
    if epochs not in (20,50) or [r['epoch'] for r in history]!=list(range(1,epochs+1)):
        raise ValueError('Independent E20/E50 require validation every epoch')
    if not all(math.isfinite(r['mIoU']) for r in history):
        raise ValueError('Validation metrics must be finite')
    best=max(history,key=lambda row:(row['mIoU'],-row['epoch']))
    if meta['best_epoch']!=best['epoch'] or not math.isclose(meta['best_val_miou'],best['mIoU']):
        raise ValueError('Head must be selected by best validation mIoU, earliest tie')
    if meta['test_evaluations']!=1:raise ValueError('Test must be evaluated exactly once')
