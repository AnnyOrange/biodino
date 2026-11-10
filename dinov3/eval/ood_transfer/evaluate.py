"""Validation and independently invoked test scoring in native coordinates."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from dinov3.eval.bio_detection.native import coco_bbox_metrics
from .data import collate
from .metrics import rsna_image_precision


@torch.inference_mode()
def predict(model, dataset, batch_size=1, workers=2):
    model.eval()
    outputs = []
    bone = dataset.recipe['task'] == 'bone_age'
    if not bone:
        # torchvision chooses the last min_size in eval; explicitly lock it.
        model.transform.min_size = (dataset.recipe['resolution'],)
    for images, targets in DataLoader(dataset, batch_size=batch_size, num_workers=workers, collate_fn=collate):
        images = [x.cuda(non_blocking=True) for x in images]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if bone:
                sex = torch.stack([x['sex'] for x in targets]).cuda()
                result = model(torch.stack(images), sex).float()
                result = result * dataset.recipe['target_std'] + dataset.recipe['target_mean']
                outputs.extend(float(x) for x in result.cpu())
            else:
                outputs.extend({k: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu()
                                for k, v in pred.items()} for pred in model(images))
    if len(outputs) != len(dataset):
        raise ValueError('Prediction population mismatch')
    return outputs


def score(dataset, predictions, threshold=None):
    task = dataset.recipe['task']
    if task == 'bone_age':
        error = np.abs(np.asarray(predictions) - np.asarray([r['age_months'] for r in dataset.records]))
        return dict(mae_months=float(error.mean()), n=len(error), selection_score=-float(error.mean()))
    if task == 'pneumonia':
        thresholds = dataset.recipe['score_threshold_grid'] if threshold is None else [threshold]
        candidates = []
        for cut in thresholds:
            values = []
            for row, pred in zip(dataset.records, predictions):
                keep = pred['scores'] >= cut
                value = rsna_image_precision(row['boxes'], pred['boxes'][keep].numpy(), pred['scores'][keep].numpy())
                if value is not None:
                    values.append(value)
            mean = float(np.mean(values)) if values else 0.0
            candidates.append(dict(rsna_mean_precision=mean, score_threshold=cut,
                                   scored_images=len(values), evaluated_images=len(dataset), selection_score=mean))
        return max(candidates, key=lambda x: (x['selection_score'], -x['score_threshold']))
    metrics = coco_bbox_metrics(dataset, predictions, dataset.classes)
    metrics['selection_score'] = metrics['bbox_ap_50_95']
    return metrics
