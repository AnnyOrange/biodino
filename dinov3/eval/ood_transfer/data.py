"""Full-field supervised inputs, with disjoint development/test manifests."""
import random

import numpy as np
from PIL import Image
import tifffile
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from .common import read_json, resolve_input


class TransferDataset(Dataset):
    def __init__(self, manifest_path, data_root, split, recipe, augment=False):
        manifest = read_json(manifest_path)
        if manifest['task'] != recipe['task']:
            raise ValueError('Dataset task differs from model recipe')
        allowed = {'train', 'val'} if manifest['role'] == 'development' else {'test'}
        if split not in allowed or any(r['split'] not in allowed for r in manifest['records']):
            raise ValueError('Development/test manifest separation violated')
        self.records = [r for r in manifest['records'] if r['split'] == split]
        self.classes = manifest.get('classes', [])
        self.root, self.recipe, self.augment = data_root, recipe, augment
        if not self.records:
            raise ValueError('Empty split: ' + split)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        path = resolve_input(self.root, row['path'])
        task = self.recipe['task']
        if task == 'pneumonia':
            image = tifffile.imread(path)
            if image.dtype != np.uint16 or image.ndim != 2:
                raise ValueError('RSNA TIFF encoding changed')
            x = torch.from_numpy(image.astype(np.float32) / 65535)[None].expand(3, -1, -1).contiguous()
        else:
            with Image.open(path) as image:
                x = TF.pil_to_tensor(image.convert('RGB')).float() / 255
        if task == 'bone_age':
            size = self.recipe['resolution']
            if self.augment:
                size = random.choice(self.recipe.get('train_sizes', [size]))
                x = TF.rotate(x, random.uniform(-10, 10), InterpolationMode.BILINEAR, expand=True)
                x = TF.adjust_brightness(x, random.uniform(.9, 1.1))
                x = TF.adjust_contrast(x, random.uniform(.9, 1.1))
            # Fit the entire transformed field, then pad; no anatomical crop.
            fraction = random.uniform(.9, 1.0) if self.augment else 1.0
            ratio = size * fraction / max(x.shape[-2:])
            height, width = [max(1, round(v * ratio)) for v in x.shape[-2:]]
            x = TF.resize(x, [height, width], antialias=True)
            left = random.randrange(size - width + 1) if self.augment else (size - width) // 2
            top = random.randrange(size - height + 1) if self.augment else (size - height) // 2
            x = TF.pad(x, [left, top, size - width - left, size - height - top])
            sex = row.get('sex', 0)
            if self.recipe.get('use_sex') and ('sex' not in row or sex not in (0, 1)):
                raise ValueError('Missing source sex metadata')
            return x, dict(age=torch.tensor(row['age_months'], dtype=torch.float32),
                           sex=torch.tensor(sex, dtype=torch.float32), image_id=torch.tensor(index))
        boxes = torch.tensor(row['boxes'], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(row['labels'], dtype=torch.int64)
        h, w = x.shape[-2:]
        if (h, w) != (row['height'], row['width']):
            raise ValueError('Native image dimensions disagree with source annotation')
        if self.augment:
            if random.random() < .5:
                x = x.flip(-1)
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if task == 'bbbc041':
                if random.random() < .5:
                    x = x.flip(-2)
                    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                for _ in range(random.randrange(4)):
                    x = x.rot90(1, [-2, -1])
                    old = boxes.clone()
                    boxes[:, 0], boxes[:, 2] = old[:, 1], old[:, 3]
                    boxes[:, 1], boxes[:, 3] = w - old[:, 2], w - old[:, 0]
                    h, w = w, h
            x = TF.adjust_brightness(x, random.uniform(.9, 1.1))
            x = TF.adjust_contrast(x, random.uniform(.9, 1.1))
            if task == 'bbbc041':
                x = TF.adjust_saturation(x, random.uniform(.9, 1.1))
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        if torch.any(area <= 0) or not torch.isfinite(boxes).all():
            raise ValueError('Invalid transformed boxes')
        return x.clamp(0, 1), dict(boxes=boxes, labels=labels, area=area,
                                  image_id=torch.tensor(index), iscrowd=torch.zeros(len(labels), dtype=torch.int64))


def collate(batch):
    return tuple(zip(*batch))
