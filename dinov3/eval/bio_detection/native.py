"""Native box detection adapters and a frozen-DINO torchvision Faster R-CNN.

This is separate from the historical center-grid proxy. Torchvision owns the
RPN, RoI sampling, losses and NMS; pycocotools owns native bounding-box AP.
"""
from __future__ import annotations

from collections import OrderedDict
import contextlib
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
from torch import nn


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_boxes(boxes, labels, width, height, classes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    labels = np.asarray(labels, dtype=np.int64)
    if len(boxes) != len(labels) or not np.isfinite(boxes).all():
        raise ValueError("Invalid or misaligned native boxes")
    if len(boxes) and (np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2] > width)
                       or np.any(boxes[:, 3] > height) or np.any(boxes[:, 2:] <= boxes[:, :2])):
        raise ValueError("Native boxes are degenerate or outside image bounds")
    if np.any(labels < 1) or np.any(labels > len(classes)):
        raise ValueError("Native class ID is outside declared class map")
    return boxes.tolist(), labels.tolist()


def validate_manifest(manifest, decode=True):
    seen, content = set(), {}
    splits = {"train": 0, "val": 0, "test": 0}
    for row in manifest["records"]:
        if row["sample_id"] in seen or row["split"] not in splits:
            raise ValueError("Duplicate sample or invalid detection split")
        seen.add(row["sample_id"])
        splits[row["split"]] += 1
        validate_boxes(row["boxes"], row["labels"], row["width"], row["height"], manifest["classes"])
        if decode:
            with Image.open(row["path"]) as image:
                image.load()
                if image.size != (row["width"], row["height"]):
                    raise ValueError("Native image/annotation dimensions differ")
                pixels = np.asarray(image.convert("RGB"))
            identity = hashlib.sha256(pixels.tobytes()).hexdigest()
            row["decoded_pixel_sha256"] = identity
            if identity in content:
                raise ValueError(f"Duplicate detection image: {content[identity]} / {row['sample_id']}")
            content[identity] = row["sample_id"]
    if not all(splits.values()):
        raise ValueError("Detection protocol needs nonempty train/val/test")
    manifest["split_counts"] = splits
    return manifest


def build_bccd_manifest(root):
    root = Path(root)
    if (root / "BCCD").is_dir():
        root = root / "BCCD"
    classes = ["RBC", "WBC", "Platelets"]
    records, source_hashes, quarantined = [], {}, []
    for split in ("train", "val", "test"):
        source = root / "ImageSets/Main" / f"{split}.txt"
        source_hashes[split] = sha256(source)
        for sample in source.read_text().split():
            path = root / "Annotations" / f"{sample}.xml"
            annotation = ET.parse(path).getroot()
            width, height = (int(annotation.findtext(f"size/{key}")) for key in ("width", "height"))
            boxes, labels = [], []
            for object_index, item in enumerate(annotation.findall("object")):
                name = item.findtext("name")
                if name not in classes:
                    raise ValueError(f"Unknown BCCD class: {name}")
                box = [float(item.findtext(f"bndbox/{key}")) for key in ("xmin", "ymin", "xmax", "ymax")]
                if box[2] == box[0] and box[3] == box[1] and sample in {
                    "BloodImage_00343", "BloodImage_00338"
                }:
                    quarantined.append({"sample_id": sample, "object_index": object_index,
                        "class": name, "box": box, "annotation_sha256": sha256(path),
                        "reason": "SOURCE_DEGENERATE_POINT_BOX", "source_url":
                        f"https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/Annotations/{sample}.xml"})
                    continue
                # Released labels are tool-generated image coordinates; zero and
                # width-bound coordinates occur. Do not invent a one-pixel shift.
                boxes.append(box)
                labels.append(classes.index(name) + 1)
            validate_boxes(boxes, labels, width, height, classes)
            records.append({"sample_id": sample, "split": split, "group": sample,
                            "path": str((root / "JPEGImages" / annotation.findtext("filename")).resolve()),
                            "width": width, "height": height, "boxes": boxes, "labels": labels,
                            "annotation_sha256": sha256(path)})
    return validate_manifest({"dataset": "BCCD", "classes": classes, "records": records,
        "source": "https://github.com/Shenggan/BCCD_Dataset", "license": "MIT",
        "split_source": "OFFICIAL", "metric_source": "ESTABLISHED_CONVENTION",
        "metric": "COCO-style bbox AP@[.50:.95], maxDets=300", "source_hashes": source_hashes,
        "box_convention": "released continuous xyxy, no implicit VOC one-pixel shift",
        "quarantined_annotations": quarantined, "cleaning_source": "PROPOSED_BY_US",
        "limitation": "Two upstream degenerate point boxes removed, retaining images and other valid objects; no official AP leaderboard or patient provenance.",
        "grouping": "released source image; patient identity unavailable"})


def build_bbbc041_manifest(root, seed=0):
    root = Path(root)
    classes = ["red blood cell", "leukocyte", "gametocyte", "ring", "trophozoite", "schizont", "difficult"]
    records, hashes = [], {}
    for source_split, filename in (("trainval", "training.json"), ("test", "test.json")):
        source = root / filename
        hashes[filename] = sha256(source)
        data = json.loads(source.read_text())
        ordered = sorted(data, key=lambda row: hashlib.sha256(
            f"{seed}:{row['image']['pathname']}".encode()).hexdigest())
        val_ids = {r["image"]["pathname"] for r in ordered[:int(np.ceil(.1 * len(ordered)))]} if source_split == "trainval" else set()
        for item in data:
            image = item["image"]
            boxes, labels = [], []
            for obj in item["objects"]:
                lo, hi = obj["bounding_box"]["minimum"], obj["bounding_box"]["maximum"]
                boxes.append([lo["c"], lo["r"], hi["c"], hi["r"]])
                labels.append(classes.index(obj["category"]) + 1)
            split = "test" if source_split == "test" else "val" if image["pathname"] in val_ids else "train"
            path = root / image["pathname"].lstrip("/")
            if hashlib.md5(path.read_bytes()).hexdigest() != image["checksum"]:
                raise ValueError(f"Official BBBC041 image checksum mismatch: {path}")
            records.append({"sample_id": image["pathname"], "path": str(path.resolve()),
                            "split": split, "source_split": source_split, "group": image["pathname"],
                            "width": image["shape"]["c"], "height": image["shape"]["r"],
                            "boxes": boxes, "labels": labels})
    return validate_manifest({"dataset": "BBBC041", "classes": classes, "records": records,
        "source": "https://bbbc.broadinstitute.org/BBBC041", "license": "CC-BY-NC-SA-3.0",
        "split_source": "OFFICIAL test / PROPOSED_BY_US image-heldout validation", "seed": seed,
        "metric_source": "ESTABLISHED_CONVENTION", "metric": "COCO-style bbox AP@[.50:.95], maxDets=300",
        "source_hashes": hashes, "grouping": "source image; researcher/patient provenance unresolved",
        "limitation": "Seven released annotation categories, including ambiguous difficult; no official AP leaderboard. JSON scope 1328 images versus portal 1364."})


class NativeDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, split):
        self.records = [row for row in manifest["records"] if row["split"] == split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from torchvision.transforms.functional import pil_to_tensor

        row = self.records[index]
        with Image.open(row["path"]) as image:
            pixels = pil_to_tensor(image.convert("RGB")).float() / 255
        boxes = torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4)
        target = {"boxes": boxes, "labels": torch.tensor(row["labels"], dtype=torch.int64),
                  "image_id": torch.tensor(index),
                  "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                  "iscrowd": torch.zeros(len(boxes), dtype=torch.int64)}
        return pixels, target


class FrozenDINOForRCNN(nn.Module):
    def __init__(self, backbone, layers):
        super().__init__()
        from dinov3.eval.detection.models.backbone import DINOBackbone

        self.body = DINOBackbone(backbone, train_backbone=False, layers_to_use=layers, use_layernorm=False)
        self.out_channels = 256
        self.projection = nn.Conv2d(self.body.num_channels[0], self.out_channels, kernel_size=1)
        self.body.eval()

    def train(self, mode=True):
        super().train(mode)
        self.body.eval()
        return self

    def forward(self, images):
        from dinov3.eval.detection.util.misc import NestedTensor

        mask = torch.zeros(images.shape[0], *images.shape[-2:], device=images.device, dtype=torch.bool)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=images.is_cuda):
            features = self.body(NestedTensor(images, mask))[0].tensors
        return OrderedDict([("0", self.projection(features.float()))])


def build_detector(backbone, layers, num_classes, min_size=512, max_size=1024):
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    return FasterRCNN(FrozenDINOForRCNN(backbone, layers), num_classes=num_classes,
                      min_size=min_size, max_size=max_size,
                      rpn_anchor_generator=AnchorGenerator(((16, 32, 64, 128, 256),), ((.5, 1., 2.),)),
                      box_roi_pool=MultiScaleRoIAlign(["0"], output_size=7, sampling_ratio=2),
                      box_detections_per_img=300, rpn_pre_nms_top_n_train=2000,
                      rpn_post_nms_top_n_train=1000)


def coco_bbox_metrics(dataset, predictions, classes):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if len(predictions) != len(dataset):
        raise ValueError("Native detections must include every evaluation image")
    annotations, detections, images = [], [], []
    for image_id, (row, pred) in enumerate(zip(dataset.records, predictions)):
        images.append({"id": image_id, "width": row["width"], "height": row["height"]})
        for box, label in zip(row["boxes"], row["labels"]):
            x, y, xx, yy = box
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": label,
                                "bbox": [x, y, xx - x, yy - y], "area": (xx-x)*(yy-y), "iscrowd": 0})
        for box, label, score in zip(pred["boxes"].detach().cpu().tolist(),
                                     pred["labels"].detach().cpu().tolist(), pred["scores"].detach().cpu().tolist()):
            x, y, xx, yy = box
            if not np.isfinite([x, y, xx, yy, score]).all() or xx <= x or yy <= y or not 1 <= label <= len(classes):
                raise ValueError("Invalid native predicted box/class/score")
            detections.append({"image_id": image_id, "category_id": label,
                               "bbox": [x, y, xx-x, yy-y], "score": score})
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = {"images": images, "annotations": annotations,
                         "categories": [{"id": i+1, "name": name} for i, name in enumerate(classes)]}
        truth.createIndex()
        if detections:
            results = truth.loadRes(detections)
        else:
            results = COCO()
            results.dataset = {**truth.dataset, "annotations": []}
            results.createIndex()
        evaluator = COCOeval(truth, results, "bbox")
        evaluator.params.maxDets = [1, 10, 300]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    # COCO summarize AP hard-codes maxDets=100; compute headline from the
    # precision tensor for the frozen microscopy cap instead of returning -1.
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    valid = precision[precision > -1]
    ap = float(valid.mean()) if len(valid) else None
    return {"bbox_ap_50_95": ap, "bbox_ap_50": float(evaluator.stats[1]),
            "bbox_ap_75": float(evaluator.stats[2]), "max_detections": 300,
            "evaluated_images": len(dataset), "native_box_metric": True}
