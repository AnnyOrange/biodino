import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from dinov3.eval.bio_detection.native import (
    NativeDetectionDataset, build_detector, coco_bbox_metrics, validate_boxes,
)


class TinyBackbone(nn.Module):
    patch_size = 16
    n_blocks = 4
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 16, stride=16)

    def get_intermediate_layers(self, images, n, reshape=True):
        count = n if isinstance(n, int) else len(n)
        return tuple(self.conv(images) for _ in range(count))


class NativeDetectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "image.png"
        Image.fromarray(np.full((64, 64, 3), 100, dtype=np.uint8)).save(path)
        self.dataset = NativeDetectionDataset({"records": [{"path": str(path), "split": "test",
            "width": 64, "height": 64, "boxes": [[10., 10., 30., 30.]], "labels": [1]}]}, "test")

    def test_box_validation_rejects_outside_and_bad_classes(self):
        for boxes, labels in (([[0, 0, 0, 10]], [1]), ([[0, 0, 65, 10]], [1]),
                              ([[0, 0, 10, 10]], [0]), ([[0, 0, float("nan"), 10]], [1])):
            with self.assertRaises(ValueError):
                validate_boxes(boxes, labels, 64, 64, ["cell"])

    def test_perfect_and_empty_native_predictions(self):
        truth = self.dataset[0][1]
        pred = {"boxes": truth["boxes"], "labels": truth["labels"], "scores": torch.ones(1)}
        self.assertAlmostEqual(coco_bbox_metrics(self.dataset, [pred], ["cell"])["bbox_ap_50_95"], 1.)
        pred = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long), "scores": torch.empty(0)}
        self.assertEqual(coco_bbox_metrics(self.dataset, [pred], ["cell"])["bbox_ap_50_95"], 0.)

    def test_real_rcnn_loss_backward_and_inference_with_frozen_backbone(self):
        torch.manual_seed(0)
        model = build_detector(TinyBackbone(), [3], 2, min_size=64, max_size=64)
        image, target = self.dataset[0]
        model.train()
        losses = model([image], [target])
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(model.roi_heads.box_predictor.cls_score.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.backbone.body.backbone.parameters()))
        self.assertFalse(model.backbone.body.training)
        model.eval()
        with torch.no_grad():
            prediction = model([image])
        metrics = coco_bbox_metrics(self.dataset, prediction, ["cell"])
        self.assertTrue(metrics["native_box_metric"])


if __name__ == "__main__":
    unittest.main()
