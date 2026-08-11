from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import dinov3.distributed as distributed
from dinov3.eval.bio_classification.common import checkpoint_stem, load_backbone, parse_autocast_dtype
from dinov3.eval.bio_segmentation.datasets.livecell import get_livecell_paths
from dinov3.eval.helpers import write_results
from dinov3.utils.bio_io import read_bio_image_as_numpy

logger = logging.getLogger("dinov3.bio_detection.center_probe")
RESULTS_FILENAME = "results-detection.csv"
_IMAGE_NET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_IMAGE_NET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
_IMAGE_MAP_CACHE: Dict[str, Dict[str, str]] = {}


class LiveCellCenterDataset(Dataset):
    """LIVECell COCO center-to-patch detection labels for a frozen-feature probe."""

    def __init__(self, data_root: str, split: str, image_size: int = 224, patch_size: int = 16, max_samples: int = 0, seed: int = 0):
        coco_json, image_root = get_livecell_paths(data_root, split)
        self.image_root = Path(image_root)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid = self.image_size // self.patch_size
        with open(coco_json) as f:
            data = json.load(f)
        anns_by_image: Dict[int, List[Tuple[float, float]]] = {}
        for ann in data.get("annotations", []):
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            anns_by_image.setdefault(int(ann["image_id"]), []).append((float(x + w / 2.0), float(y + h / 2.0)))
        image_records = list(data.get("images", []))
        if max_samples > 0 and len(image_records) > max_samples:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(image_records), size=max_samples, replace=False))
            image_records = [image_records[i] for i in idx]
        cache_key = str(self.image_root)
        if cache_key not in _IMAGE_MAP_CACHE:
            _IMAGE_MAP_CACHE[cache_key] = {p.name: str(p) for p in self.image_root.rglob("*.tif")}
        self._image_by_name = _IMAGE_MAP_CACHE[cache_key]
        samples = []
        for img in image_records:
            file_name = img["file_name"]
            path = self._find_image(file_name)
            samples.append((path, int(img["width"]), int(img["height"]), anns_by_image.get(int(img["id"]), [])))
        self.samples = samples
        if not self.samples:
            raise ValueError(f"LIVECell split={split} has no samples")
        logger.info("LIVECell center dataset split=%s size=%d grid=%dx%d", split, len(self.samples), self.grid, self.grid)

    def _find_image(self, file_name: str) -> str:
        direct = self.image_root / file_name
        if direct.is_file():
            return str(direct)
        if file_name in self._image_by_name:
            return self._image_by_name[file_name]
        raise FileNotFoundError(f"Cannot find LIVECell image {file_name} under {self.image_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, width, height, centers = self.samples[index]
        image = read_bio_image_as_numpy(path, target_channels=3, normalize=True)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(image).permute(2, 0, 1).float()
        x = (x - _IMAGE_NET_MEAN) / _IMAGE_NET_STD
        label = torch.zeros(self.grid, self.grid, dtype=torch.float32)
        sx = self.image_size / max(float(width), 1.0)
        sy = self.image_size / max(float(height), 1.0)
        for cx, cy in centers:
            gx = min(self.grid - 1, max(0, int((cx * sx) // self.patch_size)))
            gy = min(self.grid - 1, max(0, int((cy * sy) // self.patch_size)))
            label[gy, gx] = 1.0
        return x, label.reshape(-1)


class PatchFeatureModel(nn.Module):
    def __init__(self, backbone: nn.Module, autocast_dtype: torch.dtype, channel_policy: str = "auto"):
        super().__init__()
        self.backbone = backbone
        self.autocast_dtype = autocast_dtype
        self.channel_policy = channel_policy

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
            # reshape=True activates ChannelViT's channel-token collapse and
            # gives every dense evaluator one feature vector per spatial patch.
            feature_map = self.backbone.get_intermediate_layers(images, n=1, reshape=True)[0]
        return feature_map.flatten(2).transpose(1, 2).float().contiguous()


@torch.inference_mode()
def _estimate_pos_weight(loader: DataLoader) -> float:
    pos = 0.0
    total = 0.0
    for _, y in loader:
        pos += float(y.sum().item())
        total += float(y.numel())
    neg = max(total - pos, 1.0)
    return max(1.0, min(neg / max(pos, 1.0), 100.0))


def _eval(model, head, loader, threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    head.eval()
    tp = fp = fn = tn = 0.0
    losses = []
    criterion = nn.BCEWithLogitsLoss()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            logits = head(model(images)).squeeze(-1)
            losses.append(float(criterion(logits, labels).item()))
            pred = torch.sigmoid(logits) >= threshold
            lab = labels >= 0.5
            tp += float((pred & lab).sum().item())
            fp += float((pred & ~lab).sum().item())
            fn += float((~pred & lab).sum().item())
            tn += float((~pred & ~lab).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    return {"loss": float(np.mean(losses)), "patch_accuracy": acc * 100.0, "patch_precision": precision * 100.0, "patch_recall": recall * 100.0, "patch_f1": f1 * 100.0}


def run_bio_detection_eval(
    *,
    arch: str,
    weights: str,
    checkpoint: str | None,
    train_config: str | None,
    benchmark_root: str,
    output_dir: str,
    dataset: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    epochs: int,
    lr: float,
    autocast_dtype: torch.dtype,
    channel_policy: str,
    max_samples_per_split: int,
    seed: int,
) -> Dict[str, float | str | int]:
    os.makedirs(output_dir, exist_ok=True)
    if dataset.lower() != "livecell":
        raise ValueError("Currently supported bio detection datasets: livecell")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    data_root = Path(benchmark_root) / "segmentation" / "LIVECell"
    backbone = load_backbone(repo_dir=".", arch=arch, weights=weights, checkpoint=checkpoint, train_config=train_config)
    feature_model = PatchFeatureModel(
        backbone,
        autocast_dtype=autocast_dtype,
        channel_policy=channel_policy,
    ).cuda().eval()
    train_ds = LiveCellCenterDataset(str(data_root), "train", image_size=image_size, max_samples=max_samples_per_split, seed=seed)
    val_ds = LiveCellCenterDataset(str(data_root), "val", image_size=image_size, max_samples=max_samples_per_split, seed=seed)
    test_ds = LiveCellCenterDataset(str(data_root), "test", image_size=image_size, max_samples=max_samples_per_split, seed=seed)
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=train_generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    sample_images, sample_labels = next(iter(train_loader))
    sample_feats = feature_model(sample_images.cuda(non_blocking=True)).clone()
    if sample_feats.shape[1] != sample_labels.shape[1]:
        raise RuntimeError(f"Patch token count {sample_feats.shape[1]} does not match labels {sample_labels.shape[1]}; adjust --image-size.")
    # Keep the probe initialization identical across checkpoint comparisons.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = nn.Linear(int(sample_feats.shape[-1]), 1).cuda()
    pos_weight = torch.tensor([_estimate_pos_weight(train_loader)], device="cuda")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(epochs):
        head.train()
        for images, labels in train_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            feats = feature_model(images).clone()
            logits = head(feats).squeeze(-1)
            loss = criterion(logits, labels)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        logger.info("epoch=%d val=%s", epoch + 1, _eval(feature_model, head, val_loader))
    val_metrics = _eval(feature_model, head, val_loader)
    test_metrics = _eval(feature_model, head, test_loader)
    results: Dict[str, float | str | int] = {
        "dataset": dataset,
        "checkpoint": checkpoint_stem(checkpoint, weights),
        "probe": "livecell_center_patch_linear",
        "image_size": image_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "pos_weight": float(pos_weight.item()),
    }
    results.update({f"val_{k}": v for k, v in val_metrics.items()})
    results.update({f"test_{k}": v for k, v in test_metrics.items()})
    with open(os.path.join(output_dir, "results_bio_detection.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_results(results, output_dir, RESULTS_FILENAME)
    with open(os.path.join(output_dir, "bio_detection.md"), "w") as f:
        f.write("# Bio Detection Probe\n\n")
        f.write("This is a frozen-backbone LIVECell center-to-patch linear probe, not a full DETR/COCO mAP head.\n\n")
        f.write(json.dumps(results, indent=2))
        f.write("\n")
    logger.info("Bio detection results: %s", results)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Bio detection center-patch linear probe.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--arch", default="dinov3_vitb16")
    parser.add_argument("--weights", default="")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--dataset", default="livecell", choices=["livecell"])
    parser.add_argument("--output-dir", default="outputs/bio_detection/livecell")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--autocast-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--channel-policy", default="auto", choices=["auto", "native", "first3"])
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    distributed.enable(set_cuda_current_device=True, overwrite=True)
    run_bio_detection_eval(
        arch=args.arch,
        weights=args.weights,
        checkpoint=args.checkpoint,
        train_config=args.train_config,
        benchmark_root=args.benchmark_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        epochs=args.epochs,
        lr=args.lr,
        autocast_dtype=parse_autocast_dtype(args.autocast_dtype),
        channel_policy=args.channel_policy,
        max_samples_per_split=args.max_samples_per_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
