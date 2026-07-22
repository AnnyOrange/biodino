"""
Train + evaluate a DINOHoVerNet (instance-seg track, Line 2).

Mirrors the structure / results.json schema of ``../linear_probe.py`` so the two
tracks read the same way, but produces *real* instances (touching nuclei split)
and reports instance metrics (AJI / AP / bPQ / mPQ) via the SAME metric code the
specialist adapter uses.

Usage:
    python -m dinov3.eval.bio_segmentation.instance_seg.train \\
        --dataset    pannuke \\
        --data-root  /data/pannuke/extracted \\
        --checkpoint /ckpt/<iter> \\
        --train-config dinov3/configs/train/microscopy_continual_vitl16.yaml \\
        --output-dir ./outputs/instance_seg/pannuke \\
        --layers 4 11 17 23 \\
        --freeze-backbone --epochs 50 --batch-size 8 --crop-size 256
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..metrics import accumulate_instance_metrics
from .losses import HoVerNetLoss
from .model import build_dino_hovernet
from .postproc import postprocess
from .targets import make_targets
from .tiling import sliding_window_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bio_seg.instance_seg.train")


# num_types = full class count INCLUDING background (TP branch channels);
# 0 → binary dataset (TP branch disabled, semantic map derived from instances).
DATASET_NUM_TYPES = {
    "cellpose": 0,
    "pannuke": 6,
    "conic": 7,
    "monuseg": 0,
    "livecell": 0,
    "bbbc038": 0,
    "tissuenet": 0,
}


# ---------------------------------------------------------------------------
# Training dataset: native sample → random crop → HoVerNet targets
# ---------------------------------------------------------------------------

class RandomCropHoVerDataset(Dataset):
    """Native sample → (optional strong aug) → crop → HoVerNet targets.

    The wrapped ``base`` must be built with do_normalize=False (image in [0,1]);
    intensity augmentation runs in [0,1] and we re-normalize with MICRO stats at
    the end (the backbone is FROZEN — aug never touches its weights).
    """

    def __init__(self, base: Dataset, crop_size: int, num_types: int, ignore_index: int = 255,
                 seed: int = 0, aug: str = "strong", mosaic_prob: float = 0.3):
        from ..constants import MICRO_RGB_MEAN, MICRO_RGB_STD
        self.base = base
        self.crop = crop_size
        self.num_types = num_types
        self.ignore_index = ignore_index
        self.rng = np.random.default_rng(seed)
        self.aug = aug
        self.mosaic_prob = mosaic_prob
        self.mean = np.asarray(MICRO_RGB_MEAN, np.float32)
        self.std = np.asarray(MICRO_RGB_STD, np.float32)

    def __len__(self) -> int:
        return len(self.base)

    def _load_crop(self, idx: int):
        """Return (img HWC [0,1] float32, sem [crop,crop] int64, inst int64)."""
        img_t, sem_t, inst_t = self.base[idx][:3]
        img = img_t.permute(1, 2, 0).numpy().astype(np.float32)
        sem = sem_t.numpy().astype(np.int64)
        inst = inst_t.numpy().astype(np.int64)
        h, w = img.shape[:2]
        crop = self.crop
        ph, pw = max(0, crop - h), max(0, crop - w)
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            sem = np.pad(sem, ((0, ph), (0, pw)), constant_values=self.ignore_index)
            inst = np.pad(inst, ((0, ph), (0, pw)), constant_values=0)
            h, w = img.shape[:2]
        y = int(self.rng.integers(0, h - crop + 1)) if h > crop else 0
        x = int(self.rng.integers(0, w - crop + 1)) if w > crop else 0
        return (img[y:y + crop, x:x + crop], sem[y:y + crop, x:x + crop], inst[y:y + crop, x:x + crop])

    def __getitem__(self, idx: int):
        from . import augment as A
        if self.aug == "strong" and self.rng.random() < self.mosaic_prob:
            samples = [self._load_crop(int(self.rng.integers(0, len(self.base)))) for _ in range(4)]
            img, sem, inst = A.mosaic(samples, self.crop, self.rng)
        else:
            img, sem, inst = self._load_crop(idx)
            if self.aug == "strong":
                img, sem, inst = A.random_scale_rotate(img, sem, inst, self.rng)

        img, sem, inst = A.random_flip_rot(img, sem, inst, self.rng)
        if self.aug == "strong":
            img = A.cell_aware_intensity(img, inst, self.rng)
            img = A.random_intensity(img, self.rng)

        img = (img - self.mean) / self.std
        img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        t = make_targets(inst, sem, ignore_index=self.ignore_index)
        return img_t, t["np"], t["hv"], t["tp"]


# ---------------------------------------------------------------------------
# Evaluation: per-image sliding window → postproc → instance metrics
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(
    model,
    base_ds: Dataset,
    device: torch.device,
    num_types: int,
    crop_size: int,
    stride: int,
    patch_size: int,
    max_images: Optional[int] = None,
    tta: bool = False,
    fg_thresh: float = 0.5,
    energy_thresh: float = 0.4,
) -> Dict[str, float]:
    model.eval()
    preds_i: List[np.ndarray] = []
    gts_i: List[np.ndarray] = []
    preds_s: List[np.ndarray] = []
    gts_s: List[np.ndarray] = []

    n = len(base_ds) if max_images is None else min(int(max_images), len(base_ds))
    for i in tqdm(range(n), desc="Eval", leave=False):
        sample = base_ds[i]
        img, sem, inst = sample[0], sample[1], sample[2]
        out = sliding_window_predict(
            model, img.to(device), crop_size=crop_size, stride=stride,
            patch_size=patch_size, num_types=num_types, tta=tta,
        )
        pred_inst, pred_sem = postprocess(
            out["np"], out["hv"], out["tp"], fg_thresh=fg_thresh, energy_thresh=energy_thresh,
        )
        preds_i.append(pred_inst)
        gts_i.append(inst.numpy().astype(np.int32))
        if num_types > 0:
            gs = sem.numpy().astype(np.int32).copy()
            gs[gs == 255] = 0
            preds_s.append(pred_sem)
            gts_s.append(gs)

    if num_types > 0:
        return accumulate_instance_metrics(preds_i, gts_i, preds_s, gts_s, num_classes=num_types)
    return accumulate_instance_metrics(preds_i, gts_i)


# ---------------------------------------------------------------------------
# Best-checkpoint helpers
# ---------------------------------------------------------------------------

def _save_best(model, path: str):
    if model.freeze_backbone:
        # Keep the historical decoder-only format consumable by standalone evaluators.
        torch.save(model.decoder.state_dict(), path)
        return
    payload = {
        "format_version": 2,
        "backbone_mode": model.backbone_mode,
        "decoder": model.decoder.state_dict(),
    }
    trainable_names = {
        name for name, param in model.backbone.named_parameters() if param.requires_grad
    }
    payload["backbone"] = {
        name: value
        for name, value in model.backbone.state_dict().items()
        if name in trainable_names
    }
    torch.save(payload, path)


def _load_best(model, path: str, device):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and sd.get("format_version") == 2:
        model.decoder.load_state_dict(sd["decoder"])
        if "backbone" in sd:
            model.backbone.load_state_dict(sd["backbone"], strict=False)
        return
    # Backward compatibility with decoder-only and full-model checkpoints.
    if model.freeze_backbone:
        model.decoder.load_state_dict(sd)
    else:
        model.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def _build_instance_dataset(args, split: str, do_normalize: bool = True) -> Dataset:
    """Build the native-resolution dataset expected by the HoVerNet trainer.

    The semantic Cellpose loader binarizes masks and returns only two tensors.
    For this instance track, use a dedicated loader that preserves Cellpose
    instance IDs and returns (image, semantic foreground, instance map).
    """
    if args.dataset == "cellpose":
        from ..datasets.cellpose_instance import CellposeInstanceDataset, get_cellpose_instance_paths

        img_paths, mask_paths = get_cellpose_instance_paths(args.data_root, split=split)
        return CellposeInstanceDataset(
            img_paths,
            mask_paths,
            size=None,
            augment=False,
            do_normalize=do_normalize,
        )

    from ..feature_extractor import _build_dataset

    return _build_dataset(
        args.dataset,
        args.data_root,
        split,
        None,
        augment=False,
        do_normalize=do_normalize,
    )


def run(args) -> Dict:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_types = DATASET_NUM_TYPES.get(args.dataset, 0)
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_dino_hovernet(
        checkpoint=args.checkpoint,
        train_config=args.train_config,
        layers=args.layers,
        num_types=num_types,
        freeze_backbone=args.freeze_backbone,
        trainable_backbone_blocks=args.unfreeze_last_blocks,
        feature_size=args.feature_size,
        embed_proj=args.embed_proj,
        device=device,
    )
    patch_size = int(model.backbone.patch_size)

    # Datasets: native resolution; training random-crops + augments, eval tiles.
    # Train images are un-normalized ([0,1]) so intensity aug runs before MICRO-norm.
    base_train = _build_instance_dataset(args, "train", do_normalize=False)
    base_val = _build_instance_dataset(args, "val", do_normalize=True)
    train_ds = RandomCropHoVerDataset(base_train, args.crop_size, num_types, seed=args.seed,
                                      aug=args.aug, mosaic_prob=args.mosaic_prob)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    tr_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        generator=loader_generator,
    )

    criterion = HoVerNetLoss(num_types=num_types).to(device)
    if args.freeze_backbone:
        optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Fine-tune: decoder at --lr, backbone at a much lower --backbone-lr so the
        # pretrained features are adapted, not destroyed.
        bb_lr = args.backbone_lr if args.backbone_lr is not None else args.lr * 0.02
        trainable_backbone = list(model.trainable_backbone_parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": model.decoder.parameters(), "lr": args.lr},
                {"params": trainable_backbone, "lr": bb_lr},
            ],
            weight_decay=args.weight_decay,
        )
        logger.info(
            "Fine-tuning mode=%s: decoder lr=%.2e, backbone lr=%.2e, trainable backbone params=%d",
            model.backbone_mode,
            args.lr,
            bb_lr,
            sum(param.numel() for param in trainable_backbone),
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp_dtype]
    amp_enabled = device.type == "cuda" and amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    sel_key = "mPQ" if num_types > 0 else "bPQ"
    best_val = -1.0
    best_path = os.path.join(args.output_dir, "best_head.pth")
    eval_every = max(1, int(args.eval_every))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        nb = 0
        total_train_batches = len(tr_loader)
        if args.max_train_batches is not None:
            total_train_batches = min(total_train_batches, int(args.max_train_batches))
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}", leave=False)
        for bi, (img, np_t, hv_t, tp_t) in enumerate(pbar):
            if args.max_train_batches is not None and bi >= args.max_train_batches:
                break
            img = img.to(device, non_blocking=True)
            target = {
                "np": np_t.to(device, non_blocking=True),
                "hv": hv_t.to(device, non_blocking=True),
                "tp": tp_t.to(device, non_blocking=True),
            }
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred = model(img)
                loss, comps = criterion(pred, target)
                scaled_loss = loss / args.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            should_step = (bi + 1) % args.grad_accum_steps == 0 or (bi + 1) >= total_train_batches
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += comps["total"]
            nb += 1
            pbar.set_postfix(loss=f"{comps['total']:.3f}")
        scheduler.step()

        if epoch % eval_every == 0 or epoch == args.epochs:
            val = evaluate(
                model, base_val, device, num_types,
                crop_size=args.crop_size, stride=args.stride, patch_size=patch_size,
                max_images=args.max_eval_images,
            )
            logger.info(
                "Epoch %3d/%d  loss=%.4f  val_%s=%.4f  val_AJI=%.4f  val_AP50=%.4f",
                epoch, args.epochs, running / max(nb, 1), sel_key, val.get(sel_key, float("nan")),
                val.get("AJI", float("nan")), val.get("AP50", float("nan")),
            )
            if val.get(sel_key, -1.0) > best_val:
                best_val = val[sel_key]
                _save_best(model, best_path)

    # ---- final test ----
    if os.path.exists(best_path):
        _load_best(model, best_path, device)

    results: Dict = {
        "val": evaluate(
            model, base_val, device, num_types,
            crop_size=args.crop_size, stride=args.stride, patch_size=patch_size,
            max_images=args.max_eval_images, tta=args.tta,
            fg_thresh=args.fg_thresh, energy_thresh=args.energy_thresh,
        ),
        "_meta": {
            "dataset": args.dataset,
            "layers": model.layers,
            "num_types": num_types,
            "freeze_backbone": bool(args.freeze_backbone),
            "backbone_mode": model.backbone_mode,
            "unfreeze_last_blocks": args.unfreeze_last_blocks,
            "feature_size": args.feature_size,
            "embed_proj": args.embed_proj,
            "crop_size": args.crop_size,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "decoder_lr": args.lr,
            "backbone_lr": None if args.freeze_backbone else bb_lr,
            "amp_dtype": args.amp_dtype,
            "seed": args.seed,
            "select_metric": sel_key,
        },
    }
    if not args.skip_test_eval:
        try:
            base_test = _build_instance_dataset(args, "test", do_normalize=True)
            results["test"] = evaluate(
                model, base_test, device, num_types,
                crop_size=args.crop_size, stride=args.stride, patch_size=patch_size,
                max_images=args.max_eval_images, tta=args.tta,
                fg_thresh=args.fg_thresh, energy_thresh=args.energy_thresh,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Test split unavailable: %s", e)

    out_json = os.path.join(args.output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved → %s", out_json)
    for split in ("val", "test"):
        if split in results:
            logger.info("[%s] %s", split.upper(), {k: round(v, 4) for k, v in results[split].items()})
    return results


def main():
    p = argparse.ArgumentParser(description="DINOHoVerNet instance-seg train/eval")
    p.add_argument("--dataset", required=True, choices=list(DATASET_NUM_TYPES.keys()))
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True, help="DCP dir or consolidated .pth")
    p.add_argument("--train-config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="ViT layer indices to tap (any count). Default: even-4 by depth.")
    p.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true", default=True)
    p.add_argument("--finetune", dest="freeze_backbone", action="store_false",
                   help="Fine-tune the backbone end-to-end (overrides --freeze-backbone).")
    p.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=None,
        help="With --finetune, update only the last N transformer blocks plus final norm.",
    )
    p.add_argument("--feature-size", type=int, default=32)
    p.add_argument("--embed-proj", type=int, default=384)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=None,
                   help="Backbone LR when fine-tuning (default: lr*0.02). Ignored if frozen.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--amp-dtype", choices=["none", "bf16", "fp16"], default="none")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--max-eval-images", type=int, default=None,
                   help="Cap eval images for fast screening.")
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="Cap training batches per epoch (sanity/screening only).")
    p.add_argument("--skip-test-eval", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aug", choices=["none", "strong"], default="strong",
                   help="Training augmentation (backbone stays frozen): strong = cell-aware "
                        "intensity + mosaic + intensity/blur/noise + scale-rotate-flip.")
    p.add_argument("--mosaic-prob", type=float, default=0.3)
    p.add_argument("--tta", action="store_true", help="4-way flip TTA at eval.")
    p.add_argument("--fg-thresh", type=float, default=0.5)
    p.add_argument("--energy-thresh", type=float, default=0.4)
    args = p.parse_args()

    if args.freeze_backbone and args.unfreeze_last_blocks is not None:
        p.error("--unfreeze-last-blocks requires --finetune")
    if args.unfreeze_last_blocks is not None and args.unfreeze_last_blocks <= 0:
        p.error("--unfreeze-last-blocks must be positive")
    if args.grad_accum_steps <= 0:
        p.error("--grad-accum-steps must be positive")

    if args.layers is None:
        # Resolve even-4 from backbone depth lazily inside build; here pick a
        # sensible default for vit_large; the pipeline script passes --layers.
        args.layers = [4, 11, 17, 23]
    run(args)


if __name__ == "__main__":
    main()
