"""
Fine-tune or eval-only Cellpose-SAM (cpsam) with the SAME instance metrics.

Training mode fine-tunes cpsam on a dataset's train split and evaluates one split.
Eval-only mode loads an already saved cpsam model (for example
``outputs/instance_seg/cpsam_ft/<dataset>/models/cpsam_ft_<dataset>``) and scores
an official test split without retraining.

Run in an env with cellpose 4.x installed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from dinov3.eval.bio_segmentation.feature_extractor import _build_dataset
from dinov3.eval.bio_segmentation.metrics import accumulate_instance_metrics


def _collect(ds, max_n=None):
    """Return (list[HxWx3 uint8], list[HxW int instance])."""
    imgs, labels = [], []
    n = len(ds) if max_n is None else min(max_n, len(ds))
    for i in range(n):
        s = ds[i]
        img = (s[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        imgs.append(img)
        labels.append(s[2].numpy().astype(np.int32))
    return imgs, labels


def _build_cpsam(model_path: str | None, gpu: bool):
    from cellpose import models

    if model_path:
        return models.CellposeModel(gpu=gpu, pretrained_model=model_path)
    return models.CellposeModel(gpu=gpu)


def _evaluate_cpsam(model, dataset: str, data_root: str, split: str, max_eval_images: int | None):
    ev = _build_dataset(dataset, data_root, split, None, do_normalize=False)
    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    n = len(ev) if max_eval_images is None else min(max_eval_images, len(ev))
    for i in tqdm(range(n), desc=f"eval {dataset}:{split}"):
        s = ev[i]
        img = (s[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        masks = model.eval(img)[0]
        preds.append(np.asarray(masks).astype(np.int32))
        gts.append(s[2].numpy().astype(np.int32))
    return accumulate_instance_metrics(preds, gts), n


def _default_saved_model(output_dir: str, dataset: str) -> str:
    return str(Path(output_dir) / "models" / f"cpsam_ft_{dataset}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="val")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training and evaluate --model-path on --eval-split.")
    p.add_argument("--model-path", default=None,
                   help="Fine-tuned cpsam model file. Required for --eval-only; in training "
                        "mode defaults to <output-dir>/models/cpsam_ft_<dataset> after save.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--nimg-per-epoch", type=int, default=500)
    p.add_argument("--max-eval-images", type=int, default=None)
    p.add_argument("--gpu", action="store_true", default=True,
                   help="Use GPU for cpsam. Pass --no-gpu to force CPU.")
    p.add_argument("--no-gpu", dest="gpu", action="store_false")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.eval_only:
        if not args.model_path:
            raise SystemExit("--eval-only requires --model-path")
        model = _build_cpsam(args.model_path, gpu=args.gpu)
        metrics, n_eval = _evaluate_cpsam(
            model, args.dataset, args.data_root, args.eval_split, args.max_eval_images
        )
        out = {
            args.eval_split: metrics,
            "_meta": {
                "specialist": "cpsam_finetuned",
                "dataset": args.dataset,
                "mode": "eval_only",
                "model_path": args.model_path,
                "eval_split": args.eval_split,
                "n_eval": n_eval,
            },
        }
        out_json = os.path.join(args.output_dir, "results.json")
        with open(out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[eval-cpsam-ft:{args.dataset}/{args.eval_split}] "
              f"{({k: round(v, 4) for k, v in metrics.items()})}", flush=True)
        print(f"[eval-cpsam-ft:{args.dataset}] results -> {out_json}", flush=True)
        return

    from cellpose import train

    # un-normalized RGB; cellpose does its own percentile normalization.
    tr = _build_dataset(args.dataset, args.data_root, args.train_split, None, do_normalize=False)
    tr_imgs, tr_lbls = _collect(tr)
    print(f"[ft-cpsam:{args.dataset}] train images: {len(tr_imgs)}", flush=True)

    model = _build_cpsam(None, gpu=args.gpu)
    train.train_seg(
        model.net,
        train_data=tr_imgs, train_labels=tr_lbls,
        batch_size=args.batch_size, learning_rate=args.lr, n_epochs=args.epochs,
        weight_decay=0.1, normalize=True,
        nimg_per_epoch=min(args.nimg_per_epoch, len(tr_imgs)),
        save_path=args.output_dir, model_name=f"cpsam_ft_{args.dataset}",
    )
    print(f"[ft-cpsam:{args.dataset}] fine-tune done; evaluating on {args.eval_split}", flush=True)

    saved_model = args.model_path or _default_saved_model(args.output_dir, args.dataset)
    if os.path.exists(saved_model):
        model = _build_cpsam(saved_model, gpu=args.gpu)
    metrics, n_eval = _evaluate_cpsam(
        model, args.dataset, args.data_root, args.eval_split, args.max_eval_images
    )
    out = {
        args.eval_split: metrics,
        "_meta": {
            "specialist": "cpsam_finetuned",
            "dataset": args.dataset,
            "mode": "train_then_eval",
            "model_path": saved_model if os.path.exists(saved_model) else None,
            "epochs": args.epochs,
            "n_eval": n_eval,
            "n_train": len(tr_imgs),
        },
    }
    json.dump(out, open(os.path.join(args.output_dir, "results.json"), "w"), indent=2)
    print(f"[ft-cpsam:{args.dataset}] {({k: round(v,4) for k,v in metrics.items()})}", flush=True)


if __name__ == "__main__":
    main()
