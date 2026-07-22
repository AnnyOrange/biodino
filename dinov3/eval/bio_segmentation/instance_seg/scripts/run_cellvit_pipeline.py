"""
Driver for the instance-seg (CellViT/HoVerNet) track.

Loops datasets × checkpoint-iterations × {frozen, last2, last4, finetune} and invokes
``instance_seg.train`` for each. Comparison *rows* (bio-DINOv3 vs generic DINOv3
vs other FMs) come from running this script once per backbone, pointing
``--checkpoints-dir`` / ``--train-config`` at each. The decisive delta is
bio-DINOv3 minus generic DINOv3 — same harness, only the backbone changes.

Example (one backbone, two datasets, frozen):
    python -m dinov3.eval.bio_segmentation.instance_seg.scripts.run_cellvit_pipeline \\
        --datasets pannuke monuseg \\
        --checkpoints-dir /ckpt/bio_dinov3 --checkpoint-iters latest \\
        --train-config dinov3/configs/train/microscopy_continual_vitl16.yaml \\
        --data-root-base /data/segmentation_datasets \\
        --output-root ./outputs/instance_seg/bio_dinov3 \\
        --modes frozen --epochs 50 --gpu 0
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bio_seg.instance_seg.pipeline")

ARCH_DEPTH = {
    "vit_small": 12, "vit_base": 12, "vit_large": 24, "vit_so400m": 27,
    "vit_huge2": 32, "vit_huge": 32, "vit_giant2": 40, "vit_7b": 40,
}
EVEN4_BY_DEPTH = {12: [2, 5, 8, 11], 24: [4, 11, 17, 23], 27: [6, 13, 20, 26],
                  32: [7, 15, 23, 31], 40: [9, 19, 29, 39]}


def _infer_depth(train_config: str) -> int:
    cfg = OmegaConf.load(train_config)
    arch = str(OmegaConf.select(cfg, "student.arch") or "vit_large")
    return ARCH_DEPTH.get(arch, 24)


def _even4(depth: int) -> List[int]:
    if depth in EVEN4_BY_DEPTH:
        return EVEN4_BY_DEPTH[depth]
    step = max(1, depth // 4)
    return [min(depth - 1, step * (i + 1) - 1) for i in range(4)]


def _data_root(base: str, dataset: str) -> str:
    if dataset == "livecell":
        return os.path.join(base, "LIVECell")
    return os.path.join(base, dataset, "extracted")


def _discover_checkpoints(ckpt_dir: str, which: str) -> Dict[int, str]:
    root = Path(ckpt_dir)
    found: Dict[int, str] = {}
    for sub in sorted(root.iterdir()) if root.is_dir() else []:
        if not sub.is_dir() or not sub.name.isdigit():
            continue
        if (sub / ".metadata").exists():
            found[int(sub.name)] = str(sub)            # DCP directory
        elif (sub / "checkpoint.pth").exists():
            found[int(sub.name)] = str(sub / "checkpoint.pth")
    if not found:
        raise SystemExit(f"No <iter>/checkpoint.pth or <iter>/.metadata under {ckpt_dir}")
    if which == "all":
        return found
    if which == "latest":
        k = max(found)
        return {k: found[k]}
    wanted = {int(x) for x in which.split(",")}
    sel = {k: v for k, v in found.items() if k in wanted}
    if not sel:
        raise SystemExit(f"Requested iters {which} not found; available={sorted(found)}")
    return sel


def main():
    p = argparse.ArgumentParser(description="Instance-seg (CellViT/HoVerNet) pipeline driver")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--checkpoints-dir", required=True)
    p.add_argument("--checkpoint-iters", default="latest", help="latest | all | '1000,2000'")
    p.add_argument("--train-config", required=True)
    p.add_argument("--data-root-base", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Override the tapped ViT layers (any count). Default: even-4 by depth.")
    p.add_argument(
        "--modes",
        nargs="+",
        default=["frozen"],
        choices=["frozen", "last2", "last4", "finetune"],
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--amp-dtype", choices=["none", "bf16", "fp16"], default="none")
    p.add_argument("--feature-size", type=int, default=32)
    p.add_argument("--embed-proj", type=int, default=384)
    p.add_argument("--max-eval-images", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aug", choices=["none", "strong"], default="strong")
    p.add_argument("--mosaic-prob", type=float, default=0.3)
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--skip-test-eval", action="store_true")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--gpu", default="0")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    layers = args.layers or _even4(_infer_depth(args.train_config))
    ckpts = _discover_checkpoints(args.checkpoints_dir, args.checkpoint_iters)
    logger.info("Tapped layers=%s; checkpoints=%s; datasets=%s; modes=%s",
                layers, sorted(ckpts), args.datasets, args.modes)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    failures = []
    for it, ckpt_path in sorted(ckpts.items()):
        for dataset in args.datasets:
            data_root = _data_root(args.data_root_base, dataset)
            for mode in args.modes:
                out_dir = os.path.join(args.output_root, dataset, str(it), mode)
                if args.skip_completed and os.path.isfile(os.path.join(out_dir, "results.json")):
                    logger.info("[iter %s | %s | %s] already complete; skipping", it, dataset, mode)
                    continue
                cmd = [
                    sys.executable, "-m", "dinov3.eval.bio_segmentation.instance_seg.train",
                    "--dataset", dataset,
                    "--data-root", data_root,
                    "--checkpoint", ckpt_path,
                    "--train-config", args.train_config,
                    "--output-dir", out_dir,
                    "--layers", *[str(x) for x in layers],
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--grad-accum-steps", str(args.grad_accum_steps),
                    "--crop-size", str(args.crop_size),
                    "--stride", str(args.stride),
                    "--lr", str(args.lr),
                    "--weight-decay", str(args.weight_decay),
                    "--amp-dtype", args.amp_dtype,
                    "--feature-size", str(args.feature_size),
                    "--embed-proj", str(args.embed_proj),
                    "--num-workers", str(args.num_workers),
                    "--eval-every", str(args.eval_every),
                    "--seed", str(args.seed),
                    "--aug", args.aug,
                    "--mosaic-prob", str(args.mosaic_prob),
                ]
                if mode == "frozen":
                    cmd.append("--freeze-backbone")
                else:
                    cmd.append("--finetune")
                    if mode in {"last2", "last4"}:
                        cmd.extend(["--unfreeze-last-blocks", mode.removeprefix("last")])
                    if args.backbone_lr is not None:
                        cmd.extend(["--backbone-lr", str(args.backbone_lr)])
                if args.max_eval_images is not None:
                    cmd += ["--max-eval-images", str(args.max_eval_images)]
                if args.skip_test_eval:
                    cmd.append("--skip-test-eval")
                if args.tta:
                    cmd.append("--tta")

                logger.info("[iter %s | %s | %s] → %s", it, dataset, mode, out_dir)
                if args.dry_run:
                    print(shlex.join(cmd))
                    continue
                proc = subprocess.run(cmd, env=env, check=False)
                if proc.returncode != 0:
                    failures.append((it, dataset, mode, proc.returncode))
                    logger.error(
                        "[iter %s | %s | %s] failed with code %s",
                        it,
                        dataset,
                        mode,
                        proc.returncode,
                    )
                    if not args.continue_on_error:
                        raise SystemExit(proc.returncode)

    if failures:
        logger.error("Pipeline completed with %d failed jobs: %s", len(failures), failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
