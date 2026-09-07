#!/usr/bin/env python
"""Launch controlled backbone-adaptation experiments for instance segmentation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "instance_seg_tuning"
DEFAULT_CKPT = ROOT / "outputs" / "01_training_runs" / "5tb_idweak10_vitl16_robust_b1024_8gpu" / "ckpt" / "15374" / "checkpoint.pth"
DEFAULT_TRAIN_CONFIG = ROOT / "dinov3" / "configs" / "train" / "microscopy_continual_vitl16_robust_5tb_idweak10.yaml"
DATA_ROOT_BASE = Path("/mnt/huawei_deepcad/benchmark/segmentation")

PRIMARY_METRIC = {
    "pannuke": "mPQ",
    "conic": "mPQ",
    "monuseg": "AJI",
    "livecell": "SEG",
    "bbbc038": "CellposeStyleAP",
    "tissuenet": "CellposeStyleAP",
    "cellpose": "CellposeStyleAP",
}

METRIC_COLUMNS = [
    "AJI", "Dice", "AP", "AP50", "AP75", "ObjectAP", "ObjectAP50", "ObjectAP75",
    "COCOProxyAP", "COCOProxyAP50", "COCOProxyAP75", "CellposeStyleAP", "CellposeStyleAP50", "CellposeStyleAP75",
    "SEG", "bPQ", "bSQ", "bDQ", "mPQ", "mSQ", "mDQ",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def append_csv(path: Path, row: Dict[str, object], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields)
    rows: List[Dict[str, object]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing = list(reader.fieldnames or [])
            rows = list(reader)
        fields = existing + [field for field in fields if field not in existing]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def data_root(dataset: str) -> Path:
    if dataset == "livecell":
        return DATA_ROOT_BASE / "LIVECell"
    return DATA_ROOT_BASE / dataset / "extracted"


def metric_value(metrics: Dict[str, object], key: str) -> float:
    try:
        if key == "CellposeStyleAP" and key not in metrics:
            key = "ObjectAP"
        return float(metrics.get(key, float("nan")))
    except Exception:
        return float("nan")


def build_cmd(args, dataset: str, mode: str, out_dir: Path, batch_size: int, grad_accum: int) -> List[str]:
    cmd = [
        sys.executable, "-m", "dinov3.eval.bio_segmentation.instance_seg.train",
        "--dataset", dataset,
        "--data-root", str(data_root(dataset)),
        "--checkpoint", str(args.checkpoint),
        "--train-config", str(args.train_config),
        "--output-dir", str(out_dir),
        "--layers", *[str(x) for x in args.layers],
        "--epochs", str(args.epochs),
        "--batch-size", str(batch_size),
        "--grad-accum-steps", str(grad_accum),
        "--crop-size", str(args.crop_size),
        "--stride", str(args.stride),
        "--lr", str(args.lr),
        "--layer-wise-lr-decay", str(args.layer_wise_lr_decay),
        "--weight-decay", str(args.weight_decay),
        "--warmup-ratio", str(args.warmup_ratio),
        "--grad-clip-norm", str(args.grad_clip_norm),
        "--select-metric", PRIMARY_METRIC[dataset],
        "--amp-dtype", args.amp_dtype,
        "--feature-size", str(args.feature_size),
        "--embed-proj", str(args.embed_proj),
        "--fusion-mode", args.fusion_mode,
        "--decoder-variant", args.decoder_variant,
        "--num-workers", str(args.num_workers),
        "--eval-every", str(args.eval_every),
        "--seed", str(args.seed),
        "--aug", args.aug,
        "--mosaic-prob", str(args.mosaic_prob),
        "--fg-thresh", str(args.fg_thresh),
        "--energy-thresh", str(args.energy_thresh),
        "--np-loss-mode", args.np_loss_mode,
        "--focal-gamma", str(args.focal_gamma),
        "--tversky-alpha", str(args.tversky_alpha),
        "--tversky-beta", str(args.tversky_beta),
        "--skip-test-eval",
    ]
    if args.spatial_adapter:
        cmd += ["--spatial-adapter", "--spatial-adapter-width", str(args.spatial_adapter_width)]
    if mode == "frozen":
        cmd.append("--freeze-backbone")
    elif mode.startswith("lora"):
        rank_text = mode.removeprefix("lora")
        if not rank_text.isdigit():
            raise ValueError(f"LoRA mode must be lora<rank>, got {mode!r}")
        cmd += ["--freeze-backbone", "--lora-rank", rank_text, "--lora-alpha", str(args.lora_alpha)]
        cmd += ["--lora-dropout", str(args.lora_dropout), "--backbone-lr", str(args.backbone_lr)]
    elif mode == "adapter":
        cmd += ["--freeze-backbone", "--adapter", "--adapter-dim", str(args.adapter_dim)]
    elif mode.startswith("last"):
        cmd += ["--finetune", "--unfreeze-last-blocks", mode.removeprefix("last")]
        cmd += ["--backbone-lr", str(args.backbone_lr)]
    elif mode == "finetune":
        cmd += ["--finetune", "--backbone-lr", str(args.backbone_lr)]
    else:
        raise ValueError(f"Unsupported mode={mode!r}")
    if args.max_eval_images is not None:
        cmd += ["--max-eval-images", str(args.max_eval_images)]
    if args.max_train_batches is not None:
        cmd += ["--max-train-batches", str(args.max_train_batches)]
    return cmd


def record(args, dataset: str, mode: str, out_dir: Path, log_path: Path, pid: int, code: int, wall: float, batch_size: int, grad_accum: int, oom_retry: int) -> None:
    results_path = out_dir / "results.json"
    metrics: Dict[str, object] = {}
    meta: Dict[str, object] = {}
    if results_path.exists():
        data = json.loads(results_path.read_text())
        metrics = data.get("val", {})
        meta = data.get("_meta", {})
    primary = PRIMARY_METRIC[dataset]
    row = {
        "timestamp": now(),
        "dataset": dataset,
        "mode": mode,
        "layers": " ".join(str(x) for x in args.layers),
        "fusion_mode": args.fusion_mode,
        "decoder_variant": args.decoder_variant,
        "spatial_adapter": int(args.spatial_adapter),
        "spatial_adapter_fusion": "additive" if args.spatial_adapter else "none",
        "spatial_adapter_width": args.spatial_adapter_width,
        "split": "val",
        "primary_metric": primary,
        "primary_value": metric_value(metrics, primary),
        "epochs": args.epochs,
        "max_train_batches": args.max_train_batches if args.max_train_batches is not None else "",
        "max_eval_images": args.max_eval_images if args.max_eval_images is not None else "",
        "crop_size": args.crop_size,
        "stride": args.stride,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "decoder_lr": args.lr,
        "backbone_lr": "" if mode == "frozen" else args.backbone_lr,
        "warmup_ratio": args.warmup_ratio,
        "grad_clip_norm": args.grad_clip_norm,
        "layer_wise_lr_decay": args.layer_wise_lr_decay,
        "amp_dtype": args.amp_dtype,
        "feature_size": args.feature_size,
        "embed_proj": args.embed_proj,
        "lora_rank": meta.get("lora_rank", ""),
        "lora_alpha": meta.get("lora_alpha", ""),
        "lora_dropout": meta.get("lora_dropout", ""),
        "trainable_params": meta.get("trainable_params", ""),
        "trainable_backbone_params": meta.get("trainable_backbone_params", ""),
        "seed": args.seed,
        "np_loss_mode": args.np_loss_mode,
        "focal_gamma": args.focal_gamma,
        "tversky_alpha": args.tversky_alpha,
        "tversky_beta": args.tversky_beta,
        "pid": pid,
        "gpu": args.gpu,
        "exit_code": code,
        "oom_retry": oom_retry,
        "wall_seconds": wall,
        "training_seconds": meta.get("training_seconds", ""),
        "inference_seconds": meta.get("inference_seconds", ""),
        "peak_cuda_gib": meta.get("peak_cuda_memory_gib", ""),
        "results_json": str(results_path),
        "log_path": str(log_path),
    }
    for key in METRIC_COLUMNS:
        row[key] = metric_value(metrics, key)
    append_csv(OUT_ROOT / "adaptation_results.csv", row, row.keys())
    append_csv(
        OUT_ROOT / "metric_improvement_experiment_log.csv",
        {
            "timestamp": row["timestamp"],
            "dataset": dataset,
            "stage": "stage4",
            "experiment": mode,
            "changed_variable": "np_loss_mode" if args.np_loss_mode != "ce_dice" else "loss_baseline",
            "config": json.dumps(
                {
                    "mode": mode,
                    "layers": args.layers,
                    "layer_wise_lr_decay": args.layer_wise_lr_decay,
                    "np_loss_mode": args.np_loss_mode,
                    "focal_gamma": args.focal_gamma,
                    "tversky_alpha": args.tversky_alpha,
                    "tversky_beta": args.tversky_beta,
                    "meta": meta,
                },
                sort_keys=True,
            ),
            "checkpoint": str(results_path),
            "split": "val",
            "n_images": "",
            "primary_metric": primary,
            "primary_value": row["primary_value"],
            "wall_seconds": wall,
            "peak_cuda_gib": "",
            "gpu": args.gpu,
            "pid": pid,
            "exit_code": code,
            "log_path": str(log_path),
        },
        [
            "timestamp", "dataset", "stage", "experiment", "changed_variable", "config",
            "checkpoint", "split", "n_images", "primary_metric", "primary_value",
            "wall_seconds", "peak_cuda_gib", "gpu", "pid", "exit_code", "log_path",
        ],
    )


def summarize() -> None:
    csv_path = OUT_ROOT / "adaptation_results.csv"
    out = OUT_ROOT / "adaptation_analysis.md"
    lines = ["# Adaptation Analysis", ""]
    if not csv_path.exists():
        lines.append("No adaptation experiments have completed yet.")
        out.write_text("\n".join(lines) + "\n")
        return
    rows = list(csv.DictReader(csv_path.open(newline="")))
    lines += [
        "Rows compare backbone adaptation with fixed dataset, split, taps, fusion, decoder width, crop, stride, loss and validation protocol.",
        "",
        "| dataset | mode | layers | fusion | primary | value | epochs | lr | backbone_lr | trainable | exit | results |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('dataset','')} | {row.get('mode','')} | {row.get('layers','')} | "
            f"{row.get('fusion_mode','')} | {row.get('primary_metric','')} | "
            f"{metric_value(row, 'primary_value'):.4f} | {row.get('epochs','')} | "
            f"{row.get('decoder_lr','')} | {row.get('backbone_lr','')} | "
            f"{row.get('trainable_params','')} | "
            f"{row.get('exit_code','')} | `{row.get('results_json','')}` |"
        )
    out.write_text("\n".join(lines) + "\n")


def run(args) -> int:
    out_root = OUT_ROOT / "adaptation_runs" / args.run_name
    log_root = OUT_ROOT / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps({"created_at": datetime.now().isoformat(), "args": vars(args)}, indent=2) + "\n")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    failures = 0
    for dataset in args.datasets:
        for mode in args.modes:
            out_dir = out_root / dataset / mode
            if args.skip_completed and (out_dir / "results.json").exists():
                record(args, dataset, mode, out_dir, out_dir / "skipped_existing.log", os.getpid(), 0, 0.0, args.batch_size, args.grad_accum_steps, 0)
                continue
            attempts = [(args.batch_size, args.grad_accum_steps)]
            if args.oom_retry:
                attempts.append((max(1, args.batch_size // 2), args.grad_accum_steps * 2))
            final_code = 1
            final_pid = -1
            final_wall = 0.0
            final_log = log_root / "missing.log"
            final_batch = args.batch_size
            final_accum = args.grad_accum_steps
            final_retry = 0
            for retry, (batch_size, grad_accum) in enumerate(attempts):
                log_path = log_root / f"stage4_{args.run_name}_{dataset}_{mode}_try{retry}_{time.strftime('%Y%m%d_%H%M%S')}.log"
                cmd = build_cmd(args, dataset, mode, out_dir, batch_size, grad_accum)
                start = time.time()
                with log_path.open("w") as log:
                    log.write(f"$ {shlex.join(cmd)}\n")
                    log.flush()
                    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
                    final_pid = proc.pid
                    final_code = proc.wait()
                final_wall = time.time() - start
                final_log = log_path
                final_batch = batch_size
                final_accum = grad_accum
                final_retry = retry
                if final_code == 0:
                    break
                text = log_path.read_text(errors="ignore")[-12000:].lower()
                if "out of memory" not in text and "cuda error: out of memory" not in text:
                    break
            record(args, dataset, mode, out_dir, final_log, final_pid, final_code, final_wall, final_batch, final_accum, final_retry)
            if not args.preserve_existing_summary:
                summarize()
            if final_code != 0:
                failures += 1
                if not args.continue_on_error:
                    return final_code
    return 1 if failures else 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["monuseg"])
    p.add_argument("--modes", nargs="+", default=["frozen", "last4"],
                   choices=["frozen", "last2", "last4", "last8", "lora8", "lora16", "adapter", "finetune"])
    p.add_argument("--run-name", default="screen_vitl16_even4_adaptation")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--train-config", default=str(DEFAULT_TRAIN_CONFIG))
    p.add_argument("--layers", type=int, nargs="+", default=[4, 11, 17, 23])
    p.add_argument("--gpu", default="0")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--layer-wise-lr-decay", type=float, default=1.0)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--adapter-dim", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--amp-dtype", choices=["none", "bf16", "fp16"], default="bf16")
    p.add_argument("--feature-size", type=int, default=32)
    p.add_argument("--embed-proj", type=int, default=384)
    p.add_argument("--fusion-mode", choices=["bucket_concat", "weighted_sum"], default="bucket_concat")
    p.add_argument("--decoder-variant", choices=["current", "fpn", "unet", "multi_layer_fpn"], default="current")
    p.add_argument("--spatial-adapter", action="store_true")
    p.add_argument("--spatial-adapter-width", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aug", choices=["none", "strong"], default="strong")
    p.add_argument("--mosaic-prob", type=float, default=0.3)
    p.add_argument("--fg-thresh", type=float, default=0.5)
    p.add_argument("--energy-thresh", type=float, default=0.4)
    p.add_argument("--np-loss-mode", choices=["ce_dice", "focal_tversky"], default="ce_dice")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--tversky-alpha", type=float, default=0.3)
    p.add_argument("--tversky-beta", type=float, default=0.7)
    p.add_argument("--max-eval-images", type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--oom-retry", action="store_true")
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument(
        "--preserve-existing-summary",
        action="store_true",
        help="Append run records without rewriting the historical adaptation_analysis.md summary.",
    )
    p.add_argument("--summarize-only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.summarize_only:
        summarize()
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
