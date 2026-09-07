#!/usr/bin/env python
"""Launch and collect controlled instance-seg layer-fusion experiments.

This is a thin orchestration layer around the existing DINOHoVerNet trainer.
It changes only the tapped ViT layer list between jobs and records enough
metadata to make the comparison auditable.
"""

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
DEFAULT_CKPT_DIR = ROOT / "outputs" / "01_training_runs" / "5tb_idweak10_vitl16_robust_b1024_8gpu" / "ckpt"
DEFAULT_TRAIN_CONFIG = ROOT / "dinov3" / "configs" / "train" / "microscopy_continual_vitl16_robust_5tb_idweak10.yaml"
DATA_ROOT_BASE = Path("/mnt/huawei_deepcad/benchmark/segmentation")

PRIMARY_METRIC = {
    "pannuke": "mPQ",
    "conic": "mPQ",
    "monuseg": "AJI",
    "livecell": "SEG",
    "bbbc038": "ObjectAP",
    "tissuenet": "ObjectAP",
    "cellpose": "ObjectAP",
}

METRIC_COLUMNS = [
    "AJI",
    "AP",
    "AP50",
    "AP75",
    "ObjectAP",
    "ObjectAP50",
    "ObjectAP75",
    "SEG",
    "bPQ",
    "bSQ",
    "bDQ",
    "mPQ",
    "mSQ",
    "mDQ",
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
            existing_fields = list(reader.fieldnames or [])
            rows = list(reader)
        fields = existing_fields + [field for field in fields if field not in existing_fields]
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


def combo_layers(depth: int) -> Dict[str, List[int]]:
    if depth == 24:
        return {
            "final": [23],
            "shallow_single": [5],
            "mid_single": [11],
            "deep_single": [17],
            "last4": [20, 21, 22, 23],
            "even4": [4, 11, 17, 23],
            "late_heavy4": [14, 19, 22, 23],
            "even8": [2, 5, 8, 11, 14, 17, 20, 23],
        }
    if depth == 40:
        return {
            "final": [39],
            "shallow_single": [9],
            "mid_single": [19],
            "deep_single": [29],
            "last4": [36, 37, 38, 39],
            "even4": [9, 19, 29, 39],
            "late_heavy4": [24, 32, 37, 39],
            "even8": [4, 9, 14, 19, 24, 29, 34, 39],
        }
    raise ValueError(f"Unsupported depth={depth}; pass explicit --combo name:layers if needed")


def parse_combos(raw: List[str] | None, depth: int) -> Dict[str, List[int]]:
    defaults = combo_layers(depth)
    if not raw:
        return {key: defaults[key] for key in ["final", "shallow_single", "mid_single", "deep_single", "last4", "even4"]}
    combos: Dict[str, List[int]] = {}
    for item in raw:
        if ":" not in item:
            combos[item] = defaults[item]
            continue
        name, values = item.split(":", 1)
        combos[name] = [int(v) for v in values.replace(",", " ").split()]
    return combos


def train_command(args, dataset: str, combo_name: str, layers: List[int], out_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "dinov3.eval.bio_segmentation.instance_seg.train",
        "--dataset",
        dataset,
        "--data-root",
        str(data_root(dataset)),
        "--checkpoint",
        str(DEFAULT_CKPT_DIR / args.checkpoint_iter / "checkpoint.pth"),
        "--train-config",
        str(args.train_config),
        "--output-dir",
        str(out_dir),
        "--layers",
        *[str(x) for x in layers],
        "--freeze-backbone",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--crop-size",
        str(args.crop_size),
        "--stride",
        str(args.stride),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--amp-dtype",
        args.amp_dtype,
        "--feature-size",
        str(args.feature_size),
        "--embed-proj",
        str(args.embed_proj),
        "--fusion-mode",
        args.fusion_mode,
        "--num-workers",
        str(args.num_workers),
        "--eval-every",
        str(args.eval_every),
        "--seed",
        str(args.seed),
        "--aug",
        args.aug,
        "--mosaic-prob",
        str(args.mosaic_prob),
        "--fg-thresh",
        str(args.fg_thresh),
        "--energy-thresh",
        str(args.energy_thresh),
        "--skip-test-eval",
    ]
    if args.max_eval_images is not None:
        cmd += ["--max-eval-images", str(args.max_eval_images)]
    if args.max_train_batches is not None:
        cmd += ["--max-train-batches", str(args.max_train_batches)]
    return cmd


def metric_value(metrics: Dict[str, object], key: str) -> float:
    try:
        return float(metrics.get(key, float("nan")))
    except Exception:
        return float("nan")


def record_result(args, dataset: str, combo_name: str, layers: List[int], out_dir: Path, log_path: Path, pid: int, code: int, wall: float) -> None:
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
        "combo": combo_name,
        "layers": " ".join(str(x) for x in layers),
        "split": "val",
        "primary_metric": primary,
        "primary_value": metric_value(metrics, primary),
        "epochs": args.epochs,
        "max_train_batches": args.max_train_batches if args.max_train_batches is not None else "",
        "max_eval_images": args.max_eval_images if args.max_eval_images is not None else "",
        "crop_size": args.crop_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "amp_dtype": args.amp_dtype,
        "feature_size": args.feature_size,
        "embed_proj": args.embed_proj,
        "fusion_mode": args.fusion_mode,
        "seed": args.seed,
        "pid": pid,
        "gpu": args.gpu,
        "exit_code": code,
        "wall_seconds": wall,
        "results_json": str(results_path),
        "log_path": str(log_path),
    }
    for key in METRIC_COLUMNS:
        row[key] = metric_value(metrics, key)
    fields = list(row.keys())
    append_csv(OUT_ROOT / "layer_fusion_results.csv", row, fields)
    append_csv(
        OUT_ROOT / "metric_improvement_experiment_log.csv",
        {
            "timestamp": row["timestamp"],
            "dataset": dataset,
            "stage": "stage3",
            "experiment": combo_name,
            "changed_variable": "dino_feature_taps",
            "config": json.dumps(
                {
                    "layers": layers,
                    "epochs": args.epochs,
                    "max_train_batches": args.max_train_batches,
                    "max_eval_images": args.max_eval_images,
                    "freeze_backbone": True,
                    "decoder": "unchanged",
                    "fusion_mode": args.fusion_mode,
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
            "timestamp",
            "dataset",
            "stage",
            "experiment",
            "changed_variable",
            "config",
            "checkpoint",
            "split",
            "n_images",
            "primary_metric",
            "primary_value",
            "wall_seconds",
            "peak_cuda_gib",
            "gpu",
            "pid",
            "exit_code",
            "log_path",
        ],
    )


def summarize() -> None:
    csv_path = OUT_ROOT / "layer_fusion_results.csv"
    out = OUT_ROOT / "layer_fusion_analysis.md"
    lines = ["# Layer Fusion Analysis", ""]
    if not csv_path.exists():
        lines.append("No layer-fusion experiments have completed yet.")
        out.write_text("\n".join(lines) + "\n")
        return
    rows = list(csv.DictReader(csv_path.open(newline="")))
    lines += [
        "Current runs keep loss, split, crop, stride and training schedule fixed. Rows labelled bucket_concat change only DINO feature taps; weighted_sum rows change only the feature fusion operator for the same taps.",
        "",
        "| dataset | combo | layers | fusion | primary | value | epochs | exit | results |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('dataset','')} | {row.get('combo','')} | {row.get('layers','')} | "
            f"{row.get('fusion_mode') or 'bucket_concat'} | "
            f"{row.get('primary_metric','')} | {metric_value(row, 'primary_value'):.4f} | "
            f"{row.get('epochs','')} | {row.get('exit_code','')} | `{row.get('results_json','')}` |"
        )
    out.write_text("\n".join(lines) + "\n")


def run(args) -> int:
    combos = parse_combos(args.combos, args.depth)
    out_root = OUT_ROOT / "layer_fusion_runs" / args.run_name
    log_root = OUT_ROOT / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(),
        "datasets": args.datasets,
        "combos": combos,
        "args": vars(args),
        "note": "Existing artifacts are ViT-L/16 unless a vit_7b checkpoint/config is supplied.",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    failures = 0
    for dataset in args.datasets:
        for combo_name, layers in combos.items():
            out_dir = out_root / dataset / combo_name
            results_path = out_dir / "results.json"
            if args.skip_completed and results_path.exists():
                record_result(args, dataset, combo_name, layers, out_dir, out_dir / "skipped_existing.log", os.getpid(), 0, 0.0)
                continue
            cmd = train_command(args, dataset, combo_name, layers, out_dir)
            log_path = log_root / f"stage3_{args.run_name}_{dataset}_{combo_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            start = time.time()
            with log_path.open("w") as log:
                log.write(f"$ {shlex.join(cmd)}\n")
                log.flush()
                proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
                pid = proc.pid
                code = proc.wait()
            wall = time.time() - start
            record_result(args, dataset, combo_name, layers, out_dir, log_path, pid, code, wall)
            summarize()
            if code != 0:
                failures += 1
                if not args.continue_on_error:
                    return code
    return 1 if failures else 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["monuseg", "cellpose", "pannuke"])
    p.add_argument("--combos", nargs="+", default=None, help="Names or name:comma_layers. Default screens final/single/last4/even4.")
    p.add_argument("--depth", type=int, default=24, choices=[24, 40])
    p.add_argument("--run-name", default="screen_vitl16_frozen")
    p.add_argument("--checkpoint-iter", default="15374")
    p.add_argument("--train-config", default=str(DEFAULT_TRAIN_CONFIG))
    p.add_argument("--gpu", default="0")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--amp-dtype", choices=["none", "bf16", "fp16"], default="fp16")
    p.add_argument("--feature-size", type=int, default=32)
    p.add_argument("--embed-proj", type=int, default=384)
    p.add_argument("--fusion-mode", choices=["bucket_concat", "weighted_sum"], default="bucket_concat")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aug", choices=["none", "strong"], default="strong")
    p.add_argument("--mosaic-prob", type=float, default=0.3)
    p.add_argument("--fg-thresh", type=float, default=0.5)
    p.add_argument("--energy-thresh", type=float, default=0.4)
    p.add_argument("--max-eval-images", type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
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
