#!/usr/bin/env python3
"""Run the approved E20/E50 frozen dense-probe matrix for one DINO encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PIPELINE = "dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline"
DATASET_PROTOCOL = {
    "bbbc038": (512, "pad", "none"),
    "cellpose": (512, "pad", "none"),
    "conic": (256, "stretch", "sqrt_inverse"),
    "livecell": (512, "pad", "none"),
    "monuseg": (768, "pad", "none"),
    "multimodal_cellseg": (512, "pad", "none"),
    "pannuke": (256, "stretch", "none"),
    "tissuenet": (256, "stretch", "none"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_payload(checkpoint_root: Path, checkpoint_id: int) -> Path:
    directory = checkpoint_root / str(checkpoint_id)
    pth = directory / "checkpoint.pth"
    if pth.is_file():
        return pth
    if (directory / ".metadata").is_file():
        return directory
    raise FileNotFoundError(f"No checkpoint.pth or DCP metadata under {directory}")


def _data_root(base: Path, dataset: str) -> Path:
    if dataset == "livecell":
        return base / "LIVECell"
    if dataset == "cellpose":
        for candidate in (base / "Cellpose", base / "cellpose", base / "cellpose" / "extracted"):
            if candidate.exists():
                return candidate
        return base / "Cellpose"
    if dataset == "multimodal_cellseg":
        return base / "Multimodal_CellSeg" / "neurips22_cellseg"
    return base / dataset / "extracted"


def _completed_results(output_root: Path, run_name: str, dataset: str, checkpoint_id: int) -> list[Path]:
    paths = sorted(
        output_root.glob(
            f"{run_name}*/budget*/seed*/{dataset}/{checkpoint_id}/results.json"
        )
    )
    expected = 18 if dataset == "pannuke" else 6
    if len(paths) != expected:
        raise RuntimeError(
            f"Expected {expected} completed results for {dataset}, found {len(paths)}"
        )
    for path in paths:
        result = json.loads(path.read_text())
        meta = result.get("_meta", {})
        budget = int(path.parts[-5].removeprefix("budget"))
        seed = int(path.parts[-4].removeprefix("seed"))
        history = meta.get("validation_history", [])
        if not (
            int(meta.get("probe_epochs", -1)) == budget
            and int(meta.get("probe_eval_every", -1)) == 1
            and int(meta.get("seed", -1)) == seed
            and len(history) == budget
            and int(meta.get("test_evaluations", -1)) == 1
            and 1 <= int(meta.get("best_epoch", -1)) <= budget
        ):
            raise RuntimeError(f"Incomplete or mismatched result metadata: {path}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-id", type=int, required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--data-root-base", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--feature-num-workers", type=int, default=2)
    parser.add_argument("--probe-num-workers", type=int, default=2)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_PROTOCOL), required=True)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not re.fullmatch(r"[0-9a-f]{64}", args.checkpoint_sha256):
        parser.error("formal runs require a lowercase 64-character --checkpoint-sha256")

    checkpoint_root = Path(args.checkpoint_root).resolve()
    checkpoint_payload = _checkpoint_payload(checkpoint_root, args.checkpoint_id)
    train_config = Path(args.train_config).resolve()
    data_root_base = Path(args.data_root_base).resolve()
    if not train_config.is_file():
        raise FileNotFoundError(train_config)
    for dataset in args.datasets:
        root = _data_root(data_root_base, dataset)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root missing for {dataset}: {root}")

    output_root = Path(args.output_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    manifest_dir = output_root / "manifests" / args.model_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    group_tag = "_".join(args.datasets)
    manifest_path = manifest_dir / f"gpu{args.gpu}_{group_tag}.json"
    checkpoint_stat = checkpoint_payload.stat()
    code_files = [
        Path(__file__).resolve(),
        REPO / "dinov3/eval/bio_segmentation/linear_probe.py",
        REPO / "dinov3/eval/bio_segmentation/feature_extractor.py",
        REPO / "dinov3/eval/bio_segmentation/scripts/run_linear_probe_pipeline.py",
    ]
    manifest = {
        "status": "PLANNED" if args.dry_run else "RUNNING",
        "protocol_id": "seg-probe-budget-fairness-v1",
        "started_at": _utc_now(),
        "host": os.uname().nodename,
        "gpu": args.gpu,
        "model": args.model_name,
        "checkpoint_id": args.checkpoint_id,
        "checkpoint_payload": str(checkpoint_payload),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "checkpoint_sha256": args.checkpoint_sha256 or "NOT_COMPUTED",
        "train_config": str(train_config),
        "train_config_sha256": _sha256(train_config),
        "data_root_base": str(data_root_base),
        "datasets": args.datasets,
        "feature_protocol": "last1",
        "budgets": [20, 50],
        "probe_seeds": [0, 1, 2],
        "probe_eval_every": 1,
        "probe_batch_size": 32,
        "feature_batch_size": args.feature_batch_size,
        "code_sha256": {str(path.relative_to(REPO)): _sha256(path) for path in code_files},
        "jobs": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    run_name = f"{args.model_name}_last1_budget20_50_bestval"
    for dataset in args.datasets:
        img_size, resize_mode, class_weight = DATASET_PROTOCOL[dataset]
        cmd = [
            sys.executable,
            "-m",
            PIPELINE,
            "--datasets",
            dataset,
            "--checkpoints-dir",
            str(checkpoint_root),
            "--checkpoint-iters",
            str(args.checkpoint_id),
            "--train-config",
            str(train_config),
            "--data-root-base",
            str(data_root_base),
            "--protocol",
            "manual",
            "--dataset-split-protocol",
            "formal-v1",
            "--feature-img-size",
            str(img_size),
            "--resize-mode",
            resize_mode,
            "--layer-preset",
            "last1",
            "--feature-batch-size",
            str(args.feature_batch_size),
            "--feature-num-workers",
            str(args.feature_num_workers),
            "--probe-epoch-grid",
            "20",
            "50",
            "--probe-seeds",
            "0",
            "1",
            "2",
            "--probe-eval-every",
            "1",
            "--probe-batch-size",
            "32",
            "--probe-num-workers",
            str(args.probe_num_workers),
            "--probe-lr",
            "0.001",
            "--probe-weight-decay",
            "0.0001",
            "--probe-class-weight-mode",
            class_weight,
            "--channel-policy",
            "auto",
            "--chunked-cache",
            "--no-compress-cache",
            "--gpu",
            args.gpu,
            "--cache-root",
            str(cache_root),
            "--output-root",
            str(output_root),
            "--run-name",
            run_name,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        job = {"dataset": dataset, "status": "RUNNING", "cmd": cmd, "started_at": _utc_now()}
        manifest["jobs"].append(job)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        try:
            if not args.dry_run:
                try:
                    existing_results = _completed_results(
                        output_root, run_name, dataset, args.checkpoint_id
                    )
                except RuntimeError:
                    existing_results = []
                if existing_results:
                    job.update(
                        {
                            "status": "COMPLETE",
                            "finished_at": _utc_now(),
                            "validated_results": [str(path) for path in existing_results],
                            "deleted_feature_cache_dirs": [],
                            "reused_complete_results": True,
                        }
                    )
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    continue
            subprocess.run(cmd, check=True, cwd=REPO)
            if args.dry_run:
                result_paths = []
                deleted_cache_dirs = []
            else:
                result_paths = _completed_results(
                    output_root, run_name, dataset, args.checkpoint_id
                )
                cache_dirs = sorted(
                    path
                    for path in cache_root.glob(f"{run_name}*/{dataset}/{args.checkpoint_id}")
                    if path.is_dir()
                )
                for cache_dir in cache_dirs:
                    shutil.rmtree(cache_dir)
                deleted_cache_dirs = [str(path) for path in cache_dirs]
            job.update(
                {
                    "status": "COMPLETE",
                    "finished_at": _utc_now(),
                    "validated_results": [str(path) for path in result_paths],
                    "deleted_feature_cache_dirs": deleted_cache_dirs,
                }
            )
        except Exception as error:
            job.update({"status": "FAILED", "finished_at": _utc_now(), "error": f"{type(error).__name__}: {error}"})
            manifest.update({"status": "FAILED", "finished_at": _utc_now()})
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            raise
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    manifest.update({"status": "COMPLETE_VALIDATED", "finished_at": _utc_now()})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
