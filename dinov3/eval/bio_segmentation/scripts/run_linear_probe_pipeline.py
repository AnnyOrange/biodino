"""
One-command pipeline for bio-segmentation linear probe:
optional dataset extraction -> feature extraction -> cached linear probe.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bio_seg.run_linear_probe_pipeline")


SUPPORTED_DATASETS = (
    "bbbc038",
    "conic",
    "livecell",
    "monuseg",
    "pannuke",
    "tissuenet",
)

DEFAULT_IMG_SIZE_BY_DATASET = {
    "bbbc038": 512,
    "conic": 256,
    "livecell": 512,
    "monuseg": 512,
    "pannuke": 256,
    "tissuenet": 256,
}


def _run_cmd(cmd: List[str], env: Dict[str, str], dry_run: bool) -> None:
    logger.info("$ %s", shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def _expand_ckpt_tokens(tokens: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for token in tokens:
        for piece in token.split(","):
            piece = piece.strip()
            if piece:
                expanded.append(piece)
    return expanded


def _discover_checkpoints(checkpoints_dir: Path) -> Dict[int, Path]:
    found: Dict[int, Path] = {}
    for child in checkpoints_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        ckpt_file = child / "checkpoint.pth"
        if ckpt_file.is_file():
            found[int(child.name)] = ckpt_file
    return dict(sorted(found.items()))


def _select_checkpoint_iters(
    requested: Sequence[str],
    discovered: Dict[int, Path],
) -> List[int]:
    if not discovered:
        raise ValueError("No valid checkpoint found (expect <iter>/checkpoint.pth).")

    tokens = [t.lower() for t in _expand_ckpt_tokens(requested)]
    if not tokens:
        tokens = ["latest"]

    if "all" in tokens:
        return list(discovered.keys())

    latest = max(discovered.keys())
    selected: List[int] = []
    for token in tokens:
        if token == "latest":
            selected.append(latest)
            continue
        if not token.isdigit():
            raise ValueError(
                f"Unsupported checkpoint token '{token}'. Use latest / all / iter ids."
            )
        selected.append(int(token))

    missing = [iter_id for iter_id in selected if iter_id not in discovered]
    if missing:
        raise ValueError(
            f"Requested ckpt iters not found: {missing}. "
            f"Available: {list(discovered.keys())}"
        )

    # Keep deterministic order and remove duplicates.
    return sorted(set(selected))


def _resolve_data_root(data_root_base: Path, dataset: str) -> Path:
    if dataset == "livecell":
        return data_root_base / "LIVECell"
    return data_root_base / dataset / "extracted"


def _resolve_img_size(dataset: str, img_size: int) -> int:
    if img_size > 0:
        rounded = (img_size // 16) * 16
        if rounded <= 0:
            raise ValueError(f"--feature-img-size={img_size} is too small.")
        return rounded
    return DEFAULT_IMG_SIZE_BY_DATASET.get(dataset, 512)


def _resolve_layers_tag(layers: Sequence[int] | None) -> str:
    if not layers:
        return "last1"
    return "custom_" + "_".join(str(x) for x in layers)


def _check_datasets(datasets: Sequence[str]) -> None:
    bad = [d for d in datasets if d not in SUPPORTED_DATASETS]
    if bad:
        raise ValueError(f"Unsupported datasets: {bad}. Choices: {SUPPORTED_DATASETS}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command bio-seg linear probe pipeline "
                    "(extract optional + feature cache + linear probe)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bbbc038", "conic", "monuseg", "pannuke", "tissuenet"],
        choices=list(SUPPORTED_DATASETS),
        help="Datasets to run (default: bbbc038 conic monuseg pannuke tissuenet)",
    )
    parser.add_argument(
        "--checkpoints-dir",
        required=True,
        help="Checkpoint root, expected layout: <dir>/<iter>/checkpoint.pth",
    )
    parser.add_argument(
        "--checkpoint-iters",
        nargs="+",
        default=["latest"],
        help="Which checkpoint iters to run: latest / all / explicit ids (supports commas).",
    )
    parser.add_argument(
        "--train-config",
        required=True,
        help="Train YAML used to build eval backbone (must match checkpoint architecture).",
    )
    parser.add_argument(
        "--data-root-base",
        default="/mnt/huawei_deepcad/benchmark/segmentation",
        help="Dataset base root. Non-livecell uses <base>/<dataset>/extracted; "
             "livecell uses <base>/LIVECell",
    )

    # Optional extraction step
    parser.add_argument(
        "--extract-src-dir",
        default=None,
        help="If set, run extract_datasets before training (single source root).",
    )
    parser.add_argument(
        "--extract-dst-dir",
        default=None,
        help="Extraction destination root (default: --data-root-base).",
    )
    parser.add_argument(
        "--overwrite-extract",
        action="store_true",
        help="Pass --overwrite to extract_datasets.",
    )

    # Feature extraction settings
    parser.add_argument("--skip-feature-extraction", action="store_true")
    parser.add_argument("--feature-img-size", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--feature-num-workers", type=int, default=4)

    # Linear probe settings
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-num-workers", type=int, default=4)

    # Runtime/output
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--cache-root", default="./cache/linear_probe_pipeline")
    parser.add_argument("--output-root", default="./outputs/linear_probe_pipeline")
    parser.add_argument("--run-name", default=None, help="Subdir tag (default: train-config stem)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _check_datasets(args.datasets)

    checkpoints_dir = Path(args.checkpoints_dir).expanduser().resolve()
    if not checkpoints_dir.is_dir():
        parser.error(f"--checkpoints-dir is not a directory: {checkpoints_dir}")

    train_config = Path(args.train_config).expanduser().resolve()
    if not train_config.is_file():
        parser.error(f"--train-config not found: {train_config}")
    cfg_stem = train_config.stem
    run_name = args.run_name or cfg_stem
    layers_tag = _resolve_layers_tag(args.layers)

    discovered = _discover_checkpoints(checkpoints_dir)
    try:
        selected_iters = _select_checkpoint_iters(args.checkpoint_iters, discovered)
    except ValueError as err:
        parser.error(str(err))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    logger.info("Datasets: %s", args.datasets)
    logger.info("Checkpoints dir: %s", checkpoints_dir)
    logger.info("Selected ckpt iters: %s", selected_iters)
    logger.info("Train config: %s", train_config)
    logger.info("run_name=%s layers_tag=%s", run_name, layers_tag)

    if args.extract_src_dir:
        extract_dst = args.extract_dst_dir or args.data_root_base
        extract_cmd = [
            sys.executable,
            "-m",
            "dinov3.eval.bio_segmentation.scripts.extract_datasets",
            "--src-dir",
            str(Path(args.extract_src_dir).expanduser().resolve()),
            "--dst-dir",
            str(Path(extract_dst).expanduser().resolve()),
            "--datasets",
            *args.datasets,
        ]
        if args.overwrite_extract:
            extract_cmd.append("--overwrite")
        _run_cmd(extract_cmd, env=env, dry_run=args.dry_run)

    data_root_base = Path(args.data_root_base).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    for iter_id in selected_iters:
        ckpt_path = discovered[iter_id]
        for dataset in args.datasets:
            data_root = _resolve_data_root(data_root_base, dataset)
            if not args.dry_run and not data_root.exists():
                raise FileNotFoundError(
                    f"Dataset root not found for {dataset}: {data_root}\n"
                    f"Set --data-root-base correctly or use --extract-src-dir first."
                )

            cache_dir = cache_root / run_name / dataset / str(iter_id)
            output_dir = output_root / run_name / dataset / str(iter_id)
            cache_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            if not args.skip_feature_extraction:
                for split in ("train", "val", "test"):
                    feature_cmd = [
                        sys.executable,
                        "-m",
                        "dinov3.eval.bio_segmentation.feature_extractor",
                        "--dataset",
                        dataset,
                        "--data-root",
                        str(data_root),
                        "--checkpoint",
                        str(ckpt_path),
                        "--train-config",
                        str(train_config),
                        "--output-dir",
                        str(cache_dir),
                        "--split",
                        split,
                        "--img-size",
                        str(args.feature_img_size),
                        "--batch-size",
                        str(args.feature_batch_size),
                        "--num-workers",
                        str(args.feature_num_workers),
                    ]
                    if args.layers:
                        feature_cmd.extend(["--layers", *[str(x) for x in args.layers]])
                    _run_cmd(feature_cmd, env=env, dry_run=args.dry_run)

            img_size = _resolve_img_size(dataset, args.feature_img_size)
            train_cache = cache_dir / f"{dataset}_train_{cfg_stem}_{layers_tag}_s{img_size}.npz"
            val_cache = cache_dir / f"{dataset}_val_{cfg_stem}_{layers_tag}_s{img_size}.npz"
            test_cache = cache_dir / f"{dataset}_test_{cfg_stem}_{layers_tag}_s{img_size}.npz"

            if not args.dry_run:
                missing_cache = [str(p) for p in (train_cache, val_cache, test_cache) if not p.is_file()]
                if missing_cache:
                    raise FileNotFoundError(
                        "Expected cache file(s) not found:\n"
                        + "\n".join(missing_cache)
                    )

            probe_cmd = [
                sys.executable,
                "-m",
                "dinov3.eval.bio_segmentation.linear_probe",
                "--dataset",
                dataset,
                "--use-cached-features",
                "--train-cache",
                str(train_cache),
                "--val-cache",
                str(val_cache),
                "--test-cache",
                str(test_cache),
                "--output-dir",
                str(output_dir),
                "--epochs",
                str(args.probe_epochs),
                "--batch-size",
                str(args.probe_batch_size),
                "--lr",
                str(args.probe_lr),
                "--weight-decay",
                str(args.probe_weight_decay),
                "--num-workers",
                str(args.probe_num_workers),
            ]
            _run_cmd(probe_cmd, env=env, dry_run=args.dry_run)

    logger.info("Pipeline done.")


if __name__ == "__main__":
    main()
