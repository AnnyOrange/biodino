"""
One-command pipeline for bio-segmentation linear probe:
optional dataset extraction -> feature extraction -> cached linear probe.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bio_seg.run_linear_probe_pipeline")


SUPPORTED_DATASETS = (
    "bbbc038",
    "cellpose",
    "conic",
    "livecell",
    "monuseg",
    "pannuke",
    "tissuenet",
)

DEFAULT_IMG_SIZE_BY_DATASET = {
    "bbbc038": 512,
    "cellpose": 512,
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
            continue
        # Some newer DINOv3 runs are saved as torch.distributed.checkpoint
        # directories.  The bio-segmentation loader can read these directories
        # directly, so pass the checkpoint directory through as the checkpoint.
        if (child / ".metadata").is_file():
            found[int(child.name)] = child
    return dict(sorted(found.items()))


def _infer_checkpoint_id(checkpoint_file: Path) -> int:
    match = re.search(r"ep[=-]?(\d+)", checkpoint_file.stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    matches = re.findall(r"\d+", checkpoint_file.stem)
    return int(matches[-1]) if matches else 0


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
    if dataset == "cellpose":
        candidates = [
            data_root_base / "Cellpose",
            data_root_base / "cellpose",
            data_root_base / "cellpose" / "extracted",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
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


def _infer_arch_depth(train_config: Path) -> Tuple[str, int]:
    text = train_config.read_text(errors="ignore")
    match = re.search(r"^\s*arch:\s*([A-Za-z0-9_+-]+)\s*$", text, flags=re.MULTILINE)
    arch = match.group(1).strip() if match else "vit_base"
    depth_by_arch = {
        "vit_small": 12,
        "vit_base": 12,
        "vit_large": 24,
        "vit_7b": 40,
    }
    depth = depth_by_arch.get(arch)
    if depth is None:
        raise ValueError(
            f"Cannot infer layer preset depth for arch={arch!r}. "
            "Use explicit --layers instead."
        )
    return arch, depth


def _even4_layers(depth: int) -> List[int]:
    if depth == 12:
        return [2, 5, 8, 11]
    if depth == 24:
        return [4, 11, 17, 23]
    if depth == 40:
        return [9, 19, 29, 39]
    return [max(0, round((i + 1) * depth / 4) - 1) for i in range(4)]


def _last4_layers(depth: int) -> List[int]:
    return list(range(max(0, depth - 4), depth))


def _resolve_layer_jobs(
    *,
    explicit_layers: Optional[Sequence[int]],
    presets: Optional[Sequence[str]],
    train_config: Path,
) -> List[Tuple[Optional[List[int]], str]]:
    if explicit_layers is not None and presets:
        raise ValueError("Use either --layers or --layer-preset, not both.")

    if explicit_layers is not None:
        layers = [int(x) for x in explicit_layers]
        return [(layers, _resolve_layers_tag(layers))]

    if not presets:
        return [(None, "last1")]

    arch, depth = _infer_arch_depth(train_config)
    jobs: List[Tuple[Optional[List[int]], str]] = []
    for preset in presets:
        key = preset.lower().replace("-", "_")
        if key == "last1":
            jobs.append((None, "last1"))
        elif key in {"even4", "four_even", "multilayer", "multi_layer"}:
            layers = _even4_layers(depth)
            jobs.append((layers, _resolve_layers_tag(layers)))
        elif key == "last4":
            layers = _last4_layers(depth)
            jobs.append((layers, _resolve_layers_tag(layers)))
        elif key == "layerwise":
            jobs.extend(([i], _resolve_layers_tag([i])) for i in range(depth))
        else:
            raise ValueError(
                f"Unknown --layer-preset {preset!r}. "
                "Choices: last1, even4, last4, layerwise."
            )

    # Stable de-duplication lets callers combine presets such as "last1 last4".
    deduped: List[Tuple[Optional[List[int]], str]] = []
    seen = set()
    for layers, tag in jobs:
        sig = tuple(layers) if layers is not None else ()
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append((layers, tag))
    logger.info("Layer preset arch=%s depth=%d -> jobs=%s", arch, depth, [tag for _, tag in deduped])
    return deduped


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
        default=None,
        help="Checkpoint root, expected layout: <dir>/<iter>/checkpoint.pth",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Single checkpoint file to run, e.g. a .ckpt/.pth ChAda-ViT checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-id",
        type=int,
        default=None,
        help="Numeric output/cache id for --checkpoint-file (default: infer from filename, or 0).",
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
             "livecell uses <base>/LIVECell; cellpose uses <base>/Cellpose if present",
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
    parser.add_argument(
        "--layer-preset",
        nargs="+",
        default=None,
        help="Convenience layer jobs: last1, even4, last4, layerwise. "
             "Uses arch from --train-config; cannot be combined with --layers.",
    )
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--feature-num-workers", type=int, default=4)
    parser.add_argument(
        "--no-compress-cache",
        action="store_true",
        help="Pass --no-compress-cache to feature_extractor. The pipeline also "
             "auto-enables this for multi-layer jobs because compressed npz can "
             "look like a hang on very large feature tensors.",
    )
    parser.add_argument(
        "--chunked-cache",
        action="store_true",
        help="Pass --chunked-cache to feature_extractor. The pipeline also "
             "auto-enables this for multi-layer jobs to avoid a large final "
             "np.concatenate step.",
    )

    # Linear probe settings
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-num-workers", type=int, default=4)
    parser.add_argument("--probe-eval-every", type=int, default=5)
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--semantic-only", action="store_true")
    parser.add_argument(
        "--fast-eval",
        action="store_true",
        help="Fast screening mode: train/val only, 10 probe epochs, validate at the end, "
             "semantic metrics only, skip test eval. Use this to quickly decide whether a checkpoint/layer combo is worth full eval.",
    )

    # Runtime/output
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--cache-root", default="./cache/linear_probe_pipeline")
    parser.add_argument("--output-root", default="./outputs/linear_probe_pipeline")
    parser.add_argument("--run-name", default=None, help="Subdir tag (default: train-config stem)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _check_datasets(args.datasets)

    if bool(args.checkpoints_dir) == bool(args.checkpoint_file):
        parser.error("Set exactly one of --checkpoints-dir or --checkpoint-file.")

    train_config = Path(args.train_config).expanduser().resolve()
    if not train_config.is_file():
        parser.error(f"--train-config not found: {train_config}")
    cfg_stem = train_config.stem
    run_name = args.run_name or cfg_stem
    try:
        layer_jobs = _resolve_layer_jobs(
            explicit_layers=args.layers,
            presets=args.layer_preset,
            train_config=train_config,
        )
    except ValueError as err:
        parser.error(str(err))

    if args.checkpoint_file:
        checkpoint_file = Path(args.checkpoint_file).expanduser().resolve()
        if not checkpoint_file.is_file():
            parser.error(f"--checkpoint-file not found: {checkpoint_file}")
        checkpoint_id = args.checkpoint_id
        if checkpoint_id is None:
            checkpoint_id = _infer_checkpoint_id(checkpoint_file)
        discovered = {checkpoint_id: checkpoint_file}
        selected_iters = [checkpoint_id]
        checkpoints_label = str(checkpoint_file)
    else:
        checkpoints_dir = Path(args.checkpoints_dir).expanduser().resolve()
        if not checkpoints_dir.is_dir():
            parser.error(f"--checkpoints-dir is not a directory: {checkpoints_dir}")
        discovered = _discover_checkpoints(checkpoints_dir)
        try:
            selected_iters = _select_checkpoint_iters(args.checkpoint_iters, discovered)
        except ValueError as err:
            parser.error(str(err))
        checkpoints_label = str(checkpoints_dir)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    logger.info("Datasets: %s", args.datasets)
    logger.info("Checkpoint source: %s", checkpoints_label)
    logger.info("Selected ckpt iters: %s", selected_iters)
    logger.info("Train config: %s", train_config)
    logger.info("run_name=%s layer_jobs=%s", run_name, [tag for _, tag in layer_jobs])

    if args.fast_eval:
        args.probe_epochs = min(args.probe_epochs, 10)
        args.probe_eval_every = args.probe_epochs
        args.skip_test_eval = True
        args.semantic_only = True
        logger.info(
            "Fast eval enabled: probe_epochs=%d, train/val only, semantic metrics only, skip test eval.",
            args.probe_epochs,
        )

    auto_no_compress = any(layers is not None and len(layers) > 1 for layers, _ in layer_jobs)
    no_compress_cache = args.no_compress_cache or auto_no_compress
    auto_chunked_cache = any(layers is not None and len(layers) > 1 for layers, _ in layer_jobs)
    chunked_cache = args.chunked_cache or auto_chunked_cache
    if no_compress_cache:
        logger.info(
            "Feature cache compression disabled%s.",
            " automatically for multi-layer extraction" if auto_no_compress and not args.no_compress_cache else "",
        )
    if chunked_cache:
        logger.info(
            "Feature cache chunked saving enabled%s.",
            " automatically for multi-layer extraction" if auto_chunked_cache and not args.chunked_cache else "",
        )

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
        for layers, layers_tag in layer_jobs:
            effective_run_name = (
                run_name
                if len(layer_jobs) == 1 and layers_tag == "last1"
                else f"{run_name}__{layers_tag}"
            )
            for dataset in args.datasets:
                data_root = _resolve_data_root(data_root_base, dataset)
                if not args.dry_run and not data_root.exists():
                    raise FileNotFoundError(
                        f"Dataset root not found for {dataset}: {data_root}\n"
                        f"Set --data-root-base correctly or use --extract-src-dir first."
                    )

                cache_dir = cache_root / effective_run_name / dataset / str(iter_id)
                output_dir = output_root / effective_run_name / dataset / str(iter_id)
                cache_dir.mkdir(parents=True, exist_ok=True)
                output_dir.mkdir(parents=True, exist_ok=True)

                if not args.skip_feature_extraction:
                    splits = ("train", "val") if args.skip_test_eval else ("train", "val", "test")
                    for split in splits:
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
                        if layers:
                            feature_cmd.extend(["--layers", *[str(x) for x in layers]])
                        if no_compress_cache:
                            feature_cmd.append("--no-compress-cache")
                        if chunked_cache:
                            feature_cmd.append("--chunked-cache")
                        _run_cmd(feature_cmd, env=env, dry_run=args.dry_run)

                img_size = _resolve_img_size(dataset, args.feature_img_size)
                train_cache = cache_dir / f"{dataset}_train_{cfg_stem}_{layers_tag}_s{img_size}.npz"
                val_cache = cache_dir / f"{dataset}_val_{cfg_stem}_{layers_tag}_s{img_size}.npz"
                test_cache = cache_dir / f"{dataset}_test_{cfg_stem}_{layers_tag}_s{img_size}.npz"

                if not args.dry_run:
                    required_caches = (train_cache, val_cache) if args.skip_test_eval else (train_cache, val_cache, test_cache)
                    missing_cache = [str(p) for p in required_caches if not p.is_file()]
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
                    "--eval-every",
                    str(args.probe_eval_every),
                ]
                if not args.skip_test_eval:
                    probe_cmd.extend(["--test-cache", str(test_cache)])
                else:
                    probe_cmd.append("--skip-test-eval")
                if args.semantic_only:
                    probe_cmd.append("--semantic-only")
                _run_cmd(probe_cmd, env=env, dry_run=args.dry_run)

    logger.info("Pipeline done.")


if __name__ == "__main__":
    main()
