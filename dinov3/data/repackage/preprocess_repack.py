#!/usr/bin/env python3
"""CLI entry-point for the offline repackage pipeline.

Can be invoked in two ways::

    # As a package module (from dinov3/ directory)
    python -m data.repackage.preprocess_repack --help

    # As a standalone script (from anywhere)
    python data/repackage/preprocess_repack.py --help
"""

import argparse
import sys
from pathlib import Path


def _ensure_package_importable() -> None:
    """Make relative imports work when running this file directly."""
    pkg_dir = Path(__file__).resolve().parent
    if __package__ is None or __package__ == "":
        # Running as a script — bootstrap the package
        import importlib
        import types

        data_dir = pkg_dir.parent
        project_root = data_dir.parent

        # Create minimal package stubs so relative imports resolve
        if "data" not in sys.modules:
            data_mod = types.ModuleType("data")
            data_mod.__path__ = [str(data_dir)]
            sys.modules["data"] = data_mod
        if "data.repackage" not in sys.modules:
            rpkg = types.ModuleType("data.repackage")
            rpkg.__path__ = [str(pkg_dir)]
            sys.modules["data.repackage"] = rpkg

        # Ensure the repackage directory itself is on sys.path
        pkg_str = str(pkg_dir)
        if pkg_str not in sys.path:
            sys.path.insert(0, pkg_str)

        # Now set __package__ so relative imports work in this module
        globals()["__package__"] = "data.repackage"


_ensure_package_importable()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Repackage per-channel WebDataset shards into packed multi-channel shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- paths ----
    p.add_argument(
        "--input-root",
        type=Path,
        default=Path("/mnt/huawei_deepcad/webds_micro_100k_by_channel"),
        help="Root directory containing ch1/ … ch8/ subdirectories.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/huawei_deepcad/wds_packed_shards"),
        help="Where to write output tar shards.",
    )

    # ---- tiling ----
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--target-stride", type=int, default=384)
    p.add_argument("--small-image-threshold", type=int, default=900)

    # ---- filtering ----
    p.add_argument(
        "--reference-channel",
        type=int,
        default=1,
        help="1-indexed channel for variance filtering.",
    )
    p.add_argument(
        "--variance-threshold",
        type=float,
        default=20.0,
        help="Minimum pixel variance on the reference channel.",
    )
    p.add_argument(
        "--missing-ref-policy",
        choices=["fallback_first_available", "skip_sample"],
        default="fallback_first_available",
    )

    # ---- shuffle ----
    p.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=10_000,
        help="Max samples in the in-memory shuffle buffer.",
    )

    # ---- writing ----
    p.add_argument("--shard-prefix", default="filtered_mixed_train")
    p.add_argument("--max-shard-count", type=int, default=10_000)
    p.add_argument(
        "--max-shard-size",
        type=int,
        default=3 * 1024 * 1024 * 1024,
        help="Max bytes per output tar shard.",
    )

    # ---- parallelism ----
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes. "
            "Bottleneck is TIFF decode (CPU), so scaling is near-linear. "
            "Recommend 8–16 on a 40-core machine. "
            "Total RAM ≈ num_workers × (shuffle_buffer_size/num_workers) × patch_bytes."
        ),
    )

    # ---- misc ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    p.add_argument(
        "--channel-dirs",
        nargs="*",
        default=[],
        help="Explicit ch* dirs to scan (empty = auto-detect).",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from .config import RepackConfig
    from .pipeline import run_pipeline
    from .utils import setup_logging

    setup_logging(args.log_level)

    cfg = RepackConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        patch_size=args.patch_size,
        target_stride=args.target_stride,
        small_image_threshold=args.small_image_threshold,
        reference_channel=args.reference_channel,
        variance_threshold=args.variance_threshold,
        missing_ref_policy=args.missing_ref_policy,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_shard_count=args.max_shard_count,
        max_shard_size=args.max_shard_size,
        shard_prefix=args.shard_prefix,
        seed=args.seed,
        channel_dirs=args.channel_dirs or [],
        num_workers=args.num_workers,
    )

    stats = run_pipeline(cfg)
    print(stats.summary())


if __name__ == "__main__":
    main()
