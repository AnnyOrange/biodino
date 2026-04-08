"""Repackage pipeline configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal


@dataclass
class RepackConfig:
    """All tunables for the offline repackage pipeline.

    Attributes:
        input_root: Root directory containing ch1/ … ch8/ subdirectories.
        output_root: Directory where output tar shards are written.
        patch_size: Target patch side length for tiling.
        target_stride: Desired stride between patches (actual stride is
            dynamically adjusted per image).
        small_image_threshold: Images with *both* H and W <= this value
            are kept whole (no tiling).
        reference_channel: 1-indexed channel used for variance filtering.
        variance_threshold: Minimum pixel variance on the reference channel;
            patches below this are discarded as background.
        missing_ref_policy: What to do when the reference channel is absent.
        shuffle_buffer_size: Max samples held in the in-memory shuffle buffer.
        max_shard_count: Max samples per output tar shard.
        max_shard_size: Max bytes per output tar shard.
        shard_prefix: Filename prefix for output shards.
        seed: Random seed for reproducibility.
        channel_dirs: Which ch* subdirectories to scan (auto-detected if empty).
    """

    input_root: Path = Path("/mnt/huawei_deepcad/webds_micro_100k_by_channel")
    output_root: Path = Path("/mnt/huawei_deepcad/wds_packed_shards")

    # --- tiling ---
    patch_size: int = 512
    target_stride: int = 384
    small_image_threshold: int = 900

    # --- filtering ---
    reference_channel: int = 1
    variance_threshold: float = 20.0
    missing_ref_policy: Literal["fallback_first_available", "skip_sample"] = (
        "fallback_first_available"
    )

    # --- shuffle ---
    shuffle_buffer_size: int = 10_000

    # --- writing ---
    max_shard_count: int = 10_000
    max_shard_size: int = 3 * 1024 * 1024 * 1024  # 3 GB
    shard_prefix: str = "filtered_mixed_train"

    # --- misc ---
    seed: int = 42
    channel_dirs: List[str] = field(default_factory=list)
    num_workers: int = 1
