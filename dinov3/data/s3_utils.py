# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
S3 data source utilities for WebDataset integration.

Provides two cold-path helpers called once at dataset init:
- resolve_s3_stream_urls: expand shard pattern → pipe:aws s3 cp ... URLs
- sync_s3_to_cache: download shards to local cache, return local paths

Neither function is called on the training hot path.
"""

import logging
import os
import re
import subprocess
from typing import List, Optional

logger = logging.getLogger("dinov3")


def expand_braces(pattern: str) -> List[str]:
    """Expand brace expressions like 'path/shard-{0000..0099}.tar' into a URL list.

    Prefers the ``braceexpand`` package (pip install braceexpand) for full
    bash-style expansion.  Falls back to a simple {start..end} regex parser.
    """
    try:
        import braceexpand
        return list(braceexpand.braceexpand(pattern))
    except ImportError:
        pass

    match = re.search(r"\{(\d+)\.\.(\d+)\}", pattern)
    if not match:
        return [pattern]

    start_str, end_str = match.group(1), match.group(2)
    width = len(start_str)
    start_val, end_val = int(start_str), int(end_str)
    prefix, suffix = pattern[: match.start()], pattern[match.end() :]

    return [f"{prefix}{str(i).zfill(width)}{suffix}" for i in range(start_val, end_val + 1)]


def resolve_s3_stream_urls(
    s3_shard_pattern: str,
    profile: Optional[str] = None,
    region: Optional[str] = None,
) -> List[str]:
    """Expand an S3 shard pattern and convert to ``pipe:`` URLs for streaming.

    Each URL becomes ``pipe:aws s3 cp [--profile X] [--region Y] <s3_url> -``
    which WebDataset opens as a subprocess per shard.  This is efficient for
    large shards (2–3 GB each) and requires no extra Python dependencies.

    Args:
        s3_shard_pattern: e.g. ``s3://bucket/dir/shard-{0000..0099}.tar``
        profile: AWS CLI profile name (optional).
        region: AWS region (optional).

    Returns:
        List of ``pipe:...`` URLs ready for ``wds.SimpleShardList``.
    """
    s3_urls = expand_braces(s3_shard_pattern)

    base_cmd = "aws s3 cp"
    if profile:
        base_cmd += f" --profile {profile}"
    if region:
        base_cmd += f" --region {region}"

    pipe_urls = [f"pipe:{base_cmd} {url} -" for url in s3_urls]
    logger.info(f"S3 stream: resolved {len(pipe_urls)} shard pipe URLs")
    return pipe_urls


def sync_s3_to_cache(
    s3_shard_pattern: str,
    cache_root: str,
    profile: Optional[str] = None,
    region: Optional[str] = None,
) -> List[str]:
    """Download S3 shards to a local cache directory.

    Shards already present on disk are skipped.  Downloads go to a temporary
    ``.downloading`` file and are atomically renamed on completion, so
    concurrent processes on the same node will not corrupt each other.

    Args:
        s3_shard_pattern: S3 brace pattern, e.g. ``s3://bucket/ch1/train-{000000..000099}.tar``
        cache_root: Local directory to store cached shards.
        profile: AWS CLI profile (optional).
        region: AWS region (optional).

    Returns:
        List of local file paths (one per shard), in the same order as the
        expanded pattern.  These can be passed directly to
        ``wds.SimpleShardList``.
    """
    s3_urls = expand_braces(s3_shard_pattern)
    if not s3_urls:
        raise ValueError(f"No shards resolved from pattern: {s3_shard_pattern}")

    prefix = _common_s3_prefix(s3_urls)
    os.makedirs(cache_root, exist_ok=True)

    local_paths: List[str] = []
    downloaded, skipped = 0, 0

    for s3_url in s3_urls:
        rel = s3_url[len(prefix) :]
        local_path = os.path.join(cache_root, rel)
        local_paths.append(local_path)

        if os.path.exists(local_path):
            skipped += 1
            continue

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        tmp_path = local_path + ".downloading"

        cmd = ["aws", "s3", "cp"]
        if profile:
            cmd.extend(["--profile", profile])
        if region:
            cmd.extend(["--region", region])
        cmd.extend([s3_url, tmp_path])

        logger.info(f"[{downloaded + skipped + 1}/{len(s3_urls)}] downloading: {rel}")
        subprocess.run(cmd, check=True)
        os.rename(tmp_path, local_path)
        downloaded += 1

    logger.info(
        f"S3 cache sync complete: {downloaded} downloaded, {skipped} cached, "
        f"{len(local_paths)} total in {cache_root}"
    )
    return local_paths


def _common_s3_prefix(urls: List[str]) -> str:
    """Find the longest common S3 directory prefix among *urls*."""
    if len(urls) == 1:
        return urls[0].rsplit("/", 1)[0] + "/"
    prefix = os.path.commonprefix(urls)
    if not prefix.endswith("/"):
        prefix = prefix.rsplit("/", 1)[0] + "/"
    return prefix
