"""Core repackage pipeline: discover → stream → tile → filter → shuffle → write.

Supports single-process and multi-process modes.

Profiling shows the bottleneck is TIFF decode (~90% CPU), so
``num_workers > 1`` gives near-linear speedup.

Multi-process design:
  - Shard list is split into N interleaved chunks (round-robin, so each
    worker sees a balanced mix of channel types and shard sizes).
  - Each worker runs the full mini-pipeline independently and writes to
    its own set of output shards (prefix ``<shard_prefix>_w<NN>``).
  - Worker shuffle buffers are sized to ``shuffle_buffer_size // num_workers``
    to keep aggregate peak memory constant.
  - Stats from all workers are merged and returned.
"""

import logging
import random
import tarfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np

from .config import RepackConfig
from .filtering import passes_variance_filter
from .index_builder import TarInfo, discover_shards, extract_sample_id
from .io_utils import encode_tiff_uint16, ensure_channel_first, read_json_bytes, read_tiff_bytes
from .tiling import CropRegion, compute_crops
from .utils import PipelineStats, merge_stats as _merge_stats
from .writer import PackedShardWriter

logger = logging.getLogger("repackage.pipeline")


# ======================================================================== #
# Public entry point                                                        #
# ======================================================================== #

def run_pipeline(cfg: RepackConfig) -> PipelineStats:
    """Execute the full repackage pipeline.

    Delegates to ``_run_parallel`` when ``cfg.num_workers > 1``,
    otherwise runs single-process.
    """
    logger.info("Step 1 — Discovering shards …")
    shard_infos = discover_shards(cfg.input_root, cfg.channel_dirs or None)
    if not shard_infos:
        logger.error("No shards found under %s", cfg.input_root)
        return PipelineStats()

    rng = random.Random(cfg.seed)
    rng.shuffle(shard_infos)

    if cfg.num_workers > 1:
        return _run_parallel(cfg, shard_infos)
    return _run_serial(cfg, shard_infos, worker_id=None)


# ======================================================================== #
# Parallel dispatch                                                          #
# ======================================================================== #

def _run_parallel(cfg: RepackConfig, shard_infos: List[TarInfo]) -> PipelineStats:
    """Fork N worker processes; each handles an interleaved shard subset."""
    import concurrent.futures

    n = cfg.num_workers
    # Round-robin split: worker k gets shards at indices k, k+n, k+2n, …
    # This ensures each worker sees a balanced mix of channel types.
    chunks = [shard_infos[i::n] for i in range(n)]

    # Each worker gets proportionally smaller buffer to cap total RAM
    per_worker_buffer = max(100, cfg.shuffle_buffer_size // n)

    logger.info(
        "Parallel mode: %d workers, %d shards each (buf=%d/worker)",
        n,
        len(chunks[0]),
        per_worker_buffer,
    )

    worker_cfgs = []
    for worker_id in range(n):
        wcfg = RepackConfig(
            input_root=cfg.input_root,
            output_root=cfg.output_root,
            patch_size=cfg.patch_size,
            target_stride=cfg.target_stride,
            small_image_threshold=cfg.small_image_threshold,
            reference_channel=cfg.reference_channel,
            variance_threshold=cfg.variance_threshold,
            missing_ref_policy=cfg.missing_ref_policy,
            shuffle_buffer_size=per_worker_buffer,
            max_shard_count=cfg.max_shard_count,
            max_shard_size=cfg.max_shard_size,
            shard_prefix=f"{cfg.shard_prefix}_w{worker_id:02d}",
            seed=cfg.seed + worker_id,
            num_workers=1,
        )
        worker_cfgs.append(wcfg)

    all_stats: List[PipelineStats] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n) as pool:
        futures = {
            pool.submit(_worker_entry, wcfg, chunk): worker_id
            for worker_id, (wcfg, chunk) in enumerate(zip(worker_cfgs, chunks))
        }
        for future in concurrent.futures.as_completed(futures):
            wid = futures[future]
            try:
                stats = future.result()
                all_stats.append(stats)
                logger.info("Worker %d finished: %d patches kept", wid, stats.patches_kept)
            except Exception as exc:
                logger.error("Worker %d crashed: %s", wid, exc, exc_info=True)

    return _merge_stats(all_stats)


def _worker_entry(cfg: RepackConfig, shards: List[TarInfo]) -> PipelineStats:
    """Top-level function executed in each worker process (must be picklable)."""
    from .utils import setup_logging
    setup_logging("INFO")
    return _run_serial(cfg, shards, worker_id=None)


# ======================================================================== #
# Single-process pipeline                                                   #
# ======================================================================== #

def _run_serial(
    cfg: RepackConfig,
    shard_infos: List[TarInfo],
    worker_id: Optional[int],
) -> PipelineStats:
    """Process *shard_infos* sequentially in one process."""
    stats = PipelineStats()
    rng = random.Random(cfg.seed)

    writer = PackedShardWriter(
        output_dir=cfg.output_root,
        prefix=cfg.shard_prefix,
        max_count=cfg.max_shard_count,
        max_size=cfg.max_shard_size,
    )

    shuffle_buf: List[Dict[str, Any]] = []
    prefix = f"[w{worker_id}] " if worker_id is not None else ""

    try:
        for si_idx, shard in enumerate(shard_infos, 1):
            logger.info(
                "%sShard %d/%d  %s  (ch%d, ~%d samples)",
                prefix,
                si_idx,
                len(shard_infos),
                shard.tar_path.name,
                shard.channel_count,
                shard.approx_samples,
            )
            for sample_dict in _stream_tar(shard, cfg, stats):
                shuffle_buf.append(sample_dict)
                if len(shuffle_buf) >= cfg.shuffle_buffer_size:
                    _drain_buffer(shuffle_buf, writer, rng, stats)

            if si_idx % 20 == 0:
                logger.info(
                    "%s  Progress: %d/%d | written=%d | filtered=%d | errors=%d",
                    prefix,
                    si_idx,
                    len(shard_infos),
                    stats.patches_kept,
                    stats.patches_filtered,
                    stats.read_errors,
                )

        if shuffle_buf:
            rng.shuffle(shuffle_buf)
            for sample in shuffle_buf:
                writer.write(sample)
                stats.patches_kept += 1
            shuffle_buf.clear()

    finally:
        writer.close()

    logger.info(stats.summary())
    return stats


# ======================================================================== #
# Per-shard streaming                                                       #
# ======================================================================== #

def _stream_tar(
    shard: TarInfo,
    cfg: RepackConfig,
    stats: PipelineStats,
) -> Generator[Dict[str, Any], None, None]:
    """Open *shard* and yield packed sample dicts (no full-member scan)."""
    try:
        tf = tarfile.open(shard.tar_path, "r")
    except Exception as exc:
        logger.warning("Cannot open %s: %s", shard.tar_path, exc)
        stats.read_errors += 1
        return

    try:
        yield from _iterate_sample_pairs(tf, shard, cfg, stats)
    finally:
        tf.close()


def _iterate_sample_pairs(
    tf: tarfile.TarFile,
    shard: TarInfo,
    cfg: RepackConfig,
    stats: PipelineStats,
) -> Generator[Dict[str, Any], None, None]:
    """Walk tar members sequentially, pairing .json + .tif by basename."""
    pending: Dict[str, Dict[str, Any]] = {}

    for member in tf:
        if not member.isfile():
            continue
        name = member.name
        base, dot, ext = name.rpartition(".")
        if not dot:
            continue
        ext_low = ext.lower()

        if ext_low == "json":
            pending.setdefault(base, {})["json_name"] = name
            try:
                raw = tf.extractfile(member)
                if raw is not None:
                    pending[base]["json_data"] = raw.read()
            except Exception:
                pass
        elif ext_low in ("tif", "tiff"):
            pending.setdefault(base, {})["tif_name"] = name
            try:
                raw = tf.extractfile(member)
                if raw is not None:
                    pending[base]["tif_data"] = raw.read()
            except Exception:
                pass

        # When both halves are present, process immediately and free memory
        entry = pending.get(base)
        if entry and "json_data" in entry and "tif_data" in entry:
            yield from _process_sample(
                base, entry, shard.channel_count, cfg, stats,
            )
            del pending[base]

    # Handle any unpaired remainders (shouldn't happen in well-formed tars)
    for base, entry in pending.items():
        if "json_data" in entry and "tif_data" in entry:
            yield from _process_sample(
                base, entry, shard.channel_count, cfg, stats,
            )


# ======================================================================== #
# Per-sample processing: decode → tile → filter → encode                    #
# ======================================================================== #

def _process_sample(
    base: str,
    entry: Dict[str, Any],
    channel_count: int,
    cfg: RepackConfig,
    stats: PipelineStats,
) -> Generator[Dict[str, Any], None, None]:
    stats.total_samples += 1

    meta = read_json_bytes(entry["json_data"])
    if meta is None:
        stats.read_errors += 1
        return

    image = read_tiff_bytes(entry["tif_data"])
    if image is None:
        stats.read_errors += 1
        return

    try:
        image = ensure_channel_first(image)
    except ValueError as exc:
        logger.warning("Shape error for %s: %s", base, exc)
        stats.read_errors += 1
        return

    n_channels, img_h, img_w = image.shape
    sample_id = extract_sample_id(base)
    stats.record_channel_combo(n_channels)

    # ---- tiling -------------------------------------------------------
    crops = compute_crops(
        img_h, img_w,
        patch_size=cfg.patch_size,
        target_stride=cfg.target_stride,
        small_threshold=cfg.small_image_threshold,
    )
    kept_as_full = (
        len(crops) == 1 and crops[0].y0 == 0 and crops[0].x0 == 0
    )

    for crop in crops:
        stats.total_patches += 1
        patch = image[:, crop.y0:crop.y1, crop.x0:crop.x1]

        # ---- variance filter ------------------------------------------
        passed, var_val = passes_variance_filter(
            patch,
            reference_channel=cfg.reference_channel,
            variance_threshold=cfg.variance_threshold,
            missing_ref_policy=cfg.missing_ref_policy,
        )
        if not passed:
            stats.patches_filtered += 1
            continue

        # ---- encode packed sample -------------------------------------
        sample = _encode_sample(
            sample_id=sample_id,
            meta=meta,
            crop=crop,
            patch=patch,
            variance_value=var_val,
            kept_as_full=kept_as_full,
            n_channels=n_channels,
            img_h=img_h,
            img_w=img_w,
        )
        if sample is not None:
            yield sample


def _encode_sample(
    *,
    sample_id: str,
    meta: Dict[str, Any],
    crop: CropRegion,
    patch: np.ndarray,
    variance_value: float,
    kept_as_full: bool,
    n_channels: int,
    img_h: int,
    img_w: int,
) -> Optional[Dict[str, Any]]:
    """Build a packed WebDataset sample dict with encoded TIFF bytes."""
    key = f"{sample_id}_crop_{crop.y0}_{crop.x0}"

    out_meta: Dict[str, Any] = {
        "id": meta.get("id"),
        "dataset_name": meta.get("dataset_name", ""),
        "available_channels": list(range(1, n_channels + 1)),
        "crop_coordinates": [crop.x0, crop.y0, crop.x1, crop.y1],
        "original_shape": meta.get("original_shape"),
        "patch_shape": [crop.height, crop.width],
        "original_path": meta.get("original_path", ""),
        "source_sample_id": sample_id,
        "kept_as_full_image": kept_as_full,
        "variance_value": round(variance_value, 4),
        "source_channel_count": n_channels,
        "source_image_shape": [n_channels, img_h, img_w],
        "original_image_id": meta.get("original_image_id"),
        "source_crop_coordinates": meta.get("crop_coordinates"),
    }

    sample: Dict[str, Any] = {"__key__": key, "meta.json": out_meta}

    for ch_idx in range(n_channels):
        try:
            tif_bytes = encode_tiff_uint16(patch[ch_idx])
        except Exception as exc:
            logger.warning("TIFF encode error %s ch%d: %s", key, ch_idx + 1, exc)
            return None
        sample[f"ch{ch_idx + 1}.tif"] = tif_bytes

    return sample


# ======================================================================== #
# Shuffle-buffer helpers                                                    #
# ======================================================================== #

def _drain_buffer(
    buf: List[Dict[str, Any]],
    writer: PackedShardWriter,
    rng: random.Random,
    stats: PipelineStats,
) -> None:
    """Shuffle *buf* in-place and write out the first half."""
    rng.shuffle(buf)
    half = len(buf) // 2
    for sample in buf[:half]:
        writer.write(sample)
        stats.patches_kept += 1
    del buf[:half]
    logger.debug("Buffer drained: wrote %d, remaining %d", half, len(buf))
