"""Ray Master 调度逻辑（含断点续传 + OOM 防护 + .tmp 清理）。"""

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import ray

from .checkpoint import append_chunk_keys, filter_unprocessed, load_processed_keys
from .config import (
    FALLBACK_OUTPUT_DIR,
    GIGANTIC_FILE_THRESH,
    GIGANTIC_NUM_CPUS,
    ImageMeta,
    PipelineConfig,
    PRIMARY_OUTPUT_DIR,
)
from .db_client import fetch_all_tables
from .dedup_index import build_whitelist, filter_and_shuffle

logger = logging.getLogger("dinov3")


def run_ray_pipeline(config: PipelineConfig) -> None:
    """Ray 分布式流水线入口（含断点续传）。"""
    ray.init(address="auto")
    logger.info(f"Ray 集群已连接: {ray.cluster_resources()}")

    try:
        _execute(config)
    finally:
        ray.shutdown()
        logger.info("Ray 已关闭")


def _execute(config: PipelineConfig) -> None:
    """Master 端：拉取 → 白名单 → 断点续传 → 拆分 → 分发。"""
    valid_metas = _fetch_and_filter(config)
    if not valid_metas:
        logger.warning("白名单过滤后无有效数据")
        return
    worker_cfg = _build_worker_config(config)
    Path(worker_cfg["nfs_output_dir"]).mkdir(parents=True, exist_ok=True)

    processed = load_processed_keys(worker_cfg["nfs_output_dir"], config.channel_count)
    remaining = filter_unprocessed(valid_metas, processed)
    if not remaining:
        logger.info("所有数据已处理完毕（断点续传）")
        return
    chunks = _split_chunks(remaining, config.ray_chunk_size)
    logger.info(f"任务拆分: {len(remaining):,d} 条 → {len(chunks)} chunks")
    _dispatch_and_collect(chunks, worker_cfg)


def _fetch_and_filter(config: PipelineConfig) -> List[ImageMeta]:
    """Master 端: DB 拉取 + 白名单 O(1) 拦截 + 跨源 shuffle。"""
    whitelist = build_whitelist(config)
    all_metas = fetch_all_tables(config)
    return filter_and_shuffle(all_metas, whitelist)


def _split_chunks(
    metas: List[ImageMeta],
    chunk_size: int,
) -> List[List[dict]]:
    """将 ImageMeta 按 chunk_size 拆分为可序列化字典块。"""
    dicts = [asdict(m) for m in metas]
    return [
        dicts[i : i + chunk_size]
        for i in range(0, len(dicts), chunk_size)
    ]


def _build_worker_config(config: PipelineConfig) -> Dict[str, Any]:
    """构建 Worker 配置字典（含分片偏移量防覆盖）。"""
    nfs_dir = _resolve_nfs_dir()
    prefix = f"mixed_{config.channel_count}ch"
    existing = len(list(Path(nfs_dir).glob(f"{prefix}-*.tar")))
    return {"max_target_size": config.max_target_size, "nfs_output_dir": nfs_dir,
            "shard_prefix": prefix, "channel_count": config.channel_count,
            "shard_offset": existing}


def _resolve_nfs_dir() -> str:
    """
    解析 NFS 输出目录（主路径优先，空间不足时回退到备用路径）。

    与本地模式保持一致：当主路径可用空间低于阈值时，自动回退到
    /mnt/deepcad_nfs，避免主盘写满导致任务失败。
    """
    primary_root = Path(PRIMARY_OUTPUT_DIR)
    if _has_disk_space(primary_root):
        return str(primary_root / "wds_shards")

    logger.warning(f"主路径空间不足: {primary_root}, 回退到备用路径")
    fallback_root = Path(FALLBACK_OUTPUT_DIR)
    if not _has_disk_space(fallback_root):
        logger.warning(f"备用路径空间也不足: {fallback_root}, 仍尝试写入备用路径")
    return str(fallback_root / "wds_shards")


def _has_disk_space(path: Path, min_gb: float = 50.0) -> bool:
    """
    检查目录所在文件系统是否有足够剩余空间。

    Args:
        path: 目标目录。
        min_gb: 最低可用空间（GB），低于该值判定为空间不足。
    """
    try:
        stat = os.statvfs(str(path))
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        return free_gb > min_gb
    except OSError:
        return False


def _dispatch_and_collect(
    chunks: List[List[dict]],
    worker_cfg: dict,
) -> None:
    """提交所有 Chunk 到 Ray 集群并流式收集。"""
    from .ray_worker import process_chunk
    offset = worker_cfg.get("shard_offset", 0)
    if offset:
        logger.info(f"断点续传: tar 编号从 {offset} 开始")
    futures = [
        _submit_chunk(process_chunk, chunk, idx + offset, worker_cfg)
        for idx, chunk in enumerate(chunks)
    ]
    _collect_with_progress(futures, chunks, worker_cfg)


def _submit_chunk(fn: object, chunk: List[dict], idx: int, cfg: dict) -> object:
    """按 Chunk 内最大文件动态设置 CPU（巨型图按配置值提核）。"""
    max_size = max((d.get("file_size_bytes", 0) for d in chunk), default=0)
    cpus = GIGANTIC_NUM_CPUS if max_size >= GIGANTIC_FILE_THRESH else 1
    return fn.options(num_cpus=cpus).remote(chunk, idx, cfg)


def _collect_with_progress(
    futures: list,
    chunks: List[List[dict]],
    worker_cfg: dict,
) -> None:
    """ray.wait + Rich 进度条逐个收集，同时写断点检查点。"""
    progress_ctx = _build_rich_progress()
    total = len(futures)
    total_samples, failed = 0, 0
    pending = list(futures)
    offset = worker_cfg.get("shard_offset", 0)

    with progress_ctx as progress:
        task_id = progress.add_task("archive", total=total, samples=0)

        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            result = _safe_get(done[0])
            total_samples += result.get("samples", 0)
            local_idx = result.get("chunk_id", -1) - offset
            if 0 <= local_idx < len(chunks):
                append_chunk_keys(
                    worker_cfg["nfs_output_dir"],
                    worker_cfg["channel_count"],
                    chunks[local_idx],
                    tar_name=result.get("tar_name", ""),
                )
            else:
                failed += 1
            progress.update(task_id, advance=1, samples=total_samples)

    _cleanup_stale_tmp(worker_cfg)
    _log_summary(total, failed, total_samples)


def _cleanup_stale_tmp(worker_cfg: dict) -> None:
    """清理 NFS 中 Worker 崩溃残留的 .tar.tmp 文件。"""
    stale = list(Path(worker_cfg["nfs_output_dir"]).glob("*.tar.tmp"))
    if not stale:
        return
    logger.warning(f"清理 {len(stale)} 个崩溃残留 .tmp 文件")
    for f in stale:
        try:
            f.unlink()
        except OSError:
            pass


def _log_summary(total: int, failed: int, samples: int) -> None:
    """输出归档摘要（成功/失败 + 断点续传提示）。"""
    logger.info(f"归档完成: {total - failed}/{total} chunks 成功, {samples:,d} 样本")
    if failed:
        logger.warning(f"{failed} chunks 失败(OOM/崩溃)，重跑可断点续传补齐")


def _build_rich_progress() -> Any:
    """构建 Rich 进度条。"""
    from rich.progress import (
        BarColumn, Progress, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn,
    )
    return Progress(
        TextColumn("[bold cyan]归档进度"),
        BarColumn(bar_width=40),
        TextColumn("{task.completed}/{task.total} chunks ·"),
        TextColumn("[green]{task.fields[samples]:,d} samples"),
        TimeElapsedColumn(), TextColumn("eta"), TimeRemainingColumn(),
    )


def _safe_get(ref: "ray.ObjectRef") -> dict:
    """安全获取 Ray 任务结果，Worker 异常时返回空统计。"""
    try:
        return ray.get(ref)
    except (ray.exceptions.RayTaskError, RuntimeError) as exc:
        logger.warning(f"Worker 异常: {exc}")
        return {"chunk_id": -1, "samples": 0, "tar_name": ""}
