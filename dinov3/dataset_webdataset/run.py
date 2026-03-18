# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档 — CLI 入口。

"""
命令行入口脚本。

必须从仓库根目录（dinov3/ 的上一级）运行，否则本地 logging/ 目录会
遮蔽标准库 logging 模块导致 ModuleNotFoundError。

用法:
    # 切换到仓库根目录
    cd /mnt/huawei_deepcad/dinov3

    # Ray 分布式打包（默认，推荐）
    python -m dinov3.dataset_webdataset.run --table all --channels 4

    # 单机本地打包（调试用）
    python -m dinov3.dataset_webdataset.run --table all --channels 4 --mode local

    # Ray + 自定义 chunk 大小
    python -m dinov3.dataset_webdataset.run --table all --channels 4 --chunk-size 800
"""

import argparse
import logging
import sys
from typing import List

from .config import ALL_TABLES, PipelineConfig, TABLE_DYNAMIC, TABLE_STATIC

logger = logging.getLogger("dinov3")

VALID_TABLE_CHOICES: List[str] = [TABLE_STATIC, TABLE_DYNAMIC, "all"]
VALID_MODES: List[str] = ["ray", "local"]


def _parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        解析后的参数命名空间。
    """
    parser = _build_arg_parser()
    return parser.parse_args()


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器（含所有子参数定义）。"""
    p = argparse.ArgumentParser(
        description="DINOv3 WebDataset 100TB 数据归档流水线"
    )
    p.add_argument("--table", type=str, required=True,
                   choices=VALID_TABLE_CHOICES,
                   help='"all" 联合打包 (推荐), 或单表名')
    p.add_argument("--channels", type=int, required=True,
                   help="目标通道数")
    p.add_argument("--mode", type=str, default="ray",
                   choices=VALID_MODES,
                   help="ray (分布式, 默认) / local (单机)")
    p.add_argument("--max-target-size", type=int, default=4000,
                   help="动态等分切片期望的单片最大边长 (默认 4000)")
    p.add_argument("--chunk-size", type=int, default=500,
                   help="Ray 每个 Chunk 的图片数 (默认 500)")
    p.add_argument("--max-shard-gb", type=float, default=4.0,
                   help="单个 tar 上限 GB (默认 3.0, 仅 local)")
    return p


def _resolve_table_names(table_arg: str) -> List[str]:
    """
    将 CLI --table 参数解析为表名列表。

    Args:
        table_arg: "all" 或具体表名。

    Returns:
        表名列表。
    """
    if table_arg == "all":
        return list(ALL_TABLES)
    return [table_arg]


def _setup_logging() -> None:
    """配置全局日志格式和级别。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    """CLI 主函数：根据 --mode 分发到 Ray 或本地流水线。"""
    _setup_logging()
    args = _parse_args()

    config = PipelineConfig(
        table_names=_resolve_table_names(args.table),
        channel_count=args.channels,
        max_target_size=args.max_target_size,
        max_shard_bytes=int(args.max_shard_gb * 1024 ** 3),
        ray_chunk_size=args.chunk_size,
    )

    logger.info(f"配置: {config}")
    logger.info(f"执行模式: {args.mode}")

    if args.mode == "ray":
        from .ray_dispatcher import run_ray_pipeline
        run_ray_pipeline(config)
    else:
        from .pipeline import run_pipeline
        run_pipeline(config)


if __name__ == "__main__":
    main()
