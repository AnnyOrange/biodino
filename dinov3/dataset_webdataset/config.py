# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DINOv3 多源医学影像 WebDataset 归档配置模块。

"""
全局常量、数据类和配置定义。

集中管理数据库连接、路径路由、阈值参数等流水线配置。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ============================================================================
# 全局常量
# ============================================================================

DB_CONFIG: Dict[str, str] = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "pW5^sL9&hJmZ1!uN",
    "host": "172.16.0.217",
    "port": "5432",
}

DEDUP_INDEX_ROOT: str = (
    "/mnt/huawei_deepcad/ssl-data-curation/dedupIndex_100t1"
)

# 默认输出切换为 deepcad_nfs，huawei_deepcad 作为后备目录
PRIMARY_OUTPUT_DIR: str = "/mnt/huawei_deepcad"
FALLBACK_OUTPUT_DIR: str = "/mnt/deepcad_nfs"

# --- 动态切图配置 ---
MAX_TARGET_SIZE: int = 4000       # 动态等分切片期望的单片最大边长
TILING_LONG_THRESH: int = 4096    # 切分触发条件：长边 > 此阈值
TILING_SHORT_THRESH: int = 1024   # 切分触发条件：短边 > 此阈值

MAX_SHARD_BYTES: int = 3 * 1024 * 1024 * 1024  # ~3 GB
DB_FETCH_BATCH: int = 500

# --- 极速空块过滤配置 ---
QUALITY_DS_FACTOR: int = 16       # 降采样步长（计算量降 256 倍）
QUALITY_MIN_COVERAGE: float = 0.05  # 最小有效组织覆盖率
QUALITY_MIN_STD: float = 2.0     # 有效像素最低标准差

# OOM 防爆阈值：>= 1GB 的文件视为巨型图，Ray 分配 8 核以限制并发，
# 防止多个 Worker 同时 imread 大文件撑爆内存。
# 读取方式统一走 imread（已关闭 memmap/本地缓存）。
GIGANTIC_FILE_THRESH: int = 1 * 1024 * 1024 * 1024
GIGANTIC_NUM_CPUS: int = 8

# Ray 分布式常量
# 暂存目录使用 /dev/shm（tmpfs 纯内存），磁盘空间极小时避免撑爆本地盘
RAY_LOCAL_TMP: str = "/dev/shm/ray_wds_shards"
RAY_CHUNK_SIZE: int = 500

# 数据表名常量
TABLE_STATIC: str = "original_images_all"
TABLE_DYNAMIC: str = "original_image_all_2p_parsed"
ALL_TABLES: List[str] = [TABLE_STATIC, TABLE_DYNAMIC]


# ============================================================================
# 数据类
# ============================================================================

@dataclass(frozen=True)
class ImageMeta:
    """单条数据库图像元数据行。

    Attributes:
        row_id: 数据库原始 ID。
        file_path: TIFF 文件物理路径。
        height: 图像高度（像素）。
        width: 图像宽度（像素）。
        frame_count: 时间/深度帧数。
        channel_count: 通道数。
        source_table: 数据来源表名（决定重构策略与命名路由）。
        frame_idx: 目标帧索引（仅动态序列需要）。
        file_size_bytes: 文件字节数（用于 OOM 防护调度）。
    """
    row_id: int
    file_path: str
    height: int
    width: int
    frame_count: int
    channel_count: int
    source_table: str = TABLE_STATIC
    frame_idx: Optional[int] = None
    file_size_bytes: int = 0

    @property
    def is_dynamic(self) -> bool:
        """判断该样本是否来自动态序列数据池。"""
        return self.source_table == TABLE_DYNAMIC


@dataclass
class PipelineConfig:
    """流水线运行时配置。

    Attributes:
        table_names: 目标表名列表（支持多表联合打包）。
        channel_count: 目标通道数。
        max_target_size: 动态等分切片期望的单片最大边长。
        max_shard_bytes: 单个 tar 分片的最大字节数。
        ray_chunk_size: Ray 每个 Chunk 的图片数。
    """
    table_names: List[str] = field(default_factory=lambda: [TABLE_STATIC])
    channel_count: int = 3
    max_target_size: int = MAX_TARGET_SIZE
    max_shard_bytes: int = MAX_SHARD_BYTES
    ray_chunk_size: int = RAY_CHUNK_SIZE
