# dataset_webdataset — DINOv3 100TB 多源医学影像归档流水线

## 它解决什么问题？

将分散在 PostgreSQL + NFS 上的海量（100TB+）多通道 TIFF 医学影像，经过去重、张量重构、帧提取、裁切、质量过滤后，打包成训练可用的 WebDataset `.tar` 分片。

## 快速开始

```bash
# 推荐：Ray 分布式 + 两表联合打包（默认模式）
python -m dataset_webdataset.run --table all --channels 4

# Ray + 自定义 chunk 大小
python -m dataset_webdataset.run --table all --channels 4 --chunk-size 800

# 单机本地模式（调试用）
python -m dataset_webdataset.run --table all --channels 4 --mode local

# 仅打包单表
python -m dataset_webdataset.run --table original_images_all --channels 4
```

> **为什么推荐 `--table all`？**
>
> 自监督预训练的核心假设是训练样本满足 i.i.d.（独立同分布）。如果将静态图谱（`original_images_all`）和动态序列（`original_image_all_2p_parsed`）分开打包，同一个 tar 分片内的样本只会来自同一种数据源——这直接破坏了 shuffle 的混合效果。`--table all` 模式会将两张表的数据**先拉取合并、再统一 shuffle**，最终写入 tar 分片的顺序是完全随机交织的。

## 依赖

```bash
pip install psycopg2-binary tifffile numpy webdataset ray rich
```

## 目录结构与模块职责

```
dataset_webdataset/
│
├── config.py            ← 全局常量 + 数据类定义
│   • DB_CONFIG            数据库连接信息
│   • ImageMeta            单条图像元数据（含 source_table 来源标记）
│   • PipelineConfig       流水线运行时配置
│
├── run.py               ← CLI 命令行入口（--mode ray/local）
│   • main()              解析参数 → 按 mode 分发到 Ray 或本地流水线
│
├── ray_dispatcher.py    ← Ray Master 调度（默认模式）
│   • run_ray_pipeline()   DB → 白名单 → Chunk 拆分 → ray.wait → Rich 进度
│
├── ray_worker.py        ← Ray Worker 远程函数
│   • process_chunk()      TIFF→裁切→过滤→本地写tar→shutil.move至NFS
│
├── pipeline.py          ← 单机本地流水线（调试 / 回退用）
│   • run_pipeline()       端到端编排所有阶段
│
│  ┌─ 数据获取阶段 ──────────────────────────────────────────
│  │
├── db_client.py         ← PostgreSQL 批量拉取
│   │ • fetch_all_tables()  从多表联合拉取，每条记录标记来源
│   │
├── dedup_index.py       ← 去重白名单 O(1) 拦截
│   │ • build_whitelist()   多表索引目录聚合为统一哈希集
│   │ • filter_and_shuffle() 白名单过滤 + 跨源宏观打乱
│   │
│  ├─ 张量重构阶段 ──────────────────────────────────────────
│  │
├── tiff_reader.py       ← TIFF 安全读取（异常不中断流水线）
│   │
├── tensor_static.py     ← 静态图谱维度清洗 (§2.1)
│   │ • reconstruct_static()  基于 DB 元数据的确定性 (C,H,W) 重塑
│   │
├── tensor_dynamic.py    ← 动态序列 Hyperstack 解码 (§2.2)
│   │ • reconstruct_dynamic() 4D 匹配 / Fiji Hyperstack 还原 → (C,F,H,W)
│   │
├── frame_extractor.py   ← 精准帧提取 (§3)
│   │ • extract_target_frame() array[:, frame_idx, :, :] 保留所有通道
│   │
│  ├─ 裁切与过滤阶段 ────────────────────────────────────────
│  │
├── spatial_slicer.py    ← WSI 4096 自适应滑窗裁切 (§4)
│   │ • slice_wsi_patches()   边缘反弹对齐（Snap-to-Edge）
│   │
├── quality_filter.py    ← 背景质量过滤 (§5)
│   │ • passes_quality_check() 方差 + 组织覆盖率二阶段检测
│   │
│  └─ 归档阶段 ──────────────────────────────────────────────
│
├── npy_namer.py         ← 全息溯源命名法 (§6)
│   • build_npy_name()     由 meta.is_dynamic 自动路由命名策略
│
└── shard_writer.py      ← 动态闭环 Tar 分片写入 (§6)
    • ShardWriter          ~3GB 自动轮转，主路径满后回退备用路径
```

## Ray 分布式架构 (默认模式)

```
  ┌─── Master (Driver) ────────────────────────────┐
  │                                                 │
  │  PostgreSQL → fetch_all_tables()                │
  │       ↓                                         │
  │  build_whitelist() → O(1) 哈希拦截              │
  │       ↓                                         │
  │  filter_and_shuffle() → 跨源宏观 shuffle        │
  │       ↓                                         │
  │  拆分为 N 个 Chunks (每块 ≤500 条)              │
  │       ↓                                         │
  │  ray.wait() 逐个收集 + Rich 实时进度条          │
  └──────┬──────────────────────────────────────────┘
         │  process_chunk.remote()
         │  (白名单已过滤，Worker 无需哈希集)
    ┌────┴────────────────────────────────────┐
    │          Ray Worker ×1040 CPUs          │
    │                                         │
    │  TIFF 读取 (NFS 读)                     │
    │  → 张量重构 → 帧提取 → 滑窗裁切 → 过滤  │
    │  → 本地 NVMe 写 tar (/tmp/)             │
    │  → shutil.move 一次性移至 NFS           │
    └────┬────────────────────────────────────┘
         ↓
  /mnt/huawei_deepcad/wds_shards/mixed_4ch-000000.tar
  /mnt/huawei_deepcad/wds_shards/mixed_4ch-000001.tar
```

**关键 I/O 保护**：Worker 不直接写 NFS。先写入节点本地 `/tmp/ray_wds_shards/`，关闭 tar 后 `shutil.move` 整文件移动，避免 1000+ 并发流式写入导致 NFS 锁崩溃。

## 关键设计决策

### 1. 为什么 `ImageMeta` 携带 `source_table`？

每条数据记录自带来源标记（`source_table` 字段 + `is_dynamic` 属性），在流水线的任何阶段都可以根据数据来源做差异化处理——**无需向外传递 config 级别的全局开关**。

这使得联合打包时，同一个 shuffle 后的列表中的静态图谱和动态序列能各自走对应的张量重构和命名路由，互不干扰。

### 2. 去重白名单的路由逻辑

```
dedupIndex_100t/
├── ori/                   ← original_images_all 路由到这里
│   ├── 1ch/
│   │   └── filtered_kept_paths.txt
│   ├── 2ch/
│   ├── 3ch/
│   ├── ...
│   └── 10ch/
└── slfmandhighna/         ← original_image_all_2p_parsed 路由到这里
    ├── 1ch/               ← HighNA 1ch 数据
    │   └── kept_paths_eps0.01.txt
    ├── 2ch/
    ├── 3ch/
    ├── 4ch/
    ├── 5ch/
    └── slfm/              ← SLFM 数据（固定 1ch，仅当 channel_count=1 时追加）
        └── kept_paths_eps0.01.txt
```

> **slfm/ 不是回退目录**，而是 1ch SLFM 数据的专属补充源。当 `--channels 1` 时，动态表的白名单同时加载 `slfmandhighna/1ch/` 和 `slfmandhighna/slfm/` 两个目录并取并集。当 `--channels >= 2` 时 `slfm/` 不参与。

联合打包时，两个表的白名单**分别加载、合并为统一哈希集**。

### 3. NPY 命名路由差异

| 来源 | 无裁切 | 有裁切 |
|------|--------|--------|
| 静态 | `12345.npy` | `12345_X0_Y0_H4096_W4096.npy` |
| 动态 | `2p_67890_frame3.npy` | `2p_67890_X0_Y0_H4096_W4096_frame3.npy` |

由 `meta.is_dynamic` 自动路由，抛弃独立的 JSON 元数据文件。

### 4. 滑窗裁切的边缘反弹 (Snap-to-Edge)

```
图像宽度 = 10000 px, crop_size = 4096

常规裁切:  [0:4096] [4096:8192] [8192:10000] ← 最后一块只有 1808 px（畸形！）
Snap裁切:  [0:4096] [4096:8192] [5904:10000] ← 反弹到 10000-4096=5904 起始
                                     ↑ 与前一块有 2288 px 重叠（提供多视角上下文）
```

### 5. 质量过滤两阶段

1. **方差检测**：`np.var(patch)` < 1e-4 → 纯黑/纯噪点，拒绝
2. **组织覆盖率**：自适应阈值二值化后有效面积 < 5% → 空白背景，拒绝

## 输出格式

- **路径**: `/mnt/huawei_deepcad/wds_shards/mixed_4ch-000000.tar`
- **分片大小**: ~3 GB / 个
- **内容**: 每个 tar 包含大量 `.npy` 文件
- **数据精度**: 保留原始精度（如 16-bit），不做浮点转换
- **读取**: 训练时用 `data/wds_pipeline.py` 中的 WebDataset 管道加载

## 配置参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CROP_SIZE` | 4096 | WSI 滑窗尺寸 |
| `MAX_SHARD_BYTES` | 3 GB | 单个 tar 分片上限 (local 模式) |
| `DB_FETCH_BATCH` | 500 | DB 批量拉取行数 |
| `MIN_VARIANCE` | 1e-4 | 最低像素方差 |
| `MIN_TISSUE_COVERAGE` | 0.05 | 最低组织覆盖率 (5%) |
| `RAY_CHUNK_SIZE` | 500 | Ray 模式每个 Chunk 的图片数 |
| `RAY_LOCAL_TMP` | `/tmp/ray_wds_shards` | Worker 本地临时写入目录 |
| `PRIMARY_OUTPUT_DIR` | `/mnt/huawei_deepcad` | 主输出路径 |
| `FALLBACK_OUTPUT_DIR` | `/mnt/deepcad_nfs` | 备用输出路径 |

