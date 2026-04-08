# data/repackage — 多通道显微图像重打包流水线

将按通道分开存放的 WebDataset tar 分片（`ch1/` … `ch8/`），重新打包为**每个样本包含所有实际存在通道**的新 tar 分片，供 DINOv3 + ChannelViT 训练直接使用。

---

## 解决的问题

原始数据按通道分目录存储，训练时如果并行读取多个通道流再 zip，会产生三个问题：

1. **通道错位**：各通道 shuffle 独立，拼出来的多通道图像可能来自不同的物理视野
2. **I/O 压力**：8 个并发 tar 流 × worker 数，NFS 句柄和带宽开销翻倍
3. **变通道处理粗糙**：缺失通道只能靠循环填充，模型看到假数据

重打包后每条样本保证同一物理位置的所有通道严格对齐，缺失通道用 `available_channels` 字段显式标记。

---

## 输出格式

每条输出样本是标准 WebDataset 格式，包含：

```
__key__    :  id000268476_oid31371344_crop_0_0
ch1.tif    :  uint16 TIFF, shape (H, W)
ch2.tif    :  uint16 TIFF, shape (H, W)
...
chN.tif    :  uint16 TIFF, shape (H, W)
meta.json  :  见下文
```

`meta.json` 字段：

| 字段 | 含义 |
|------|------|
| `id` | 原始数据库 ID |
| `dataset_name` | 数据集名称 |
| `available_channels` | 实际存在的通道编号列表（如 `[1,3,5]`） |
| `crop_coordinates` | 本次裁切坐标 `[x0, y0, x1, y1]` |
| `original_shape` | 原始图像形状 |
| `patch_shape` | 输出 patch 的 `[H, W]` |
| `original_path` | 原始 TIFF 在 NFS 上的路径 |
| `source_sample_id` | 来源样本 ID（不含 crop 坐标） |
| `kept_as_full_image` | `true` 表示未切图，保留了整图 |
| `variance_value` | 参考通道的像素方差（过滤依据） |
| `source_image_shape` | 输入图像的 `[C, H, W]` |
| `source_crop_coordinates` | 输入数据中已有的裁切坐标 |

---

## 快速开始

```bash
# 最简调用（自动检测 ch1–ch8，默认参数）
python data/repackage/preprocess_repack.py

# 完整参数示例
python data/repackage/preprocess_repack.py \
  --input-root  /mnt/huawei_deepcad/webds_micro_100k_by_channel \
  --output-root /mnt/huawei_deepcad/wds_packed_shards \
  --reference-channel 1 \
  --variance-threshold 20.0 \
  --num-workers 16 \
  --shuffle-buffer-size 3000

# 仅处理指定通道
python data/repackage/preprocess_repack.py \
  --channel-dirs ch1 ch3 ch5 \
  --num-workers 8 \
  --shuffle-buffer-size 2000
```

---

## 项目结构

```
data/repackage/
├── preprocess_repack.py   # CLI 入口
├── pipeline.py            # 核心编排：发现 → 流读 → 切图 → 过滤 → 洗牌 → 写出
├── config.py              # RepackConfig dataclass（所有可调参数）
├── index_builder.py       # 分片发现（读 manifest.json，O(1) per dir）
├── tiling.py              # 动态切图逻辑
├── filtering.py           # 方差过滤
├── writer.py              # PackedShardWriter（webdataset.ShardWriter 包装）
├── io_utils.py            # TIFF / JSON 编解码、channel-first 归一化
├── utils.py               # PipelineStats、日志设置、stats 合并
├── __init__.py
├── __main__.py            # 支持 python -m data.repackage
└── requirements.txt
```

---

## 流水线五步

### Step 1 — 分片发现（`index_builder.py`）

扫描 `ch*/manifest.json`（毫秒级，不读 tar 内容），得到所有待处理 tar 的路径和通道号。若无 manifest，则回退到 `glob("*.tar")`。

```
ch1/manifest.json  →  130 shards
ch2/manifest.json  →    6 shards
ch3/manifest.json  →  1147 shards
...
```

宏观 shuffle：将所有 shard 按 `seed` 随机重排，混合不同通道的数据。

---

### Step 2 — 动态切图（`tiling.py`）

规则（无 padding，无 resize）：

| 图像尺寸 | 行为 |
|----------|------|
| H ≤ 900 **且** W ≤ 900 | 保留完整原图，生成 1 个 crop |
| 任一边 > 900 | 滑窗切图 |

切图参数：

- `patch_size = 512`
- `target_stride = 384`（实际 stride 动态调整，保证最后一块严格贴边，避免过度重叠）

H 和 W 各自独立计算起点列表，再做笛卡尔积，确保覆盖所有角落。

---

### Step 3 — 方差过滤（`filtering.py`）

在**参考通道**上计算像素方差，低于阈值的 patch 视为背景废图丢弃：

```
var(patch[ref_channel]) < variance_threshold  →  丢弃
```

参考通道缺失时的策略（`--missing-ref-policy`）：
- `fallback_first_available`（默认）：改用第一个实际存在的通道
- `skip_sample`：直接丢弃该 patch

---

### Step 4 — 有界全局洗牌（`pipeline.py`）

采用**有界 buffer** 模式，避免全量加载：

```
while 有数据:
    填充 buffer 直到 shuffle_buffer_size
    random.shuffle(buffer)
    写出前一半
    保留后一半继续填充
```

多 worker 时 buffer 自动按 `shuffle_buffer_size // num_workers` 分配，保持总内存预算不变。训练时 WebDataset 的内置 shuffle 在 shard 级别做进一步混洗。

---

### Step 5 — 重打包写出（`writer.py`）

使用 `webdataset.ShardWriter` 写出，达到 `max_shard_size` 或 `max_shard_count` 时自动轮转：

```
wds_packed_shards/
  filtered_mixed_train_w00-000000.tar
  filtered_mixed_train_w00-000001.tar
  filtered_mixed_train_w01-000000.tar   ← worker 1 的输出
  ...
```

---

## 多进程并行（`--num-workers`）

**性能瓶颈实测**：TIFF decode 占约 90% 的处理时间（CPU 密集），NFS 读约 10%。多进程接近线性加速。

实现方式：`ProcessPoolExecutor`，每个 worker 处理 shard 列表的一个**交错子集**（round-robin 分配，保证各 worker 通道分布均衡），写入独立的 tar 文件（前缀带 `_w<NN>`）。

| workers | 推荐 `shuffle-buffer-size` | 预期加速 |
|---------|--------------------------|----------|
| 1 | 500（ch3 大图） | 基准 |
| 8 | 2000 | ~6–7x |
| 16 | 3000 | ~12–14x |
| 32 | 5000 | ~20x+ |

> **ch3 内存警告**：ch3 中存在 ~1.5 GB 的单张 TIFF（约 16K×16K px）。buffer 中每个 512×512 3ch patch ≈ 1.5 MB，`buffer=10000` 单 worker 需要约 15 GB。推荐 ch3 使用 `--shuffle-buffer-size 2000 --num-workers 16`。

---

## 配置参数速查（`config.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_root` | `/mnt/huawei_deepcad/webds_micro_100k_by_channel` | 输入根目录 |
| `output_root` | `/mnt/huawei_deepcad/wds_packed_shards` | 输出目录 |
| `patch_size` | `512` | 切图 patch 边长 |
| `target_stride` | `384` | 滑窗目标步长 |
| `small_image_threshold` | `900` | 两边都 ≤ 此值则不切图 |
| `reference_channel` | `1` | 方差过滤参考通道（1-indexed） |
| `variance_threshold` | `20.0` | 最低方差门槛（uint16 量纲） |
| `missing_ref_policy` | `fallback_first_available` | 参考通道缺失时的策略 |
| `shuffle_buffer_size` | `10000` | 洗牌 buffer 容量（总预算，自动按 worker 数分配） |
| `max_shard_count` | `10000` | 单个输出 tar 最大样本数 |
| `max_shard_size` | `3 GB` | 单个输出 tar 最大字节数 |
| `shard_prefix` | `filtered_mixed_train` | 输出文件名前缀 |
| `seed` | `42` | 随机种子 |
| `num_workers` | `1` | 并行进程数 |

---

## 接入 DINOv3 训练

输出 shard 可直接用 `wds:` 前缀加载。解码端需要一个能处理 `ch*.tif` + `meta.json` 格式的 decoder，示例框架：

```python
# data/loaders.py 中添加：
dataset_str = "wds:/mnt/huawei_deepcad/wds_packed_shards/filtered_mixed_train_w*-{000000..000999}.tar"

# 自定义 decode_sample（替换 wds_pipeline.py 中现有的 decode_sample）：
def decode_packed_sample(sample: dict, target_channels: int = 8):
    import json, tifffile, io, torch
    meta = json.loads(sample["meta.json"])
    available = meta["available_channels"]
    h, w = meta["patch_shape"]

    result = torch.zeros(target_channels, h, w, dtype=torch.float32)
    for ch in available:
        key = f"ch{ch}.tif"
        if key in sample:
            arr = tifffile.imread(io.BytesIO(sample[key])).astype("float32") / 65535.0
            result[ch - 1] = torch.from_numpy(arr)
    return result
```

`meta.json` 中的 `available_channels` 字段可直接传给 ChannelViT 的 channel mask，告知模型哪些通道有实际信号。
