# 1TB WebDataset 检查记录

数据目录：

`/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle`

## 目录结构

当前数据是 packed WebDataset shards：

- 总大小：`966G`
- 顶层 tar shard 数：`326`
- 单个主 shard 大多约 `3.2GB`
- 另有 `test_pack_spec/`，包含一个 `402M` 的小测试包

示例 shard：

`filtered_mixed_train_w00-000000.tar`

内部样本结构：

```text
<sample_key>.ch1.tif
<sample_key>.ch2.tif
<sample_key>.ch3.tif
<sample_key>.meta.json
```

样本并不总是 3 通道；有的只有 `ch1.tif`，有的有 `ch1/ch2/ch3`。

## TIFF 格式

抽样检查到的 TIFF：

- 尺寸：`512 x 512`
- dtype：`uint16`
- TIFF compression：`none`
- 单通道未压缩 TIFF 大小约 `524KB`

这和代码里的打包逻辑一致：

`dinov3/data/repackage/io_utils.py` 里当前是：

```python
tifffile.imwrite(buf, array_2d, compression=None)
```

所以当前 966G 的主要空间来自未压缩 16-bit TIFF，而不是 tar 容器本身。

## 一个主 shard 的内容分布

对 `filtered_mixed_train_w00-000000.tar` 抽样统计：

```text
entries: 9973
json samples: 3270
tif channel files: 6703
total payload bytes: 3221964710
tif payload bytes: 3220217600
json payload bytes: 1747110
```

结论：JSON metadata 可以忽略，空间几乎全部在 TIFF。

## 小测试包压缩结果

测试包：

`test_pack_spec/filtered_mixed_train_w00-000000.tar`

测试包信息：

```text
tar size: 402M
samples: 100
tif channel files: 200
```

外层 tar 压缩：

| 方法 | 输出大小 | 约占原始 tar |
| --- | ---: | ---: |
| zstd -1 | 100M | 24.9% |
| lz4 -1 | 148M | 36.8% |

TIFF 内部无损压缩：

| 方法 | TIFF payload 大小 | 约占原始 TIFF payload |
| --- | ---: | ---: |
| 原始 TIFF | 421078016 bytes | 100% |
| TIFF deflate | 103456460 bytes | 24.6% |
| TIFF LZW | 114736846 bytes | 27.2% |

基于这个小包的粗略外推：

- `966G` 原始数据用 TIFF deflate 后可能约 `240G`
- `966G` 原始数据用 TIFF LZW 后可能约 `260G`
- `966G` 原始 tar 用 zstd -1 外压后可能约 `240G`
- `966G` 原始 tar 用 lz4 -1 外压后可能约 `355G`

这个外推只用于定方向，不能直接作为最终数值。正式实验应至少抽 `10-20` 个 shard 重新测。

## 对实验设计的影响

这批数据已经是 WebDataset tar shards，所以优先方案应该是：

1. 保持 `.tar` shard 结构不变。
2. 保持 `.chN.tif + .meta.json` 样本格式不变。
3. 只把 TIFF 从 `compression=None` 改成 lossless compression，例如 `deflate` 或 `zstd`。
4. 先验证现有 `tifffile.imread` 解码器能直接读取压缩 TIFF。
5. 再测 dataloader 吞吐和 GPU 利用率。

不建议第一步就上 JPEG/WebP，因为当前是 16-bit microscopy/patch 数据，JPEG/WebP 会引入强 lossy 假设，也可能破坏通道强度分布。更合理的顺序是：

1. TIFF deflate / zstd 等无损内部压缩。
2. 如果仍放不下，再试 `uint16 -> uint8` 量化。
3. 最后才考虑降分辨率、patch 筛选或 feature 去重。

