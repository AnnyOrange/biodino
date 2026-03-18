# DINOv3 Medical Project

这是一个基于 Meta `DINOv3` 改造的多通道医学影像自监督训练工程。仓库保留了原始 DINOv3 的核心训练框架，但当前版本的重点已经不是官方模型展示，而是面向你自己的项目需求：

- 支持多通道输入与 `ChannelViT`
- 支持 `LoRA + ChannelViT` 轻量微调
- 支持 `wds:` 前缀的 `WebDataset` 训练入口
- 提供面向 PostgreSQL + NFS 的大规模医学影像归档流水线
- 提供面向多通道/多尺寸数据的 smart sharding 脚本

## 项目定位

原始 DINOv3 仓库更偏向通用视觉基础模型发布与论文复现；这个仓库已经被扩展为一个面向多通道生物/医学影像的训练工程。

当前仓库最重要的不是官方预训练权重表，而是下面这三条主线：

1. `ChannelViT` 多通道建模
2. `WebDataset` 大规模流式训练
3. 医学影像数据归档与分片生成

## 主要改动

### 1. ChannelViT 集成到 ViT 主干

在 `dinov3/models/vision_transformer.py` 中增加了 `enable_channelvit` 开关。

- 当 `enable_channelvit=false` 时，保持原始 DINOv3 patch embedding 路径
- 当 `enable_channelvit=true` 时，切换到按通道处理的 `PatchEmbedPerChannel`
- 新增 `channel_embed` 参数，用于多通道特征编码
- `student.in_chans` / `teacher.in_chans` 可以配置为 3 以外的通道数

适用场景：

- 荧光显微镜
- 多染色生物图像
- 多模态医学图像
- 非 RGB 遥感/科学成像

### 2. LoRA + ChannelViT

仓库增加了 `LoRA` 与 `ChannelViT` 的联合配置，可在保留多通道能力的同时降低微调成本。

相关配置：

- `dinov3/configs/train/dinov3_vit7b16_pretrain_channelvit.yaml`
- `dinov3/configs/train/dinov3_vit7b16_pretrain_lora_channelvit.yaml`
- `dinov3/docs/LORA_CHANNELVIT_USAGE.md`

### 3. WebDataset 训练入口

在 `dinov3/data/loaders.py` 中，`make_dataset()` 已支持 `wds:` 前缀的数据路径；当 `train.dataset_path` 以 `wds:` 开头时，会自动进入 WebDataset 流式加载分支。

当前训练侧新增模块：

- `dinov3/data/wds_pipeline.py`
- `dinov3/data/wds_decoder.py`
- `dinov3/data/wds_example.py`

特点：

- 自动识别 `IterableDataset`
- DataLoader 自动绕过标准 Sampler
- shuffle 由 WebDataset 管道内部完成
- 支持多通道图像解码为 `(C, H, W)`

### 4. 医学影像 WebDataset 归档流水线

`dinov3/dataset_webdataset/` 是这个项目里新增的核心数据工程模块，用于将 PostgreSQL/NFS 上的大规模医学影像整理成训练可用的 tar shards。

流水线目标：

- 从数据库读取样本元信息
- 基于 dedup 白名单进行高优先级过滤
- 对静态/动态 TIFF 做确定性张量重构
- 保留全部通道，只在帧维度上抽取目标切片
- 对超大图执行自适应裁切
- 过滤纯背景或低质量 patch
- 将结果写成 WebDataset tar 分片

### 5. Smart sharding 工具

`dinov3/smart_sharding/` 目录提供了一组更早期、偏任务拆分/预分桶的脚本，用于：

- 按分辨率聚类
- 按通道数拆分
- 为后续 worker 生成 JSON 任务文件

这部分适合做数据摸底、预切任务生成或离线分桶，不等同于最终的 tar 归档流水线。

## 仓库重点目录

```text
dinov3/
├── configs/train/
│   ├── dinov3_vit7b16_pretrain_channelvit.yaml
│   ├── dinov3_vit7b16_pretrain_lora_channelvit.yaml
│   └── dinov3_vit7b16_pretrain_webdataset.yaml
├── data/
│   ├── loaders.py
│   ├── wds_pipeline.py
│   ├── wds_decoder.py
│   └── wds_example.py
├── dataset_webdataset/
│   ├── run.py
│   ├── config.py
│   ├── db_client.py
│   ├── dedup_index.py
│   ├── tensor_static.py
│   ├── tensor_dynamic.py
│   ├── frame_extractor.py
│   ├── spatial_slicer.py
│   ├── quality_filter.py
│   ├── npy_namer.py
│   ├── shard_writer.py
│   └── README.md
├── smart_sharding/
│   ├── sharding.py
│   ├── sharding_multi_channel.py
│   ├── sharding_highNA.py
│   └── ...
├── docs/
│   └── LORA_CHANNELVIT_USAGE.md
└── models/
    └── vision_transformer.py
```

## 安装

基础环境沿用原始 DINOv3：

```shell
micromamba env create -f conda.yaml
micromamba activate dinov3
```

如果你要使用当前仓库里的多通道训练和数据归档能力，建议额外安装：

```shell
pip install webdataset tifffile psycopg2-binary ray rich
```

## 快速开始

### 1. 使用 ChannelViT 训练多通道数据

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py   --nodes 1   --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_channelvit.yaml   --output-dir <PATH/TO/OUTPUT/DIR>   train.dataset_path=<PATH/TO/YOUR/DATASET>   student.in_chans=4   teacher.in_chans=4   student.enable_channelvit=true   teacher.enable_channelvit=true
```

适用前提：

- 你的数据已经能被训练代码直接读取
- 输入张量形状为 `(B, C, H, W)`
- `C` 与 `student.in_chans` / `teacher.in_chans` 一致

### 2. 使用 LoRA + ChannelViT 微调

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py   --nodes 1   --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_lora_channelvit.yaml   --output-dir <PATH/TO/OUTPUT/DIR>   train.dataset_path=<PATH/TO/YOUR/DATASET>   student.resume_from_teacher_chkpt=<PATH/TO/CHECKPOINT>   student.in_chans=4   teacher.in_chans=4   student.enable_channelvit=true   teacher.enable_channelvit=true
```

适用场景：

- GPU 显存有限
- 需要做领域适配
- 不想全量微调 7B 主干

### 3. 使用 WebDataset 流式训练

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py   --nodes 1   --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_webdataset.yaml   --output-dir <PATH/TO/OUTPUT/DIR>   train.dataset_path="wds:/path/to/shards-{0000..0999}.tar"   student.in_chans=4   teacher.in_chans=4   student.enable_channelvit=true   teacher.enable_channelvit=true
```

当前逻辑说明：

- 训练侧通过 `dinov3/data/loaders.py` 自动识别 `wds:`
- `dinov3/data/wds_pipeline.py` 构建 iterable pipeline
- DataLoader 会自动关闭标准 sampler 路径

## 数据归档流水线

`dinov3/dataset_webdataset/` 面向海量医学影像数据归档，适合在正式训练前先把数据库/NFS 上的原始 TIFF 整理成 tar shards。

### 支持的数据来源

- `original_images_all`
- `original_image_all_2p_parsed`
- `--table all` 联合打包

### 入口命令

推荐在仓库根目录执行：

```shell
python -m dinov3.dataset_webdataset.run --table all --channels 4
python -m dinov3.dataset_webdataset.run --table all --channels 4 --chunk-size 800
python -m dinov3.dataset_webdataset.run --table all --channels 4 --mode local
```

参数说明：

- `--table`: `all`、`original_images_all`、`original_image_all_2p_parsed`
- `--channels`: 目标通道数
- `--mode`: `ray` 或 `local`
- `--chunk-size`: Ray 模式每个任务块包含的样本数
- `--max-target-size`: 切图目标边长

### 流水线过程

1. 数据库读取元数据
2. 读取 dedup 白名单并建立哈希集合
3. 根据静态/动态来源选择不同张量重构逻辑
4. 对动态序列提取目标帧，但始终保留全部通道
5. 对大图做空间裁切
6. 过滤低方差和低覆盖率背景块
7. 写入 tar shard

### 输出位置

默认输出目录定义在 `dinov3/dataset_webdataset/config.py`：

- 主路径：`/mnt/huawei_deepcad`
- 后备路径：`/mnt/deepcad_nfs`

写出的 shards 默认位于：

```text
/mnt/huawei_deepcad/wds_shards/
/mnt/deepcad_nfs/wds_shards/
```

## Smart Sharding

`dinov3/smart_sharding/` 里的脚本主要用于离线任务拆解和预分桶。

典型用途：

- 将不同通道数的数据拆成多个子任务
- 将长尾分辨率归入混合桶
- 生成供后续处理程序消费的 JSON shard/task 文件

例如：

- `sharding.py`: 单通道数据分组与 hybrid bucket 归档
- `sharding_multi_channel.py`: 多通道任务裂变，为每个 `extract_channel_idx` 生成独立任务

如果你的目标是“为 worker 生成任务清单”，看 `smart_sharding/`；如果目标是“直接写 tar 训练集”，看 `dataset_webdataset/`。

## 关键配置文件

### `dinov3_vit7b16_pretrain_channelvit.yaml`

用于多通道自监督预训练，关键字段：

- `student.in_chans`
- `teacher.in_chans`
- `student.enable_channelvit`
- `teacher.enable_channelvit`

### `dinov3_vit7b16_pretrain_lora_channelvit.yaml`

用于 LoRA + ChannelViT 微调，除上面的通道参数外，还新增：

- `lora.r`
- `lora.alpha`
- `lora.dropout`
- `lora.train_heads`

### `dinov3_vit7b16_pretrain_webdataset.yaml`

用于 WebDataset 流式训练，重点关注：

- `train.dataset_path`
- `train.num_workers`
- `student.in_chans`
- `teacher.in_chans`
- `student.enable_channelvit`
- `teacher.enable_channelvit`

## 当前仓库状态说明

这个仓库已经明显偏离原始官方 README 的结构，建议把它理解成一个“基于 DINOv3 的医学影像训练工程”。

当前需要注意两点：

1. `dinov3/data/wds_pipeline.py` 训练侧目前实现的是 TIFF 解码路径。
2. `dinov3/dataset_webdataset/` 归档侧当前写出的样本名和内容以 `.npy` 为主。

这意味着：

- 训练侧 WebDataset 入口已经有了
- 大规模归档侧也已经有了
- 但如果你要直接把 `dataset_webdataset` 产出的 `.npy` tar shards 无缝接到当前训练侧，还需要统一解码协议

## 与原始 DINOv3 的关系

本仓库保留了原始 DINOv3 的核心训练/评估代码和许可证信息，但 README 不再以官方预训练模型表和论文复现为中心。

如果你只是想了解这个项目，可以优先看：

- `dinov3/models/vision_transformer.py`
- `dinov3/data/loaders.py`
- `dinov3/data/wds_pipeline.py`
- `dinov3/dataset_webdataset/README.md`
- `dinov3/docs/LORA_CHANNELVIT_USAGE.md`

## 原始项目信息

基础项目来自 Meta AI Research 的 `DINOv3`：

- Paper: <https://arxiv.org/abs/2508.10104>
- Official project: <https://ai.meta.com/dinov3/>

## License

本仓库代码与模型相关内容遵循 DINOv3 License。详情见 `LICENSE.md`。

## Citation

如果你在学术工作中使用了这个仓库，建议同时引用原始 DINOv3：

```bibtex
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{'e}e and Moutakanni, Th{'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{'e}gou, Herv{'e} and Labatut, Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104},
}
```
