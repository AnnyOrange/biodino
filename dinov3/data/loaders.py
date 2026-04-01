# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar, Union

import torch
from torch.utils.data import Sampler

from .datasets import ADE20K, CocoCaptions, ImageNet, ImageNet22k, NYU
from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler
from .wds_pipeline import is_webdataset

logger = logging.getLogger("dinov3")


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


def _make_bool_str(b: bool) -> str:
    return "yes" if b else "no"


def _make_sample_transform(
    image_transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    def transform(sample):
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def _parse_dataset_str(dataset_str: str):
    tokens = dataset_str.split(":")

    name = tokens[0]
    kwargs = {}

    for token in tokens[1:]:
        key, value = token.split("=")
        assert key in ("root", "extra", "split")
        kwargs[key] = value

    if name == "ImageNet":
        class_ = ImageNet
        if "split" in kwargs:
            kwargs["split"] = ImageNet.Split[kwargs["split"]]
    elif name == "ImageNet22k":
        class_ = ImageNet22k
    elif name == "ADE20K":
        class_ = ADE20K
        if "split" in kwargs:
            kwargs["split"] = ADE20K.Split[kwargs["split"]]
    elif name == "CocoCaptions":
        class_ = CocoCaptions
        if "split" in kwargs:
            kwargs["split"] = CocoCaptions.Split[kwargs["split"]]
    elif name == "NYU":
        class_ = NYU
        if "split" in kwargs:
            kwargs["split"] = NYU.Split[kwargs["split"]]
    else:
        raise ValueError(f'Unsupported dataset "{name}"')

    return class_, kwargs


def make_dataset(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    transforms: Optional[Callable] = None,
    target_channels: Optional[int] = None,
    s3_cache_root: Optional[str] = None,
    aws_profile: Optional[str] = None,
    aws_region: Optional[str] = None,
):
    """
    Creates a dataset with the specified parameters.

    Prefix-based routing (resolved once at init, not on the hot path):
        - no prefix          → DINOv3 native dataset (ImageNet:split=TRAIN …)
        - ``wds:``           → local WebDataset shards
        - ``s3wds:``         → S3 streaming WebDataset (pipe:aws s3 cp …)
        - ``cachewds:``      → download S3 shards to *s3_cache_root*, then
                               read as local WebDataset

    Args:
        dataset_str: Dataset descriptor string.
        transform: Image transform.
        target_transform: Target transform.
        transforms: Joint image+target transform.
        target_channels: Align decoded samples to this channel count (WebDataset only).
        s3_cache_root: Local cache directory for ``cachewds:`` mode.
        aws_profile: AWS CLI profile name for S3 modes (optional).
        aws_region: AWS region for S3 modes (optional).

    Returns:
        The created dataset.
    """
    logger.info(f'using dataset: "{dataset_str}"')

    if dataset_str.startswith("wds:"):
        return _make_webdataset(dataset_str[4:], transform, target_channels=target_channels)

    if dataset_str.startswith("s3wds:"):
        return _make_s3_stream_webdataset(
            dataset_str[6:], transform,
            target_channels=target_channels,
            aws_profile=aws_profile,
            aws_region=aws_region,
        )

    if dataset_str.startswith("cachewds:"):
        return _make_s3_cache_webdataset(
            dataset_str[9:], transform,
            target_channels=target_channels,
            cache_root=s3_cache_root,
            aws_profile=aws_profile,
            aws_region=aws_region,
        )

    # DINOv3 native dataset path
    class_, kwargs = _parse_dataset_str(dataset_str)
    dataset = class_(transform=transform, target_transform=target_transform, transforms=transforms, **kwargs)

    logger.info(f"# of dataset samples: {len(dataset):,d}")

    if not hasattr(dataset, "transform"):
        dataset.transform = transform
    if not hasattr(dataset, "target_transform"):
        dataset.target_transform = target_transform
    if not hasattr(dataset, "transforms"):
        dataset.transforms = transforms

    return dataset


def _make_webdataset(
    shard_urls: Union[str, List[str]],
    transform: Optional[Callable] = None,
    target_channels: Optional[int] = None,
):
    """Create a WebDataset pipeline from local shards or pre-resolved URL list.

    Args:
        shard_urls: Brace pattern string **or** explicit list of URLs
            (local paths, ``pipe:`` URLs, etc.).
        transform: Image transform.
        target_channels: Align decoded samples to this channel count.

    Returns:
        WebDataset IterableDataset pipeline.
    """
    from .wds_pipeline import WdsConfig, build_wds_pipeline

    logger.info(f"creating WebDataset from: {shard_urls if isinstance(shard_urls, str) else f'{len(shard_urls)} URLs'}")

    config = WdsConfig(
        shard_urls=shard_urls,
        shuffle_buffer=1000,
        target_channels=target_channels,
    )
    pipeline = build_wds_pipeline(config, transform=transform)

    logger.info("WebDataset pipeline created (infinite streaming)")
    return pipeline


def _make_s3_stream_webdataset(
    s3_shard_pattern: str,
    transform: Optional[Callable] = None,
    target_channels: Optional[int] = None,
    aws_profile: Optional[str] = None,
    aws_region: Optional[str] = None,
):
    """S3 streaming mode: expand pattern → pipe:aws s3 cp URLs → WebDataset."""
    from .s3_utils import resolve_s3_stream_urls

    pipe_urls = resolve_s3_stream_urls(s3_shard_pattern, profile=aws_profile, region=aws_region)
    return _make_webdataset(pipe_urls, transform, target_channels=target_channels)


def _make_s3_cache_webdataset(
    s3_shard_pattern: str,
    transform: Optional[Callable] = None,
    target_channels: Optional[int] = None,
    cache_root: Optional[str] = None,
    aws_profile: Optional[str] = None,
    aws_region: Optional[str] = None,
):
    """S3 cache mode: download shards to *cache_root*, then read locally."""
    from .s3_utils import sync_s3_to_cache

    if not cache_root:
        raise ValueError(
            "cachewds: mode requires 'train.s3_cache_root' in config — "
            "set it to a local directory for shard caching"
        )

    local_paths = sync_s3_to_cache(
        s3_shard_pattern, cache_root, profile=aws_profile, region=aws_region,
    )
    return _make_webdataset(local_paths, transform, target_channels=target_channels)


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
    worker_init_fn: Optional[Callable[[List[T]], Any]] = None,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        sampler_advance: How many samples to skip (when applicable).
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
        worker_init_fn: Optional init function for each dataloader worker.
    """
    # WebDataset (IterableDataset) 兼容处理：绕过 Sampler
    if is_webdataset(dataset):
        return _make_webdataset_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
        )

    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
        advance=sampler_advance,
    )

    logger.info("using PyTorch data loader")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader


def _make_webdataset_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable] = None,
    worker_init_fn: Optional[Callable] = None,
) -> torch.utils.data.DataLoader:
    """
    为 WebDataset (IterableDataset) 创建 DataLoader。

    WebDataset 自带 shuffle 和分布式支持，必须绕过 Sampler。

    Args:
        dataset: WebDataset IterableDataset 管道。
        batch_size: 批次大小。
        num_workers: 工作进程数。
        drop_last: 是否丢弃最后不完整的批次。
        persistent_workers: 是否保持工作进程存活。
        collate_fn: 批次整理函数。
        worker_init_fn: 工作进程初始化函数。

    Returns:
        配置好的 DataLoader。
    """
    logger.info("using WebDataset (IterableDataset) data loader")
    logger.info("sampler: none (WebDataset handles shuffling internally)")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=None,  # WebDataset 不使用 Sampler
        shuffle=False,  # 强制关闭，shuffle 由 WebDataset 内部处理
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )

    logger.info("infinite WebDataset data loader")
    return data_loader
