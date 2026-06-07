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
    wds_shuffle_buffer: int = 1000,
):
    """
    Creates a dataset with the specified parameters.

    Prefix-based routing (resolved once at init, not on the hot path):
        - no prefix          → DINOv3 native dataset (ImageNet:split=TRAIN …)
        - ``packwds:``       → packed multi-channel shards from ``data/repackage``.
                               Each sample bundles all channels (``ch1.tif`` …
                               ``chN.tif``) in one tar entry; missing channels
                               are zero-filled.  Requires ``target_channels``
                               to match ``student.in_chans`` in the YAML.
                               Example:
                               ``packwds:/data/packed/filtered_mixed_train_w*-{000000..000999}.tar``
        - ``packwds_chvit:`` → packed ChannelViT shards.  Samples a fixed-size
                               subset from actually present channels and returns
                               channel ids; no dataset-level zero padding/copying.
                               Optional suffix: ``::sample_channels=K`` caps
                               the number of sampled channels per sample.
        - ``packwds_robust:`` → like ``packwds:`` but per-channel percentile
                               normalization + single-channel replication
                               (grayscale → replicated, not zero-filled).
                               Optional suffix ``::pct=low,high`` (default 1,99).

    Args:
        dataset_str: Dataset descriptor string.
        transform: Image transform.
        target_transform: Target transform.
        transforms: Joint image+target transform.
        target_channels: Align decoded samples to this channel count (WebDataset only).

    Returns:
        The created dataset.
    """
    logger.info(f'using dataset: "{dataset_str}"')

    if dataset_str.startswith("packwds_robust:"):
        return _make_packed_robust_webdataset(
            dataset_str[len("packwds_robust:") :],
            transform,
            target_channels=target_channels,
            shuffle_buffer=wds_shuffle_buffer,
        )

    if dataset_str.startswith("packwds_chvit:"):
        return _make_packed_channelvit_webdataset(
            dataset_str[len("packwds_chvit:") :],
            transform,
            target_channels=target_channels,
            shuffle_buffer=wds_shuffle_buffer,
        )

    if dataset_str.startswith("packwds:"):
        return _make_packed_webdataset(
            dataset_str[8:],
            transform,
            target_channels=target_channels,
            shuffle_buffer=wds_shuffle_buffer,
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


def _make_packed_webdataset(
    shard_spec: str,
    transform: Optional[Callable] = None,
    target_channels: Optional[int] = None,
    shuffle_buffer: int = 1000,
):
    """Create a pipeline for packed multi-channel shards (``packwds:`` prefix).

    Packed shards are produced by ``data/repackage``.  Each sample contains
    ``ch<N>.tif`` files for all available channels plus a ``meta.json``.
    Missing channels are zero-filled to produce a fixed
    ``(target_channels, H, W)`` tensor.

    The shard spec is resolved in two stages:
    1. Brace expansion: ``{000000..000999}`` → individual numbers.
    2. Shell glob expansion: ``w*`` → all matching worker prefixes.

    This means patterns like
    ``/data/packed/filtered_mixed_train_w*-{000000..000999}.tar``
    work correctly even when worker-numbered files exist on disk.

    Args:
        shard_spec: Brace/glob pattern or ``;``-separated list of patterns.
        transform: DINOv3-style transform.
        target_channels: Number of output channels (default 8).
            Must match ``student.in_chans`` / ``teacher.in_chans`` in the YAML.

    Returns:
        WebDataset IterableDataset pipeline.
    """
    from .wds_pipeline import WdsConfig, build_packed_wds_pipeline

    raw_patterns = [s.strip() for s in shard_spec.split(";") if s.strip()]
    shard_urls: List[str] = _expand_shard_patterns(raw_patterns)

    if not shard_urls:
        raise FileNotFoundError(
            f"packwds: no tar shards found matching: {shard_spec}\n"
            "Check that the output directory exists and the pattern is correct."
        )

    logger.info(
        "creating packed WebDataset from %d pattern(s) → %d shards",
        len(raw_patterns),
        len(shard_urls),
    )

    effective_channels = target_channels or 8
    config = WdsConfig(
        shard_urls=shard_urls,
        shuffle_buffer=shuffle_buffer,
        target_channels=effective_channels,
    )
    pipeline = build_packed_wds_pipeline(config, transform=transform)
    logger.info(
        "Packed WebDataset pipeline created (target_channels=%d)", effective_channels
    )
    return pipeline


def _make_packed_channelvit_webdataset(
    shard_spec: str,
    transform: Optional[Callable] = None,
    target_channels: Optional[int] = None,
    shuffle_buffer: int = 1000,
):
    """Create a true ChannelViT pipeline for packed multi-channel shards.

    ``target_channels`` is interpreted as the channel embedding table capacity
    / maximum accepted channel id count.  ``sample_channels`` optionally caps
    the number of real channels sampled per sample.  Samples with fewer
    channels are kept with their actual channel count.
    """
    from .wds_pipeline import WdsConfig, build_packed_channelvit_wds_pipeline

    sample_channels = None
    if "::" in shard_spec:
        shard_spec, opts_str = shard_spec.rsplit("::", 1)
        for opt in [o.strip() for o in opts_str.split(",") if o.strip()]:
            key, value = opt.split("=", 1)
            if key == "sample_channels":
                sample_channels = int(value)
            else:
                raise ValueError(f"Unknown packwds_chvit option: {key}")

    raw_patterns = [s.strip() for s in shard_spec.split(";") if s.strip()]
    shard_urls: List[str] = _expand_shard_patterns(raw_patterns)
    if not shard_urls:
        raise FileNotFoundError(
            f"packwds_chvit: no tar shards found matching: {shard_spec}\n"
            "Check that the output directory exists and the pattern is correct."
        )

    max_channels = target_channels or 8
    if sample_channels is not None and sample_channels > max_channels:
        raise ValueError(
            f"packwds_chvit sample_channels ({sample_channels}) must be <= "
            f"target_channels/student.in_chans ({max_channels})"
        )

    config = WdsConfig(
        shard_urls=shard_urls,
        shuffle_buffer=shuffle_buffer,
        target_channels=max_channels,
    )
    pipeline = build_packed_channelvit_wds_pipeline(
        config,
        transform=transform,
        sample_channels=sample_channels,
    )
    logger.info(
        "Packed ChannelViT WebDataset pipeline created (max_channels=%d, sample_channels=%s)",
        max_channels,
        sample_channels if sample_channels is not None else "all-present",
    )
    return pipeline


def _make_packed_robust_webdataset(
    shard_spec: str,
    transform: Optional[Callable] = None,
    target_channels: Optional[int] = None,
    shuffle_buffer: int = 1000,
):
    """Create a pipeline for packed shards with robust per-channel normalization
    (``packwds_robust:`` prefix).

    Same packed shards as ``packwds:`` but each channel is percentile-clipped
    and rescaled to [0, 1], and single-channel samples are replicated across
    ``target_channels`` instead of zero-filled.  Optional suffix
    ``::pct=low,high`` sets the clip percentiles (default ``1,99``).

    The original ``packwds:`` path (:func:`_make_packed_webdataset`) is unchanged.
    """
    from .wds_pipeline import WdsConfig, build_packed_robust_wds_pipeline

    p_low, p_high = 1.0, 99.0
    if "::" in shard_spec:
        shard_spec, opts_str = shard_spec.rsplit("::", 1)
        opts_str = opts_str.strip()
        if opts_str.startswith("pct="):
            vals = [v.strip() for v in opts_str[len("pct=") :].split(",")]
            if len(vals) != 2:
                raise ValueError(
                    f"packwds_robust: ::pct expects 'pct=low,high', got: {opts_str}"
                )
            p_low, p_high = float(vals[0]), float(vals[1])
        elif opts_str:
            raise ValueError(f"Unknown packwds_robust option: {opts_str}")

    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError(
            f"packwds_robust: pct must satisfy 0 <= low < high <= 100, got {p_low},{p_high}"
        )

    raw_patterns = [s.strip() for s in shard_spec.split(";") if s.strip()]
    shard_urls: List[str] = _expand_shard_patterns(raw_patterns)
    if not shard_urls:
        raise FileNotFoundError(
            f"packwds_robust: no tar shards found matching: {shard_spec}\n"
            "Check that the output directory exists and the pattern is correct."
        )

    effective_channels = target_channels or 8
    config = WdsConfig(
        shard_urls=shard_urls,
        shuffle_buffer=shuffle_buffer,
        target_channels=effective_channels,
    )
    pipeline = build_packed_robust_wds_pipeline(
        config, transform=transform, p_low=p_low, p_high=p_high
    )
    logger.info(
        "Packed ROBUST WebDataset pipeline created (target_channels=%d, pct=[%s,%s])",
        effective_channels,
        p_low,
        p_high,
    )
    return pipeline


def _expand_shard_patterns(patterns: List[str]) -> List[str]:
    """Expand brace expressions and shell globs; return sorted, deduplicated paths.

    Handles patterns that mix both notations, e.g.::

        /data/packed/filtered_mixed_train_w*-{000000..000999}.tar

    Steps:
      1. ``braceexpand`` turns ``{a..b}`` into individual strings.
      2. ``glob.glob`` resolves ``*``, ``?``, ``[...]`` against the filesystem.
      3. Results are sorted and deduplicated.
    """
    import glob as _glob

    from braceexpand import braceexpand

    resolved: List[str] = []
    for pattern in patterns:
        brace_expanded = list(braceexpand(pattern))
        for bp in brace_expanded:
            if any(c in bp for c in ("*", "?", "[")):
                matches = sorted(_glob.glob(bp))
                resolved.extend(matches)
            else:
                resolved.append(bp)

    # Deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for path in resolved:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


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
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = None,
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
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
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
    loader_kwargs = dict(
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    data_loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

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
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = None,
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

    loader_kwargs = dict(
        sampler=None,  # WebDataset 不使用 Sampler
        shuffle=False,  # 强制关闭，shuffle 由 WebDataset 内部处理
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    data_loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    logger.info("WebDataset DataLoader created (dataset controls finiteness/infinite streaming)")
    return data_loader
