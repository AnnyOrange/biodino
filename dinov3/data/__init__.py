# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .adapters import DatasetWithEnumeratedTargets
from .augmentations import DataAugmentationDINO
from .collate import collate_data_and_cast
from .loaders import SamplerType, make_data_loader, make_dataset
from .meta_loaders import CombinedDataLoader
from .masking import MaskingGenerator
from .transforms import make_classification_eval_transform, make_classification_train_transform
from .s3_utils import resolve_s3_stream_urls, sync_s3_to_cache
from .wds_decoder import create_tiff_decoder, decode_tiff_bytes
from .wds_pipeline import WdsConfig, build_wds_pipeline, is_webdataset
