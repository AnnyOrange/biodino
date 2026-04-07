# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import random

import numpy as np
import torch
from torch import nn, Tensor
from torchvision.transforms import v2

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, GaussianBlur

logger = logging.getLogger("dinov3")


# ---------------------------------------------------------------------------
# Channel-agnostic replacements for torchvision v2 RGB-only transforms.
# torchvision v2 ColorJitter / RandomGrayscale / RandomSolarize all enforce
# channels ∈ {1, 3}.  The implementations below operate on arbitrary C.
# ---------------------------------------------------------------------------

class ChannelAgnosticColorJitter(nn.Module):
    """Per-channel brightness & contrast jitter that works for any C.

    Unlike torchvision ColorJitter we skip saturation and hue which are
    inherently RGB concepts.  For multi-channel bio images, per-channel
    brightness/contrast perturbation provides analogous regularisation.
    """

    def __init__(self, brightness: float = 0.4, contrast: float = 0.4, p: float = 0.8):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def forward(self, img: Tensor) -> Tensor:
        if random.random() > self.p:
            return img
        was_uint8 = not img.is_floating_point()
        if was_uint8:
            img = img.to(torch.float32) / 255.0
        C = img.shape[-3]
        if random.random() > 0.5:
            factors = 1.0 + (torch.rand(C, 1, 1, device=img.device) * 2 - 1) * self.brightness
            img = img * factors
        if random.random() > 0.5:
            means = img.mean(dim=(-2, -1), keepdim=True)
            factors = 1.0 + (torch.rand(C, 1, 1, device=img.device) * 2 - 1) * self.contrast
            img = (img - means) * factors + means
        img = img.clamp(0.0, 1.0)
        if was_uint8:
            img = (img * 255.0).to(torch.uint8)
        return img


class ChannelAgnosticRandomGrayscale(nn.Module):
    """Average all channels → repeat back to C.  Works for any C."""

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, img: Tensor) -> Tensor:
        if random.random() > self.p:
            return img
        gray = img.mean(dim=-3, keepdim=True)
        return gray.expand_as(img)


class ChannelAgnosticRandomSolarize(nn.Module):
    """Invert pixels above *threshold*.  Works for any C and dtype."""

    def __init__(self, threshold: float = 0.5, p: float = 0.2):
        super().__init__()
        self.threshold = threshold
        self.p = p

    def forward(self, img: Tensor) -> Tensor:
        if random.random() > self.p:
            return img
        if img.is_floating_point():
            inverted = 1.0 - img
            threshold = self.threshold
        else:
            inverted = 255 - img.to(torch.int16)
            inverted = inverted.to(img.dtype)
            threshold = self.threshold
        return torch.where(img >= threshold, inverted, img)


# ---------------------------------------------------------------------------
# Main augmentation class
# ---------------------------------------------------------------------------

class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        float_input=False,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std
        self.float_input = float_input
        self._logged_channel_stats_adapt = False

        solarize_threshold = 0.5 if float_input else 128

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info(f"float_input: {float_input} (solarize threshold: {solarize_threshold})")
        logger.info("Using channel-agnostic color augmentations (supports 1-N channels)")
        logger.info("###################################")

        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()
        self.resize_global_post_transf = nn.Identity()
        self.resize_gram_teacher = None
        if gram_teacher_crops_size is not None:
            if gram_teacher_no_distortions:
                resize_global = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            else:
                self.resize_global_post_transf = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            self.resize_gram_teacher = v2.Resize(
                gram_teacher_crops_size,
                interpolation=v2.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # Channel-agnostic color distortions
        color_jittering = v2.Compose(
            [
                ChannelAgnosticColorJitter(brightness=0.4, contrast=0.4, p=0.8),
                ChannelAgnosticRandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)
        global_transfo2_extra = v2.Compose(
            [
                GaussianBlur(p=0.1),
                ChannelAgnosticRandomSolarize(threshold=solarize_threshold, p=0.2),
            ]
        )
        local_transfo_extra = GaussianBlur(p=0.5)

        if float_input:
            pre_norm_clamp = v2.Lambda(lambda x: x.clamp(0.0, 1.0))
        else:
            pre_norm_clamp = nn.Identity()

        self.normalize = v2.Compose(
            [
                pre_norm_clamp,
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(self._normalize_with_adaptive_stats),
            ]
        )

        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = v2.Compose([resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = v2.Compose([resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = v2.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = v2.Compose(
                [resize_global, color_jittering, global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = v2.Compose(
                [resize_global, color_jittering, global_transfo2_extra, self.normalize]
            )
            self.local_transfo = v2.Compose([color_jittering, local_transfo_extra, self.normalize])

    # ------------------------------------------------------------------

    def _adapt_stats_to_channels(self, stats, channels: int):
        stats_list = list(stats)
        if len(stats_list) == channels:
            return stats_list
        if len(stats_list) == 0:
            return [0.0] * channels
        if len(stats_list) > channels:
            return stats_list[:channels]
        repeats = (channels + len(stats_list) - 1) // len(stats_list)
        expanded = (stats_list * repeats)[:channels]
        if not self._logged_channel_stats_adapt:
            logger.info(
                f"Normalize stats length ({len(stats_list)}) != input channels ({channels}); "
                f"cycling stats to {channels} channels."
            )
            self._logged_channel_stats_adapt = True
        return expanded

    def _normalize_with_adaptive_stats(self, image: Tensor) -> Tensor:
        channels = int(image.shape[-3])
        mean = torch.tensor(
            self._adapt_stats_to_channels(self.mean, channels),
            dtype=image.dtype, device=image.device,
        ).view(-1, 1, 1)
        std = torch.tensor(
            self._adapt_stats_to_channels(self.std, channels),
            dtype=image.dtype, device=image.device,
        ).view(-1, 1, 1)
        std = torch.clamp(std, min=1e-6)
        return (image - mean) / std

    # ------------------------------------------------------------------

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True

        if self.share_color_jitter:
            image = self.color_jittering(image)

        # global crops
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image))
                for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        return output
