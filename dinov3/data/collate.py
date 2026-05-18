# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import random

import torch


def _pad_crop_channels(crop, target_channels):
    current_channels = crop.shape[0]
    if current_channels == target_channels:
        return crop
    if current_channels > target_channels:
        raise ValueError(
            f"crop has {current_channels} channels, target padding is {target_channels}"
        )
    padding_shape = (target_channels - current_channels, *crop.shape[1:])
    return torch.cat([crop, crop.new_zeros(padding_shape)], dim=0)


def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    random_circular_shift=False,
    local_batch_size=None,
):
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])
    has_channel_ids = "channel_ids" in samples_list[0][0]
    max_sample_channels = (
        max(int(s[0]["channel_ids"].shape[0]) for s in samples_list)
        if has_channel_ids
        else None
    )

    if has_channel_ids:
        collated_global_crops = torch.stack(
            [
                _pad_crop_channels(s[0]["global_crops"][i], max_sample_channels)
                for i in range(n_global_crops)
                for s in samples_list
            ]
        )
        collated_local_crops = torch.stack(
            [
                _pad_crop_channels(s[0]["local_crops"][i], max_sample_channels)
                for i in range(n_local_crops)
                for s in samples_list
            ]
        )
    else:
        collated_global_crops = torch.stack(
            [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
        )  # [n_global_crops, B, ...]
        collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    if "gram_teacher_crops" in samples_list[0][0]:
        if has_channel_ids:
            collated_gram_teacher_crops = torch.stack(
                [
                    _pad_crop_channels(s[0]["gram_teacher_crops"][i], max_sample_channels)
                    for i in range(n_global_crops)
                    for s in samples_list
                ]
            )
        else:
            collated_gram_teacher_crops = torch.stack(
                [s[0]["gram_teacher_crops"][i] for i in range(n_global_crops) for s in samples_list]
            )  # [n_global_crops, B, ...]
    else:
        collated_gram_teacher_crops = None

    if has_channel_ids:
        padded_channel_ids = []
        channel_valid_masks = []
        for s in samples_list:
            ids = s[0]["channel_ids"]
            valid = torch.zeros(max_sample_channels, dtype=torch.bool)
            valid[: ids.shape[0]] = True
            if ids.shape[0] < max_sample_channels:
                pad = ids.new_zeros(max_sample_channels - ids.shape[0])
                ids = torch.cat([ids, pad], dim=0)
            padded_channel_ids.append(ids)
            channel_valid_masks.append(valid)
        collated_global_channel_ids = torch.stack(
            [ids for _ in range(n_global_crops) for ids in padded_channel_ids]
        )
        collated_local_channel_ids = torch.stack(
            [ids for _ in range(n_local_crops) for ids in padded_channel_ids]
        )
        collated_global_channel_valid_mask = torch.stack(
            [valid for _ in range(n_global_crops) for valid in channel_valid_masks]
        )
        collated_local_channel_valid_mask = torch.stack(
            [valid for _ in range(n_local_crops) for valid in channel_valid_masks]
        )
        if collated_gram_teacher_crops is not None:
            collated_gram_teacher_channel_ids = torch.stack(
                [ids for _ in range(n_global_crops) for ids in padded_channel_ids]
            )
            collated_gram_teacher_channel_valid_mask = torch.stack(
                [valid for _ in range(n_global_crops) for valid in channel_valid_masks]
            )
        else:
            collated_gram_teacher_channel_ids = None
            collated_gram_teacher_channel_valid_mask = None
    else:
        collated_global_channel_ids = None
        collated_local_channel_ids = None
        collated_global_channel_valid_mask = None
        collated_local_channel_valid_mask = None
        collated_gram_teacher_channel_ids = None
        collated_gram_teacher_channel_valid_mask = None

    if local_batch_size is not None:
        # multi-distillation case, number of masks is different because the number of samples masked
        # is different of the number of samples passed into the teacher initially
        B = n_global_crops * local_batch_size
    else:
        B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(N * prob_max)))
        if random_circular_shift:  # apply le random circular shift to
            shift_x, shift_y = (
                random.randint(0, mask.shape[0] - 1),
                random.randint(0, mask.shape[1] - 1),
            )
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list.append(mask)
        upperbound += int(N * prob_max)
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    out = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
    if collated_gram_teacher_crops is not None:
        out["collated_gram_teacher_crops"] = collated_gram_teacher_crops.to(dtype)
    if collated_global_channel_ids is not None:
        out["collated_global_channel_ids"] = collated_global_channel_ids
        out["collated_local_channel_ids"] = collated_local_channel_ids
        out["collated_global_channel_valid_mask"] = collated_global_channel_valid_mask
        out["collated_local_channel_valid_mask"] = collated_local_channel_valid_mask
        if collated_gram_teacher_channel_ids is not None:
            out["collated_gram_teacher_channel_ids"] = collated_gram_teacher_channel_ids
            out["collated_gram_teacher_channel_valid_mask"] = collated_gram_teacher_channel_valid_mask
    return out


# def get_batch_subset(collated_data_batch, target_bs):
def get_batch_subset(collated_data_batch, divide_by):
    old_bs = collated_data_batch["collated_global_crops"].shape[0] // 2
    target_bs = (old_bs + divide_by - 1) // divide_by
    collated_global_crops = (
        collated_data_batch["collated_global_crops"].unflatten(0, (2, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )
    collated_local_crops = (
        collated_data_batch["collated_local_crops"].unflatten(0, (-1, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )
    if "collated_global_channel_ids" in collated_data_batch:
        collated_global_channel_ids = (
            collated_data_batch["collated_global_channel_ids"]
            .unflatten(0, (2, old_bs))
            .narrow(1, 0, target_bs)
            .flatten(0, 1)
        )
        collated_local_channel_ids = (
            collated_data_batch["collated_local_channel_ids"]
            .unflatten(0, (-1, old_bs))
            .narrow(1, 0, target_bs)
            .flatten(0, 1)
        )
        collated_global_channel_valid_mask = (
            collated_data_batch["collated_global_channel_valid_mask"]
            .unflatten(0, (2, old_bs))
            .narrow(1, 0, target_bs)
            .flatten(0, 1)
        )
        collated_local_channel_valid_mask = (
            collated_data_batch["collated_local_channel_valid_mask"]
            .unflatten(0, (-1, old_bs))
            .narrow(1, 0, target_bs)
            .flatten(0, 1)
        )
    else:
        collated_global_channel_ids = None
        collated_local_channel_ids = None
        collated_global_channel_valid_mask = None
        collated_local_channel_valid_mask = None

    masks_old_bs = collated_data_batch["collated_masks"].shape[0] // 2
    masks_target_bs = masks_old_bs // divide_by
    collated_masks = (
        collated_data_batch["collated_masks"]
        .unflatten(0, (2, masks_old_bs))
        .narrow(1, 0, masks_target_bs)
        .flatten(0, 1)
    )
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    while mask_indices_list.shape[0] == 0:
        _unbind = list(collated_data_batch["collated_masks"].unbind(0))
        random.shuffle(_unbind)
        _bind = torch.stack(_unbind, dim=0)
        collated_masks = _bind.unflatten(0, (2, masks_old_bs)).narrow(1, 0, masks_target_bs).flatten(0, 1)
        mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    upperbound = collated_data_batch["upperbound"]

    new_batch = {
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": collated_local_crops,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }

    if "global_batch_size" in collated_data_batch.keys():
        new_batch["global_batch_size"] = collated_data_batch["global_batch_size"] // divide_by
    if collated_global_channel_ids is not None:
        new_batch["collated_global_channel_ids"] = collated_global_channel_ids
        new_batch["collated_local_channel_ids"] = collated_local_channel_ids
        new_batch["collated_global_channel_valid_mask"] = collated_global_channel_valid_mask
        new_batch["collated_local_channel_valid_mask"] = collated_local_channel_valid_mask

    return new_batch
