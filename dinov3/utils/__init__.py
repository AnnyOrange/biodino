# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .dtype import as_torch_dtype
from .utils import (
    cat_keep_shapes,
    count_parameters,
    fix_random_seeds,
    get_conda_env,
    get_sha,
    named_apply,
    named_replace,
    uncat_with_shapes,
)
from .lora_utils import (
    apply_lora_to_model,
    apply_lora_to_vit_backbone,
    freeze_non_lora_params,
    get_lora_params,
    get_trainable_params_info,
    load_lora_checkpoint,
    merge_lora_weights,
    save_lora_checkpoint,
    unmerge_lora_weights,
)
