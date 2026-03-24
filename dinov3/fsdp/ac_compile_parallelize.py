# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Any, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import register_fsdp_forward_method
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
from torch.utils.checkpoint import create_selective_checkpoint_contexts

from dinov3.utils import utils

logger = logging.getLogger("dinov3")


def _build_world_mesh(cfg) -> DeviceMesh:
    """Build a 1-D (pure FSDP) or 2-D (HSDP) DeviceMesh.

    Configure via ``compute_precision.hsdp_shards`` in the YAML:

    * ``hsdp_shards: 1``  (default) — classic FSDP, model sharded across ALL
      world_size GPUs.  One all-gather/reduce-scatter per layer per step
      involving every GPU → high communication overhead with many cards.

    * ``hsdp_shards: N``  (N > 1) — Hybrid Sharded Data Parallel.
      GPUs are grouped into ``world_size // N`` replica groups of N cards each.
      Within each group, parameters are sharded (FSDP).
      Across groups, gradients are all-reduced (DDP).

    Example with 8 GPUs and hsdp_shards=2:
        Mesh (4, 2):
          dp_replicate dim (size 4): groups [0,1], [2,3], [4,5], [6,7]
          dp_shard     dim (size 2): within each group, model cut in 2

        Communication pattern:
          - param all-gather/reduce-scatter:  2-card bandwidth  (fast)
          - gradient all-reduce:              4-card bandwidth  (moderate)
        vs. pure FSDP: every op touches all 8 cards.
    """
    world_size = dist.get_world_size()
    hsdp_shards = getattr(cfg.compute_precision, "hsdp_shards", 1)

    if hsdp_shards > 1:
        if world_size % hsdp_shards != 0:
            raise ValueError(
                f"hsdp_shards={hsdp_shards} does not evenly divide "
                f"world_size={world_size}. Choose a divisor of {world_size}."
            )
        dp_replicas = world_size // hsdp_shards
        mesh = init_device_mesh(
            "cuda",
            mesh_shape=(dp_replicas, hsdp_shards),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        logger.info(
            f"HSDP mesh: {dp_replicas} replica-groups × {hsdp_shards} shards/group  "
            f"(world_size={world_size})"
        )
    else:
        mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size,),
            mesh_dim_names=("dp",),
        )
        logger.info(f"FSDP mesh: 1-D, {world_size} shards")

    return mesh


def map_modules_and_blocks(models: list[nn.ModuleDict], callable) -> None:
    for m in models:
        assert isinstance(m, nn.ModuleDict)
        for k in m.keys():
            if k == "backbone":
                assert isinstance(m[k].blocks, nn.ModuleList)
                for block_id, block in enumerate(m[k].blocks):
                    m[k].blocks[block_id] = callable(block, is_backbone_block=True)
            else:
                m[k] = callable(m[k], is_backbone_block=False)


def ac_compile_parallelize(
    trained_model: nn.ModuleDict,
    inference_only_models: List[nn.ModuleDict],
    cfg: Any,
    trained_model_process_group: Optional[dist.ProcessGroup] = None,
    inference_only_models_process_groups: Optional[List[dist.ProcessGroup]] = None,
) -> None:
    """
    Order of the wrappers:
    1/ Activation checkpointing on blocks
    2/ Compile blocks
    3/ FSDP blocks + global model
    """
    assert (
        isinstance(trained_model, nn.ModuleDict) and "backbone" in trained_model.keys()
    ), f"{trained_model} does not contain a backbone?"
    logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
    if utils.has_batchnorms(trained_model):
        raise NotImplementedError

    # 1/ AC on blocks
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    backbone = trained_model.backbone
    if cfg.train.checkpointing:
        if cfg.train.checkpointing_full:
            _checkpointing_wrapper = checkpoint_wrapper
            logger.info("using selective checkpointing on backbone with full checkpointing policy")
        else:
            _save_list = [
                # mm
                torch.ops.aten.mm.default,
                torch.ops.aten._scaled_mm.default,
                # attentions
                torch.ops.aten._scaled_dot_product_efficient_attention.default,
                torch.ops.aten._scaled_dot_product_flash_attention.default,
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
            ]
            _checkpointing_wrapper = partial(
                checkpoint_wrapper,
                context_fn=partial(create_selective_checkpoint_contexts, _save_list),
                preserve_rng_state=True,
            )
            logger.info("using selective checkpointing on backbone with selective policy")
        for i, b in enumerate(backbone.blocks):
            backbone.blocks[i] = _checkpointing_wrapper(b)

    # 2/ Compile blocks
    all_models = [trained_model] + inference_only_models
    if trained_model_process_group is None and inference_only_models_process_groups is None:
        all_pgs = [None] * len(all_models)
    elif trained_model_process_group is None:
        all_pgs = [None] + inference_only_models_process_groups
    elif inference_only_models_process_groups is None:
        all_pgs = [trained_model_process_group] + [None] * len(inference_only_models_process_groups)
    else:
        all_pgs = [trained_model_process_group] + inference_only_models_process_groups

    def wrap_compile_block(m: nn.Module, is_backbone_block: bool) -> nn.Module:
        if cfg.train.compile:
            if is_backbone_block and cfg.train.cudagraphs:
                m.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": True})
            else:
                m.compile()
        return m

    map_modules_and_blocks(all_models, wrap_compile_block)

    # 3/ Wrap submodules with FSDP (or HSDP when hsdp_shards > 1)
    DTYPE_MAP = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    mp_policy = MixedPrecisionPolicy(
        param_dtype=DTYPE_MAP[cfg.compute_precision.param_dtype],
        reduce_dtype=DTYPE_MAP[cfg.compute_precision.reduce_dtype],
    )

    world_mesh = _build_world_mesh(cfg)

    for m, pg in zip(all_models, all_pgs):
        # When a custom process-subgroup is provided (and it is NOT the global
        # WORLD group), fall back to a 1-D mesh derived from that group.
        # In practice get_process_subgroup() currently always returns WORLD,
        # so we only hit this branch if the caller explicitly passes a real
        # sub-group (e.g. for a future koleo-replica topology).
        if pg is not None and pg is not dist.group.WORLD:
            mesh = DeviceMesh.from_group(pg, "cuda")
            logger.info(f"FSDP: using explicit process-group mesh ({dist.get_world_size(pg)} ranks)")
        else:
            mesh = world_mesh
        fsdp_config = {"mesh": mesh, "mp_policy": mp_policy}
        for k in m.keys():
            if k != "backbone":
                m[k] = fully_shard(m[k], **fsdp_config, reshard_after_forward=True)
                continue
            # Backbone - FSDP every block
            blocks = m[k].blocks

            assert isinstance(blocks, nn.ModuleList)
            for block_id, block in enumerate(blocks):
                block_reshard: int | bool = True
                # if m is trained_model and dist.get_world_size() % 8 == 0 and dist.get_world_size() > 8:
                #     block_reshard = 8
                blocks[block_id] = fully_shard(block, **fsdp_config, reshard_after_forward=block_reshard)
            prev_block: FSDPState
            next_block: FSDPState
            for prev_block, next_block in zip(blocks, blocks[1:]):
                prev_block.set_modules_to_forward_prefetch([next_block])
                next_block.set_modules_to_backward_prefetch([prev_block])
            fully_shard(m.backbone, **fsdp_config, reshard_after_forward=True)
            register_fsdp_forward_method(m.backbone, "get_intermediate_layers")

    # 4/ Move to `cuda` device
    for model in all_models:
        model.to_empty(device="cuda")

    # 5/ FSDP2: Reshard immediately after forward for inference-only models
    for model in inference_only_models:
        for k in model.keys():
            fsdp_state: FSDPState = model[k]._get_fsdp_state()
            if not fsdp_state._fsdp_param_group:
                continue
            mi = fsdp_state._fsdp_param_group.post_forward_mesh_info
            fsdp_state._lazy_init()
            fsdp_state._fsdp_param_group.post_forward_mesh_info = mi
