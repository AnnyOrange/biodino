# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Suggested file structure:

output_dir/
|-- ckpt/
|   |-- 0/
|   |-- 99/
|   |-- 199/
|   |-- 199_keep/
|   |-- 299/
|   `-- ...
`-- eval/
    `-- 0/
    `-- 99/
        `-- ckpt/

Distributed checkpointer docs:
- https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
- https://pytorch.org/docs/stable/distributed.checkpoint.html
"""

import inspect
import logging
import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import List, Sequence, Set

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.filesystem as dcpfs
import torch.distributed.checkpoint.state_dict as dcpsd
from torch.distributed.checkpoint.stateful import Stateful

logger = logging.getLogger("dinov3")

# PyTorch 2.2 exposes DTensor only through the private module; newer releases
# also provide the public ``torch.distributed.tensor`` namespace.
_DISTRIBUTE_TENSOR = getattr(getattr(torch.distributed, "tensor", None), "distribute_tensor", None)
if _DISTRIBUTE_TENSOR is None:
    from torch.distributed._tensor import distribute_tensor as _DISTRIBUTE_TENSOR

_DISTRIBUTE_TENSOR_HAS_SRC_DATA_RANK = "src_data_rank" in inspect.signature(_DISTRIBUTE_TENSOR).parameters


def _distribute_checkpoint_tensor(tensor: torch.Tensor, *, device_mesh, placements):
    """Shard a full checkpoint tensor across PyTorch 2.6 and newer releases."""
    kwargs = {"device_mesh": device_mesh, "placements": placements}
    if _DISTRIBUTE_TENSOR_HAS_SRC_DATA_RANK:
        # Every rank loads the same full checkpoint, so no source-rank broadcast is needed.
        kwargs["src_data_rank"] = None
    return _DISTRIBUTE_TENSOR(tensor, **kwargs)


def _torch_load_trusted(path: str | Path, *, map_location="cpu"):
    """torch.load with weights_only=False for trusted checkpoints (PyTorch 2.6+ default is True)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _checkpoint_tensor_shape(tensor) -> tuple[int, ...] | None:
    if tensor is None or not hasattr(tensor, "shape"):
        return None
    return tuple(int(x) for x in tensor.shape)


def _adapt_checkpoint_tensor_for_model(
    key: str,
    tensor: torch.Tensor,
    target_tensor,
) -> tuple[torch.Tensor, bool]:
    """Adapt known-compatible checkpoint tensors whose layouts changed locally.

    The important case is bootstrapping ChannelViT from official RGB DINOv3
    weights: standard PatchEmbed stores Conv2d weights as [D, 3, P, P], while
    PatchEmbedPerChannel stores a shared per-channel Conv3d as [D, 1, 1, P, P].
    We initialize that shared filter with the mean RGB filter instead of leaving
    the whole patch projection random.
    """
    target_shape = _checkpoint_tensor_shape(target_tensor)
    if target_shape is None or tuple(tensor.shape) == target_shape:
        return tensor, False

    normalized_key = key.removeprefix("backbone.")
    if (
        normalized_key == "patch_embed.proj.weight"
        and tensor.ndim == 4
        and len(target_shape) == 5
    ):
        out_ch, _in_ch, patch_h, patch_w = tuple(tensor.shape)
        tgt_out, tgt_in, tgt_depth, tgt_h, tgt_w = target_shape
        if (
            out_ch == tgt_out
            and tgt_in == 1
            and tgt_depth == 1
            and patch_h == tgt_h
            and patch_w == tgt_w
        ):
            adapted = tensor.float().mean(dim=1, keepdim=True).unsqueeze(2)
            adapted = adapted.to(dtype=tensor.dtype)
            logger.info(
                "[CKPT] adapted RGB PatchEmbed Conv2d -> ChannelViT Conv3d for %s: %s -> %s",
                key,
                tuple(tensor.shape),
                target_shape,
            )
            return adapted, True

    return tensor, False


def _add_zero_channel_embed_if_missing(chkpt: dict, model_state: dict) -> None:
    """Zero-init ChannelViT channel embeddings when loading RGB-only weights."""
    for key, target_tensor in model_state.items():
        normalized_key = key.removeprefix("backbone.")
        if normalized_key != "channel_embed" or key in chkpt:
            continue
        target_shape = _checkpoint_tensor_shape(target_tensor)
        if target_shape is None:
            continue
        dtype = getattr(target_tensor, "dtype", torch.float32)
        chkpt[key] = torch.zeros(target_shape, dtype=dtype)
        logger.info(
            "[CKPT] initialized missing ChannelViT channel_embed with zeros: %s %s",
            key,
            target_shape,
        )


def _checkpoint_tensor_initial_value(target_tensor, fallback: float):
    """Return the model-initialized tensor value when available, else fallback."""
    target_shape = _checkpoint_tensor_shape(target_tensor)
    if target_shape is None:
        return None
    dtype = getattr(target_tensor, "dtype", torch.float32)
    try:
        if hasattr(target_tensor, "to_local"):
            local = target_tensor.to_local()
            if isinstance(local, torch.Tensor) and not local.is_meta:
                return local.detach().cpu().clone().to(dtype=dtype)
        if isinstance(target_tensor, torch.Tensor) and not target_tensor.is_meta:
            return target_tensor.detach().cpu().clone().to(dtype=dtype)
    except Exception:
        pass
    return torch.full(target_shape, fallback, dtype=dtype)


def _remap_dualroute_patch_embed(chkpt: dict, model_state: dict) -> None:
    """Bootstrap a #1 dual-route stem from a standard PatchEmbed checkpoint.

    A dual-route model holds two stems: ``patch_embed.rgb`` (Conv2d, 3-in) and
    ``patch_embed.pool`` (shared per-channel Conv2d, 1-in). A pretrained / #4 /
    #5 checkpoint only has the original ``patch_embed.proj.{weight,bias}``.
    We route the pretrained stem to BOTH:
      * ``patch_embed.rgb.proj.*`` <- exact copy (preserves the 1.7B RGB prior),
      * ``patch_embed.pool.proj.weight`` <- mean over the 3 input channels
        (``[D,3,P,P] -> [D,1,P,P]``, same trick as the ChannelViT adapter),
        ``patch_embed.pool.proj.bias`` <- exact copy.
    The content-descriptor MLP / attention query are absent from the checkpoint
    and keep their ``init_weights()`` values. No-op unless the model is
    dual-route, so all other load paths are unaffected.
    """
    rgb_marker = "patch_embed.rgb.proj.weight"
    pool_marker = "patch_embed.pool.proj.weight"
    if not any(k.endswith(rgb_marker) for k in model_state):
        return  # not a dual-route model
    if not any(k.endswith(pool_marker) for k in model_state):
        return  # another RGB-submodule stem (for example residual_mc)
    src_w_key = next((k for k in chkpt if k.endswith("patch_embed.proj.weight")), None)
    if src_w_key is None:
        return  # checkpoint already dual-route, or no standard stem to remap
    prefix = src_w_key[: -len("patch_embed.proj.weight")]  # e.g. "backbone."
    src_b_key = prefix + "patch_embed.proj.bias"
    src_w = chkpt[src_w_key]
    rgb_w_key = prefix + "patch_embed.rgb.proj.weight"
    rgb_b_key = prefix + "patch_embed.rgb.proj.bias"
    pool_w_key = prefix + "patch_embed.pool.proj.weight"
    pool_b_key = prefix + "patch_embed.pool.proj.bias"

    if rgb_w_key in model_state and rgb_w_key not in chkpt:
        chkpt[rgb_w_key] = src_w
        if src_b_key in chkpt and rgb_b_key in model_state:
            chkpt[rgb_b_key] = chkpt[src_b_key]
    if pool_w_key in model_state and pool_w_key not in chkpt and src_w.ndim == 4:
        chkpt[pool_w_key] = src_w.float().mean(dim=1, keepdim=True).to(src_w.dtype)
        if src_b_key in chkpt and pool_b_key in model_state:
            chkpt[pool_b_key] = chkpt[src_b_key]
    # Drop the now-orphaned standard-stem keys (no target in a dual-route model).
    chkpt.pop(src_w_key, None)
    chkpt.pop(src_b_key, None)
    logger.info(
        "[CKPT] remapped standard PatchEmbed -> dual-route stem (rgb=exact copy, pool=channel-mean): %s",
        src_w_key,
    )


def _remap_residual_multichannel_patch_embed(chkpt: dict, model_state: dict) -> None:
    """Bootstrap a residual multi-channel stem from a standard RGB PatchEmbed.

    The residual stem has an exact RGB base path plus a 1-channel extra branch:
      * ``patch_embed.rgb.proj.*`` <- exact RGB PatchEmbed copy,
      * ``patch_embed.extra.weight`` <- mean over RGB input-channel weights,
      * ``patch_embed.extra_scale`` <- tiny scalar gate.

    The extra branch has no bias by design, so the RGB PatchEmbed bias is not
    duplicated in the residual path.
    """
    rgb_marker = "patch_embed.rgb.proj.weight"
    extra_marker = "patch_embed.extra.weight"
    if not any(k.endswith(rgb_marker) for k in model_state):
        return
    if not any(k.endswith(extra_marker) for k in model_state):
        return  # dual-route also has patch_embed.rgb; this is not residual_mc

    src_w_key = next((k for k in chkpt if k.endswith("patch_embed.proj.weight")), None)
    if src_w_key is None:
        return  # checkpoint already has residual_mc keys, or no standard stem
    prefix = src_w_key[: -len("patch_embed.proj.weight")]
    src_b_key = prefix + "patch_embed.proj.bias"
    src_w = chkpt[src_w_key]

    rgb_w_key = prefix + "patch_embed.rgb.proj.weight"
    rgb_b_key = prefix + "patch_embed.rgb.proj.bias"
    extra_w_key = prefix + "patch_embed.extra.weight"
    extra_scale_key = prefix + "patch_embed.extra_scale"

    if rgb_w_key in model_state and rgb_w_key not in chkpt:
        chkpt[rgb_w_key] = src_w
        if src_b_key in chkpt and rgb_b_key in model_state:
            chkpt[rgb_b_key] = chkpt[src_b_key]
    if extra_w_key in model_state and extra_w_key not in chkpt and src_w.ndim == 4:
        chkpt[extra_w_key] = src_w.float().mean(dim=1, keepdim=True).to(src_w.dtype)
    if extra_scale_key in model_state and extra_scale_key not in chkpt:
        init_value = _checkpoint_tensor_initial_value(model_state[extra_scale_key], fallback=1e-3)
        if init_value is not None:
            chkpt[extra_scale_key] = init_value

    # Drop the now-orphaned standard-stem keys (no target in residual_mc).
    chkpt.pop(src_w_key, None)
    chkpt.pop(src_b_key, None)
    logger.info(
        "[CKPT] remapped standard PatchEmbed -> residual multi-channel stem "
        "(rgb=exact copy, extra=channel-mean, scale=model-init): %s",
        src_w_key,
    )


class CheckpointRetentionPolicy(Enum):
    ALL = "all"  # keep all checkpoints
    BEST = "best"
    LAST = "last"
    LAST_AND_BEST = "last_and_best"
    NONE = "none"  # do not keep any checkpoints

    @property
    def keep_filters(self) -> Set[str]:
        """Files that match these patterns are not deleted by cleanup"""
        if self == CheckpointRetentionPolicy.LAST:
            return set(["final"])
        if self == CheckpointRetentionPolicy.BEST:
            return set(["best"])
        if self == CheckpointRetentionPolicy.LAST_AND_BEST:
            return set(["final", "best"])
        if self == CheckpointRetentionPolicy.ALL:
            return set()
        return set()

    @property
    def max_to_keep(self) -> int | None:
        """
        maximum "periodic" checkpoints to keep concurrently, ie. saved with `step` and not `save`. `None` for keep all
        """
        if self == CheckpointRetentionPolicy.ALL:
            return None
        return 1


def _materialize_to_cpu(obj):
    """Recursively move tensors / DTensors to CPU for consolidated torch.save (collective-safe)."""
    if isinstance(obj, DTensor):
        # full_tensor() is collective; all ranks must execute it
        return obj.full_tensor().detach().cpu()
    elif torch.is_tensor(obj):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: _materialize_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_materialize_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_materialize_to_cpu(v) for v in obj)
    else:
        return obj


def save_checkpoint(
    ckpt_dir: str | Path,
    *,
    iteration: int | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    overwrite: bool = True,
    sharded: bool = False,
    process_group: dist.ProcessGroup = None,
    **others: Stateful,
):
    """
    Save either a consolidated ``checkpoint.pth`` or a DCP sharded checkpoint.

    Consolidated files are convenient for small models and downstream tools.
    DCP avoids materializing the full model and optimizer on every rank, which
    is required for practical 7B training and supports resharded resume.
    """
    rank = torch.distributed.get_rank(group=process_group)
    ckpt_dir = Path(ckpt_dir)

    ckpt_dir_exists = [ckpt_dir.exists() if rank == 0 else None]
    src_rank = 0
    if process_group is not None:
        src_rank = torch.distributed.get_global_rank(group=process_group, group_rank=0)
    torch.distributed.broadcast_object_list(ckpt_dir_exists, src=src_rank, group=process_group)
    ckpt_dir_exists = ckpt_dir_exists[0]

    if ckpt_dir_exists:
        if overwrite:
            if rank == 0:
                if ckpt_dir.is_dir():
                    shutil.rmtree(ckpt_dir)
                else:
                    ckpt_dir.unlink()
                logger.info(f"Deleted: {ckpt_dir}")
            torch.distributed.barrier(group=process_group)
        else:
            raise RuntimeError(f"Checkpoint already exists: {ckpt_dir}")

    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir_tmp = [tempfile.mkdtemp(dir=ckpt_dir.parent, prefix=ckpt_dir.name) if rank == 0 else None]
    torch.distributed.broadcast_object_list(ckpt_dir_tmp, src=src_rank, group=process_group)
    ckpt_dir_tmp = Path(ckpt_dir_tmp[0])

    to_save = {"iteration": iteration}
    to_save["model"] = dcpsd.get_model_state_dict(model)
    if optimizer is not None:
        to_save["optimizer"] = dcpsd.get_optimizer_state_dict(model, optimizer)
    to_save.update(others)

    if sharded:
        dcp.save(
            to_save,
            storage_writer=dcpfs.FileSystemWriter(ckpt_dir_tmp),
            process_group=process_group,
        )
        if rank == 0:
            ckpt_dir_tmp.rename(ckpt_dir)
        torch.distributed.barrier(group=process_group)
        logger.info("Saved DCP sharded checkpoint: %s", ckpt_dir)
        return

    # All ranks participate: DTensor.full_tensor() is collective.
    to_save = _materialize_to_cpu(to_save)

    if rank == 0:
        torch.save(to_save, ckpt_dir_tmp / "checkpoint.pth")

    torch.distributed.barrier(group=process_group)

    if rank == 0:
        ckpt_dir_tmp.rename(ckpt_dir)

    torch.distributed.barrier(group=process_group)
    logger.info(f"Saved consolidated checkpoint: {ckpt_dir / 'checkpoint.pth'}")


def _iteration_to_python(iteration):
    if iteration is None:
        return None
    if torch.is_tensor(iteration):
        return int(iteration.item())
    if hasattr(iteration, "item") and callable(iteration.item):
        try:
            return int(iteration.item())
        except (TypeError, ValueError):
            return iteration
    return iteration


def load_checkpoint(
    ckpt_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict_loading: bool = True,
    process_group: dist.ProcessGroup = None,
    **others: Stateful,
) -> int | None:
    """
    Load either:
      1) a DCP checkpoint directory, or
      2) a consolidated file checkpoint: ``ckpt_dir/checkpoint.pth``

    For consolidated ``.pth`` resume, restores the iteration, model, and
    optimizer state saved by :func:`save_checkpoint`.
    """
    ckpt_dir = Path(ckpt_dir)
    pth_file = ckpt_dir / "checkpoint.pth"

    if pth_file.is_file():
        logger.info("Loading consolidated checkpoint file: %s", pth_file)
        raw = _torch_load_trusted(pth_file, map_location="cpu")

        iteration = _iteration_to_python(raw.get("iteration", None))

        if "model" not in raw:
            raise KeyError(f"'model' key not found in consolidated checkpoint: {pth_file}")

        ckpt_model = raw["model"]
        model_state = model.state_dict()
        converted_model = {}
        for key, tensor in ckpt_model.items():
            if key not in model_state:
                continue
            target_tensor = model_state[key]
            if isinstance(target_tensor, DTensor):
                converted_model[key] = _distribute_checkpoint_tensor(
                    tensor,
                    device_mesh=target_tensor.device_mesh,
                    placements=target_tensor.placements,
                )
            else:
                converted_model[key] = tensor

        incompatible = model.load_state_dict(converted_model, strict=False)
        missing = incompatible.missing_keys
        unexpected = incompatible.unexpected_keys
        logger.info(
            "Loaded consolidated model checkpoint with %d missing keys and %d unexpected keys",
            len(missing),
            len(unexpected),
        )
        if strict_loading and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError(
                f"Consolidated checkpoint load not strict: "
                f"missing={list(missing)[:10]}, unexpected={list(unexpected)[:10]}"
            )

        if optimizer is not None and "optimizer" in raw:
            dcpsd.set_optimizer_state_dict(
                model,
                optimizer,
                raw["optimizer"],
                options=dcpsd.StateDictOptions(full_state_dict=True),
            )
            logger.info("Restored optimizer state from consolidated checkpoint")

        logger.info("Loaded consolidated checkpoint: %s", pth_file)
        return iteration

    to_load = {"iteration": None}
    to_load["model"] = dcpsd.get_model_state_dict(model)
    if optimizer is not None:
        to_load["optimizer"] = dcpsd.get_optimizer_state_dict(model, optimizer)
    to_load.update(others)
    dcp.load(
        to_load,
        storage_reader=dcpfs.FileSystemReader(ckpt_dir),
        planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=not strict_loading),
        process_group=process_group,
    )
    iteration = to_load["iteration"]
    dcpsd.set_model_state_dict(model, to_load["model"])
    if optimizer is not None:
        dcpsd.set_optimizer_state_dict(model, optimizer, to_load["optimizer"])
    logger.info("Loaded DCP checkpoint: %s", ckpt_dir)
    return iteration


def register_dont_save_hooks(module: torch.nn.Module, dont_save: Sequence[str]):
    """
    Registers save/load state dict hooks such that the weights in `dont_save` are not persisted in the checkpoint.

    Typical use case: a classification model composed of a frozen backbone and a trainable head.
    If the frozen backbone is loaded from torch hub, it does't make sense to save a copy of it in each checkpoint.
    """

    def state_dict_post_hook(module, state_dict, prefix, local_metadata):
        # Remove frozen weights so they won't get saved.
        # If this module is not the top-level module, its weights will have a prefix in the state dict.
        nonlocal _dont_save
        for k in _dont_save:
            del state_dict[prefix + k]

    def load_state_dict_pre_hook(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # This pre hook exists only to pass the prefix to the post hook when loading the state dict.
        nonlocal _prefix
        assert _prefix is None
        _prefix = prefix

    def load_state_dict_post_hook(module, incompatible_keys):
        # Remove the frozen weights from the missing keys so they don't raise an error.
        nonlocal _prefix
        assert _prefix is not None
        to_remove = []
        for missing_key in incompatible_keys.missing_keys:
            k = missing_key.removeprefix(_prefix)
            k = k.replace("_checkpoint_wrapped_module.", "")  # Added by activation checkpointing
            if k in _dont_save:
                to_remove.append(missing_key)
        for r in to_remove:
            incompatible_keys.missing_keys.remove(r)
        _prefix = None

    _dont_save = set(name.replace("_checkpoint_wrapped_module.", "") for name in dont_save)
    _prefix = None
    module.register_state_dict_post_hook(state_dict_post_hook)
    module.register_load_state_dict_pre_hook(load_state_dict_pre_hook)
    module.register_load_state_dict_post_hook(load_state_dict_post_hook)


def find_all_checkpoints(ckpt_dir: Path | str) -> list[Path]:
    """Find all checkpoints in a directory, i.e. subdirs with integer name. Sorted from first to last."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    checkpoints = [p for p in ckpt_dir.iterdir() if p.is_dir() and _is_int(p.name)]
    checkpoints.sort(key=lambda p: int(p.name))
    return checkpoints


def find_latest_checkpoint(ckpt_dir: Path | str) -> Path | None:
    """Find the latest checkpoint in a directory, i.e. the subdir with the highest integer name."""
    checkpoints = find_all_checkpoints(ckpt_dir)
    if len(checkpoints) == 0:
        return None
    return checkpoints[-1]


def keep_last_n_checkpoints(ckpt_dir: Path | str, n: int | None):
    """In a directory with integer-named subdirs, keep only the n subdirs with the highest number."""
    if n is None:
        return
    checkpoints = find_all_checkpoints(ckpt_dir)
    for ckpt_dir in checkpoints[:-n]:
        try:
            shutil.rmtree(ckpt_dir)
            logger.info(f"Deleted: {ckpt_dir}")
        except Exception:
            logger.exception(f"Failed to delete: {ckpt_dir}")


def keep_checkpoint_copy(src: Path | str):
    """Copy a file/directory next to itself with a _keep suffix. Files are hardlinked."""
    src = Path(src)
    dst = src.parent / f"{src.name}_keep"
    subprocess.check_output(["cp", "--recursive", "--link", src, dst])
    logger.info(f"Copied: {src} -> {dst}")


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


# Initialize a FSDP2 model from DCP or PyTorch standard checkpoint
def init_fsdp_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    skip_load_keys: List[str] | None = None,
    keys_not_sharded: List[str] | None = None,
    process_group: dist.ProcessGroup = None,
    checkpoint_state_prefix: str | None = None,
):
    if not Path(checkpoint_path).is_dir():  # PyTorch standard checkpoint
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        raw = _torch_load_trusted(checkpoint_path, map_location="cpu")
        if checkpoint_state_prefix is not None:
            # Training checkpoints store the complete meta-architecture under
            # ``model`` (for example ``teacher.backbone.*``).  A frozen
            # auxiliary backbone needs one explicitly selected state instead
            # of treating the checkpoint wrapper as a flat official release.
            state = raw.get("model", raw) if isinstance(raw, dict) else raw
            if not isinstance(state, dict):
                raise ValueError(
                    "checkpoint_state_prefix requires a mapping checkpoint state, got "
                    f"{type(state).__name__}"
                )
            selected = {
                f"backbone.{key.removeprefix(checkpoint_state_prefix)}": tensor
                for key, tensor in state.items()
                if key.startswith(checkpoint_state_prefix)
            }
            if not selected:
                raise ValueError(
                    f"No checkpoint keys begin with {checkpoint_state_prefix!r} in {checkpoint_path}"
                )
            chkpt = selected
            logger.info(
                "Selected %d keys with training-checkpoint prefix %s",
                len(chkpt),
                checkpoint_state_prefix,
            )
        elif isinstance(raw, dict) and "teacher" in raw:
            chkpt = raw["teacher"]
        else:
            # Flat backbone-only checkpoint (e.g. official released weights).
            # Keys need a "backbone." prefix to match the ModuleDict layout.
            state = raw if not isinstance(raw, dict) else raw
            first_key = next(iter(state))
            if not first_key.startswith("backbone."):
                chkpt = {f"backbone.{k}": v for k, v in state.items()}
                logger.info("Detected flat backbone checkpoint — added 'backbone.' prefix to all keys")
            else:
                chkpt = dict(state)
        skip_load_keys = skip_load_keys or []
        keys_not_sharded = keys_not_sharded or []
        model_state = model.state_dict()
        _add_zero_channel_embed_if_missing(chkpt, model_state)
        _remap_dualroute_patch_embed(chkpt, model_state)
        _remap_residual_multichannel_patch_embed(chkpt, model_state)
        converted_chkpt = {}
        for key, tensor in chkpt.items():
            if any(key_not_sharded in key for key_not_sharded in keys_not_sharded):
                converted_chkpt[key] = tensor
                continue
            target_tensor = model_state.get(key)
            if target_tensor is not None and isinstance(tensor, torch.Tensor):
                tensor, _ = _adapt_checkpoint_tensor_for_model(key, tensor, target_tensor)
            if (
                target_tensor is not None
                and isinstance(tensor, torch.Tensor)
                and isinstance(target_tensor, torch.Tensor)
                and tuple(tensor.shape) != tuple(target_tensor.shape)
            ):
                logger.warning(
                    "[CKPT] skip shape-mismatch key %s (pre-convert): ckpt %s vs model %s",
                    key,
                    tuple(tensor.shape),
                    tuple(target_tensor.shape),
                )
                continue
            if isinstance(target_tensor, DTensor):
                converted_chkpt[key] = _distribute_checkpoint_tensor(
                    tensor,
                    device_mesh=target_tensor.device_mesh,
                    placements=target_tensor.placements,
                )
            else:
                converted_chkpt[key] = tensor
        chkpt = converted_chkpt
        filtered_chkpt = {}
        for key, tensor in chkpt.items():
            if any(skip_load_key in key for skip_load_key in skip_load_keys):
                continue
            target_tensor = model_state.get(key)
            if target_tensor is not None and isinstance(tensor, torch.Tensor):
                tensor, _ = _adapt_checkpoint_tensor_for_model(key, tensor, target_tensor)
            if (
                target_tensor is not None
                and isinstance(tensor, torch.Tensor)
                and isinstance(target_tensor, torch.Tensor)
                and tuple(tensor.shape) != tuple(target_tensor.shape)
            ):
                logger.warning(
                    "[CKPT] skip shape-mismatch key %s: ckpt %s vs model %s",
                    key,
                    tuple(tensor.shape),
                    tuple(target_tensor.shape),
                )
                continue
            filtered_chkpt[key] = tensor
        incompatible = model.load_state_dict(filtered_chkpt, strict=False)
        missing = incompatible.missing_keys
        unexpected = incompatible.unexpected_keys

        # Classify missing keys: backbone core vs heads/centers (expected to be missing for backbone-only ckpt)
        backbone_missing = [
            k for k in missing
            if "backbone." in k and "head" not in k and "center" not in k
        ]
        other_missing = [k for k in missing if k not in backbone_missing]

        if backbone_missing:
            logger.warning(
                f"[CKPT] backbone keys MISSING — likely prefix mismatch! "
                f"({len(backbone_missing)} keys, e.g. {backbone_missing[:5]})"
            )
        else:
            logger.info("[CKPT] backbone fully loaded (0 backbone core keys missing)")
        logger.info(
            f"[CKPT] non-backbone missing (random init): {len(other_missing)} keys, "
            f"e.g. {other_missing[:3]}"
        )
        if unexpected:
            logger.warning(
                f"[CKPT] unexpected keys (ignored): {len(unexpected)} keys, "
                f"e.g. {unexpected[:3]}"
            )
    else:  # DCP checkpoint
        load_checkpoint(ckpt_dir=checkpoint_path, model=model, process_group=process_group)


# Initialize a standard non distributed PyTorch model from PyTorch standard checkpoint for evals
def init_model_from_checkpoint_for_evals(
    model: torch.nn.Module, pretrained_weights: str | Path, checkpoint_key: str = None
):
    state_dict = _torch_load_trusted(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info("Take key %s in provided checkpoint dict", checkpoint_key)
        state_dict = state_dict[checkpoint_key]

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Full training checkpoints: narrow to backbone weights for bare ViT load.
    if any(k.startswith("teacher.backbone.") for k in state_dict.keys()):
        state_dict = {
            k[len("teacher.backbone.") :]: v
            for k, v in state_dict.items()
            if k.startswith("teacher.backbone.")
        }
    elif any(k.startswith("backbone.") for k in state_dict.keys()):
        state_dict = {
            k[len("backbone.") :]: v for k, v in state_dict.items() if k.startswith("backbone.")
        }

    model_state = model.state_dict()
    _add_zero_channel_embed_if_missing(state_dict, model_state)
    _remap_dualroute_patch_embed(state_dict, model_state)
    _remap_residual_multichannel_patch_embed(state_dict, model_state)
    filtered_state_dict = {}
    for key, tensor in state_dict.items():
        target_tensor = model_state.get(key)
        if target_tensor is None:
            filtered_state_dict[key] = tensor
            continue
        if isinstance(tensor, torch.Tensor):
            tensor, _ = _adapt_checkpoint_tensor_for_model(key, tensor, target_tensor)
            target_shape = _checkpoint_tensor_shape(target_tensor)
            if target_shape is not None and tuple(tensor.shape) != target_shape:
                logger.warning(
                    "[CKPT] skip eval shape-mismatch key %s: ckpt %s vs model %s",
                    key,
                    tuple(tensor.shape),
                    target_shape,
                )
                continue
        filtered_state_dict[key] = tensor

    msg = model.load_state_dict(filtered_state_dict, strict=False)
    logger.info("Pretrained weights at %s loaded with msg: %s", pretrained_weights, msg)


def cleanup_checkpoint(ckpt_dir: str, checkpoint_retention_policy: CheckpointRetentionPolicy):
    """
    ckpt_dir is the directory containing each individual checkpoint directories (either at iteration, best (validation performance) or final)
    |-- ckpt_dir/
    |   |-- 0/
    |       |--checkpoint.pth  or dcp_sharded_checkpoint_dir
    |   |-- 99/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    |   |-- 199/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    |   |-- best/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    |   |-- 299/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    |   |-- final/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    checkpoint_filters = checkpoint_retention_policy.keep_filters
    checkpoints = [p for p in ckpt_dir.iterdir() if p.is_dir()]
    for checkpoint in checkpoints:
        if checkpoint in checkpoint_filters:
            continue
        try:
            shutil.rmtree(checkpoint)
            logger.info(f"Deleted: {checkpoint}")
        except Exception:
            logger.exception(f"Failed to delete: {checkpoint}")
