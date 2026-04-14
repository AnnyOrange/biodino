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


def _torch_load_trusted(path: str | Path, *, map_location="cpu"):
    """torch.load with weights_only=False for trusted checkpoints (PyTorch 2.6+ default is True)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


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
    process_group: dist.ProcessGroup = None,
    **others: Stateful,
):
    """
    Save a consolidated single-file checkpoint:
        ckpt_dir/checkpoint.pth

    Avoids dcp.save() and its NCCL gather_object planner path; directory layout
    (ckpt/<iter>/) stays the same for find_latest_checkpoint / train loop.
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

    For consolidated ``.pth`` resume, restores iteration and model weights.
    Optimizer state in the file is ignored for now (fresh optimizer on resume).
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
                converted_model[key] = torch.distributed.tensor.distribute_tensor(
                    tensor,
                    device_mesh=target_tensor.device_mesh,
                    placements=target_tensor.placements,
                    src_data_rank=None,
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
            logger.warning(
                "Consolidated .pth contains optimizer state, but optimizer restore is skipped "
                "in resume mode; training continues with the current optimizer."
            )

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
):
    if not Path(checkpoint_path).is_dir():  # PyTorch standard checkpoint
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        raw = _torch_load_trusted(checkpoint_path, map_location="cpu")
        if isinstance(raw, dict) and "teacher" in raw:
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
        model_state = model.state_dict()
        converted_chkpt = {}
        for key, tensor in chkpt.items():
            if any(key_not_sharded in key for key_not_sharded in keys_not_sharded):
                converted_chkpt[key] = tensor
                continue
            target_tensor = model_state.get(key)
            if isinstance(target_tensor, torch.distributed.tensor.DTensor):
                converted_chkpt[key] = torch.distributed.tensor.distribute_tensor(
                    tensor,
                    device_mesh=target_tensor.device_mesh,
                    placements=target_tensor.placements,
                    src_data_rank=None,
                )
            else:
                converted_chkpt[key] = tensor
        chkpt = converted_chkpt
        filtered_chkpt = {
            key: tensor
            for key, tensor in chkpt.items()
            if not any(skip_load_key in key for skip_load_key in skip_load_keys)
        }
        missing, unexpected = model.load_state_dict(filtered_chkpt, strict=False)

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

    msg = model.load_state_dict(state_dict, strict=False)
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
