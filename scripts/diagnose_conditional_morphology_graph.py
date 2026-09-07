#!/usr/bin/env python3
"""Test whether omitted microscopy channels carry stable local morphology.

The diagnostic compares three views of the same channel subset S:
  full:      the original available channel set C;
  subset:    S only;
  shuffled:  S plus omitted channels copied from other samples in the batch.

The two global views use the same crop geometry but independent photometric
augmentations.  This makes patch positions comparable while retaining a useful
invariance test.  A usable conditional morphology signal should be (a) stable
across those views and (b) more predictable from S than the channel-shuffled
counterfactual.  The script is deliberately diagnostic only: it never updates
the DINO backbone.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from functools import partial
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor, nn

from dinov3.configs import get_default_config
from dinov3.data import DataAugmentationDINO, collate_data_and_cast, make_dataset
from dinov3.data.loaders import make_data_loader
from dinov3.data.masking import MaskingGenerator
from dinov3.eval.bio_segmentation.model_utils import load_dinov3_backbone
from dinov3.loss.conditional_morphology_graph_loss import ConditionalEdgeGraphPredictor
from dinov3.loss.nested_channel_innovation_loss import ConditionalFeaturePredictor
from dinov3.train.ssl_meta_arch import _sample_channel_subset_mask


LOGGER = logging.getLogger("cmgi_diagnostic")


def _build_augmentation(cfg):
    return DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
        gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
        local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
        share_color_jitter=cfg.crops.share_color_jitter,
        horizontal_flips=cfg.crops.horizontal_flips,
        mean=cfg.crops.rgb_mean,
        std=cfg.crops.rgb_std,
        float_input=bool(getattr(cfg.crops, "float_input", False)),
        augmentation_policy=str(getattr(cfg.crops, "augmentation_policy", "dinov3")),
        paired_global_geometry=True,
    )


def _make_loader(cfg, *, batch_size: int, workers: int):
    image_size = int(cfg.crops.global_crops_size)
    patch_size = int(cfg.student.patch_size)
    n_tokens = (image_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(image_size // patch_size, image_size // patch_size),
        max_num_patches=0.5 * n_tokens,
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype=torch.float32,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        random_circular_shift=False,
        local_batch_size=None,
    )
    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=_build_augmentation(cfg),
        target_transform=lambda _: (),
        target_channels=cfg.student.in_chans,
        wds_shuffle_buffer=int(getattr(cfg.train, "wds_shuffle_buffer", 100)),
    )
    return make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        seed=int(cfg.train.seed),
        drop_last=True,
        pin_memory=True,
        prefetch_factor=1 if workers > 0 else None,
        collate_fn=collate_fn,
    )


def _local_edge_index(num_patches: int, radius: int, device: torch.device) -> tuple[Tensor, Tensor]:
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} patches")
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    coords = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=1)
    dy = (coords[:, None, 0] - coords[None, :, 0]).abs()
    dx = (coords[:, None, 1] - coords[None, :, 1]).abs()
    keep = (dy <= radius) & (dx <= radius) & ((dy + dx) > 0)
    return keep.nonzero(as_tuple=True)[0].to(device), keep.nonzero(as_tuple=True)[1].to(device)


def _edge_kernel(features: Tensor, src: Tensor, dst: Tensor) -> Tensor:
    """Return local cosine similarities for ``[B, P, D]`` patch features."""
    features = F.normalize(features.float(), dim=-1)
    return (features[:, src] * features[:, dst]).sum(dim=-1)


def _counterfactual_inputs(
    images: Tensor,
    valid_mask: Tensor,
    subset_mask: Tensor,
    source_indices: list[Tensor],
) -> Tensor:
    """Replace only omitted valid channels with a same-index donor channel."""
    out = images.clone()
    for channel, sources in enumerate(source_indices):
        targets = (valid_mask[:, channel] & ~subset_mask[:, channel]).nonzero(as_tuple=False).flatten()
        if targets.numel() == 0:
            continue
        if sources.numel() != targets.numel():
            raise RuntimeError("Counterfactual donor map is inconsistent with target mask")
        out[targets, channel] = images[sources, channel]
    return out


def _sample_counterfactual_sources(valid_mask: Tensor, subset_mask: Tensor) -> list[Tensor]:
    """Choose donors with the same physical channel, avoiding self-copy when possible."""
    source_indices: list[Tensor] = []
    for channel in range(valid_mask.shape[1]):
        targets = (valid_mask[:, channel] & ~subset_mask[:, channel]).nonzero(as_tuple=False).flatten()
        donors = valid_mask[:, channel].nonzero(as_tuple=False).flatten()
        choices = []
        for target in targets.tolist():
            candidates = donors[donors != target]
            if candidates.numel() == 0:
                candidates = donors
            choices.append(candidates[torch.randint(candidates.numel(), ()).item()])
        source_indices.append(torch.stack(choices) if choices else torch.empty(0, dtype=torch.long))
    return source_indices


def _mean_std(values: list[Tensor]) -> dict[str, float]:
    all_values = torch.cat([x.detach().float().cpu().reshape(-1) for x in values])
    return {
        "mean": float(all_values.mean()),
        "median": float(all_values.median()),
        "std": float(all_values.std(unbiased=False)),
        "stderr": float(all_values.std(unbiased=False) / max(1.0, all_values.numel() ** 0.5)),
        "n": int(all_values.numel()),
        "positive_fraction": float((all_values > 0).float().mean()),
    }


def _prepare_predictor_data(records: Iterable[dict[str, Tensor]], *, train: bool, patches_per_image: int) -> tuple[Tensor, Tensor, Tensor]:
    selected = list(records)
    if not selected:
        raise RuntimeError("No active multi-channel samples were collected")
    split = max(1, int(0.7 * len(selected)))
    selected = selected[:split] if train else selected[split:]
    if not selected:
        selected = list(records)[-1:]
    subset_tokens, full_tokens, shuffled_tokens = [], [], []
    for record in selected:
        for view in range(record["subset"].shape[0]):
            patch_count = record["subset"].shape[2]
            take = min(patches_per_image, patch_count)
            idx = torch.randperm(patch_count)[:take]
            subset_tokens.append(record["subset"][view, :, idx].reshape(-1, record["subset"].shape[-1]))
            full_tokens.append(record["full"][view, :, idx].reshape(-1, record["full"].shape[-1]))
            shuffled_tokens.append(record["shuffled"][view, :, idx].reshape(-1, record["shuffled"].shape[-1]))
    return torch.cat(subset_tokens), torch.cat(full_tokens), torch.cat(shuffled_tokens)


def _train_predictor(
    subset: Tensor,
    full: Tensor,
    *,
    validation_subset: Tensor,
    validation_full: Tensor,
    device: torch.device,
    updates: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[nn.Module, float]:
    predictor = ConditionalFeaturePredictor(subset.shape[-1]).to(device)
    predictor.reset_parameters()
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=learning_rate, weight_decay=1e-4)
    subset = F.normalize(subset.float(), dim=-1)
    full = F.normalize(full.float(), dim=-1)
    validation_subset = F.normalize(validation_subset.float(), dim=-1)
    validation_full = F.normalize(validation_full.float(), dim=-1)
    validation_size = min(batch_size, validation_subset.shape[0])

    def validation_loss() -> float:
        index = torch.randperm(validation_subset.shape[0])[:validation_size]
        x = validation_subset[index].to(device, non_blocking=True)
        target = validation_full[index].to(device, non_blocking=True)
        with torch.no_grad():
            return float(F.mse_loss(F.normalize(predictor(x), dim=-1), target))

    best_loss = validation_loss()
    best_state = {name: value.detach().cpu().clone() for name, value in predictor.state_dict().items()}
    for step in range(updates):
        index = torch.randint(subset.shape[0], (min(batch_size, subset.shape[0]),))
        x = subset[index].to(device, non_blocking=True)
        target = full[index].to(device, non_blocking=True)
        prediction = F.normalize(predictor(x), dim=-1)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0 or step + 1 == updates:
            current_validation_loss = validation_loss()
            if current_validation_loss < best_loss:
                best_loss = current_validation_loss
                best_state = {name: value.detach().cpu().clone() for name, value in predictor.state_dict().items()}
    if best_state is not None:
        predictor.load_state_dict(best_state)
    predictor.eval()
    return predictor, best_loss


def _prepare_edge_predictor_data(
    records: Iterable[dict[str, Tensor]], *, train: bool
) -> tuple[Tensor, Tensor, Tensor]:
    """Flatten matched views into image-level local-graph training examples."""
    selected = list(records)
    split = max(1, int(0.7 * len(selected)))
    selected = selected[:split] if train else selected[split:]
    if not selected:
        selected = list(records)[-1:]
    subset = torch.cat([record["subset"].flatten(0, 1) for record in selected])
    full = torch.cat([record["full"].flatten(0, 1) for record in selected])
    shuffled = torch.cat([record["shuffled"].flatten(0, 1) for record in selected])
    return subset, full, shuffled


def _edge_prediction(
    predictor: ConditionalEdgeGraphPredictor,
    subset: Tensor,
    *,
    src: Tensor,
    dst: Tensor,
) -> tuple[Tensor, Tensor]:
    subset_graph = _edge_kernel(subset, src, dst)
    prediction = predictor(
        subset.unsqueeze(0),
        src,
        dst,
        subset_graph.unsqueeze(0),
    ).squeeze(0)
    return prediction, subset_graph


def _train_edge_predictor(
    subset: Tensor,
    full: Tensor,
    *,
    validation_subset: Tensor,
    validation_full: Tensor,
    device: torch.device,
    src: Tensor,
    dst: Tensor,
    edge_dim: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[ConditionalEdgeGraphPredictor, float]:
    predictor = ConditionalEdgeGraphPredictor(
        subset.shape[-1], edge_dim=edge_dim
    ).to(device)
    predictor.reset_parameters()
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=learning_rate, weight_decay=1e-4)
    validation_size = min(batch_size, validation_subset.shape[0])

    def validation_loss() -> float:
        index = torch.randperm(validation_subset.shape[0])[:validation_size]
        source = validation_subset[index].to(device, non_blocking=True).float()
        target_features = validation_full[index].to(device, non_blocking=True).float()
        target = _edge_kernel(target_features, src, dst)
        with torch.no_grad():
            prediction, _ = _edge_prediction(predictor, source, src=src, dst=dst)
            return float(F.mse_loss(prediction, target))

    best_loss = validation_loss()
    best_state = {name: value.detach().cpu().clone() for name, value in predictor.state_dict().items()}
    for step in range(updates):
        index = torch.randint(subset.shape[0], (min(batch_size, subset.shape[0]),))
        source = subset[index].to(device, non_blocking=True).float()
        target_features = full[index].to(device, non_blocking=True).float()
        target = _edge_kernel(target_features, src, dst)
        prediction, _ = _edge_prediction(predictor, source, src=src, dst=dst)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0 or step + 1 == updates:
            current_validation_loss = validation_loss()
            if current_validation_loss < best_loss:
                best_loss = current_validation_loss
                best_state = {name: value.detach().cpu().clone() for name, value in predictor.state_dict().items()}
    predictor.load_state_dict(best_state)
    predictor.eval()
    return predictor, best_loss


def _evaluate_predictor(
    predictor: nn.Module,
    records: Iterable[dict[str, Tensor]],
    *,
    device: torch.device,
    src: Tensor,
    dst: Tensor,
) -> dict[str, dict[str, float]]:
    full_errors, shuffled_errors, identity_errors = [], [], []
    graph_true_energy, graph_shuffled_energy = [], []
    graph_true_stability, graph_shuffled_stability = [], []
    with torch.no_grad():
        for record in records:
            subset = record["subset"].to(device, non_blocking=True).float()
            full = record["full"].to(device, non_blocking=True).float()
            shuffled = record["shuffled"].to(device, non_blocking=True).float()
            predicted = predictor(subset.flatten(0, 2)).reshape_as(subset)
            predicted = F.normalize(predicted, dim=-1)
            subset_norm = F.normalize(subset, dim=-1)
            full_norm = F.normalize(full, dim=-1)
            shuffled_norm = F.normalize(shuffled, dim=-1)
            full_errors.append((predicted - full_norm).square().mean(dim=(-1, -2)))
            shuffled_errors.append((predicted - shuffled_norm).square().mean(dim=(-1, -2)))
            identity_errors.append((subset_norm - full_norm).square().mean(dim=(-1, -2)))

            view_count, sample_count = subset.shape[:2]
            edge_subset = _edge_kernel(subset.flatten(0, 1), src, dst)
            edge_full = _edge_kernel(full.flatten(0, 1), src, dst)
            edge_shuffled = _edge_kernel(shuffled.flatten(0, 1), src, dst)
            edge_prediction = _edge_kernel(predicted.flatten(0, 1), src, dst)
            residual_true = edge_full - edge_prediction
            residual_shuffled = edge_shuffled - edge_prediction
            graph_true_energy.append(residual_true.square().mean(dim=-1).sqrt())
            graph_shuffled_energy.append(residual_shuffled.square().mean(dim=-1).sqrt())
            residual_true = residual_true.unflatten(0, (view_count, sample_count))
            residual_shuffled = residual_shuffled.unflatten(0, (view_count, sample_count))
            graph_true_stability.append(F.cosine_similarity(residual_true[0], residual_true[1], dim=-1))
            graph_shuffled_stability.append(F.cosine_similarity(residual_shuffled[0], residual_shuffled[1], dim=-1))

    full_summary = _mean_std(full_errors)
    shuffled_summary = _mean_std(shuffled_errors)
    identity_summary = _mean_std(identity_errors)
    mse_full = full_summary["mean"]
    mse_identity = identity_summary["mean"]
    return {
        "predict_full_mse": full_summary,
        "predict_shuffled_mse": shuffled_summary,
        "identity_to_full_mse": identity_summary,
        "predictor_relative_improvement": {
            "mean": float((mse_identity - mse_full) / max(mse_identity, 1e-12)),
            "shuffled_penalty": float(shuffled_summary["mean"] - mse_full),
        },
        "conditional_graph_true_energy": _mean_std(graph_true_energy),
        "conditional_graph_shuffled_energy": _mean_std(graph_shuffled_energy),
        "conditional_graph_true_stability": _mean_std(graph_true_stability),
        "conditional_graph_shuffled_stability": _mean_std(graph_shuffled_stability),
        "conditional_graph_stability_margin": _mean_std(
            [a - b for a, b in zip(graph_true_stability, graph_shuffled_stability)]
        ),
    }


def _evaluate_edge_predictor(
    predictor: ConditionalEdgeGraphPredictor,
    records: Iterable[dict[str, Tensor]],
    *,
    device: torch.device,
    src: Tensor,
    dst: Tensor,
) -> dict[str, dict[str, float]]:
    """Evaluate the conditional graph projection on held-out frozen images."""
    predicted_errors, subset_errors, shuffled_errors = [], [], []
    predicted_absolute, subset_absolute = [], []
    true_stability, shuffled_stability = [], []
    with torch.no_grad():
        for record in records:
            subset = record["subset"].to(device, non_blocking=True).float()
            full = record["full"].to(device, non_blocking=True).float()
            shuffled = record["shuffled"].to(device, non_blocking=True).float()
            n_views, batch_size = subset.shape[:2]
            source = subset.flatten(0, 1)
            full_graph = _edge_kernel(full.flatten(0, 1), src, dst)
            shuffled_graph = _edge_kernel(shuffled.flatten(0, 1), src, dst)
            predicted_graph, subset_graph = _edge_prediction(predictor, source, src=src, dst=dst)
            predicted_errors.append((predicted_graph - full_graph).square().mean(dim=-1))
            subset_errors.append((subset_graph - full_graph).square().mean(dim=-1))
            shuffled_errors.append((predicted_graph - shuffled_graph).square().mean(dim=-1))
            predicted_absolute.append((predicted_graph - full_graph).abs().mean(dim=-1))
            subset_absolute.append((subset_graph - full_graph).abs().mean(dim=-1))
            residual_true = (full_graph - predicted_graph).unflatten(0, (n_views, batch_size))
            residual_shuffled = (shuffled_graph - predicted_graph).unflatten(0, (n_views, batch_size))
            true_stability.append(F.cosine_similarity(residual_true[0], residual_true[1], dim=-1))
            shuffled_stability.append(F.cosine_similarity(residual_shuffled[0], residual_shuffled[1], dim=-1))

    predicted_summary = _mean_std(predicted_errors)
    subset_summary = _mean_std(subset_errors)
    absolute_delta = _mean_std([a - b for a, b in zip(predicted_absolute, subset_absolute)])
    return {
        "predict_full_graph_mse": predicted_summary,
        "subset_to_full_graph_mse": subset_summary,
        "predict_shuffled_graph_mse": _mean_std(shuffled_errors),
        "relative_mse_improvement": {
            "mean": float(
                (subset_summary["mean"] - predicted_summary["mean"])
                / max(subset_summary["mean"], 1.0e-12)
            ),
        },
        "predictor_vs_subset_absolute_edge_delta": absolute_delta,
        "conditional_graph_true_stability": _mean_std(true_stability),
        "conditional_graph_shuffled_stability": _mean_std(shuffled_stability),
        "conditional_graph_stability_margin": _mean_std(
            [a - b for a, b in zip(true_stability, shuffled_stability)]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-subset-channels", type=int, default=1)
    parser.add_argument("--max-subset-channels", type=int, default=3)
    parser.add_argument("--local-radius", type=int, default=2)
    parser.add_argument(
        "--predictor-mode",
        choices=("feature", "edge"),
        default="feature",
        help="Use legacy token prediction or direct conditional edge regression.",
    )
    parser.add_argument("--edge-predictor-dim", type=int, default=64)
    parser.add_argument("--predictor-patches", type=int, default=32)
    parser.add_argument("--predictor-updates", type=int, default=300)
    parser.add_argument("--predictor-batch-size", type=int, default=1024)
    parser.add_argument("--predictor-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(args.train_config))
    # The diagnostic needs matched spatial coordinates.  DINO-style augmentation
    # keeps post-crop geometry deterministic while still changing photometrics.
    cfg.crops.augmentation_policy = "dinov3"
    cfg.crops.paired_global_geometry = True
    cfg.train.batch_size_per_gpu = args.batch_size
    cfg.train.num_workers = args.workers
    if "min_channels=" not in str(cfg.train.dataset_path):
        raise ValueError("CMGI diagnostic requires a packwds_chvit dataset with channel metadata")

    backbone = load_dinov3_backbone(
        args.checkpoint,
        args.train_config,
        device=device,
        freeze=True,
    )
    loader = _make_loader(cfg, batch_size=args.batch_size, workers=args.workers)

    records: list[dict[str, Tensor]] = []
    raw_true_energy, raw_shuffled_energy = [], []
    raw_true_stability, raw_shuffled_stability = [], []
    edge_src = edge_dst = None
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.batches:
            break
        crops = batch["collated_global_crops"].to(device, non_blocking=True)
        ids = batch["collated_global_channel_ids"].to(device, non_blocking=True)
        valid = batch["collated_global_channel_valid_mask"].to(device, non_blocking=True)
        n_views = 2
        base_batch = crops.shape[0] // n_views
        crops = crops.unflatten(0, (n_views, base_batch))
        ids = ids.unflatten(0, (n_views, base_batch))
        valid = valid.unflatten(0, (n_views, base_batch))
        base_valid = valid[0]
        subset = _sample_channel_subset_mask(
            base_valid,
            min_channels=args.min_subset_channels,
            max_channels=args.max_subset_channels,
            require_omission=True,
        )
        active = base_valid.sum(dim=1) > subset.sum(dim=1)
        if not active.any():
            continue
        sources = _sample_counterfactual_sources(base_valid.cpu(), subset.cpu())
        subset_views = subset.repeat(n_views, 1).unflatten(0, (n_views, base_batch))
        shuffled_views = []
        for view in range(n_views):
            shuffled_views.append(
                _counterfactual_inputs(crops[view], base_valid, subset, [x.to(device) for x in sources])
            )
        shuffled = torch.stack(shuffled_views)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            flat_full = backbone(
                crops.flatten(0, 1),
                channel_ids=ids.flatten(0, 1),
                channel_valid_mask=valid.flatten(0, 1),
                is_training=True,
            )["x_norm_patchtokens"].unflatten(0, (n_views, base_batch))
            flat_subset = backbone(
                crops.flatten(0, 1),
                channel_ids=ids.flatten(0, 1),
                channel_valid_mask=subset_views.flatten(0, 1),
                is_training=True,
            )["x_norm_patchtokens"].unflatten(0, (n_views, base_batch))
            flat_shuffled = backbone(
                shuffled.flatten(0, 1),
                channel_ids=ids.flatten(0, 1),
                channel_valid_mask=valid.flatten(0, 1),
                is_training=True,
            )["x_norm_patchtokens"].unflatten(0, (n_views, base_batch))

        full = flat_full[:, active]
        subset_features = flat_subset[:, active]
        shuffled_features = flat_shuffled[:, active]
        if edge_src is None:
            edge_src, edge_dst = _local_edge_index(full.shape[-2], args.local_radius, device)
        active_count = int(active.sum())
        edge_subset = _edge_kernel(subset_features.flatten(0, 1), edge_src, edge_dst).unflatten(
            0, (n_views, active_count)
        )
        edge_full = _edge_kernel(full.flatten(0, 1), edge_src, edge_dst).unflatten(0, (n_views, active_count))
        edge_shuffled = _edge_kernel(shuffled_features.flatten(0, 1), edge_src, edge_dst).unflatten(
            0, (n_views, active_count)
        )
        raw_true = edge_full - edge_subset
        raw_shuffled = edge_shuffled - edge_subset
        raw_true_energy.append(raw_true.square().mean(dim=-1).sqrt())
        raw_shuffled_energy.append(raw_shuffled.square().mean(dim=-1).sqrt())
        raw_true_stability.append(F.cosine_similarity(raw_true[0], raw_true[1], dim=-1))
        raw_shuffled_stability.append(F.cosine_similarity(raw_shuffled[0], raw_shuffled[1], dim=-1))
        records.append(
            {
                "full": full.cpu().to(torch.float16),
                "subset": subset_features.cpu().to(torch.float16),
                "shuffled": shuffled_features.cpu().to(torch.float16),
            }
        )
        LOGGER.info(
            "batch=%d active=%d/%d true_energy=%.5f shuffled_energy=%.5f",
            batch_index,
            int(active.sum()),
            base_batch,
            float(raw_true_energy[-1].mean()),
            float(raw_shuffled_energy[-1].mean()),
        )

    if len(records) < 2:
        raise RuntimeError(f"Only collected {len(records)} active batches; need at least two")
    held_out_records = records[max(1, int(0.7 * len(records))) :]
    if args.predictor_mode == "feature":
        train_subset, train_full, _ = _prepare_predictor_data(
            records, train=True, patches_per_image=args.predictor_patches
        )
        validation_subset, validation_full, _ = _prepare_predictor_data(
            records, train=False, patches_per_image=args.predictor_patches
        )
        predictor, best_validation_loss = _train_predictor(
            train_subset,
            train_full,
            validation_subset=validation_subset,
            validation_full=validation_full,
            device=device,
            updates=args.predictor_updates,
            batch_size=args.predictor_batch_size,
            learning_rate=args.predictor_lr,
        )
        metrics = _evaluate_predictor(
            predictor,
            held_out_records,
            device=device,
            src=edge_src,
            dst=edge_dst,
        )
    else:
        train_subset, train_full, _ = _prepare_edge_predictor_data(records, train=True)
        validation_subset, validation_full, _ = _prepare_edge_predictor_data(records, train=False)
        predictor, best_validation_loss = _train_edge_predictor(
            train_subset,
            train_full,
            validation_subset=validation_subset,
            validation_full=validation_full,
            device=device,
            src=edge_src,
            dst=edge_dst,
            edge_dim=args.edge_predictor_dim,
            updates=args.predictor_updates,
            batch_size=args.predictor_batch_size,
            learning_rate=args.predictor_lr,
        )
        metrics = _evaluate_edge_predictor(
            predictor,
            held_out_records,
            device=device,
            src=edge_src,
            dst=edge_dst,
        )
    output = {
        "experiment": "conditional_morphology_graph_diagnostic",
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "design": {
            "full": "all available channels C",
            "subset": "random S subset with at least one omitted valid channel",
            "counterfactual": "omitted channels replaced by same-physical-index channels from another batch sample",
            "geometry": "same global crop geometry, independent DINO-style photometric transforms",
            "graph": f"local patch cosine graph, Chebyshev radius={args.local_radius}",
            "predictor_mode": args.predictor_mode,
        },
        "batches_collected": len(records),
        "predictor_best_validation_mse": best_validation_loss,
        "raw_full_minus_subset_energy": _mean_std(raw_true_energy),
        "raw_shuffled_minus_subset_energy": _mean_std(raw_shuffled_energy),
        "raw_true_stability": _mean_std(raw_true_stability),
        "raw_shuffled_stability": _mean_std(raw_shuffled_stability),
        "raw_stability_margin": _mean_std([a - b for a, b in zip(raw_true_stability, raw_shuffled_stability)]),
        "predictor": metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
