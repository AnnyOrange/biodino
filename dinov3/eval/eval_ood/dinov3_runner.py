from __future__ import annotations

import argparse
import csv
import logging
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dinov3.configs import get_default_config
from dinov3.models import build_model_from_cfg
from dinov3.checkpointer import init_model_from_checkpoint_for_evals
from dinov3.eval.bio_classification.common import parse_autocast_dtype
from dinov3.eval.bio_frozen_eval.encoder import CHANNEL_POLICIES, Dinov3CkptEncoder
from dinov3.eval.bio_segmentation.model_utils import load_dinov3_backbone
from dinov3.eval.eval_ood.datasets import (
    CryoParticleDataset,
    XrayTomogramSliceDataset,
    build_id_reference_dataset,
)
from dinov3.eval.eval_ood.metrics import (
    binary_classification_probe,
    classification_probe,
    clustering_metrics,
    dump_json,
    encode_strings,
    first_by_group,
    id_vs_ood_knn,
    mean_by_group,
    regression_probe,
    retrieval_metrics,
    xray_pair_retrieval,
)


logger = logging.getLogger("dinov3.eval_ood")


DINOV3_ROOT = Path(os.environ.get("DINOV3_ROOT", Path(__file__).resolve().parents[3]))
DEFAULT_BENCHMARK_ROOT = Path(os.environ.get("BENCHMARK_ROOT", "/mnt/huawei_deepcad/benchmark"))
DEFAULT_OOD_ROOT = Path(os.environ.get("OOD_ROOT", DEFAULT_BENCHMARK_ROOT / "ood"))


@dataclass(frozen=True)
class RunSpec:
    name: str
    output_dir: Path
    ckpt_root: Path
    train_config: Path
    suggested_iters: tuple[str, ...]


RUN_SPECS: dict[str, RunSpec] = {
    "base": RunSpec(
        name="base",
        output_dir=Path("/mnt/huawei_deepcad/dinov3/outputs/bio_continue_1025_a100_grad_acc_2_base"),
        ckpt_root=Path("/mnt/huawei_deepcad/dinov3/outputs/bio_continue_1025_a100_grad_acc_2_base/ckpt"),
        train_config=Path("/mnt/huawei_deepcad/dinov3/outputs/bio_continue_1025_a100_grad_acc_2_base/config.yaml"),
        suggested_iters=("1024", "4099", "8199", "16399"),
    ),
    "hplus_rgb3": RunSpec(
        name="hplus_rgb3",
        output_dir=Path("/mnt/huawei_deepcad/dinov3/outputs/bio_continue_rgb3_vith16plus"),
        ckpt_root=Path("/mnt/huawei_deepcad/dinov3/outputs/bio_continue_rgb3_vith16plus/ckpt"),
        train_config=Path("/mnt/huawei_deepcad/dinov3/outputs/bio_continue_rgb3_vith16plus/config.yaml"),
        suggested_iters=("1024", "2049", "4099", "14349"),
    ),
    "channelvit_s6_fixed": RunSpec(
        name="channelvit_s6_fixed",
        output_dir=Path("/mnt/huawei_deepcad/dinov3/outputs/bio_continue_true_channelvit_sample6_fixedinit"),
        ckpt_root=Path("/mnt/huawei_deepcad/dinov3/outputs/bio_continue_true_channelvit_sample6_fixedinit/ckpt"),
        train_config=Path("/mnt/huawei_deepcad/dinov3/outputs/bio_continue_true_channelvit_sample6_fixedinit/config.yaml"),
        suggested_iters=("1024", "8199", "16383", "30749"),
    ),
    "vitl_oep1025": RunSpec(
        name="vitl_oep1025",
        output_dir=Path("/mnt/huawei_deepcad/dinov3/outputs/bio_continue_vitL16_OEP1025_ep15_b1024_1025"),
        ckpt_root=Path("/mnt/huawei_deepcad/dinov3/outputs/bio_continue_vitL16_OEP1025_ep15_b1024_1025/ckpt"),
        train_config=Path("/mnt/huawei_deepcad/dinov3/outputs/bio_continue_vitL16_OEP1025_ep15_b1024_1025/config.yaml"),
        suggested_iters=("1024", "4099", "8199", "15374"),
    ),
}


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def maybe_init_dist_for_dcp(checkpoint: Path) -> None:
    if not checkpoint.is_dir():
        return
    import torch.distributed as dist

    if dist.is_initialized():
        return
    port = _free_tcp_port()
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=0,
        world_size=1,
    )


def resolve_checkpoint(ckpt_root: str | Path, ckpt_iter: str) -> Path:
    ckpt_root = Path(ckpt_root)
    if ckpt_iter == "latest":
        numeric = sorted(int(p.name) for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit())
        if not numeric:
            raise FileNotFoundError(f"No numeric checkpoint directories under {ckpt_root}")
        ckpt_iter = str(numeric[-1])
    ckpt_dir = ckpt_root / str(ckpt_iter)
    pth = ckpt_dir / "checkpoint.pth"
    if pth.exists():
        return pth
    if ckpt_dir.is_dir() and (ckpt_dir / ".metadata").exists():
        return ckpt_dir
    raise FileNotFoundError(f"No checkpoint.pth or DCP .metadata under {ckpt_dir}")


def load_local_backbone_fast(
    checkpoint: Path,
    train_config: Path,
    device: torch.device,
    checkpoint_key: str | None = "model",
) -> torch.nn.Module:
    """Load consolidated local DINOv3 checkpoints without AutoModel.

    The shared bio loader is still used for DCP directories.  For ordinary
    ``checkpoint.pth`` files this direct path avoids repeatedly reading multi-GB
    checkpoints while keeping the official DINOv3 eval model construction.
    """
    if checkpoint.is_dir():
        return load_dinov3_backbone(str(checkpoint), str(train_config), device=device, freeze=True)
    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(train_config))
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    model.to_empty(device=device)
    # Local training checkpoints are full SSL checkpoints whose top-level
    # ``model`` entry contains ``teacher.backbone.*``.  The eval checkpointer
    # then narrows that nested dict to the bare teacher backbone.
    init_model_from_checkpoint_for_evals(model, checkpoint, checkpoint_key=checkpoint_key)
    if str(getattr(cfg.student, "arch", "")) == "vit_7b":
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("Backbone ready: embed_dim=%s, patch_size=%s", model.embed_dim, model.patch_size)
    return model


def _collate(batch):
    images, labels, metas = zip(*batch)
    return list(images), np.asarray(labels), list(metas)


class Dinov3OODEncoder:
    def __init__(
        self,
        *,
        checkpoint: Path,
        train_config: Path,
        device: str,
        n_last_blocks: int,
        use_avgpool: bool,
        autocast_dtype: torch.dtype,
        resize_size: int,
        crop_size: int,
        checkpoint_key: str | None,
        channel_policy: str,
        channel_tta_samples: int,
        channel_policy_seed: int,
    ):
        if checkpoint_key not in {None, "model"}:
            raise ValueError("The shared frozen encoder currently supports checkpoint_key='model' only")
        self.encoder = Dinov3CkptEncoder(
            checkpoint=checkpoint,
            train_config=train_config,
            device=device,
            n_last_blocks=n_last_blocks,
            use_avgpool=use_avgpool,
            autocast_dtype=autocast_dtype,
            image_size=crop_size,
            resize_size=resize_size,
            channel_policy=channel_policy,
            channel_tta_samples=channel_tta_samples,
            channel_policy_seed=channel_policy_seed,
        )
        # Keep PIL/tensor inputs untransformed in workers. The shared encoder
        # applies the same resize, normalization, and channel policy as the
        # classification/retrieval frozen probes.
        self.transform = None

    @torch.inference_mode()
    def encode_batch(self, images: list) -> np.ndarray:
        return self.encoder.encode_images(images).astype(np.float32)


def extract_features(
    dataset,
    encoder: Dinov3OODEncoder,
    *,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if output_path.exists() and not overwrite:
        return load_feature_cache(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    metas_all: list[dict[str, Any]] = []
    for i, (images, y, metas) in enumerate(loader, 1):
        features.append(encoder.encode_batch(images))
        labels.append(np.asarray(y))
        metas_all.extend(metas)
        if i == 1 or i % 25 == 0 or i == len(loader):
            logger.info("[%s] %d/%d samples", desc, len(metas_all), len(dataset))
    feats = np.concatenate(features, axis=0)
    labs = np.concatenate(labels, axis=0)
    np.savez_compressed(
        output_path,
        features=feats.astype(np.float16),
        labels=labs,
        metas=np.asarray(metas_all, dtype=object),
    )
    return feats, labs, metas_all


def load_feature_cache(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature cache does not exist: {path}")
    with np.load(path, allow_pickle=True) as pack:
        metas = pack["metas"].tolist() if "metas" in pack.files else []
        return pack["features"].astype(np.float32), np.asarray(pack["labels"]), metas


def _xray_metrics(features: np.ndarray, metas: list[dict[str, Any]], *, seed: int) -> dict[str, float]:
    volume_ids = np.asarray([m["volume_id"] for m in metas])
    vol_features, unique_volumes = mean_by_group(features, volume_ids)
    tomo_ids = first_by_group(np.asarray([m["tomo_id"] for m in metas]), volume_ids)
    variants = first_by_group(np.asarray([m["variant"] for m in metas]), volume_ids)
    doses = first_by_group(np.asarray([m["dose"] for m in metas], dtype=np.float32), volume_ids)
    resolutions = first_by_group(np.asarray([m["resolution"] for m in metas], dtype=np.float32), volume_ids)
    sample_labels, _ = encode_strings(first_by_group(np.asarray([m["sample_id"] for m in metas]), volume_ids))
    resin_labels, _ = encode_strings(first_by_group(np.asarray([m["resin_id"] for m in metas]), volume_ids))
    variant_labels, _ = encode_strings(variants)

    out: dict[str, float] = {
        "xray_n_slices": int(len(features)),
        "xray_n_volumes": int(len(vol_features)),
    }
    out.update({f"xray_{k}": v for k, v in xray_pair_retrieval(vol_features, tomo_ids, variants).items()})
    out.update({f"xray_dose_{k}": v for k, v in regression_probe(vol_features, doses, seed=seed).items()})
    out.update({f"xray_resolution_{k}": v for k, v in regression_probe(vol_features, resolutions, seed=seed).items()})
    out.update({f"xray_variant_{k}": v for k, v in classification_probe(vol_features, variant_labels, seed=seed).items()})
    if len(np.unique(sample_labels)) > 1:
        out.update({f"xray_sample_{k}": v for k, v in classification_probe(vol_features, sample_labels, seed=seed).items()})
    if len(np.unique(resin_labels)) > 1:
        out.update({f"xray_resin_{k}": v for k, v in classification_probe(vol_features, resin_labels, seed=seed).items()})
    return out


def _cryo_metrics(features: np.ndarray, metas: list[dict[str, Any]], *, seed: int) -> dict[str, float]:
    project_ids = np.asarray([m["project_id"] for m in metas])
    class_ids = np.asarray([m["class_id"] for m in metas], dtype=np.int64)
    project_labels, _ = encode_strings(project_ids)
    project_class_labels, _ = encode_strings([f"{p}:{c}" for p, c in zip(project_ids, class_ids)])
    quality = np.asarray([m["quality_score"] for m in metas], dtype=np.float32)

    out: dict[str, float] = {
        "cryo_n_particles": int(len(features)),
        "cryo_n_projects": int(len(np.unique(project_ids))),
        "cryo_n_project_classes": int(len(np.unique(project_class_labels))),
    }
    out.update({f"cryo_project_{k}": v for k, v in classification_probe(features, project_labels, seed=seed).items()})
    out.update({f"cryo_class_{k}": v for k, v in classification_probe(features, project_class_labels, seed=seed).items()})
    out.update({f"cryo_retrieval_{k}": v for k, v in retrieval_metrics(features, project_class_labels).items()})
    out.update({f"cryo_cluster_{k}": v for k, v in clustering_metrics(features, project_class_labels, seed=seed).items()})

    # The Cryo-IEF paper uses 0.7/0.3 quality tiers. Keep the middle tier out of
    # binary AUROC/AP so the metric focuses on clear good-vs-junk particles.
    mask = np.isfinite(quality) & ((quality >= 0.7) | (quality <= 0.3))
    if mask.any():
        q_labels = (quality[mask] >= 0.7).astype(np.int64)
        out.update({f"cryo_quality_{k}": v for k, v in binary_classification_probe(features[mask], q_labels, seed=seed).items()})
        out.update({f"cryo_quality_score_{k}": v for k, v in regression_probe(features[np.isfinite(quality)], quality[np.isfinite(quality)], seed=seed).items()})
        out["cryo_quality_n_binary"] = int(mask.sum())
    else:
        out["cryo_quality_auroc"] = float("nan")
        out["cryo_quality_average_precision"] = float("nan")
        out["cryo_quality_n_binary"] = 0
    return out


def append_summary(path: Path, row: dict[str, Any]) -> None:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        exists = path.exists()
        fieldnames = sorted(row.keys())
        if exists:
            with path.open(newline="") as f:
                existing = csv.DictReader(f)
                fieldnames = list(dict.fromkeys((existing.fieldnames or []) + fieldnames))
                old_rows = list(existing)
            if any(k not in (existing.fieldnames or []) for k in row):
                with path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for old in old_rows:
                        writer.writerow({k: old.get(k, "") for k in fieldnames})
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_one(args) -> dict[str, Any]:
    checkpoint = resolve_checkpoint(args.ckpt_root, args.ckpt_iter)
    model_name = f"{args.model_name}-{args.ckpt_iter}"
    out_dir = Path(args.output_dir) / args.model_name / str(args.ckpt_iter)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = None
    if args.phase != "metrics":
        logger.info("Loading %s checkpoint=%s", model_name, checkpoint)
        encoder = Dinov3OODEncoder(
            checkpoint=checkpoint,
            train_config=Path(args.train_config),
            device=args.device,
            n_last_blocks=args.n_last_blocks,
            use_avgpool=not args.no_avgpool,
            autocast_dtype=parse_autocast_dtype(args.autocast_dtype),
            resize_size=args.resize_size,
            crop_size=args.crop_size,
            checkpoint_key=args.checkpoint_key,
            channel_policy=args.channel_policy,
            channel_tta_samples=args.channel_tta_samples,
            channel_policy_seed=args.channel_policy_seed,
        )
    row: dict[str, Any] = {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "train_config": str(args.train_config),
        "n_last_blocks": int(args.n_last_blocks),
        "use_avgpool": not args.no_avgpool,
        "resize_size": int(args.resize_size),
        "crop_size": int(args.crop_size),
        "channel_policy": args.channel_policy,
        "channel_tta_samples": int(args.channel_tta_samples),
        "xray_input_mode": args.xray_input_mode,
        "xray_slices_per_volume": int(args.xray_slices_per_volume),
        "cryo_invert": bool(args.cryo_invert),
    }

    id_features = None
    if "ood" in args.metrics:
        id_cache = out_dir / "features/id_reference.npz"
        if args.phase == "metrics":
            id_features, _id_labels, _id_metas = load_feature_cache(id_cache)
        else:
            id_ds = build_id_reference_dataset(
                args.benchmark_root,
                transform=encoder.transform,
                dataset_names=args.id_datasets,
                max_samples=args.id_max_samples,
                seed=args.seed,
            )
            id_features, _id_labels, _id_metas = extract_features(
                id_ds,
                encoder,
                output_path=id_cache,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                overwrite=args.overwrite_features,
                desc=f"{model_name}:id",
            )

    if "xray" in args.tasks:
        xray_cache = out_dir / f"features/xray_{args.xray_input_mode}_spv{args.xray_slices_per_volume}.npz"
        if args.phase == "metrics":
            xray_features, _xray_labels, xray_metas = load_feature_cache(xray_cache)
        else:
            xray_ds = XrayTomogramSliceDataset(
                args.ood_root,
                transform=encoder.transform,
                slices_per_volume=args.xray_slices_per_volume,
                input_mode=args.xray_input_mode,
                percentiles=(args.percentile_low, args.percentile_high),
                max_volumes=args.xray_max_volumes,
            )
            xray_features, _xray_labels, xray_metas = extract_features(
                xray_ds,
                encoder,
                output_path=xray_cache,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                overwrite=args.overwrite_features,
                desc=f"{model_name}:xray",
            )
        if args.phase != "extract":
            row.update(_xray_metrics(xray_features, xray_metas, seed=args.seed))
            if id_features is not None:
                row.update({f"xray_ood_{k}": v for k, v in id_vs_ood_knn(id_features, xray_features, seed=args.seed).items()})

    if "cryo" in args.tasks:
        suffix = "inv" if args.cryo_invert else "raw"
        cryo_cache = out_dir / f"features/cryo_{suffix}_mpp{args.cryo_max_particles_per_project}.npz"
        if args.phase == "metrics":
            cryo_features, _cryo_labels, cryo_metas = load_feature_cache(cryo_cache)
        else:
            cryo_ds = CryoParticleDataset(
                args.ood_root,
                transform=encoder.transform,
                percentiles=(args.percentile_low, args.percentile_high),
                invert=args.cryo_invert,
                max_projects=args.cryo_max_projects,
                max_particles_per_project=args.cryo_max_particles_per_project,
                max_per_class=args.cryo_max_per_class,
                seed=args.seed,
            )
            cryo_features, _cryo_labels, cryo_metas = extract_features(
                cryo_ds,
                encoder,
                output_path=cryo_cache,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                overwrite=args.overwrite_features,
                desc=f"{model_name}:cryo",
            )
        if args.phase != "extract":
            row.update(_cryo_metrics(cryo_features, cryo_metas, seed=args.seed))
            if id_features is not None:
                row.update({f"cryo_ood_{k}": v for k, v in id_vs_ood_knn(id_features, cryo_features, seed=args.seed).items()})

    if args.phase == "extract":
        dump_json(
            out_dir / "features_complete.json",
            {"model": model_name, "tasks": list(args.tasks), "feature_dir": str(out_dir / "features")},
        )
        logger.info("Finished feature extraction for %s", model_name)
        return row

    dump_json(out_dir / "last_result.json", row)
    append_summary(Path(args.output_dir) / "summary.csv", row)
    append_summary(out_dir / "summary.csv", row)
    logger.info("Finished %s; result=%s", model_name, out_dir / "last_result.json")
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DINOv3 frozen-feature OOD benchmark for X-ray tomography and cryo-EM.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--ckpt-iter", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--output-dir", default="benchmark_runs/eval_ood")
    parser.add_argument("--ood-root", default=str(DEFAULT_OOD_ROOT))
    parser.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT))
    parser.add_argument("--tasks", nargs="+", default=["xray", "cryo"], choices=["xray", "cryo"])
    parser.add_argument("--metrics", nargs="+", default=["ood"], choices=["ood"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-last-blocks", type=int, default=1)
    parser.add_argument("--no-avgpool", action="store_true")
    parser.add_argument("--autocast-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--checkpoint-key", default="model", help="Top-level key for consolidated .pth checkpoints; local training runs use 'model'.")
    parser.add_argument("--channel-policy", default="auto", choices=CHANNEL_POLICIES)
    parser.add_argument("--channel-tta-samples", type=int, default=8)
    parser.add_argument("--channel-policy-seed", type=int, default=0)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--percentile-low", type=float, default=0.5)
    parser.add_argument("--percentile-high", type=float, default=99.5)
    parser.add_argument("--xray-input-mode", default="three_slices", choices=["slice", "three_slices"])
    parser.add_argument("--xray-slices-per-volume", type=int, default=8)
    parser.add_argument("--xray-max-volumes", type=int)
    parser.add_argument("--cryo-invert", action="store_true")
    parser.add_argument("--cryo-max-projects", type=int)
    parser.add_argument("--cryo-max-particles-per-project", type=int, default=20000)
    parser.add_argument("--cryo-max-per-class", type=int)
    parser.add_argument("--id-max-samples", type=int, default=3000)
    parser.add_argument("--id-datasets", nargs="+", default=["bloodmnist", "bbbc048", "cyclops"])
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "extract", "metrics"],
        help="Run end-to-end, cache GPU features only, or compute metrics from existing caches only.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    os.chdir(DINOV3_ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    args = parse_args(argv)
    run_one(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
