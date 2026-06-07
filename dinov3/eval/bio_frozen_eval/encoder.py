# Frozen DINOv3 feature extractor shared by the classification/regression and
# retrieval/clustering entry points.
#
# This is the exact encoder used by the reference harness
# (`benchmark_model/run_dinov3_ckpt_benchmark.py`): it reuses the repo's own
# backbone loader, eval transform (resize 256 / center-crop 224 / ImageNet
# norm), and `LinearFeatureModel`, then L2-normalises and casts to float16 —
# all of which are part of the protocol that produced the reported numbers.
from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dinov3.data.transforms import make_classification_eval_transform
from dinov3.eval.bio_classification.common import LinearFeatureModel, parse_autocast_dtype  # re-exported
from dinov3.eval.bio_segmentation.model_utils import load_dinov3_backbone

__all__ = [
    "parse_autocast_dtype",
    "resolve_checkpoint",
    "Dinov3CkptEncoder",
    "extract_features",
    "completed",
    "pil_collate",
]


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def maybe_init_dist_for_dcp(checkpoint: Path) -> None:
    """A DCP (torch.distributed.checkpoint) directory needs a process group to load."""
    if not Path(checkpoint).is_dir():
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


def resolve_checkpoint(ckpt_root: Path, ckpt_iter: str) -> Path:
    """Resolve <ckpt_root>/<iter> to a consolidated checkpoint.pth or a DCP dir."""
    ckpt_dir = Path(ckpt_root) / str(ckpt_iter)
    pth = ckpt_dir / "checkpoint.pth"
    if pth.exists():
        return pth
    if ckpt_dir.is_dir() and (ckpt_dir / ".metadata").exists():
        return ckpt_dir
    raise FileNotFoundError(f"No checkpoint.pth or DCP .metadata under {ckpt_dir}")


def pil_collate(batch):
    imgs, labels, paths = zip(*batch)
    return list(imgs), np.asarray(labels), list(paths)


class Dinov3CkptEncoder:
    def __init__(
        self,
        checkpoint: Path,
        train_config: Path,
        device: str,
        n_last_blocks: int,
        use_avgpool: bool,
        autocast_dtype: torch.dtype,
    ):
        self.device = torch.device(device)
        maybe_init_dist_for_dcp(checkpoint)
        backbone = load_dinov3_backbone(str(checkpoint), str(train_config), device=self.device, freeze=True)
        self.model = LinearFeatureModel(
            backbone, n_last_blocks=n_last_blocks, use_avgpool=use_avgpool, autocast_dtype=autocast_dtype
        )
        self.model.to(self.device).eval()
        self.transform = make_classification_eval_transform(resize_size=256, crop_size=224)

    @torch.inference_mode()
    def encode_pil(self, images: list) -> np.ndarray:
        x = torch.stack([self.transform(img) for img in images]).to(self.device, non_blocking=True)
        feat = self.model(x).float()
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy().astype(np.float16)


def extract_features(
    dataset,
    encoder: Dinov3CkptEncoder,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
    model_name: str,
    save_features: bool = True,
    save_paths: bool = False,
):
    """Extract (and optionally cache) frozen features. Returns (features_f32, labels)."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        try:
            pack = np.load(output_path, allow_pickle=True)
            return pack["features"].astype(np.float32), pack["labels"]
        except Exception as exc:
            print(f"[features] ignoring unreadable cache {output_path}: {type(exc).__name__}: {exc}", flush=True)
    if save_features:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=pil_collate, pin_memory=True,
    )
    feats, labels, paths = [], [], []
    for i, (imgs, y, p) in enumerate(loader, 1):
        feats.append(encoder.encode_pil(imgs))
        labels.append(np.asarray(y))
        paths.extend(p)
        if i == 1 or i % 50 == 0 or i == len(loader):
            print(f"[features] {model_name}: {len(paths)}/{len(dataset)}", flush=True)
    features = np.concatenate(feats, axis=0)
    labels_arr = np.concatenate(labels, axis=0)
    if save_features:
        payload = {"features": features, "labels": labels_arr, "model": model_name}
        if save_paths:
            payload["paths"] = np.asarray(paths)
        np.savez(output_path, **payload)
    return features.astype(np.float32), labels_arr


def completed(summary_path: Path, dataset: str, model: str) -> bool:
    """True if (dataset, model) already has an error-free row in summary.csv."""
    import csv

    summary_path = Path(summary_path)
    if not summary_path.exists():
        return False
    try:
        with summary_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("dataset") == dataset and row.get("model") == model and not row.get("error"):
                    return True
    except Exception:
        return False
    return False
