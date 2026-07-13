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
import torch.nn.functional as F
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
    "CHANNEL_POLICIES",
]

ROBUST_MC_MEAN = (0.514666, 0.488834, 0.498267)
ROBUST_MC_STD = (0.338707, 0.339202, 0.336091)
CHANNEL_POLICIES = ("auto", "native", "first3", "compact3", "zerofill3", "mean3", "sample3_tta")


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


def _config_get(container, key: str):
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    return getattr(container, key, None)


def _load_multichannel_stats(train_config: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Read robust microscopy eval stats from the train config if available."""
    mean, std = ROBUST_MC_MEAN, ROBUST_MC_STD
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(train_config)
        crops = _config_get(cfg, "crops")
        cfg_mean = _config_get(crops, "rgb_mean")
        cfg_std = _config_get(crops, "rgb_std")
        if cfg_mean is not None and cfg_std is not None:
            mean = tuple(float(x) for x in cfg_mean)
            std = tuple(float(x) for x in cfg_std)
            return mean, std
    except Exception:
        pass
    try:
        import yaml

        with Path(train_config).open() as f:
            cfg = yaml.safe_load(f) or {}
        crops = cfg.get("crops", {}) if isinstance(cfg, dict) else {}
        cfg_mean = crops.get("rgb_mean")
        cfg_std = crops.get("rgb_std")
        if cfg_mean is not None and cfg_std is not None:
            mean = tuple(float(x) for x in cfg_mean)
            std = tuple(float(x) for x in cfg_std)
    except Exception:
        pass
    return mean, std


def _cycle_stats(values: tuple[float, ...], channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not values:
        values = (0.0,)
    repeats = (channels + len(values) - 1) // len(values)
    expanded = (list(values) * repeats)[:channels]
    return torch.tensor(expanded, device=device, dtype=dtype).view(1, channels, 1, 1)


class Dinov3CkptEncoder:
    def __init__(
        self,
        checkpoint: Path,
        train_config: Path,
        device: str,
        n_last_blocks: int,
        use_avgpool: bool,
        autocast_dtype: torch.dtype,
        image_size: int = 224,
        resize_size: int = 256,
        channel_policy: str = "auto",
        channel_tta_samples: int = 8,
        channel_policy_seed: int = 0,
    ):
        if channel_policy not in CHANNEL_POLICIES:
            raise ValueError(f"Unknown channel_policy={channel_policy!r}; expected one of {CHANNEL_POLICIES}")
        if channel_tta_samples <= 0:
            raise ValueError(f"channel_tta_samples must be positive, got {channel_tta_samples}")
        self.device = torch.device(device)
        maybe_init_dist_for_dcp(checkpoint)
        backbone = load_dinov3_backbone(str(checkpoint), str(train_config), device=self.device, freeze=True)
        self.model = LinearFeatureModel(
            backbone, n_last_blocks=n_last_blocks, use_avgpool=use_avgpool, autocast_dtype=autocast_dtype
        )
        self.model.to(self.device).eval()
        self.image_size = image_size
        self.resize_size = resize_size
        self.transform = make_classification_eval_transform(resize_size=resize_size, crop_size=image_size)
        self.mc_mean, self.mc_std = _load_multichannel_stats(train_config)
        self.channel_policy = channel_policy
        self.channel_tta_samples = int(channel_tta_samples)
        self.channel_rng = torch.Generator(device="cpu")
        self.channel_rng.manual_seed(int(channel_policy_seed))

    @torch.inference_mode()
    def encode_pil(self, images: list) -> np.ndarray:
        x = torch.stack([self.transform(img) for img in images]).to(self.device, non_blocking=True)
        feat = self.model(x).float()
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy().astype(np.float16)

    def _resize_center_crop_tensor(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError(f"Expected tensor image shaped C,H,W, got {tuple(image.shape)}")
        image = image.to(dtype=torch.float32).clamp(0.0, 1.0)
        _, height, width = image.shape
        short = max(1, min(height, width))
        scale = float(self.resize_size) / float(short)
        new_h = max(self.image_size, int(round(height * scale)))
        new_w = max(self.image_size, int(round(width * scale)))
        image = image.unsqueeze(0)
        try:
            image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        except TypeError:
            image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
        image = image.squeeze(0)
        top = max(0, (new_h - self.image_size) // 2)
        left = max(0, (new_w - self.image_size) // 2)
        return image[:, top : top + self.image_size, left : left + self.image_size].contiguous()

    def _normalize_tensor_batch(self, x: torch.Tensor) -> torch.Tensor:
        channels = int(x.shape[1])
        mean = _cycle_stats(self.mc_mean, channels, x.device, x.dtype)
        std = _cycle_stats(self.mc_std, channels, x.device, x.dtype).clamp_min(1e-6)
        return (x - mean) / std

    @staticmethod
    def _real_channel_indices(valid_row: torch.Tensor, total_channels: int) -> torch.Tensor:
        idx = valid_row.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            idx = torch.arange(total_channels, dtype=torch.long)[:1]
        return idx.cpu()

    def _collapse_to_three_channels_once(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        policy: str,
    ) -> torch.Tensor:
        if policy == "auto":
            policy = "first3"
        if policy == "native":
            raise ValueError("channel_policy='native' requires a true multichannel backbone")
        if policy == "sample3_tta":
            policy = "sample3"

        bsz, total_channels, height, width = x.shape
        out = x.new_zeros(bsz, 3, height, width)
        for i in range(bsz):
            real_idx = self._real_channel_indices(valid[i], total_channels)
            n_real = int(real_idx.numel())
            if policy == "zerofill3":
                take = real_idx[:3]
                out[i, : int(take.numel())] = x[i, take.to(x.device)]
                continue
            if policy == "mean3":
                mean = x[i, real_idx.to(x.device)].mean(dim=0, keepdim=True)
                out[i] = mean.expand(3, -1, -1)
                continue
            if policy in {"first3", "compact3"}:
                take = real_idx[:3]
                if take.numel() < 3:
                    pad = take[-1:].expand(3 - take.numel())
                    take = torch.cat([take, pad], dim=0)
                out[i] = x[i, take.to(x.device)]
                continue
            if policy == "sample3":
                if n_real >= 3:
                    perm = torch.randperm(n_real, generator=self.channel_rng)[:3]
                    take = real_idx[perm]
                else:
                    draw = torch.randint(n_real, (3,), generator=self.channel_rng)
                    take = real_idx[draw]
                out[i] = x[i, take.to(x.device)]
                continue
            raise ValueError(f"Unknown channel collapse policy: {policy}")
        return out

    def _encode_collapsed_batch(self, batch: torch.Tensor, valid: torch.Tensor, policy: str) -> torch.Tensor:
        if policy == "sample3_tta":
            features = []
            for _ in range(self.channel_tta_samples):
                x = self._collapse_to_three_channels_once(batch, valid, "sample3_tta")
                x = self._normalize_tensor_batch(x).to(self.device, non_blocking=True)
                feat = self.model(x).float()
                features.append(torch.nn.functional.normalize(feat, dim=1))
            return torch.nn.functional.normalize(torch.stack(features, dim=0).mean(dim=0), dim=1)

        x = self._collapse_to_three_channels_once(batch, valid, policy)
        x = self._normalize_tensor_batch(x).to(self.device, non_blocking=True)
        feat = self.model(x).float()
        return torch.nn.functional.normalize(feat, dim=1)

    @torch.inference_mode()
    def encode_tensor(self, images: list[torch.Tensor]) -> np.ndarray:
        tensors = [self._resize_center_crop_tensor(img) for img in images]
        max_channels = max(int(t.shape[0]) for t in tensors)
        batch = torch.zeros(
            len(tensors),
            max_channels,
            self.image_size,
            self.image_size,
            dtype=torch.float32,
        )
        valid = torch.zeros(len(tensors), max_channels, dtype=torch.bool)
        for i, tensor in enumerate(tensors):
            channels = int(tensor.shape[0])
            batch[i, :channels] = tensor
            valid[i, :channels] = True

        backbone = self.model.backbone
        true_multichannel = getattr(backbone, "stem_type", None) in {
            "dualroute",
            "residual_mc",
            "rgb_extra_residual",
            "residual_mc_v2",
            "rgb_extra_residual_v2",
        } or getattr(backbone, "enable_channelvit", False)
        if true_multichannel and self.channel_policy in {"auto", "native"}:
            x = self._normalize_tensor_batch(batch).to(self.device, non_blocking=True)
            valid = valid.to(self.device, non_blocking=True)
            channel_ids = torch.arange(max_channels, dtype=torch.long, device=self.device)
            feat = self.model(x, channel_ids=channel_ids, channel_valid_mask=valid).float()
            feat = torch.nn.functional.normalize(feat, dim=1)
        else:
            if self.channel_policy == "native":
                raise ValueError("channel_policy='native' was requested for a non-multichannel backbone")
            feat = self._encode_collapsed_batch(batch, valid, self.channel_policy)
        return feat.cpu().numpy().astype(np.float16)

    @torch.inference_mode()
    def encode_images(self, images: list) -> np.ndarray:
        if images and torch.is_tensor(images[0]):
            return self.encode_tensor(images)
        return self.encode_pil(images)


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
        feats.append(encoder.encode_images(imgs))
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


def completed(
    summary_path: Path,
    dataset: str,
    model: str,
    image_size: int | None = None,
    resize_size: int | None = None,
    split: str | None = None,
    channel_policy: str | None = None,
    channel_tta_samples: int | None = None,
) -> bool:
    """True if (dataset/model/protocol) already has an error-free row in summary.csv."""
    import csv

    summary_path = Path(summary_path)
    if not summary_path.exists():
        return False
    try:
        with summary_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("dataset") == dataset and row.get("model") == model and not row.get("error"):
                    if split is not None:
                        row_split = row.get("split")
                        # Legacy rows have no split column and were produced by
                        # the historical internal 80/20 probe. Reuse them only
                        # for datasets that still use that same protocol.
                        if row_split:
                            if row_split != split:
                                continue
                        elif split != "internal-80-20":
                            continue
                    if image_size is None:
                        if _row_matches_channel_policy(row, channel_policy, channel_tta_samples):
                            return True
                        continue
                    row_image_size = row.get("image_size")
                    row_resize_size = row.get("resize_size")
                    if not _row_matches_channel_policy(row, channel_policy, channel_tta_samples):
                        continue
                    if not row_image_size and image_size == 224 and resize_size in (None, 256):
                        return True
                    if row_image_size == str(image_size) and row_resize_size == str(resize_size):
                        return True
    except Exception:
        return False
    return False


def _row_matches_channel_policy(
    row: dict,
    channel_policy: str | None,
    channel_tta_samples: int | None,
) -> bool:
    if channel_policy in (None, "", "auto"):
        return True
    row_policy = row.get("channel_policy")
    if row_policy != channel_policy:
        return False
    if channel_policy == "sample3_tta" and channel_tta_samples is not None:
        row_tta = row.get("channel_tta_samples")
        return row_tta in ("", None) or row_tta == str(channel_tta_samples)
    return True
