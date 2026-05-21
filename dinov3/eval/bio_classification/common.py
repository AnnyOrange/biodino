from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dinov3.data.transforms import CROP_DEFAULT_SIZE, RESIZE_DEFAULT_SIZE, make_classification_eval_transform
from dinov3.eval.bio_segmentation.model_utils import load_dinov3_backbone

logger = logging.getLogger(__name__)


def create_linear_input(x_tokens_list, use_n_blocks: int, use_avgpool: bool) -> torch.Tensor:
    """Same feature convention as DINOv3 eval.linear.create_linear_input."""
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat((output, torch.mean(intermediate_output[-1][0], dim=1)), dim=-1)
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearFeatureModel(torch.nn.Module):
    """Backbone wrapper that returns the exact vector used by DINOv3 linear heads."""

    def __init__(self, backbone: torch.nn.Module, n_last_blocks: int, use_avgpool: bool, autocast_dtype: torch.dtype):
        super().__init__()
        self.backbone = backbone
        self.n_last_blocks = n_last_blocks
        self.use_avgpool = use_avgpool
        self.autocast_dtype = autocast_dtype

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
            tokens = self.backbone.get_intermediate_layers(
                images,
                n=self.n_last_blocks,
                reshape=False,
                return_class_token=True,
            )
        return create_linear_input(tokens, use_n_blocks=self.n_last_blocks, use_avgpool=self.use_avgpool)


class ModelWithNormalize(torch.nn.Module):
    """DINOv3 k-NN convention: L2-normalize backbone output features."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model(samples), dim=1, p=2)


def parse_autocast_dtype(name: str) -> torch.dtype:
    key = name.lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported autocast dtype: {name}")



def _load_official_backbone_direct(arch: str, weights: str) -> torch.nn.Module:
    from dinov3.hub import backbones

    if not hasattr(backbones, arch):
        raise ValueError(f"Unknown DINOv3 backbone arch {arch!r}; expected a function in dinov3.hub.backbones")
    factory = getattr(backbones, arch)
    model = factory(weights=weights, pretrained=bool(weights))
    model.cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def load_backbone(
    *,
    repo_dir: str,
    arch: str,
    weights: str,
    checkpoint: str | None = None,
    train_config: str | None = None,
    normalize: bool = False,
) -> torch.nn.Module:
    """Load either a hub backbone or a train-compatible DINOv3 checkpoint."""
    if checkpoint:
        if not train_config:
            raise ValueError("--train-config is required when --checkpoint is used.")
        logger.info("Loading train-compatible backbone checkpoint=%s config=%s", checkpoint, train_config)
        model = load_dinov3_backbone(checkpoint, train_config_path=train_config, device=torch.device("cuda"), freeze=True)
    else:
        del repo_dir  # Kept for CLI compatibility; direct import avoids hubconf importing optional eval deps.
        logger.info("Loading official backbone arch=%s weights=%s", arch, weights)
        model = _load_official_backbone_direct(arch, weights)

    if normalize:
        model = ModelWithNormalize(model)
    return model


def build_camelyonpatch_dataset(
    *,
    data_root: str,
    split: str,
    train_transform: bool = False,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
):
    # DINOv3 k-NN/log-regression use deterministic eval transforms for feature extraction.
    # Linear pre-extraction should do the same unless the caller explicitly requests train aug.
    if train_transform:
        from dinov3.data.transforms import make_classification_train_transform

        transform = make_classification_train_transform(crop_size=crop_size)
    else:
        transform = make_classification_eval_transform(resize_size=resize_size, crop_size=crop_size)
    from dinov3.eval.bio_classification.datasets.camelyonpatch import CamelyonPatchDataset

    return CamelyonPatchDataset(root=data_root, split=split, transform=transform)


def build_classification_transform(
    *,
    train_transform: bool = False,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
):
    if train_transform:
        from dinov3.data.transforms import make_classification_train_transform

        return make_classification_train_transform(crop_size=crop_size)
    return make_classification_eval_transform(resize_size=resize_size, crop_size=crop_size)


def get_num_classes_from_dataset(dataset) -> int:
    if getattr(dataset, "NUM_CLASSES", None):
        return int(dataset.NUM_CLASSES)
    labels = dataset.get_targets() if hasattr(dataset, "get_targets") else [dataset.get_target(i) for i in range(len(dataset))]
    labels_t = torch.as_tensor(labels, dtype=torch.long).view(-1)
    return int(labels_t.max().item() + 1)


@torch.inference_mode()
def extract_features_simple(
    model: torch.nn.Module,
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    features, labels = [], []
    for i, (images, targets) in enumerate(loader):
        if i % 20 == 0:
            logger.info("Extracting %s: batch %d/%d", desc, i + 1, len(loader))
        images = images.cuda(non_blocking=True)
        features.append(model(images).float().cpu())
        labels.append(targets.long().cpu().view(-1))
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def compute_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    batch_size: int = 1024,
) -> Dict[str, float]:
    """Compute DINO-style classification metrics without importing torchmetrics."""
    del batch_size
    logits = logits.float().cpu()
    labels = labels.long().cpu().view(-1)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    accuracy = (preds == labels).float().mean().item() * 100.0

    per_class_acc = []
    per_class_f1 = []
    for cls in range(num_classes):
        target_pos = labels == cls
        pred_pos = preds == cls
        support = int(target_pos.sum().item())
        if support > 0:
            per_class_acc.append((preds[target_pos] == cls).float().mean().item())
        tp = (target_pos & pred_pos).sum().float()
        fp = ((~target_pos) & pred_pos).sum().float()
        fn = (target_pos & (~pred_pos)).sum().float()
        denom = 2 * tp + fp + fn
        if denom.item() > 0:
            per_class_f1.append((2 * tp / denom).item())
    balanced_accuracy = (sum(per_class_acc) / len(per_class_acc) * 100.0) if per_class_acc else float("nan")
    macro_f1 = (sum(per_class_f1) / len(per_class_f1) * 100.0) if per_class_f1 else float("nan")

    macro_auroc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        if num_classes == 2:
            macro_auroc = float(roc_auc_score(labels.numpy(), probs[:, 1].numpy())) * 100.0
        else:
            macro_auroc = float(roc_auc_score(labels.numpy(), probs.numpy(), multi_class="ovr", average="macro")) * 100.0
    except Exception as exc:  # pragma: no cover - optional sklearn / degenerate labels
        logger.warning("Could not compute macro_auroc: %s", exc)

    return {
        "accuracy_top1": accuracy,
        "balanced_accuracy_top1": balanced_accuracy,
        "macro_f1": macro_f1,
        "macro_auroc": macro_auroc,
    }


def topk_accuracy_from_scores(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    topks: Sequence[int] = (1,),
    average: str = "micro",
    num_classes: int | None = None,
) -> Dict[str, float]:
    labels = labels.long().view(-1)
    max_k = min(max(topks), scores.shape[1])
    top = scores.topk(max_k, dim=1).indices.cpu()
    out: Dict[str, float] = {}
    for k in topks:
        kk = min(k, scores.shape[1])
        hit = (top[:, :kk] == labels[:, None]).any(dim=1)
        if average == "macro":
            assert num_classes is not None
            vals = []
            for cls in range(num_classes):
                mask = labels == cls
                if mask.any():
                    vals.append(hit[mask].float().mean().item())
            value = sum(vals) / len(vals) if vals else float("nan")
        else:
            value = hit.float().mean().item()
        out[f"top-{k}"] = value * 100.0
    return out


def ensure_main_process_write() -> bool:
    import dinov3.distributed as distributed

    return (not torch.cuda.is_available()) or distributed.is_main_process()


def checkpoint_stem(checkpoint: str | None, weights: str) -> str:
    source = checkpoint or weights
    if not source:
        return "none"
    path = Path(source)
    if path.name == "checkpoint.pth" and path.parent.name.isdigit():
        return path.parent.name
    return path.stem
