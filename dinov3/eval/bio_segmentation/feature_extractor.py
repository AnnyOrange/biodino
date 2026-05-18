"""
DINOv3 patch-feature pre-extractor for biological segmentation datasets.

Runs the frozen backbone ONCE over an entire dataset and caches the spatial
patch features to disk as a compressed .npz file.  Subsequent training of
linear probes then reads the cached features directly, avoiding repeated
expensive backbone forward passes.

Cache file format (.npz):
    features    : float16  [N, D, H_p, W_p]   D = embed_dim * n_layers
    sem_masks   : int16    [N, H, W]            semantic labels (255 = ignore)
    inst_maps   : int32    [N, H, W]            instance IDs (0 = bg); all-zero if unavailable
    orig_H      : int32    scalar               original (after resize) image height
    orig_W      : int32    scalar               original (after resize) image width
    patch_size  : int32    scalar
    embed_dim   : int32    scalar
    n_layers    : int32    scalar

Usage:
    python -m dinov3.eval.bio_segmentation.feature_extractor \\
        --dataset    monuseg \\
        --data-root  /data1/xuzijing/dataset/monuseg/extracted \\
        --checkpoint /data1/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \\
        --train-config dinov3/configs/train/microscopy_continual_vitl16.yaml \\
        --output-dir ./cache/monuseg \\
        --split train \\
        --img-size 448 \\
        --n-layers 4 \\
        --batch-size 8

    # Run for all splits:
    for SPLIT in train val test; do
        python -m dinov3.eval.bio_segmentation.feature_extractor \\
            --dataset monuseg --split $SPLIT --data-root ... --checkpoint ... --train-config ...
    done
"""

import argparse
import gc
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import zipfile

import numpy as np
import torch
import torch.nn as nn
from numpy.lib import format as npy_format
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .constants import MICRO_RGB_MEAN, MICRO_RGB_STD
from .model_utils import load_dinov3_backbone

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# ============================================================================
# Per-dataset canonical image sizes
# ============================================================================

# These sizes are chosen to:
#   1. Match the natural image size of each dataset (no unnecessary up/downscale).
#   2. Be a multiple of the ViT patch size (16).
#   3. For large-image datasets, use 512 which aligns with the official
#      DINOv3 segmentation crop_size=512 ("whole" inference mode).
#
# Reference: eval/segmentation/inference.py line 75 — official whole-image
# inference resizes to 512×512 before backbone forward.
#
# Using 0 as a placeholder means "no override; rely on the dataset class's
# own default size (usually its native resolution rounded to patch multiples)".

DATASET_DEFAULT_IMG_SIZES: Dict[str, int] = {
    'bbbc038':   512,  # variable original sizes → canonical 512
    'conic':     256,  # all patches are natively 256×256 (no resize needed)
    'livecell':  512,  # ~520×696 tif images → 512
    'monuseg':   512,  # 1000×1000 H&E → 512  (slide inference for M2F)
    'pannuke':   256,  # all patches are natively 256×256
    'tissuenet': 256,  # fluorescence patches, native ~256
    'cellpose':  512,
    'csc':       512,
}
logger = logging.getLogger('feature_extractor')


# ============================================================================
# Feature extraction
# ============================================================================

def _prepare_input_channels(imgs: torch.Tensor, backbone: nn.Module) -> torch.Tensor:
    """Match eval image channels to the backbone without padding ChannelViT inputs."""
    expected = int(getattr(backbone, "in_chans", imgs.shape[1]))
    current = int(imgs.shape[1])

    if getattr(backbone, "enable_channelvit", False):
        if not getattr(backbone, "_bioseg_channel_align_logged", False):
            if current <= expected:
                logger.info(
                    "ChannelViT eval uses %s real input channel(s) with channel_embed capacity=%s; "
                    "not padding missing channels.",
                    current,
                    expected,
                )
            else:
                logger.info(
                    "ChannelViT eval input has %s channel(s), exceeding channel_embed capacity=%s; "
                    "trimming to the first %s channel(s).",
                    current,
                    expected,
                    expected,
                )
            setattr(backbone, "_bioseg_channel_align_logged", True)

        if current > expected:
            return imgs[:, :expected]
        return imgs

    if current == expected:
        return imgs

    if not getattr(backbone, "_bioseg_channel_align_logged", False):
        logger.info(
            "Aligning eval input channels from %s to backbone.in_chans=%s.",
            current,
            expected,
        )
        setattr(backbone, "_bioseg_channel_align_logged", True)

    if current < expected:
        pad_shape = (imgs.shape[0], expected - current, imgs.shape[2], imgs.shape[3])
        padding = imgs.new_zeros(pad_shape)
        return torch.cat([imgs, padding], dim=1)

    return imgs[:, :expected]


def _channelvit_spatial_features(
    backbone: nn.Module,
    imgs: torch.Tensor,
    n_layers: Union[int, List[int]],
    channel_aggregation: str = "mean",
) -> Tuple[torch.Tensor, ...]:
    """
    Convert ChannelViT's C*H*W token sequence into dense H*W features.

    ChannelViT emits one token per input channel and spatial patch.  For a
    segmentation probe we need one feature vector per spatial patch, so the
    channel axis is folded back explicitly instead of using the standard ViT
    reshape path.
    """
    token_outputs = backbone.get_intermediate_layers(
        imgs,
        n=n_layers,
        reshape=False,
        return_class_token=False,
    )
    patch_size = int(getattr(backbone, "patch_size", 16))
    h_patch = imgs.shape[2] // patch_size
    w_patch = imgs.shape[3] // patch_size
    spatial_tokens = h_patch * w_patch

    features: List[torch.Tensor] = []
    for tokens in token_outputs:
        bsz, n_tokens, dim = tokens.shape
        if n_tokens % spatial_tokens != 0:
            raise RuntimeError(
                "Cannot reshape ChannelViT tokens: "
                f"tokens={n_tokens}, spatial={spatial_tokens}."
            )

        n_channels = n_tokens // spatial_tokens
        tokens = tokens.reshape(bsz, n_channels, h_patch, w_patch, dim)
        if channel_aggregation == "mean":
            feat = tokens.mean(dim=1).permute(0, 3, 1, 2).contiguous()
        elif channel_aggregation == "concat":
            feat = tokens.permute(0, 1, 4, 2, 3).reshape(
                bsz, n_channels * dim, h_patch, w_patch
            ).contiguous()
        else:
            raise ValueError(f"Unknown channel_aggregation={channel_aggregation!r}")
        features.append(feat)

    return tuple(features)


@torch.inference_mode()
def extract_features(
    backbone:    nn.Module,
    dataset:     Dataset,
    n_layers:    Union[int, List[int]] = 4,
    batch_size:  int = 8,
    num_workers: int = 2,
    device:      torch.device = torch.device('cuda'),
    desc:        str = 'Extracting',
    return_chunks: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the backbone over every sample in *dataset* and collect:
        1. Spatial patch features  [N, D, H_p, W_p]  (float16)
        2. Semantic masks          [N, H, W]           (int16)
        3. Instance maps           [N, H, W]           (int32)

    The dataset's ``__getitem__`` must return one of:
        (img_tensor [3,H,W], sem_mask [H,W])
        (img_tensor [3,H,W], sem_mask [H,W], inst_map [H,W])

    Args:
        backbone    : frozen DINOv3 backbone in eval mode.
        dataset     : any dataset returning the format above.
        n_layers    : layer specification for get_intermediate_layers.
                      - int  → last n layers (e.g. n=4 → layers [-4,-3,-2,-1]).
                      - List → specific layer indices (e.g. [4,11,17,23] for ViT-L).
                      Use a List to align with the official FOUR_EVEN_INTERVALS
                      multi-scale strategy for dense prediction.
        batch_size  : inference batch size.
        num_workers : DataLoader workers.
        device      : computation device.
        desc        : tqdm description string.

    Returns:
        (features, sem_masks, inst_maps) as NumPy arrays.
    """
    backbone = backbone.to(device)
    backbone.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    all_feats:  List[np.ndarray] = []
    all_sem:    List[np.ndarray] = []
    all_inst:   List[np.ndarray] = []

    for batch in tqdm(loader, desc=desc):
        # Unpack batch: supports 2-tuple (img, sem) or 3-tuple (img, sem, inst)
        if len(batch) == 3:
            imgs, sem, inst = batch
        else:
            imgs, sem = batch
            # sem here may be a binary mask OR an instance map depending on the
            # dataset.  For instance-only datasets the inst field carries the
            # instance IDs; sem becomes a derived binary mask.
            inst = torch.zeros_like(sem)

        imgs = imgs.to(device)   # [B, C, H, W]
        imgs = _prepare_input_channels(imgs, backbone)

        # -------------------------------------------------------------------
        # Backbone forward: extract intermediate spatial patch features.
        # get_intermediate_layers(n=k) returns the LAST k layers when k is
        # an int; returns tuple of tensors [B, C, H_p, W_p] when reshape=True
        # and return_class_token=False.
        # -------------------------------------------------------------------
        with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
            if getattr(backbone, "enable_channelvit", False):
                feats_list = _channelvit_spatial_features(backbone, imgs, n_layers)
            else:
                feats_list = backbone.get_intermediate_layers(
                    imgs,
                    n=n_layers,
                    reshape=True,
                    return_class_token=False,
                )
            # Each element: [B, C, H_p, W_p] – concatenate along channel axis
            feats = torch.cat(feats_list, dim=1).float()  # [B, D, H_p, W_p]

        feats = feats  # already float32 after .float() above

        all_feats.append(feats.half().cpu().numpy())  # store as float16 to save disk space
        all_sem.append(sem.numpy().astype(np.int16))   # semantic class map
        all_inst.append(inst.numpy().astype(np.int32)) # instance IDs (0 if unavailable)

    if return_chunks:
        n_samples = sum(x.shape[0] for x in all_feats)
        logger.info(
            "Extracted %d samples as %d cache chunks for %s: first feature chunk %s",
            n_samples,
            len(all_feats),
            desc,
            all_feats[0].shape if all_feats else None,
        )
        return all_feats, all_sem, all_inst

    logger.info("Concatenating %d feature batches for %s", len(all_feats), desc)
    features  = np.concatenate(all_feats, axis=0)  # [N, D, H_p, W_p]
    sem_masks = np.concatenate(all_sem,   axis=0)  # [N, H, W]
    inst_maps = np.concatenate(all_inst,  axis=0)  # [N, H, W]

    logger.info(
        f"Extracted {len(features)} samples: "
        f"features {features.shape}, dtype={features.dtype}"
    )
    return features, sem_masks, inst_maps


def save_cache(
    out_path: str,
    features:  Union[np.ndarray, List[np.ndarray]],
    sem_masks: Union[np.ndarray, List[np.ndarray]],
    inst_maps: Union[np.ndarray, List[np.ndarray]],
    patch_size: int,
    embed_dim:  int,
    n_layers:   int,
    compressed: bool = True,
):
    """Save pre-extracted features and labels to a .npz file."""
    first_sem = sem_masks[0] if isinstance(sem_masks, list) else sem_masks
    orig_H, orig_W = first_sem.shape[1], first_sem.shape[2]
    save_fn = np.savez_compressed if compressed else np.savez
    logger.info(
        "Saving cache (%s) -> %s",
        "compressed" if compressed else "uncompressed",
        out_path,
    )
    metadata = {
        'orig_H': np.int32(orig_H),
        'orig_W': np.int32(orig_W),
        'patch_size': np.int32(patch_size),
        'embed_dim': np.int32(embed_dim),
        'n_layers': np.int32(n_layers),
    }
    if isinstance(features, list):
        arrays = {
            'chunked': np.int8(1),
            'num_chunks': np.int32(len(features)),
            'num_samples': np.int32(sum(x.shape[0] for x in features)),
            **metadata,
        }
        for i, (feat, sem, inst) in enumerate(zip(features, sem_masks, inst_maps)):
            key = f"{i:04d}"
            arrays[f'features_{key}'] = feat
            arrays[f'sem_masks_{key}'] = sem
            arrays[f'inst_maps_{key}'] = inst
        save_fn(out_path, **arrays)
    else:
        save_fn(
            out_path,
            features   = features,
            sem_masks  = sem_masks,
            inst_maps  = inst_maps,
            chunked    = np.int8(0),
            **metadata,
        )
    size_mb = os.path.getsize(out_path + '.npz') / 1024 / 1024 if os.path.exists(out_path + '.npz') \
              else os.path.getsize(out_path) / 1024 / 1024
    logger.info(f"Saved cache → {out_path}  ({size_mb:.1f} MB)")


def _write_npz_array(zf: zipfile.ZipFile, key: str, array: np.ndarray) -> None:
    """Write one .npy member into an open .npz zip without keeping all arrays in RAM."""
    with zf.open(f"{key}.npy", "w", force_zip64=True) as handle:
        npy_format.write_array(handle, np.asarray(array), allow_pickle=False)


@torch.inference_mode()
def extract_features_to_cache(
    out_path: str,
    backbone: nn.Module,
    dataset: Dataset,
    n_layers: Union[int, List[int]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str,
    patch_size: int,
    embed_dim: int,
    n_layers_scalar: int,
    compressed: bool = False,
) -> None:
    """
    Streaming chunked cache writer.

    The older chunked path still accumulated every batch in Python lists before
    writing the .npz.  TissueNet/PanNuke runs can therefore use tens of GB per
    process.  This path writes each batch directly as a separate .npy member in
    the .npz archive, so peak RAM is roughly one batch plus model/dataset state.
    """
    backbone = backbone.to(device)
    backbone.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    num_chunks = 0
    num_samples = 0
    orig_H: Optional[int] = None
    orig_W: Optional[int] = None

    logger.info(
        "Streaming chunked cache (%s) -> %s",
        "compressed" if compressed else "uncompressed",
        out_path,
    )

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=compression, allowZip64=True) as zf:
            for batch in tqdm(loader, desc=desc):
                if len(batch) == 3:
                    imgs, sem, inst = batch
                else:
                    imgs, sem = batch
                    inst = torch.zeros_like(sem)

                imgs = imgs.to(device)
                imgs = _prepare_input_channels(imgs, backbone)

                with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
                    if getattr(backbone, "enable_channelvit", False):
                        feats_list = _channelvit_spatial_features(backbone, imgs, n_layers)
                    else:
                        feats_list = backbone.get_intermediate_layers(
                            imgs,
                            n=n_layers,
                            reshape=True,
                            return_class_token=False,
                        )
                    feats = torch.cat(feats_list, dim=1).float()

                feat_np = feats.half().cpu().numpy()
                sem_np = sem.numpy().astype(np.int16)
                inst_np = inst.numpy().astype(np.int32)

                if orig_H is None:
                    orig_H, orig_W = int(sem_np.shape[1]), int(sem_np.shape[2])

                key = f"{num_chunks:04d}"
                _write_npz_array(zf, f"features_{key}", feat_np)
                _write_npz_array(zf, f"sem_masks_{key}", sem_np)
                _write_npz_array(zf, f"inst_maps_{key}", inst_np)
                num_chunks += 1
                num_samples += int(feat_np.shape[0])

                del imgs, sem, inst, feats, feats_list, feat_np, sem_np, inst_np
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if num_chunks % 50 == 0:
                    gc.collect()

            if orig_H is None:
                orig_H = orig_W = 0

            metadata = {
                'chunked': np.int8(1),
                'num_chunks': np.int32(num_chunks),
                'num_samples': np.int32(num_samples),
                'orig_H': np.int32(orig_H),
                'orig_W': np.int32(orig_W),
                'patch_size': np.int32(patch_size),
                'embed_dim': np.int32(embed_dim),
                'n_layers': np.int32(n_layers_scalar),
            }
            for key, value in metadata.items():
                _write_npz_array(zf, key, value)

        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    logger.info(
        "Saved streaming cache -> %s  (%d samples, %d chunks, %.1f MB)",
        out_path,
        num_samples,
        num_chunks,
        size_mb,
    )


def load_cache(cache_path: str) -> Dict[str, object]:
    """Load a feature cache created by ``save_cache``."""
    data = np.load(cache_path)
    return {
        'features':   data['features'],    # [N, D, H_p, W_p] float16
        'sem_masks':  data['sem_masks'],   # [N, H, W]         int16
        'inst_maps':  data['inst_maps'],   # [N, H, W]         int32
        'orig_H':     int(data['orig_H']),
        'orig_W':     int(data['orig_W']),
        'patch_size': int(data['patch_size']),
        'embed_dim':  int(data['embed_dim']),
        'n_layers':   int(data['n_layers']),
    }


# ============================================================================
# Dataset builder (supports all registered datasets)
# ============================================================================

def _build_dataset(
    dataset_name: str,
    data_root: str,
    split: str,
    img_size: Optional[int],
    augment: bool = False,
    rgb_mean=MICRO_RGB_MEAN,
    rgb_std=MICRO_RGB_STD,
    do_normalize: bool = True,
) -> Dataset:
    """
    Build a dataset instance from the registry.

    Args:
        img_size : target square side length for resizing every sample.
                   Pass None (or 0) to keep images at native resolution —
                   required for Mask2Former (random-crop training + sliding-
                   window evaluation).

    Loader types:
        'file'  : get_paths_fn(root, split) → (img_paths, mask_paths)
        'coco'  : get_paths_fn(root, split) → (coco_json, img_root)
        'array' : dataset-specific constructor arguments

    Split availability per dataset:
        train / val / test : LIVECell, TissueNet, CoNIC (auto-split)
        train / val* / test: BBBC038, MoNuSeg  (*auto val subset from train)
        train / val* / test: PanNuke  (folds 1+2=train, fold 3=val+test)
    """
    from .datasets import DATASET_REGISTRY

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    DatasetClass, get_paths_fn, loader_type = DATASET_REGISTRY[dataset_name]
    # size=None → native resolution (Mask2Former); tuple → fixed resize (feature cache)
    size = None if (img_size is None or img_size == 0) else (img_size, img_size)

    if loader_type == 'file':
        img_paths, mask_paths = get_paths_fn(data_root, split=split)
        return DatasetClass(
            img_paths,
            mask_paths,
            size=size,
            augment=augment,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            do_normalize=do_normalize,
        )

    elif loader_type == 'coco':
        coco_json, img_root = get_paths_fn(data_root, split=split)
        return DatasetClass(
            coco_json,
            img_root,
            size=size,
            augment=augment,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            do_normalize=do_normalize,
        )

    elif loader_type == 'array':
        if dataset_name == 'conic':
            images_npy, labels_npy, indices = get_paths_fn(data_root, split=split)
            return DatasetClass(
                images_npy,
                labels_npy,
                indices=indices,
                size=size,
                augment=augment,
                rgb_mean=rgb_mean,
                rgb_std=rgb_std,
                do_normalize=do_normalize,
            )
        elif dataset_name == 'pannuke':
            fold_dirs = get_paths_fn(data_root)
            split_map = {'train': [1, 2], 'val': [3], 'test': [3]}
            folds = split_map.get(split, [1, 2, 3])
            return DatasetClass(
                fold_dirs,
                split_folds=folds,
                size=size,
                augment=augment,
                rgb_mean=rgb_mean,
                rgb_std=rgb_std,
                do_normalize=do_normalize,
            )
        elif dataset_name == 'tissuenet':
            npz_path = get_paths_fn(data_root, split=split)
            return DatasetClass(
                npz_path,
                size=size,
                augment=augment,
                rgb_mean=rgb_mean,
                rgb_std=rgb_std,
                do_normalize=do_normalize,
            )
        else:
            raise ValueError(f"Unsupported array dataset: {dataset_name}")

    raise ValueError(f"Unknown loader type '{loader_type}' for dataset '{dataset_name}'")


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Pre-extract DINOv3 features for bio-segmentation')
    parser.add_argument('--dataset',    required=True,
                        help='Dataset name (e.g. monuseg, livecell, pannuke ...)')
    parser.add_argument('--data-root',  required=True,
                        help='Path to extracted dataset root')
    parser.add_argument('--checkpoint', required=True,
                        help='DCP ckpt dir, or consolidated .pth (teacher/model/state_dict/flat)')
    parser.add_argument(
        '--train-config',
        required=True,
        help='Training YAML merged with ssl_default_config; must match checkpoint architecture.',
    )
    parser.add_argument('--output-dir', required=True,
                        help='Directory where .npz cache files will be saved')
    parser.add_argument('--split', default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to process (default: train)')
    parser.add_argument('--img-size',   type=int, default=0,
                        help='Image size for resizing to a square (H=W). '
                             'Use 0 (default) to apply the per-dataset canonical '
                             'size from DATASET_DEFAULT_IMG_SIZES (e.g. 256 for '
                             'CoNIC/PanNuke, 512 for MoNuSeg/LIVECell). '
                             'Sizes are automatically rounded to multiples of 16 '
                             '(ViT patch size). Aligns with the official DINOv3 '
                             'segmentation crop_size=512 strategy.')
    parser.add_argument('--layers',     type=int, nargs='+', default=None,
                        help='Specific layer indices to extract (e.g. --layers 4 11 17 23). '
                             'Default (not set): last layer only (n=1), matching the official '
                             'backbone_out_layers: LAST used for linear segmentation. '
                             'For multi-layer experiments: '
                             'ViT-L  → --layers 4 11 17 23 (FOUR_EVEN_INTERVALS), '
                             'ViT-7B → --layers 9 19 29 39.')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Inference batch size (default: 8)')
    parser.add_argument('--num-workers',type=int, default=4)
    parser.add_argument('--no-compress-cache', action='store_true',
                        help='Save cache with np.savez instead of np.savez_compressed. '
                             'This uses more disk but avoids very slow CPU compression '
                             'for large multi-layer feature tensors.')
    parser.add_argument('--chunked-cache', action='store_true',
                        help='Save each extraction batch as a separate npz entry. '
                             'This avoids a large final np.concatenate step and is '
                             'recommended for multi-layer feature tensors.')
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Determine which layers to extract.
    #
    # Official FOUR_EVEN_INTERVALS (from eval/segmentation/models/__init__.py):
    #   ViT-L (24 blocks): [4, 11, 17, 23]   (hardcoded in the paper)
    #   ViT-g / 7B (40 blocks): [9, 19, 29, 39]  (i * 10 - 1 for i in 1..4)
    #
    # Using specific indices instead of "last-4" is critical for segmentation:
    # the [9,19,29,39] strategy captures shallow, mid, and deep features,
    # preserving spatial detail that gets lost in the final layers.
    # -----------------------------------------------------------------------
    if args.layers is not None:
        layers_to_extract: Union[int, List[int]] = args.layers
        layers_tag = 'custom_' + '_'.join(map(str, args.layers))
    else:
        # Default: LAST layer only.
        #
        # Rationale: the official DINOv3 linear segmentation config
        # (eval/segmentation/configs/config-ade20k-linear-training.yaml)
        # uses backbone_out_layers: LAST — only the final block's patch tokens.
        # Feature dimension: 1024 (ViT-L) or 4096 (ViT-g/7B) per patch.
        #
        # Mask2Former does NOT use this cache — it uses DINOv3_Adapter which
        # handles FOUR_EVEN_INTERVALS internally (see config-ade20k-m2f-inference.yaml).
        #
        # If you want to experiment with multi-layer features, pass e.g.
        #   --layers 4 11 17 23    (ViT-L, FOUR_EVEN_INTERVALS)
        #   --layers 9 19 29 39    (ViT-g / 7B, FOUR_EVEN_INTERVALS)
        layers_to_extract = 1   # int → get_intermediate_layers(n=1) → last layer
        layers_tag        = 'last1'

    logger.info(f"Layer extraction strategy: {layers_to_extract}  (tag: {layers_tag})")

    # -----------------------------------------------------------------------
    # Determine effective image size.
    #
    # Official DINOv3 segmentation (eval/segmentation/inference.py line 75)
    # uses 512×512 for "whole-image" inference.  For datasets whose patches are
    # natively smaller (e.g. CoNIC/PanNuke at 256×256), using 512 would
    # unnecessarily upsample and waste memory — use native 256 instead.
    # -----------------------------------------------------------------------
    if args.img_size == 0:
        img_size = DATASET_DEFAULT_IMG_SIZES.get(args.dataset, 512)
        logger.info(
            f"--img-size 0 → using per-dataset canonical size: {img_size} "
            f"(override with --img-size N if needed)"
        )
    else:
        # Round user-supplied size to nearest multiple of 16 (ViT patch size)
        img_size = (args.img_size // 16) * 16
        if img_size != args.img_size:
            logger.warning(
                f"--img-size {args.img_size} rounded to {img_size} "
                f"(must be a multiple of the ViT patch size 16)"
            )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build dataset
    logger.info(f"Building dataset: {args.dataset} / {args.split}  img_size={img_size}")
    dataset = _build_dataset(args.dataset, args.data_root, args.split, img_size)
    logger.info(f"Dataset size: {len(dataset)}")

    cfg_tag = Path(args.train_config).stem

    # Load backbone
    backbone = load_dinov3_backbone(
        args.checkpoint,
        train_config_path=args.train_config,
        device=device,
        freeze=True,
    )

    # Save path includes layer strategy and img_size for clarity.
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.split}_{cfg_tag}_{layers_tag}_s{img_size}.npz"
    )
    n_layers_scalar = (len(layers_to_extract) if isinstance(layers_to_extract, list)
                       else layers_to_extract)

    if args.chunked_cache:
        extract_features_to_cache(
            out_path,
            backbone,
            dataset,
            n_layers=layers_to_extract,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            desc=f'{args.dataset}/{args.split}',
            patch_size=backbone.patch_size,
            embed_dim=backbone.embed_dim,
            n_layers_scalar=n_layers_scalar,
            compressed=not args.no_compress_cache,
        )
        return

    # Extract
    features, sem_masks, inst_maps = extract_features(
        backbone, dataset,
        n_layers=layers_to_extract,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        desc=f'{args.dataset}/{args.split}',
        return_chunks=False,
    )

    save_cache(
        out_path, features, sem_masks, inst_maps,
        patch_size=backbone.patch_size,
        embed_dim=backbone.embed_dim,
        n_layers=n_layers_scalar,
        compressed=not args.no_compress_cache,
    )


if __name__ == '__main__':
    main()
