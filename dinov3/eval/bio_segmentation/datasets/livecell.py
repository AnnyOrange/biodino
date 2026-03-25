"""
LIVECell dataset loader.

Dataset layout (after extraction):
    <root>/
        images/
            train/   <cell_type>/<img_name>.tif
            val/     ...
            test/    ...
        annotations/
            livecell_coco_train.json
            livecell_coco_val.json
            livecell_coco_test.json

Alternatively, a flat layout is also supported:
    <root>/
        train/  images/ + annotations/
        val/    ...
        test/   ...

Annotation format: COCO JSON with polygonal instance masks (single category "cell").

Usage:
    from dinov3.eval.bio_segmentation.datasets.livecell import LIVECellDataset, get_livecell_paths
    img_paths, ann_paths = get_livecell_paths('/data1/xuzijing/dataset/livecell/extracted', 'train')
    dataset = LIVECellDataset(img_paths, ann_paths, size=(448, 448))
"""

import json
import logging
import os
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dinov3.utils.bio_io import read_bio_image_as_numpy
from dinov3.eval.bio_segmentation.preprocessing import apply_preprocessing

logger = logging.getLogger(__name__)


# ============================================================================
# COCO polygon → instance mask converter (no pycocotools dependency)
# ============================================================================

def _poly_to_mask(polygons: List[List[float]], h: int, w: int) -> np.ndarray:
    """Rasterize a list of polygon vertex lists into a binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def _build_instance_map(annotations: List[dict], h: int, w: int) -> np.ndarray:
    """
    Convert a list of COCO annotation dicts for one image into an integer
    instance map (0 = background, 1..N = instance IDs).
    """
    inst_map = np.zeros((h, w), dtype=np.int64)
    for inst_id, ann in enumerate(annotations, start=1):
        seg = ann.get('segmentation', [])
        if isinstance(seg, list) and seg:
            binary = _poly_to_mask(seg, h, w)
            inst_map[binary > 0] = inst_id
        elif isinstance(seg, dict):
            # RLE format - decode with pycocotools if available
            try:
                from pycocotools import mask as coco_mask
                binary = coco_mask.decode(seg)
                inst_map[binary > 0] = inst_id
            except ImportError:
                logger.warning("pycocotools not available; skipping RLE annotation.")
    return inst_map


# ============================================================================
# Dataset class
# ============================================================================

class LIVECellDataset(Dataset):
    """
    LIVECell single-category cell instance segmentation dataset.

    Items are loaded lazily: masks are rasterised from COCO JSON on __getitem__.

    Returns:
        img_tensor  : [3, H, W] float32 in [0, 1]
        inst_tensor : [H, W] int64, instance IDs (0 = background)
    """

    def __init__(
        self,
        coco_json: str,
        img_root: str,
        size:    Optional[Tuple[int, int]] = None,
        augment: bool = False,
    ):
        """
        Args:
            coco_json : path to the COCO annotation JSON file for this split.
            img_root  : root directory containing image files.
            size      : (H, W) to resize to, or None to keep native resolution.
            augment   : random horizontal/vertical flips.
        """
        with open(coco_json) as f:
            data = json.load(f)

        self.img_root = img_root
        self.size     = size
        self.augment  = augment

        # Index images
        self._images: Dict[int, dict] = {img['id']: img for img in data['images']}
        # Group annotations by image_id
        self._anns: Dict[int, List[dict]] = {}
        for ann in data.get('annotations', []):
            self._anns.setdefault(ann['image_id'], []).append(ann)

        self._img_ids = list(self._images.keys())
        logger.info(f"LIVECell: {len(self._img_ids)} images from {coco_json}")

    def __len__(self) -> int:
        return len(self._img_ids)

    def _find_image_path(self, file_name: str) -> str:
        """Search for the image file recursively under img_root."""
        direct = os.path.join(self.img_root, file_name)
        if os.path.exists(direct):
            return direct
        # COCO file_name might include subdirectory like "train/..."
        base = os.path.basename(file_name)
        for root, _, files in os.walk(self.img_root):
            if base in files:
                return os.path.join(root, base)
        raise FileNotFoundError(f"Image not found: {file_name} under {self.img_root}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a 3-tuple:
            img_t  : [3, H, W] float32 in [0, 1]
            sem_t  : [H, W] int64  binary semantic (0=bg, 1=cell)
            inst_t : [H, W] int64  instance IDs   (0=bg, 1..N)
        """
        img_id   = self._img_ids[idx]
        img_info = self._images[img_id]
        img_path = self._find_image_path(img_info['file_name'])

        # Load image
        img = read_bio_image_as_numpy(img_path, target_channels=3, normalize=True)  # [H, W, 3]

        # Build instance map at original resolution, then resize together
        orig_h, orig_w = img.shape[:2]
        anns     = self._anns.get(img_id, [])
        inst_map = _build_instance_map(anns, orig_h, orig_w)

        # Resize only when a fixed output size is requested
        if self.size is not None:
            h, w     = self.size
            img      = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            inst_map = cv2.resize(inst_map.astype(np.float32), (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int64)
        sem_map = (inst_map > 0).astype(np.int64)

        # Augment
        if self.augment:
            if np.random.rand() > 0.5:
                img      = np.flip(img,      axis=1).copy()
                inst_map = np.flip(inst_map, axis=1).copy()
                sem_map  = np.flip(sem_map,  axis=1).copy()
            if np.random.rand() > 0.5:
                img      = np.flip(img,      axis=0).copy()
                inst_map = np.flip(inst_map, axis=0).copy()
                sem_map  = np.flip(sem_map,  axis=0).copy()

        img_t  = torch.from_numpy(img).permute(2, 0, 1).float()
        sem_t  = torch.from_numpy(sem_map).long()
        inst_t = torch.from_numpy(inst_map).long()
        return img_t, sem_t, inst_t

    def get_instance_map(self, idx: int) -> np.ndarray:
        """Return original-resolution instance map for metric computation."""
        img_id   = self._img_ids[idx]
        img_info = self._images[img_id]
        img_path = self._find_image_path(img_info['file_name'])
        img      = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read: {img_path}")
        h, w     = img.shape[:2]
        anns     = self._anns.get(img_id, [])
        return _build_instance_map(anns, h, w)

    def get_binary_mask(self, idx: int) -> np.ndarray:
        """Binary foreground mask (0/1) at output size."""
        _, inst_t = self[idx]
        return (inst_t.numpy() > 0).astype(np.int64)


# ============================================================================
# Path discovery
# ============================================================================

def get_livecell_paths(
    data_root: str,
    split: str = 'train',
) -> Tuple[str, str]:
    """
    Discover the COCO JSON path and image root for a given split.

    Expected layout after running extract_datasets.py for LIVECell
    (images.zip is extracted in-place inside LIVECell_dataset_2021/):

        <data_root>/
            LIVECell_dataset_2021/
                annotations/LIVECell/
                    livecell_coco_train.json
                    livecell_coco_val.json
                    livecell_coco_test.json
                images/
                    livecell_train_val_images/  *.tif
                    livecell_test_images/       *.tif

    Pass ``--data-root /data1/xuzijing/dataset/LIVECell`` when running
    experiments.

    Returns:
        (coco_json_path, image_root)
            image_root is the 'images/' directory (or data_root if not found).
    """
    import glob as _glob

    split_json_names = {
        'train': 'livecell_coco_train.json',
        'val':   'livecell_coco_val.json',
        'test':  'livecell_coco_test.json',
    }
    if split not in split_json_names:
        raise ValueError(f"Unknown split '{split}'. Choose 'train', 'val', or 'test'.")

    json_name = split_json_names[split]

    # Search recursively for the COCO JSON
    json_cands = sorted(_glob.glob(os.path.join(data_root, '**', json_name), recursive=True))
    if not json_cands:
        raise FileNotFoundError(
            f"Cannot find {json_name} under {data_root}.\n"
            f"Make sure to extract images.zip via extract_datasets.py first:\n"
            f"  python -m dinov3.eval.bio_segmentation.scripts.extract_datasets \\\n"
            f"      --datasets livecell \\\n"
            f"      --src-dir {data_root}/LIVECell_dataset_2021 \\\n"
            f"      --dst-dir <parent_dir>"
        )
    coco_json = json_cands[0]

    # Image root: find the 'images/' directory that holds the .tif files.
    # Priority: same tree as the JSON → fallback to data_root.
    img_root = data_root
    img_candidates = sorted(
        _glob.glob(os.path.join(data_root, '**', 'images'), recursive=True)
    )
    for p in img_candidates:
        if os.path.isdir(p) and any(
            _glob.glob(os.path.join(p, '**', '*.tif'), recursive=True)
        ):
            img_root = p
            break

    logger.info(f"[LIVECell {split}] json={coco_json}, img_root={img_root}")
    return coco_json, img_root
