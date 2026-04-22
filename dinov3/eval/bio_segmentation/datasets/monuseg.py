"""
MoNuSeg (Multi-organ Nuclei Segmentation) dataset loader.

Dataset layout (after extraction):
    <root>/
        MoNuSegTrainingData/
            <img_id>.tif          H&E image  (1000 × 1000, RGB)
            <img_id>.xml          polygon annotation
        MoNuSegTestData/
            <img_id>.tif
            <img_id>.xml

XML annotation format:
    <Annotations>
      <Annotation>
        <Regions>
          <Region>
            <Vertices>
              <Vertex X="..." Y="..."/>
              ...
            </Vertices>
          </Region>
          ...
        </Regions>
      </Annotation>
    </Annotations>

Each Region is one nucleus instance.

Usage:
    from dinov3.eval.bio_segmentation.datasets.monuseg import MoNuSegDataset, get_monuseg_paths
    img_paths, xml_paths = get_monuseg_paths('/data1/xuzijing/dataset/monuseg/extracted', 'train')
    dataset = MoNuSegDataset(img_paths, xml_paths, size=(448, 448))
"""

import logging
import os
import xml.etree.ElementTree as ET
from glob import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np

import torch
from dinov3.utils.bio_io import read_bio_image_as_numpy, _normalize_to_float32
from .base import BioSegDataset

logger = logging.getLogger(__name__)


# ============================================================================
# XML → instance map
# ============================================================================

def _parse_xml_to_instance_map(xml_path: str, height: int, width: int) -> np.ndarray:
    """
    Parse a MoNuSeg XML annotation file and rasterise each nucleus Region
    into an integer instance map (0 = background, 1..N = nuclei).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # cv2.fillPoly requires int32 canvas (int64 is unsupported by OpenCV C++ backend)
    inst_map = np.zeros((height, width), dtype=np.int32)
    inst_id  = 1

    for region in root.iter('Region'):
        vertices = region.find('Vertices')
        if vertices is None:
            continue
        pts = []
        for vertex in vertices.findall('Vertex'):
            x = float(vertex.get('X', 0))
            y = float(vertex.get('Y', 0))
            pts.append([x, y])
        if len(pts) < 3:
            continue
        pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(inst_map, [pts_arr.reshape(-1, 2)], inst_id)
        inst_id += 1

    return inst_map


# ============================================================================
# Dataset class
# ============================================================================

class MoNuSegDataset(BioSegDataset):
    """
    MoNuSeg nucleus instance segmentation dataset.

    img_paths  : paths to .tif images
    mask_paths : paths to .xml annotation files (paired by basename)

    __getitem__ returns a 3-tuple:
        img_tensor  : [3, H, W] float32 in [0, 1]
        sem_tensor  : [H, W] int64  binary (0=bg, 1=nucleus)
        inst_tensor : [H, W] int64  instance IDs (0=bg, 1..N)
    """

    def load_image(self, path: str) -> np.ndarray:
        """
        Return raw image array (channels aligned).
        Normalization is handled centrally in __getitem__ (same contract as BioSegDataset).
        """
        return read_bio_image_as_numpy(path, target_channels=3, normalize=False)

    def _get_hw(self, img_path: str) -> Tuple[int, int]:
        """Return (H, W) of the image without loading the full array."""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image to get H,W: {img_path}")
        return img.shape[:2]

    def load_mask(self, xml_path: str) -> np.ndarray:
        """Binary semantic mask (0=bg, 1=nucleus). Paired image used for H,W."""
        # Infer image path: same directory + same stem, different extension
        base = os.path.splitext(xml_path)[0]
        img_path = None
        for ext in ('.tif', '.tiff', '.png', '.jpg'):
            cand = base + ext
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            # Fallback: same stem, search in Tissue Images/
            stem = os.path.basename(base)
            tissue_dir = os.path.join(os.path.dirname(os.path.dirname(xml_path)), 'Tissue Images')
            for ext in ('.tif', '.tiff', '.png'):
                cand = os.path.join(tissue_dir, stem + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
        if img_path is None:
            raise FileNotFoundError(f"Cannot find image paired with {xml_path}")
        h, w = self._get_hw(img_path)
        inst_map = _parse_xml_to_instance_map(xml_path, h, w)
        return (inst_map > 0).astype(np.int64)

    def get_instance_map(self, idx: int) -> np.ndarray:
        """Return integer instance map at native resolution (not resized)."""
        xml_path = self.mask_paths[idx]
        # Determine image size via load_mask helper logic
        base = os.path.splitext(xml_path)[0]
        img_path = None
        for ext in ('.tif', '.tiff', '.png', '.jpg'):
            cand = base + ext
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            stem      = os.path.basename(base)
            tissue_dir = os.path.join(os.path.dirname(os.path.dirname(xml_path)), 'Tissue Images')
            for ext in ('.tif', '.tiff', '.png'):
                cand = os.path.join(tissue_dir, stem + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
        if img_path is None:
            img_path = self.img_paths[idx]  # final fallback: use stored img_path
        h, w = self._get_hw(img_path)
        return _parse_xml_to_instance_map(xml_path, h, w)

    def __getitem__(self, idx: int):
        """
        Override to return 3-tuple (img, sem, inst) so that the feature
        extractor can cache both semantic and instance annotations.
        """
        img = self.load_image(self.img_paths[idx])
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        img = _normalize_to_float32(img)

        inst_map = self.get_instance_map(idx)

        # Only resize when a fixed output size is requested (feature-extractor
        # caching mode).  For Mask2Former, size=None keeps the native 1000×1000
        # resolution so that sliding-window evaluation is meaningful.
        if self.size is not None:
            h, w = self.size
            img      = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            inst_map = cv2.resize(inst_map.astype(np.float32), (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int64)

        sem_map = (inst_map > 0).astype(np.int64)

        if self.augment:
            if np.random.rand() > 0.5:
                img      = np.flip(img,      axis=1).copy()
                inst_map = np.flip(inst_map, axis=1).copy()
                sem_map  = np.flip(sem_map,  axis=1).copy()
            if np.random.rand() > 0.5:
                img      = np.flip(img,      axis=0).copy()
                inst_map = np.flip(inst_map, axis=0).copy()
                sem_map  = np.flip(sem_map,  axis=0).copy()

        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        if self.do_normalize:
            img_t = (img_t - self.rgb_mean) / self.rgb_std
        sem_t = torch.from_numpy(sem_map).long()
        inst_t = torch.from_numpy(inst_map).long()
        return img_t, sem_t, inst_t


# ============================================================================
# Path discovery
# ============================================================================

def get_monuseg_paths(
    data_root: str,
    split: str = 'train',
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Discover MoNuSeg (image_path, xml_path) pairs.

    MoNuSeg has no official validation split.  When split='val' is requested
    a random ``val_ratio`` fraction of the training data is held out
    (deterministic with ``seed``).  The held-out indices are saved to a file.

    Supported post-extraction layouts:

    Layout A – original download structure:
        <root>/MoNuSeg 2018 Training Data/
                  Tissue Images/  *.tif
                  Annotations/    *.xml
        <root>/MoNuSegTestData/
                  *.tif + *.xml  (same directory)

    Layout B – pre-split naming:
        <root>/MoNuSegTrainingData/   images & annotations side-by-side
        <root>/MoNuSegTestData/

    Args:
        data_root : root directory (typically <dst>/monuseg/extracted).
        split     : 'train', 'val', or 'test'.
        val_ratio : fraction of training data used as validation.
        seed      : RNG seed.

    Returns:
        (img_paths, xml_paths) – matched pairs.
    """
    import glob as _glob

    # val uses a subset of train data
    need_train = split in ('train', 'val')
    raw_split  = 'train' if need_train else 'test'

    if raw_split == 'train':
        candidates = [
            'MoNuSeg 2018 Training Data',
            'MoNuSegTrainingData',
            'Training',
        ]
    else:
        candidates = [
            'MoNuSegTestData',
            'MoNuSegTestingData',
            'Testing',
            'Test',
        ]

    # Search up to 2 levels deep
    split_dir = None
    for candidate in candidates:
        for depth_pat in [
            os.path.join(data_root, candidate),
            os.path.join(data_root, '*', candidate),
        ]:
            hits = sorted(_glob.glob(depth_pat))
            if hits and os.path.isdir(hits[0]):
                split_dir = hits[0]
                break
        if split_dir:
            break

    if split_dir is None:
        raise FileNotFoundError(
            f"Could not find '{raw_split}' split directory under {data_root}.\n"
            f"Expected one of: {candidates}\n"
            f"Contents of {data_root}: {os.listdir(data_root)}"
        )

    # Locate XML files (may be in Annotations/ sub-dir)
    xml_files = sorted(set(
        _glob.glob(os.path.join(split_dir, '*.xml')) +
        _glob.glob(os.path.join(split_dir, 'Annotations', '*.xml')) +
        _glob.glob(os.path.join(split_dir, '**', '*.xml'), recursive=True)
    ))
    xml_files = [x for x in xml_files if not os.path.basename(x).startswith('.')]

    all_imgs, all_xmls = [], []
    for xml_p in xml_files:
        base_name = os.path.splitext(os.path.basename(xml_p))[0]
        search_dirs = [
            os.path.dirname(xml_p),
            os.path.join(split_dir, 'Tissue Images'),
            os.path.join(split_dir, 'images'),
            split_dir,
        ]
        found_img = None
        for sdir in search_dirs:
            for ext in ('.tif', '.tiff', '.png', '.jpg'):
                cand = os.path.join(sdir, base_name + ext)
                if os.path.exists(cand):
                    found_img = cand
                    break
            if found_img:
                break
        if found_img:
            all_imgs.append(found_img)
            all_xmls.append(xml_p)
        else:
            logger.debug(f"[MoNuSeg] No image found for XML: {xml_p}")

    # For test split, return as-is
    if split == 'test':
        logger.info(f"[MoNuSeg test] {len(all_imgs)} pairs under {split_dir}")
        return all_imgs, all_xmls

    # Train / val split from training data
    total   = len(all_imgs)
    idx_file = os.path.join(data_root, 'monuseg_val_indices.npy')
    if os.path.exists(idx_file):
        val_idx = set(np.load(idx_file).tolist())
    else:
        rng     = np.random.default_rng(seed)
        perm    = rng.permutation(total)
        n_val   = max(1, int(total * val_ratio))
        val_idx = set(perm[:n_val].tolist())
        np.save(idx_file, np.array(sorted(val_idx)))

    if split == 'val':
        sel = [i for i in range(total) if i in val_idx]
    else:
        sel = [i for i in range(total) if i not in val_idx]

    img_paths = [all_imgs[i] for i in sel]
    xml_paths = [all_xmls[i] for i in sel]
    logger.info(f"[MoNuSeg {split}] {len(img_paths)}/{total} pairs under {split_dir}")
    return img_paths, xml_paths
