"""NeurIPS 2022 Multimodal CellSeg loader.

Certified manifests preserve the author's 1000 training, 101 tuning and all52
public test images, including both WSI images. Masks are paired within their
own release split. The existing probe evaluates binary semantic foreground;
these are not official instance-segmentation leaderboard scores.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from dinov3.utils.bio_io import read_bio_image_as_numpy

from .base import BioSegDataset

logger = logging.getLogger(__name__)


class MultimodalCellSegDataset(BioSegDataset):
    """RGB/gray microscopy cell segmentation images with instance-label masks."""

    def load_image(self, path: str) -> np.ndarray:
        return read_bio_image_as_numpy(path, target_channels=3, normalize=False)

    def load_mask(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Cannot read mask: {path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return (mask > 0).astype(np.int64)


def _image_num_pixels(path: Path) -> int | None:
    try:
        from PIL import Image

        with Image.open(path) as pil_img:
            width, height = pil_img.size
        return int(width) * int(height)
    except Exception:
        return None


def get_multimodal_cellseg_paths(
    data_root: str,
    split: str = "train",
    max_pixels: int = 50_000_000,
    split_protocol: str = "legacy",
) -> Tuple[List[str], List[str]]:
    """Read checksum-pinned author splits; retain every image, including WSI.

    ``max_pixels`` is retained for caller compatibility and does not filter data.
    The previous global basename-joined split is explicitly rejected.
    """

    root = Path(data_root)
    if split_protocol != 'neurips22-author-public52-v2':
        raise ValueError('Withdrawn Multimodal CellSeg basename-joined split: Tuning images were paired to Training masks. Use certified neurips22-author-public52-v2.')
    import hashlib,json
    project = Path(__file__).resolve().parents[4]
    readiness = json.loads((project/'outputs/02_eval_runs/unified_paper/multimodal_official_readiness.json').read_text())
    manifest = Path(readiness['manifests'][split])
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == readiness['manifest_sha256'][split]
    with manifest.open() as stream:rows = list(csv.DictReader(stream))
    images = [str(root/r['image_path']) for r in rows]
    masks = [str(root/r['mask_path']) for r in rows]
    assert len(images) == readiness['counts'][split]
    assert all(Path(p).is_file() for p in images+masks)
    return images,masks
