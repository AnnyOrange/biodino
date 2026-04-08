"""Dynamic tiling (no padding, dynamic stride).

Rules (from pipeline.md §Step 2):

* Both H and W ≤ ``small_threshold`` (900) → keep the full image, one crop.
* Otherwise, tile each axis independently:
    - axis ≤ ``patch_size`` (512): single start at 0, extent = axis length
    - axis > ``patch_size``:
        n = ⌈(axis − patch_size) / target_stride⌉
        stride = ⌊(axis − patch_size) / n⌋
        last start = axis − patch_size  (guarantees boundary coverage)
"""

import math
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class CropRegion:
    """Pixel-coordinate rectangle [y0, x0, y1, x1) in the source image."""

    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


def compute_crops(
    height: int,
    width: int,
    patch_size: int = 512,
    target_stride: int = 384,
    small_threshold: int = 900,
) -> List[CropRegion]:
    """Return the list of crop regions for an image of size (H, W).

    No padding is ever applied.  Small images (both sides ≤ 900) produce
    exactly one crop covering the full image.
    """
    if height <= small_threshold and width <= small_threshold:
        return [CropRegion(0, 0, height, width)]

    y_starts = _axis_starts(height, patch_size, target_stride)
    x_starts = _axis_starts(width, patch_size, target_stride)

    crops: List[CropRegion] = []
    for ys in y_starts:
        y_end = min(ys + patch_size, height)
        for xs in x_starts:
            x_end = min(xs + patch_size, width)
            crops.append(CropRegion(ys, xs, y_end, x_end))
    return crops


def _axis_starts(dim: int, patch_size: int, target_stride: int) -> List[int]:
    """Compute patch start positions along a single axis."""
    if dim <= patch_size:
        return [0]

    n_steps = math.ceil((dim - patch_size) / target_stride)
    if n_steps == 0:
        return [0]

    dynamic_stride = (dim - patch_size) // n_steps
    starts = [i * dynamic_stride for i in range(n_steps)]

    last_start = dim - patch_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts
