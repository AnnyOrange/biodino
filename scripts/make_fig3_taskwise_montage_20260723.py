#!/usr/bin/env python3
"""Arrange existing task-wise report panels into a Fig. 3 montage preview."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("outputs/00_reports/20260708_taskwise_fm_figures_vertical_white")
OUTPUT = ROOT / "fig3_id_taskwise_montage_preview"

CANVAS_SIZE = (3600, 4580)
MARGIN_X = 105
MARGIN_Y = 90
GAP_X = 70
GAP_Y = 65
LABEL_SPACE = 58

PANELS = {
    "a": ROOT / "id_classification_balanced_accuracy.png",
    "b": ROOT / "id_segmentation_mdice.png",
    "c": ROOT / "id_retrieval_recall_at_1.png",
    "d": ROOT / "id_clustering_nmi.png",
    "e": ROOT / "id_regression_r2_mae_combined.png",
}


def trim_white(image: Image.Image, threshold: int = 250, padding: int = 18) -> Image.Image:
    rgb = image.convert("RGB")
    gray = rgb.convert("L")
    foreground = gray.point(lambda value: 255 if value < threshold else 0)
    bbox = foreground.getbbox()
    if bbox is None:
        return rgb
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(rgb.width, right + padding)
    bottom = min(rgb.height, bottom + padding)
    return rgb.crop((left, top, right, bottom))


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    return fitted


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def paste_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    label: str,
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    available = (right - left, bottom - top - LABEL_SPACE)
    fitted = fit_panel(image, available)
    x = left + (available[0] - fitted.width) // 2
    y = top + LABEL_SPACE + (available[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    draw.text((left + 4, top), label, fill="#1E2522", font=font(51))


def main() -> None:
    missing = [str(path) for path in PANELS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source panels: " + ", ".join(missing))

    images = {label: trim_white(Image.open(path)) for label, path in PANELS.items()}
    canvas = Image.new("RGB", CANVAS_SIZE, "white")
    draw = ImageDraw.Draw(canvas)

    inner_width = CANVAS_SIZE[0] - 2 * MARGIN_X
    column_width = (inner_width - GAP_X) // 2
    row_small = 1130
    row_large = 1940

    x1 = MARGIN_X
    x2 = MARGIN_X + column_width + GAP_X
    y1 = MARGIN_Y
    y2 = y1 + row_small + GAP_Y
    y3 = y2 + row_small + GAP_Y

    boxes = {
        "a": (x1, y1, x1 + column_width, y1 + row_small),
        "b": (x2, y1, x2 + column_width, y1 + row_small),
        "c": (x1, y2, x1 + column_width, y2 + row_small),
        "d": (x2, y2, x2 + column_width, y2 + row_small),
        "e": (MARGIN_X, y3, CANVAS_SIZE[0] - MARGIN_X, y3 + row_large),
    }

    for label in PANELS:
        paste_panel(canvas, draw, label, images[label], boxes[label])

    canvas.save(OUTPUT.with_suffix(".png"), dpi=(300, 300), optimize=True)

    figure = plt.figure(figsize=(12, 15.267), dpi=300, facecolor="white")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.imshow(canvas)
    axis.axis("off")
    figure.savefig(OUTPUT.with_suffix(".pdf"), dpi=300, facecolor="white")
    plt.close(figure)
    print(f"Wrote {OUTPUT}.png/.pdf")


if __name__ == "__main__":
    main()
