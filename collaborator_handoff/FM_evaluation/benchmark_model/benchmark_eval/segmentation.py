from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.draw import polygon


BENCHMARK_SEG_ROOT = Path("/mnt/huawei_deepcad/benchmark/segmentation")


@dataclass
class SegSample:
    dataset: str
    sample_id: str
    image: np.ndarray
    mask: np.ndarray


def to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = np.concatenate([arr, np.zeros_like(arr[..., :1])], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo = float(np.nanpercentile(arr, 0.5))
        hi = float(np.nanpercentile(arr, 99.5))
        if hi <= lo:
            lo = float(np.nanmin(arr))
            hi = float(np.nanmax(arr))
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def relabel_instance_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    out = np.zeros(mask.shape[:2], dtype=np.int32)
    vals = np.unique(mask)
    vals = vals[vals > 0]
    for new_id, old_id in enumerate(vals, start=1):
        out[mask == old_id] = new_id
    return out


def combine_pannuke_mask(mask: np.ndarray) -> np.ndarray:
    """PanNuke stores instance ids in five cell-type channels plus background."""
    out = np.zeros(mask.shape[:2], dtype=np.int32)
    next_id = 1
    for ch in range(min(5, mask.shape[-1])):
        vals = np.unique(mask[..., ch])
        for val in vals[vals > 0]:
            pix = mask[..., ch] == val
            if np.any(pix):
                out[pix] = next_id
                next_id += 1
    return out


def load_bbbc038_mask(sample_dir: Path) -> np.ndarray:
    image_path = next((sample_dir / "images").glob("*"))
    with Image.open(image_path) as img:
        w, h = img.size
    out = np.zeros((h, w), dtype=np.int32)
    for idx, mask_path in enumerate(sorted((sample_dir / "masks").glob("*.png")), start=1):
        with Image.open(mask_path) as mask_img:
            mask = np.asarray(mask_img) > 0
        out[mask] = idx
    return out


def load_monuseg_xml(xml_path: Path, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.int32)
    root = ET.parse(xml_path).getroot()
    idx = 1
    for region in root.iter("Region"):
        vertices = []
        for vertex in region.iter("Vertex"):
            x = float(vertex.attrib.get("X", "nan"))
            y = float(vertex.attrib.get("Y", "nan"))
            if math.isfinite(x) and math.isfinite(y):
                vertices.append((y, x))
        if len(vertices) < 3:
            continue
        rr, cc = polygon(
            np.asarray([v[0] for v in vertices]),
            np.asarray([v[1] for v in vertices]),
            shape=shape,
        )
        out[rr, cc] = idx
        idx += 1
    return out


def livecell_instances_from_coco(ann_file: Path, image_root: Path) -> Iterator[SegSample]:
    from skimage.draw import polygon as draw_polygon

    data = json.loads(ann_file.read_text())
    images = {img["id"]: img for img in data["images"]}
    anns_by_image: dict[int, list[dict]] = {img_id: [] for img_id in images}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for image_id, img_info in images.items():
        file_name = img_info["file_name"]
        image_path = image_root / file_name
        if not image_path.exists():
            image_path = image_root / "livecell_test_images" / file_name
        if not image_path.exists():
            continue
        with Image.open(image_path) as img:
            image = to_rgb_uint8(np.asarray(img))
        mask = np.zeros((int(img_info["height"]), int(img_info["width"])), dtype=np.int32)
        next_id = 1
        for ann in anns_by_image.get(image_id, []):
            seg = ann.get("segmentation", [])
            if not isinstance(seg, list):
                continue
            for poly in seg:
                if len(poly) < 6:
                    continue
                xs = np.asarray(poly[0::2], dtype=np.float32)
                ys = np.asarray(poly[1::2], dtype=np.float32)
                rr, cc = draw_polygon(ys, xs, shape=mask.shape)
                mask[rr, cc] = next_id
            next_id += 1
        yield SegSample("livecell_test", Path(file_name).stem, image, mask)


def iter_dataset(name: str, max_samples: int | None = None) -> Iterator[SegSample]:
    n = 0

    def take(sample: SegSample) -> bool:
        nonlocal n
        if max_samples is not None and n >= max_samples:
            return False
        n += 1
        return True

    if name == "conic_test":
        root = BENCHMARK_SEG_ROOT / "conic" / "extracted"
        images = np.load(root / "images.npy", mmap_mode="r")
        labels = np.load(root / "labels.npy", mmap_mode="r")
        indices = np.load(root / "indices_test.npy")
        for idx in indices:
            sample = SegSample(name, str(int(idx)), to_rgb_uint8(images[idx]), relabel_instance_mask(labels[idx, ..., 0]))
            if not take(sample):
                return
            yield sample
        return

    if name in {"tissuenet_test_cells", "tissuenet_test_nuclei"}:
        channel = 0 if name.endswith("cells") else 1
        data = np.load(BENCHMARK_SEG_ROOT / "tissuenet" / "extracted" / "tissuenet_v1.1_test.npz", allow_pickle=False)
        images = data["X"]
        labels = data["y"]
        for idx in range(labels.shape[0]):
            sample = SegSample(name, str(idx), to_rgb_uint8(images[idx]), relabel_instance_mask(labels[idx, ..., channel]))
            if not take(sample):
                return
            yield sample
        return

    if name in {"bbbc038_val", "bbbc038_test"}:
        root = BENCHMARK_SEG_ROOT / "bbbc038" / "extracted"
        split_key = "val_idx" if name.endswith("val") else "test_idx"
        split = np.load(root / "bbbc038_splits.npz")[split_key]
        sample_dirs = sorted([p for p in (root / "stage1_train").iterdir() if p.is_dir()])
        for idx in split:
            sample_dir = sample_dirs[int(idx)]
            image_path = next((sample_dir / "images").glob("*"))
            with Image.open(image_path) as img:
                image = to_rgb_uint8(np.asarray(img))
            sample = SegSample(name, sample_dir.name, image, load_bbbc038_mask(sample_dir))
            if not take(sample):
                return
            yield sample
        return

    if name.startswith("pannuke_fold"):
        fold = int(name.replace("pannuke_fold", ""))
        root = BENCHMARK_SEG_ROOT / "pannuke" / "extracted" / f"Fold {fold}"
        images = np.load(root / "images" / f"fold{fold}" / "images.npy", mmap_mode="r")
        masks = np.load(root / "masks" / f"fold{fold}" / "masks.npy", mmap_mode="r")
        for idx in range(images.shape[0]):
            sample = SegSample(name, str(idx), to_rgb_uint8(images[idx]), combine_pannuke_mask(masks[idx]))
            if not take(sample):
                return
            yield sample
        return

    if name == "monuseg_test":
        root = BENCHMARK_SEG_ROOT / "monuseg" / "extracted" / "MoNuSegTestData"
        for image_path in sorted(root.glob("*.tif")):
            xml_path = image_path.with_suffix(".xml")
            if not xml_path.exists():
                continue
            with Image.open(image_path) as img:
                image = to_rgb_uint8(np.asarray(img))
            sample = SegSample(name, image_path.stem, image, load_monuseg_xml(xml_path, image.shape[:2]))
            if not take(sample):
                return
            yield sample
        return

    if name == "cellpose_test":
        root = BENCHMARK_SEG_ROOT / "Cellpose" / "test" / "test"
        for image_path in sorted(root.glob("*_img.*")):
            mask_path = image_path.with_name(image_path.name.replace("_img", "_masks"))
            if not mask_path.exists():
                continue
            with Image.open(image_path) as img:
                image = to_rgb_uint8(np.asarray(img))
            with Image.open(mask_path) as mask_img:
                mask = relabel_instance_mask(np.asarray(mask_img))
            sample = SegSample(name, image_path.stem.replace("_img", ""), image, mask)
            if not take(sample):
                return
            yield sample
        return

    if name == "livecell_test":
        root = BENCHMARK_SEG_ROOT / "LIVECell" / "LIVECell_dataset_2021"
        ann_file = root / "annotations" / "LIVECell" / "livecell_coco_test.json"
        image_root = root / "images" / "livecell_test_images"
        for sample in livecell_instances_from_coco(ann_file, image_root):
            if not take(sample):
                return
            yield sample
        return

    raise KeyError(f"Unknown segmentation dataset: {name}")


def instance_metrics(gt: np.ndarray, pred: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    gt = relabel_instance_mask(gt)
    pred = relabel_instance_mask(pred)
    gt_fg = gt > 0
    pred_fg = pred > 0
    inter_fg = int(np.logical_and(gt_fg, pred_fg).sum())
    gt_area_fg = int(gt_fg.sum())
    pred_area_fg = int(pred_fg.sum())
    dice = 1.0 if gt_area_fg + pred_area_fg == 0 else 2.0 * inter_fg / (gt_area_fg + pred_area_fg)
    union_fg = gt_area_fg + pred_area_fg - inter_fg
    fg_iou = 1.0 if union_fg == 0 else inter_fg / union_fg

    n_gt = int(gt.max())
    n_pred = int(pred.max())
    if n_gt == 0 and n_pred == 0:
        tp = fp = fn = 0
    elif n_gt == 0:
        tp, fp, fn = 0, n_pred, 0
    elif n_pred == 0:
        tp, fp, fn = 0, 0, n_gt
    else:
        gt_areas = np.bincount(gt.ravel(), minlength=n_gt + 1).astype(np.float64)
        pred_areas = np.bincount(pred.ravel(), minlength=n_pred + 1).astype(np.float64)
        pair = gt.astype(np.int64) * (n_pred + 1) + pred.astype(np.int64)
        overlap = np.bincount(pair.ravel(), minlength=(n_gt + 1) * (n_pred + 1)).reshape(n_gt + 1, n_pred + 1)
        inter = overlap[1:, 1:].astype(np.float64)
        union = gt_areas[1:, None] + pred_areas[None, 1:] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, inter / union, 0.0)
        rows, cols = linear_sum_assignment(-iou)
        matched = iou[rows, cols] >= threshold
        tp = int(matched.sum())
        fp = n_pred - tp
        fn = n_gt - tp

    precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 1.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    return {
        "dice": float(dice),
        "foreground_iou": float(fg_iou),
        "instance_f1_iou50": float(f1),
        "instance_precision_iou50": float(precision),
        "instance_recall_iou50": float(recall),
        "gt_count": float(n_gt),
        "pred_count": float(n_pred),
        "abs_count_error": float(abs(n_pred - n_gt)),
        "tp_iou50": float(tp),
        "fp_iou50": float(fp),
        "fn_iou50": float(fn),
    }


DETAIL_FIELDS = [
    "model",
    "dataset",
    "sample_id",
    "height",
    "width",
    "dice",
    "foreground_iou",
    "instance_f1_iou50",
    "instance_precision_iou50",
    "instance_recall_iou50",
    "gt_count",
    "pred_count",
    "abs_count_error",
    "tp_iou50",
    "fp_iou50",
    "fn_iou50",
]


def append_detail(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in DETAIL_FIELDS})


def completed_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {row["sample_id"] for row in csv.DictReader(f)}


def summarize_detail(detail_csv: Path, summary_csv: Path) -> dict:
    rows = list(csv.DictReader(detail_csv.open()))
    if not rows:
        raise ValueError(f"No rows in {detail_csv}")
    numeric = [f for f in DETAIL_FIELDS if f not in {"model", "dataset", "sample_id"}]
    summary: dict[str, str | float | int] = {
        "model": rows[0]["model"],
        "dataset": rows[0]["dataset"],
        "n_samples": len(rows),
        "detail_file": str(detail_csv),
    }
    for field in numeric:
        vals = np.asarray([float(r[field]) for r in rows], dtype=np.float64)
        summary[f"mean_{field}"] = float(vals.mean())
    tp = sum(float(r["tp_iou50"]) for r in rows)
    fp = sum(float(r["fp_iou50"]) for r in rows)
    fn = sum(float(r["fn_iou50"]) for r in rows)
    summary["pooled_instance_f1_iou50"] = 1.0 if (2 * tp + fp + fn) == 0 else float(2 * tp / (2 * tp + fp + fn))

    fields = [
        "model",
        "dataset",
        "n_samples",
        "mean_dice",
        "mean_foreground_iou",
        "mean_instance_f1_iou50",
        "pooled_instance_f1_iou50",
        "mean_instance_precision_iou50",
        "mean_instance_recall_iou50",
        "mean_gt_count",
        "mean_pred_count",
        "mean_abs_count_error",
        "detail_file",
    ]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if summary_csv.exists():
        existing = list(csv.DictReader(summary_csv.open()))
        existing = [r for r in existing if not (r["model"] == summary["model"] and r["dataset"] == summary["dataset"])]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in existing:
            writer.writerow({k: row.get(k, "") for k in fields})
        writer.writerow({k: summary.get(k, "") for k in fields})
    return summary

