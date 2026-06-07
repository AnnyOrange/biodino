#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "uint8_vs_16bit_compare"

UINT8_ROOTS = {
    "B": REPO_ROOT / "outputs" / "uint8_vitb16_b1024" / "eval_full",
    "L": REPO_ROOT / "outputs" / "uint8_vitl16_b1024" / "eval_full",
}

BENCH_ROOTS = {
    "B": Path("/mnt/huawei_deepcad/benchmark_model/benchmark_runs/dinov3_b_ckpts"),
    "L": Path("/mnt/huawei_deepcad/benchmark_model/benchmark_runs/dinov3_l_ckpts"),
}

MODEL_LABELS = {"B": "ViT-B", "L": "ViT-L"}
PRECISION_LABELS = {"16bit": "16-bit", "uint8": "uint8"}

DATASETS = [
    {
        "task": "classification",
        "dataset": "bloodmnist",
        "uint8_dataset": "bloodmnist",
        "bench_dataset": "bloodmnist",
    },
    {
        "task": "classification",
        "dataset": "bbbc048",
        "uint8_dataset": "bbbc048",
        "bench_dataset": "bbbc048-cellcycle",
    },
    {
        "task": "classification",
        "dataset": "cyclops",
        "uint8_dataset": "cyclops",
        "bench_dataset": "cyclops-protein-loc",
    },
    {
        "task": "classification",
        "dataset": "midog25",
        "uint8_dataset": "midog25",
        "bench_dataset": "midog25-atypical",
    },
    {
        "task": "regression",
        "dataset": "bbbc013",
        "uint8_dataset": "bbbc013",
        "bench_dataset": "bbbc013",
    },
]

CLASSIFICATION_METRICS = {
    "accuracy": ("test_accuracy_top1", 100.0, True),
    "balanced_accuracy": ("test_balanced_accuracy_top1", 100.0, True),
    "macro_f1": ("test_macro_f1", 100.0, True),
}
REGRESSION_METRICS = {
    "mae": ("test_mae", 1.0, False),
    "r2": ("test_r2", 1.0, True),
}
BENCH_METRICS = {
    "classification": {
        "accuracy": ("accuracy", True),
        "balanced_accuracy": ("balanced_accuracy", True),
        "macro_f1": ("macro_f1", True),
    },
    "regression": {
        "mae": ("mae", False),
        "r2": ("r2", True),
    },
}
PRIMARY_METRIC = {"classification": "accuracy", "regression": "r2"}
CLASSIFICATION_METRIC_ORDER = ["accuracy", "balanced_accuracy", "macro_f1"]
CLASSIFICATION_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro-F1",
}

COMPRESSION_PATHS = {
    "Raw uint16 TIFF": Path("/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle"),
    "Lossless TIFF deflate": Path("/mnt/huawei_deepcad/compression/deflate_lossless"),
    "uint8 + TIFF deflate": Path("/mnt/huawei_deepcad/compression/uint8_deflate"),
}

COLORS = {
    "16bit": (37, 73, 116),
    "uint8": (212, 98, 43),
    "delta": (34, 139, 94),
    "negative": (190, 64, 55),
    "grid": (220, 225, 230),
    "axis": (60, 65, 70),
    "text": (30, 35, 40),
    "muted": (110, 118, 128),
    "panel": (255, 255, 255),
    "bg": (247, 249, 251),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = font(28, True)
FONT_SUBTITLE = font(18)
FONT_LABEL = font(16)
FONT_SMALL = font(13)
FONT_SMALL_BOLD = font(13, True)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def maybe_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def uint8_ckpts(root: Path) -> list[int]:
    source = root / "bio_classification" / "bloodmnist"
    return sorted(int(p.name) for p in source.iterdir() if p.is_dir() and p.name.isdigit())


def bench_ckpts(root: Path) -> list[int]:
    return sorted(int(p.name) for p in root.iterdir() if p.is_dir() and (p / "summary.csv").exists() and p.name.isdigit())


def discover_ckpts() -> list[int]:
    ckpt_sets = []
    for root in UINT8_ROOTS.values():
        ckpt_sets.append(set(uint8_ckpts(root)))
    for root in BENCH_ROOTS.values():
        ckpt_sets.append(set(bench_ckpts(root)))
    shared = sorted(set.intersection(*ckpt_sets))
    return shared[:15]


def load_uint8_metrics(model_size: str, ckpt: int, spec: dict) -> tuple[dict[str, float], Path]:
    root = UINT8_ROOTS[model_size]
    if spec["task"] == "classification":
        path = root / "bio_classification" / spec["uint8_dataset"] / str(ckpt) / "results_bio_linear.json"
        data = read_json(path)
        metrics = {
            metric: float(data[key]) / scale
            for metric, (key, scale, _higher) in CLASSIFICATION_METRICS.items()
        }
        return metrics, path

    if spec["task"] == "regression":
        path = root / "bio_regression" / spec["uint8_dataset"] / str(ckpt) / "results_bio_regression.json"
        data = read_json(path)
        metrics = {
            metric: float(data[key]) / scale
            for metric, (key, scale, _higher) in REGRESSION_METRICS.items()
        }
        return metrics, path

    raise ValueError(f"Unsupported task: {spec['task']}")


def load_bench_row(model_size: str, ckpt: int, dataset: str) -> tuple[dict, Path]:
    path = BENCH_ROOTS[model_size] / str(ckpt) / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("dataset") == dataset:
                return row, path
    raise KeyError(f"{dataset} not found in {path}")


def load_bench_metrics(model_size: str, ckpt: int, spec: dict) -> tuple[dict[str, float], Path]:
    row, path = load_bench_row(model_size, ckpt, spec["bench_dataset"])
    metrics = {}
    for metric, (column, _higher) in BENCH_METRICS[spec["task"]].items():
        metrics[metric] = maybe_float(row.get(column))
    return metrics, path


def higher_is_better(task: str, metric: str) -> bool:
    if task == "classification":
        return CLASSIFICATION_METRICS[metric][2]
    if task == "regression":
        return REGRESSION_METRICS[metric][2]
    raise ValueError(task)


def load_rows(ckpts: list[int]) -> list[dict]:
    rows = []
    for model_size in ["B", "L"]:
        for epoch, ckpt in enumerate(ckpts, start=1):
            for spec in DATASETS:
                for precision, loader in [("16bit", load_bench_metrics), ("uint8", load_uint8_metrics)]:
                    metrics, source_path = loader(model_size, ckpt, spec)
                    for metric, value in metrics.items():
                        if not math.isfinite(value):
                            continue
                        rows.append(
                            {
                                "model_size": model_size,
                                "precision": precision,
                                "epoch": epoch,
                                "ckpt": ckpt,
                                "task": spec["task"],
                                "dataset": spec["dataset"],
                                "bench_dataset": spec["bench_dataset"],
                                "metric": metric,
                                "value": value,
                                "higher_is_better": higher_is_better(spec["task"], metric),
                                "source_path": str(source_path),
                            }
                        )
    return rows


def primary_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["metric"] == PRIMARY_METRIC[r["task"]]]


def aggregate_task_means(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int, int, str], list[float]] = defaultdict(list)
    for r in primary_rows(rows):
        groups[(r["model_size"], r["precision"], r["epoch"], r["ckpt"], r["task"])].append(r["value"])
    out = []
    for (model_size, precision, epoch, ckpt, task), vals in sorted(groups.items()):
        out.append(
            {
                "model_size": model_size,
                "precision": precision,
                "epoch": epoch,
                "ckpt": ckpt,
                "task": task,
                "metric": PRIMARY_METRIC[task],
                "value": mean(vals),
            }
        )
    return out


def epoch15_deltas(rows: list[dict], primary_only: bool) -> list[dict]:
    source_rows = primary_rows(rows) if primary_only else rows
    max_epoch = max(r["epoch"] for r in rows)
    values = {
        (r["model_size"], r["task"], r["dataset"], r["metric"], r["precision"]): r["value"]
        for r in source_rows
        if r["epoch"] == max_epoch
    }
    out = []
    keys = sorted({key[:4] for key in values})
    for model_size, task, dataset, metric in keys:
        key16 = (model_size, task, dataset, metric, "16bit")
        key8 = (model_size, task, dataset, metric, "uint8")
        if key16 not in values or key8 not in values:
            continue
        val16 = values[key16]
        val8 = values[key8]
        higher = higher_is_better(task, metric)
        delta = val8 - val16
        improvement = delta if higher else -delta
        out.append(
            {
                "model_size": model_size,
                "task": task,
                "dataset": dataset,
                "metric": metric,
                "higher_is_better": higher,
                "value_16bit": val16,
                "value_uint8": val8,
                "delta_uint8_minus_16bit": delta,
                "improvement": improvement,
            }
        )
    return out


def task_mean_deltas(task_rows: list[dict]) -> list[dict]:
    max_epoch = max(r["epoch"] for r in task_rows)
    values = {
        (r["model_size"], r["task"], r["precision"]): r["value"]
        for r in task_rows
        if r["epoch"] == max_epoch
    }
    out = []
    for model_size in ["B", "L"]:
        for task in ["classification", "regression"]:
            val16 = values[(model_size, task, "16bit")]
            val8 = values[(model_size, task, "uint8")]
            out.append(
                {
                    "model_size": model_size,
                    "task": task,
                    "metric": PRIMARY_METRIC[task],
                    "value_16bit": val16,
                    "value_uint8": val8,
                    "delta_uint8_minus_16bit": val8 - val16,
                }
            )
    return out


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill=COLORS["text"],
    fnt=FONT_LABEL,
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, fill=fill, font=fnt, anchor=anchor)


def chart_area(origin: tuple[int, int], size: tuple[int, int], margins=(58, 28, 26, 48)) -> tuple[int, int, int, int]:
    x, y = origin
    w, h = size
    left, top, right, bottom = margins
    return x + left, y + top, x + w - right, y + h - bottom


def nice_range(values: list[float], include_zero: bool = False, clamp_unit: bool = False) -> tuple[float, float]:
    clean = [v for v in values if math.isfinite(v)]
    if include_zero:
        clean.append(0.0)
    if not clean:
        return 0.0, 1.0
    ymin, ymax = min(clean), max(clean)
    if abs(ymax - ymin) < 1e-9:
        ymin -= 0.05
        ymax += 0.05
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad
    if clamp_unit:
        ymin = max(0.0, ymin)
        ymax = min(1.0, ymax)
        if ymax <= ymin:
            ymax = ymin + 0.05
    return ymin, ymax


def plot_series_panel(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    size: tuple[int, int],
    title: str,
    series: dict[str, list[tuple[int, float]]],
    y_label: str,
    include_zero: bool = False,
    clamp_unit: bool = False,
) -> None:
    x0, y0 = origin
    w, h = size
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=8, fill=COLORS["panel"], outline=(226, 230, 235), width=1)
    draw_text(draw, (x0 + 18, y0 + 14), title, fnt=FONT_SUBTITLE)

    px0, py0, px1, py1 = chart_area(origin, size)
    values = [v for pts in series.values() for _, v in pts if math.isfinite(v)]
    if not values:
        return
    ymin, ymax = nice_range(values, include_zero=include_zero, clamp_unit=clamp_unit)

    for i in range(5):
        yy = py1 - (py1 - py0) * i / 4
        draw.line([(px0, yy), (px1, yy)], fill=COLORS["grid"], width=1)
        val = ymin + (ymax - ymin) * i / 4
        draw_text(draw, (px0 - 8, int(yy) - 7), f"{val:.2f}", fill=COLORS["muted"], fnt=FONT_SMALL, anchor="ra")
    if ymin < 0 < ymax:
        zy = py1 - (py1 - py0) * (0 - ymin) / (ymax - ymin)
        draw.line([(px0, zy), (px1, zy)], fill=(150, 155, 160), width=2)
    draw.line([(px0, py1), (px1, py1)], fill=COLORS["axis"], width=1)
    draw.line([(px0, py0), (px0, py1)], fill=COLORS["axis"], width=1)

    epochs = sorted({e for pts in series.values() for e, _ in pts})
    xmin, xmax = min(epochs), max(epochs)
    for e in [xmin, 5, 10, xmax]:
        if e < xmin or e > xmax:
            continue
        xx = px0 + (px1 - px0) * (e - xmin) / max(1, xmax - xmin)
        draw.line([(xx, py1), (xx, py1 + 4)], fill=COLORS["axis"], width=1)
        draw_text(draw, (int(xx), py1 + 8), str(e), fill=COLORS["muted"], fnt=FONT_SMALL, anchor="ma")
    draw_text(draw, ((px0 + px1) // 2, y0 + h - 18), "Epoch", fill=COLORS["muted"], fnt=FONT_SMALL, anchor="ma")
    draw_text(draw, (x0 + 15, (py0 + py1) // 2), y_label, fill=COLORS["muted"], fnt=FONT_SMALL)

    for name, pts in series.items():
        color = COLORS[name]
        coords = []
        for epoch, value in pts:
            xx = px0 + (px1 - px0) * (epoch - xmin) / max(1, xmax - xmin)
            yy = py1 - (py1 - py0) * (value - ymin) / (ymax - ymin)
            coords.append((xx, yy))
        if len(coords) >= 2:
            draw.line(coords, fill=color, width=3)
        for xx, yy in coords:
            draw.ellipse([xx - 3, yy - 3, xx + 3, yy + 3], fill=color)


def make_task_mean_plot(task_rows: list[dict], path: Path) -> None:
    img = Image.new("RGB", (1500, 980), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw_text(draw, (40, 28), "uint8 vs 16-bit benchmark_model: primary task scores", fnt=FONT_TITLE)
    draw_text(draw, (40, 66), "Primary metrics: classification accuracy, regression R2. ViT-B is restricted to the first 15 shared checkpoints.", fill=COLORS["muted"], fnt=FONT_LABEL)

    panels = {
        ("B", "classification"): (40, 110),
        ("B", "regression"): (770, 110),
        ("L", "classification"): (40, 535),
        ("L", "regression"): (770, 535),
    }
    for (model_size, task), origin in panels.items():
        series = {}
        for precision in ["16bit", "uint8"]:
            pts = [
                (r["epoch"], r["value"])
                for r in task_rows
                if r["model_size"] == model_size and r["task"] == task and r["precision"] == precision
            ]
            series[precision] = sorted(pts)
        label = "Accuracy" if task == "classification" else "R2"
        plot_series_panel(
            draw,
            origin,
            (690, 390),
            f"{MODEL_LABELS[model_size]} {task} mean",
            series,
            label,
            clamp_unit=(task == "classification"),
        )

    lx, ly = 1150, 42
    for i, precision in enumerate(["16bit", "uint8"]):
        x = lx + i * 135
        draw.line([(x, ly), (x + 36, ly)], fill=COLORS[precision], width=5)
        draw_text(draw, (x + 44, ly - 9), PRECISION_LABELS[precision], fnt=FONT_LABEL)
    img.save(path)


def make_delta_over_epoch_plot(task_rows: list[dict], path: Path) -> None:
    img = Image.new("RGB", (1500, 980), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw_text(draw, (40, 28), "uint8 minus 16-bit: primary task deltas", fnt=FONT_TITLE)
    draw_text(draw, (40, 66), "Positive values mean uint8 is higher on the primary metric.", fill=COLORS["muted"], fnt=FONT_LABEL)

    panels = {
        ("B", "classification"): (40, 110),
        ("B", "regression"): (770, 110),
        ("L", "classification"): (40, 535),
        ("L", "regression"): (770, 535),
    }
    for (model_size, task), origin in panels.items():
        values = {
            (r["epoch"], r["precision"]): r["value"]
            for r in task_rows
            if r["model_size"] == model_size and r["task"] == task
        }
        epochs = sorted({epoch for epoch, _precision in values})
        pts = [(epoch, values[(epoch, "uint8")] - values[(epoch, "16bit")]) for epoch in epochs]
        label = "Accuracy delta" if task == "classification" else "R2 delta"
        plot_series_panel(
            draw,
            origin,
            (690, 390),
            f"{MODEL_LABELS[model_size]} {task} delta",
            {"delta": pts},
            label,
            include_zero=True,
        )
    img.save(path)


def draw_bar_panel(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    size: tuple[int, int],
    title: str,
    rows: list[dict],
) -> None:
    x0, y0 = origin
    w, h = size
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=8, fill=COLORS["panel"], outline=(226, 230, 235), width=1)
    draw_text(draw, (x0 + 18, y0 + 14), title, fnt=FONT_SUBTITLE)

    px0, py0, px1, py1 = x0 + 160, y0 + 60, x0 + w - 28, y0 + h - 55
    vals = [r["delta_uint8_minus_16bit"] for r in rows]
    max_abs = max([abs(v) for v in vals] + [0.02]) * 1.15
    zero_x = px0 + (px1 - px0) / 2
    for i in range(-4, 5):
        xx = zero_x + (px1 - px0) * i / 8
        draw.line([(xx, py0), (xx, py1)], fill=COLORS["grid"], width=1)
        val = max_abs * i / 4
        draw_text(draw, (int(xx), py1 + 10), f"{val:+.2f}", fill=COLORS["muted"], fnt=FONT_SMALL, anchor="ma")
    draw.line([(zero_x, py0), (zero_x, py1)], fill=COLORS["axis"], width=2)

    bar_h = min(28, max(16, int((py1 - py0) / max(1, len(rows)) * 0.62)))
    gap = ((py1 - py0) - bar_h * len(rows)) / max(1, len(rows) + 1)
    y = py0 + gap
    for row in rows:
        delta = row["delta_uint8_minus_16bit"]
        x = zero_x + (px1 - px0) * delta / (2 * max_abs)
        color = COLORS["delta"] if delta >= 0 else COLORS["negative"]
        draw.rounded_rectangle([min(zero_x, x), y, max(zero_x, x), y + bar_h], radius=4, fill=color)
        label = f"{row['model_size']} {row['dataset']}"
        draw_text(draw, (px0 - 10, int(y + bar_h / 2 - 8)), label, fnt=FONT_SMALL, anchor="ra")
        draw_text(
            draw,
            (int(x + (8 if delta >= 0 else -8)), int(y + bar_h / 2 - 8)),
            f"{delta:+.3f}",
            fill=color,
            fnt=FONT_SMALL_BOLD,
            anchor="la" if delta >= 0 else "ra",
        )
        y += bar_h + gap
    draw_text(draw, ((px0 + px1) // 2, y0 + h - 22), "uint8 - 16-bit", fill=COLORS["muted"], fnt=FONT_SMALL, anchor="ma")


def make_epoch15_bar_plot(dataset_deltas: list[dict], path: Path) -> None:
    classification = [r for r in dataset_deltas if r["task"] == "classification"]
    regression = [r for r in dataset_deltas if r["task"] == "regression"]
    img = Image.new("RGB", (1500, 760), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw_text(draw, (40, 28), "Epoch 15 primary deltas by dataset", fnt=FONT_TITLE)
    draw_text(draw, (40, 66), "Positive bars mean uint8 is higher on accuracy or R2.", fill=COLORS["muted"], fnt=FONT_LABEL)
    draw_bar_panel(draw, (40, 110), (880, 590), "Classification accuracy delta", classification)
    draw_bar_panel(draw, (950, 110), (510, 590), "Regression R2 delta", regression)
    img.save(path)


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def make_common_metric_heatmap(all_metric_deltas: list[dict], path: Path) -> None:
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "r2"]
    rows = []
    for model_size in ["B", "L"]:
        for spec in DATASETS:
            rows.append((model_size, spec["task"], spec["dataset"]))

    values = {
        (r["model_size"], r["task"], r["dataset"], r["metric"]): r["delta_uint8_minus_16bit"]
        for r in all_metric_deltas
        if r["metric"] in metrics
    }
    column_scale = {}
    for metric in metrics:
        vals = [abs(v) for key, v in values.items() if key[3] == metric]
        column_scale[metric] = max(vals + [0.02])

    img = Image.new("RGB", (1500, 900), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw_text(draw, (40, 28), "Epoch 15 common metric deltas", fnt=FONT_TITLE)
    draw_text(draw, (40, 66), "Cells show uint8 - 16-bit. Classification deltas are shown in percentage points; R2 is shown on its native scale.", fill=COLORS["muted"], fnt=FONT_LABEL)

    x0, y0 = 270, 130
    cell_w, cell_h = 280, 54
    headers = ["accuracy", "balanced acc.", "macro-F1", "R2"]
    for j, header in enumerate(headers):
        x = x0 + j * cell_w
        draw.rounded_rectangle([x, y0 - 44, x + cell_w - 10, y0 - 10], radius=6, fill=(232, 237, 242))
        draw_text(draw, (x + cell_w // 2 - 5, y0 - 35), header, fnt=FONT_SMALL_BOLD, anchor="ma")

    white = (252, 253, 254)
    for i, (model_size, task, dataset) in enumerate(rows):
        y = y0 + i * cell_h
        label = f"{MODEL_LABELS[model_size]} {dataset}"
        draw_text(draw, (x0 - 15, y + 16), label, fnt=FONT_SMALL, anchor="ra")
        for j, metric in enumerate(metrics):
            x = x0 + j * cell_w
            value = values.get((model_size, task, dataset, metric))
            if value is None:
                fill = (238, 241, 244)
                text = ""
            else:
                scale = column_scale[metric]
                intensity = abs(value) / scale
                fill = blend(white, COLORS["delta"] if value >= 0 else COLORS["negative"], 0.20 + 0.65 * intensity)
                text = f"{value * 100:+.1f}" if metric != "r2" else f"{value:+.3f}"
            draw.rounded_rectangle([x, y, x + cell_w - 10, y + cell_h - 8], radius=6, fill=fill, outline=(226, 230, 235))
            if text:
                draw_text(draw, (x + cell_w // 2 - 5, y + 14), text, fnt=FONT_SMALL_BOLD, anchor="ma")

    draw_text(draw, (x0 + cell_w * 1.5, 820), "Green: uint8 higher. Red: 16-bit higher.", fill=COLORS["muted"], fnt=FONT_LABEL, anchor="ma")
    img.save(path)


def _series_lookup(rows: list[dict], *, task: str, dataset: str, metric: str) -> dict[tuple[str, str], list[tuple[int, float]]]:
    out: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row["task"] != task or row["dataset"] != dataset or row["metric"] != metric:
            continue
        out[(row["model_size"], row["precision"])].append((row["epoch"], row["value"]))
    return {key: sorted(vals) for key, vals in out.items()}


def classification_best_scores(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["task"] != "classification" or row["metric"] not in CLASSIFICATION_METRIC_ORDER:
            continue
        groups[(row["model_size"], row["precision"], row["dataset"], row["metric"])].append(row)

    out = []
    for (model_size, precision, dataset, metric), vals in sorted(groups.items()):
        best = max(vals, key=lambda r: (r["value"], -r["epoch"]))
        out.append(
            {
                "model_size": model_size,
                "precision": precision,
                "dataset": dataset,
                "metric": metric,
                "best_epoch": best["epoch"],
                "best_ckpt": best["ckpt"],
                "best_value": best["value"],
            }
        )
    return out


def classification_best_metric_deltas(best_rows: list[dict]) -> list[dict]:
    values = {
        (row["model_size"], row["dataset"], row["metric"], row["precision"]): row
        for row in best_rows
    }
    out = []
    datasets = [spec["dataset"] for spec in DATASETS if spec["task"] == "classification"]
    for model_size in ["B", "L"]:
        for dataset in datasets:
            for metric in CLASSIFICATION_METRIC_ORDER:
                row16 = values[(model_size, dataset, metric, "16bit")]
                row8 = values[(model_size, dataset, metric, "uint8")]
                out.append(
                    {
                        "model_size": model_size,
                        "dataset": dataset,
                        "metric": metric,
                        "best_epoch_16bit": row16["best_epoch"],
                        "best_ckpt_16bit": row16["best_ckpt"],
                        "best_value_16bit": row16["best_value"],
                        "best_epoch_uint8": row8["best_epoch"],
                        "best_ckpt_uint8": row8["best_ckpt"],
                        "best_value_uint8": row8["best_value"],
                        "delta_uint8_minus_16bit": row8["best_value"] - row16["best_value"],
                    }
                )
    return out


def classification_avg_best_rows(best_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in best_rows:
        groups[(row["model_size"], row["precision"], row["metric"])].append(row["best_value"])
    out = []
    for model_size in ["B", "L"]:
        for metric in CLASSIFICATION_METRIC_ORDER:
            for precision in ["16bit", "uint8"]:
                vals = groups[(model_size, precision, metric)]
                out.append(
                    {
                        "model_size": model_size,
                        "precision": precision,
                        "metric": metric,
                        "avg_best_value": mean(vals),
                        "n_datasets": len(vals),
                    }
                )
    return out


def make_classification_model_dataset_subplots(rows: list[dict], model_size: str, path: Path) -> None:
    datasets = [spec["dataset"] for spec in DATASETS if spec["task"] == "classification"]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), sharex=True)
    fig.suptitle(f"{MODEL_LABELS[model_size]} classification: dataset subplots", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.94,
        "Metric: accuracy. Each dataset subplot compares uint8 vs 16-bit over the first 15 shared checkpoints.",
        ha="center",
        fontsize=10,
        color="#58606a",
    )
    style = {
        "16bit": {"color": "#254974", "linestyle": "-", "label": "16-bit"},
        "uint8": {"color": "#d4622b", "linestyle": "-", "label": "uint8"},
    }
    for ax, dataset in zip(axes.flat, datasets):
        lookup = _series_lookup(rows, task="classification", dataset=dataset, metric="accuracy")
        values = []
        for precision in ["16bit", "uint8"]:
            pts = lookup.get((model_size, precision), [])
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [p[1] * 100.0 for p in pts]
            values.extend(y)
            ax.plot(x, y, marker="o", linewidth=2.2, markersize=3.4, **style[precision])
            best_idx = max(range(len(y)), key=lambda i: y[i])
            ax.scatter([x[best_idx]], [y[best_idx]], s=42, color=style[precision]["color"], edgecolor="white", linewidth=0.9, zorder=5)
        ax.set_title(dataset, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.28)
        ax.set_xlim(1, 15)
        ax.set_xticks([1, 5, 10, 15])
        if values:
            lo = max(0.0, min(values) - 5.0)
            hi = min(100.0, max(values) + 5.0)
            if hi - lo < 12:
                mid = (hi + lo) / 2
                lo, hi = max(0.0, mid - 6), min(100.0, mid + 6)
            ax.set_ylim(lo, hi)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_classification_best_heatmap(best_delta_rows: list[dict], path: Path) -> None:
    from matplotlib.colors import TwoSlopeNorm

    row_keys = []
    for model_size in ["B", "L"]:
        for spec in DATASETS:
            if spec["task"] == "classification":
                row_keys.append((model_size, spec["dataset"]))

    values = {
        (row["model_size"], row["dataset"], row["metric"]): row["delta_uint8_minus_16bit"] * 100.0
        for row in best_delta_rows
    }
    matrix = np.array(
        [[values[(model_size, dataset, metric)] for metric in CLASSIFICATION_METRIC_ORDER] for model_size, dataset in row_keys],
        dtype=float,
    )
    vmax = max(10.0, float(np.nanmax(np.abs(matrix))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
    fig.suptitle("Classification best-epoch metric deltas", fontsize=15, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.945,
        "Each cell is max over first 15 epochs: uint8 best - 16-bit best, in percentage points.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#58606a",
    )
    ax.set_xticks(range(len(CLASSIFICATION_METRIC_ORDER)))
    ax.set_xticklabels([CLASSIFICATION_METRIC_LABELS[m] for m in CLASSIFICATION_METRIC_ORDER])
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels([f"{MODEL_LABELS[m]} {d}" for m, d in row_keys])
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=10, fontweight="bold", color="#111827")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("percentage points")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_classification_avg_best_bar(avg_rows: list[dict], path: Path) -> None:
    values = {
        (row["model_size"], row["precision"], row["metric"]): row["avg_best_value"] * 100.0
        for row in avg_rows
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4), sharey=False)
    fig.suptitle("Classification average best score: uint8 vs 16-bit", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.91,
        "For each dataset, take the best epoch in the first 15 checkpoints, then average across classification datasets.",
        ha="center",
        fontsize=10,
        color="#58606a",
    )
    colors = {"16bit": "#254974", "uint8": "#d4622b"}
    for ax, metric in zip(axes, CLASSIFICATION_METRIC_ORDER):
        x = np.arange(2)
        width = 0.34
        vals16 = [values[(m, "16bit", metric)] for m in ["B", "L"]]
        vals8 = [values[(m, "uint8", metric)] for m in ["B", "L"]]
        b1 = ax.bar(x - width / 2, vals16, width, color=colors["16bit"], label="16-bit")
        b2 = ax.bar(x + width / 2, vals8, width, color=colors["uint8"], label="uint8")
        ax.set_title(CLASSIFICATION_METRIC_LABELS[metric], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["ViT-B", "ViT-L"])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ymax = max(vals16 + vals8)
        ymin = min(vals16 + vals8)
        ax.set_ylim(max(0, ymin - 8), min(100, ymax + 8))
        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=9)
        for idx, model_size in enumerate(["B", "L"]):
            delta = vals8[idx] - vals16[idx]
            ax.text(idx, max(vals16[idx], vals8[idx]) + 4.2, f"Δ {delta:+.1f}", ha="center", fontsize=9, color="#374151")
    axes[0].set_ylabel("Average best score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 0.86))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def tar_size_bytes(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    files = sorted(p for p in path.glob("*.tar") if p.is_file())
    return sum(p.stat().st_size for p in files), len(files)


def compression_rows() -> list[dict]:
    raw_bytes, raw_shards = tar_size_bytes(COMPRESSION_PATHS["Raw uint16 TIFF"])
    rows = []
    for name, path in COMPRESSION_PATHS.items():
        size, shards = tar_size_bytes(path)
        ratio = size / raw_bytes if raw_bytes else math.nan
        rows.append(
            {
                "version": name,
                "path": str(path),
                "shards": shards,
                "bytes": size,
                "gib": size / 1024**3,
                "gb": size / 1e9,
                "ratio_vs_raw": ratio,
                "compression_factor_vs_raw": (raw_bytes / size) if size else math.nan,
                "raw_shards": raw_shards,
            }
        )
    return rows


def make_compression_size_plot(rows: list[dict], path: Path) -> None:
    labels = [r["version"] for r in rows]
    sizes = [r["gib"] for r in rows]
    ratios = [r["ratio_vs_raw"] * 100.0 for r in rows]
    colors = ["#6b7280", "#254974", "#d4622b"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.bar(labels, sizes, color=colors, width=0.58)
    ax.set_title("Compression practice: on-disk WebDataset size", fontsize=15, fontweight="bold")
    ax.set_ylabel("Size (GiB)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=12)
    for bar, size, ratio in zip(bars, sizes, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sizes) * 0.025,
            f"{size:.0f} GiB\n{ratio:.1f}% of raw",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.annotate(
        "lossless TIFF DEFLATE\nsame uint16 pixels",
        xy=(1, sizes[1]),
        xytext=(0.55, max(sizes) * 0.67),
        arrowprops={"arrowstyle": "->", "color": "#4b5563"},
        fontsize=10,
        color="#374151",
    )
    ax.annotate(
        "global uint16->uint8 quantization\nthen TIFF DEFLATE",
        xy=(2, sizes[2]),
        xytext=(1.38, max(sizes) * 0.43),
        arrowprops={"arrowstyle": "->", "color": "#4b5563"},
        fontsize=10,
        color="#374151",
    )
    ax.set_ylim(0, max(sizes) * 1.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fmt_score(metric: str, value: float) -> str:
    if metric in {"accuracy", "balanced_accuracy", "macro_f1"}:
        return f"{value * 100:.2f}"
    return f"{value:.4f}"


def write_markdown(
    path: Path,
    ckpts: list[int],
    dataset_deltas: list[dict],
    task_deltas: list[dict],
    comp_rows: list[dict],
    class_best_delta_rows: list[dict],
    class_avg_rows: list[dict],
) -> None:
    avg_lookup = {
        (row["model_size"], row["precision"], row["metric"]): row["avg_best_value"]
        for row in class_avg_rows
    }
    lines = [
        "# Classification uint8 vs 16-bit benchmark_model comparison",
        "",
        "## Scope",
        "",
        f"- Compared checkpoints: `{', '.join(map(str, ckpts))}`.",
        "- uint8 sources: `outputs/uint8_vitb16_b1024/eval_full`, `outputs/uint8_vitl16_b1024/eval_full`.",
        "- 16-bit sources: `/mnt/huawei_deepcad/benchmark_model/benchmark_runs/dinov3_b_ckpts`, `/mnt/huawei_deepcad/benchmark_model/benchmark_runs/dinov3_l_ckpts`.",
        "- Plots now use classification only: `bloodmnist`, `bbbc048`, `cyclops`, `midog25`.",
        "- Regression is excluded from the plots because BBBC013 is not protocol-matched across the two sources. `benchmark_model` uses raw targets, an 80/20 split, and Ridge alpha=1; the uint8 bio-regression output uses `log1p` targets, a 70/15/15 split, and validation-selected alpha.",
        "- Best-epoch tables use the maximum score within the first 15 shared checkpoints for each dataset/metric/precision.",
        "",
        "## Classification average best scores",
        "",
        "| model | metric | 16-bit | uint8 | uint8 - 16-bit |",
        "|---|---|---:|---:|---:|",
    ]
    for model_size in ["B", "L"]:
        for metric in CLASSIFICATION_METRIC_ORDER:
            val16 = avg_lookup[(model_size, "16bit", metric)]
            val8 = avg_lookup[(model_size, "uint8", metric)]
            lines.append(
                f"| {MODEL_LABELS[model_size]} | {metric} | {fmt_score(metric, val16)} | {fmt_score(metric, val8)} | {fmt_score(metric, val8 - val16)} |"
            )

    lines.extend(
        [
            "",
            "## Classification best per dataset",
            "",
            "| model | dataset | metric | 16-bit best | 16-bit epoch | uint8 best | uint8 epoch | uint8 - 16-bit |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in class_best_delta_rows:
        metric = row["metric"]
        lines.append(
            f"| {MODEL_LABELS[row['model_size']]} | {row['dataset']} | {metric} | {fmt_score(metric, row['best_value_16bit'])} | {row['best_epoch_16bit']} | {fmt_score(metric, row['best_value_uint8'])} | {row['best_epoch_uint8']} | {fmt_score(metric, row['delta_uint8_minus_16bit'])} |"
        )

    lines.extend(
        [
            "",
            "## Epoch 15 classification accuracy reference",
            "",
            "| model | task | dataset | metric | 16-bit | uint8 | uint8 - 16-bit |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in dataset_deltas:
        if row["task"] != "classification":
            continue
        metric = row["metric"]
        lines.append(
            f"| {MODEL_LABELS[row['model_size']]} | {row['task']} | {row['dataset']} | {metric} | {fmt_score(metric, row['value_16bit'])} | {fmt_score(metric, row['value_uint8'])} | {fmt_score(metric, row['delta_uint8_minus_16bit'])} |"
        )

    lines.extend(
        [
            "",
            "## Compression practice",
            "",
            "| version | shards | size GiB | size GB | ratio vs raw | compression factor | path |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in comp_rows:
        lines.append(
            f"| {row['version']} | {row['shards']} | {row['gib']:.2f} | {row['gb']:.2f} | {row['ratio_vs_raw'] * 100:.1f}% | {row['compression_factor_vs_raw']:.2f}x | `{row['path']}` |"
        )
    lines.extend(
        [
            "",
            "### Compression method notes",
            "",
            "- Raw data is packed WebDataset tar shards. Samples keep `.chN.tif + .meta.json`; TIFF payload dominates size.",
            "- Lossless TIFF deflate keeps the same shard/member layout and the same `uint16` pixel values, only changing TIFF internal compression to DEFLATE.",
            "- `uint8 + TIFF deflate` keeps the same shard/member layout but applies a lossy global full-range map `v8 = round(v16 * 255 / 65535)` before DEFLATE.",
            "- Practical difference: lossless deflate should be benchmark-equivalent apart from IO/decode effects; uint8 is smaller but irreversible and can lose low-intensity detail.",
        ]
    )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `classification_vitb_dataset_subplots.png`",
            "- `classification_vitl_dataset_subplots.png`",
            "- `classification_best_epoch_delta_heatmap.png`",
            "- `classification_avg_best_epoch_bar.png`",
            "- `compression_size_summary.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpts = discover_ckpts()
    rows = load_rows(ckpts)
    primary = primary_rows(rows)
    task_rows = aggregate_task_means(rows)
    primary_deltas = epoch15_deltas(rows, primary_only=True)
    all_metric_deltas = epoch15_deltas(rows, primary_only=False)
    task_deltas = task_mean_deltas(task_rows)
    comp_rows = compression_rows()
    class_best_rows = classification_best_scores(rows)
    class_best_delta_rows = classification_best_metric_deltas(class_best_rows)
    class_avg_rows = classification_avg_best_rows(class_best_rows)

    write_csv(
        OUTPUT_DIR / "all_common_metric_scores.csv",
        rows,
        [
            "model_size",
            "precision",
            "epoch",
            "ckpt",
            "task",
            "dataset",
            "bench_dataset",
            "metric",
            "value",
            "higher_is_better",
            "source_path",
        ],
    )
    write_csv(
        OUTPUT_DIR / "primary_metric_scores.csv",
        primary,
        [
            "model_size",
            "precision",
            "epoch",
            "ckpt",
            "task",
            "dataset",
            "bench_dataset",
            "metric",
            "value",
            "higher_is_better",
            "source_path",
        ],
    )
    write_csv(
        OUTPUT_DIR / "primary_task_mean_scores.csv",
        task_rows,
        ["model_size", "precision", "epoch", "ckpt", "task", "metric", "value"],
    )
    write_csv(
        OUTPUT_DIR / "epoch15_primary_dataset_delta.csv",
        primary_deltas,
        [
            "model_size",
            "task",
            "dataset",
            "metric",
            "higher_is_better",
            "value_16bit",
            "value_uint8",
            "delta_uint8_minus_16bit",
            "improvement",
        ],
    )
    write_csv(
        OUTPUT_DIR / "epoch15_all_metric_delta.csv",
        all_metric_deltas,
        [
            "model_size",
            "task",
            "dataset",
            "metric",
            "higher_is_better",
            "value_16bit",
            "value_uint8",
            "delta_uint8_minus_16bit",
            "improvement",
        ],
    )
    write_csv(
        OUTPUT_DIR / "epoch15_primary_task_mean_delta.csv",
        task_deltas,
        ["model_size", "task", "metric", "value_16bit", "value_uint8", "delta_uint8_minus_16bit"],
    )
    write_csv(
        OUTPUT_DIR / "compression_sizes.csv",
        comp_rows,
        ["version", "path", "shards", "bytes", "gib", "gb", "ratio_vs_raw", "compression_factor_vs_raw", "raw_shards"],
    )
    write_csv(
        OUTPUT_DIR / "classification_best_scores.csv",
        class_best_rows,
        ["model_size", "precision", "dataset", "metric", "best_epoch", "best_ckpt", "best_value"],
    )
    write_csv(
        OUTPUT_DIR / "classification_best_metric_delta.csv",
        class_best_delta_rows,
        [
            "model_size",
            "dataset",
            "metric",
            "best_epoch_16bit",
            "best_ckpt_16bit",
            "best_value_16bit",
            "best_epoch_uint8",
            "best_ckpt_uint8",
            "best_value_uint8",
            "delta_uint8_minus_16bit",
        ],
    )
    write_csv(
        OUTPUT_DIR / "classification_avg_best_scores.csv",
        class_avg_rows,
        ["model_size", "precision", "metric", "avg_best_value", "n_datasets"],
    )

    make_classification_model_dataset_subplots(rows, "B", OUTPUT_DIR / "classification_vitb_dataset_subplots.png")
    make_classification_model_dataset_subplots(rows, "L", OUTPUT_DIR / "classification_vitl_dataset_subplots.png")
    make_classification_best_heatmap(class_best_delta_rows, OUTPUT_DIR / "classification_best_epoch_delta_heatmap.png")
    make_classification_avg_best_bar(class_avg_rows, OUTPUT_DIR / "classification_avg_best_epoch_bar.png")
    make_compression_size_plot(comp_rows, OUTPUT_DIR / "compression_size_summary.png")
    write_markdown(OUTPUT_DIR / "summary.md", ckpts, primary_deltas, task_deltas, comp_rows, class_best_delta_rows, class_avg_rows)
    print(f"Wrote comparison outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
