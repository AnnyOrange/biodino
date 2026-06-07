#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "vitb_vitl_compare"

MODELS = {
    "vitb": REPO_ROOT / "outputs" / "uint8_vitb16_b1024" / "eval_full",
    "vitl": REPO_ROOT / "outputs" / "uint8_vitl16_b1024" / "eval_full",
}

CKPT_DIRS = {
    "vitb": REPO_ROOT / "outputs" / "uint8_vitb16_b1024" / "ckpt",
    "vitl": REPO_ROOT / "outputs" / "uint8_vitl16_b1024" / "ckpt",
}

TASKS = {
    "classification": {
        "datasets": ["bloodmnist", "bbbc048", "cyclops", "midog25"],
        "path": "bio_classification/{dataset}/{ckpt}/results_bio_linear.json",
        "metric": "test_accuracy_top1",
        "scale": 100.0,
        "label": "Accuracy",
    },
    "regression": {
        "datasets": ["bbbc013"],
        "path": "bio_regression/{dataset}/{ckpt}/results_bio_regression.json",
        "metric": "test_r2",
        "scale": 1.0,
        "label": "R2",
    },
    "detection": {
        "datasets": ["livecell"],
        "path": "bio_detection/{dataset}/{ckpt}/results_bio_detection.json",
        "metric": "test_patch_f1",
        "scale": 100.0,
        "label": "Patch F1",
    },
    "segmentation": {
        "datasets": ["bbbc038", "conic", "monuseg", "pannuke", "tissuenet"],
        "path": "bio_segmentation/bio_eval/{dataset}/{ckpt}/results.json",
        "metric": "test.mIoU",
        "scale": 1.0,
        "label": "mIoU",
    },
}

COLORS = {
    "vitb": (37, 73, 116),
    "vitl": (210, 98, 43),
    "grid": (220, 225, 230),
    "axis": (60, 65, 70),
    "text": (30, 35, 40),
    "muted": (110, 118, 128),
    "positive": (34, 139, 94),
    "negative": (190, 64, 55),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


FONT_TITLE = font(28, True)
FONT_SUBTITLE = font(18)
FONT_LABEL = font(16)
FONT_SMALL = font(13)
FONT_SMALL_BOLD = font(13, True)


def get_nested(data: dict, dotted: str) -> float:
    cur = data
    for part in dotted.split("."):
        cur = cur[part]
    return float(cur)


def discover_ckpts() -> list[int]:
    per_model = []
    for ckpt_dir in CKPT_DIRS.values():
        ckpts = sorted(int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit())
        per_model.append(ckpts)
    shared = sorted(set(per_model[0]).intersection(*map(set, per_model[1:])))
    return shared[:15]


def load_rows(ckpts: list[int]) -> list[dict]:
    rows: list[dict] = []
    for model, root in MODELS.items():
        for epoch, ckpt in enumerate(ckpts, start=1):
            for task, spec in TASKS.items():
                for dataset in spec["datasets"]:
                    path = root / spec["path"].format(dataset=dataset, ckpt=ckpt)
                    if not path.exists():
                        raise FileNotFoundError(path)
                    data = json.loads(path.read_text())
                    raw_value = get_nested(data, spec["metric"])
                    score = raw_value / spec["scale"]
                    rows.append(
                        {
                            "model": model,
                            "epoch": epoch,
                            "ckpt": ckpt,
                            "task": task,
                            "dataset": dataset,
                            "metric": spec["metric"],
                            "raw_value": raw_value,
                            "score": score,
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    by_task_epoch: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    by_dataset_last: dict[tuple[str, str, str], float] = {}
    by_dataset_all: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    max_epoch = max(r["epoch"] for r in rows)
    for r in rows:
        by_task_epoch[(r["model"], r["task"], r["epoch"], r["ckpt"])].append(r["score"])
        by_dataset_all[(r["model"], r["task"], r["dataset"])].append(r["score"])
        if r["epoch"] == max_epoch:
            by_dataset_last[(r["model"], r["task"], r["dataset"])] = r["score"]

    task_rows = [
        {"model": model, "task": task, "epoch": epoch, "ckpt": ckpt, "score": mean(vals)}
        for (model, task, epoch, ckpt), vals in sorted(by_task_epoch.items())
    ]

    dataset_rows = []
    for task, spec in TASKS.items():
        for dataset in spec["datasets"]:
            b = by_dataset_last[("vitb", task, dataset)]
            l = by_dataset_last[("vitl", task, dataset)]
            dataset_rows.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "metric": spec["label"],
                    "vitb_epoch15": b,
                    "vitl_epoch15": l,
                    "delta_l_minus_b": l - b,
                }
            )

    overall_rows = []
    for task, spec in TASKS.items():
        b_vals = [by_dataset_last[("vitb", task, ds)] for ds in spec["datasets"]]
        l_vals = [by_dataset_last[("vitl", task, ds)] for ds in spec["datasets"]]
        overall_rows.append(
            {
                "task": task,
                "metric": spec["label"],
                "vitb_epoch15_mean": mean(b_vals),
                "vitl_epoch15_mean": mean(l_vals),
                "delta_l_minus_b": mean(l_vals) - mean(b_vals),
            }
        )
    all_b = [r["vitb_epoch15_mean"] for r in overall_rows]
    all_l = [r["vitl_epoch15_mean"] for r in overall_rows]
    overall_rows.append(
        {
            "task": "all_task_mean",
            "metric": "mean of task means",
            "vitb_epoch15_mean": mean(all_b),
            "vitl_epoch15_mean": mean(all_l),
            "delta_l_minus_b": mean(all_l) - mean(all_b),
        }
    )
    return task_rows, dataset_rows, overall_rows


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=COLORS["text"], fnt=FONT_LABEL, anchor=None) -> None:
    draw.text(xy, text, fill=fill, font=fnt, anchor=anchor)


def chart_area(origin: tuple[int, int], size: tuple[int, int], margins=(56, 24, 30, 48)) -> tuple[int, int, int, int]:
    x, y = origin
    w, h = size
    left, top, right, bottom = margins
    return x + left, y + top, x + w - right, y + h - bottom


def plot_series_panel(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    size: tuple[int, int],
    title: str,
    series: dict[str, list[tuple[int, float]]],
    y_label: str,
    y_range: tuple[float, float] | None = None,
) -> None:
    x0, y0 = origin
    w, h = size
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=10, fill=(255, 255, 255), outline=(226, 230, 235), width=1)
    draw_text(draw, (x0 + 18, y0 + 14), title, fnt=FONT_SUBTITLE)

    px0, py0, px1, py1 = chart_area(origin, size)
    values = [v for pts in series.values() for _, v in pts if math.isfinite(v)]
    if not values:
        return
    ymin, ymax = y_range or (min(values), max(values))
    if abs(ymax - ymin) < 1e-9:
        ymin -= 0.05
        ymax += 0.05
    pad = (ymax - ymin) * 0.08
    ymin = max(0.0, ymin - pad)
    ymax = min(1.0, ymax + pad) if ymax <= 1.0 else ymax + pad

    for i in range(5):
        yy = py1 - (py1 - py0) * i / 4
        draw.line([(px0, yy), (px1, yy)], fill=COLORS["grid"], width=1)
        val = ymin + (ymax - ymin) * i / 4
        draw_text(draw, (px0 - 8, int(yy) - 7), f"{val:.2f}", fill=COLORS["muted"], fnt=FONT_SMALL, anchor="ra")
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
    draw_text(draw, (x0 + 16, (py0 + py1) // 2), y_label, fill=COLORS["muted"], fnt=FONT_SMALL)

    for name, pts in series.items():
        color = COLORS[name]
        coords = []
        for e, v in pts:
            xx = px0 + (px1 - px0) * (e - xmin) / max(1, xmax - xmin)
            yy = py1 - (py1 - py0) * (v - ymin) / (ymax - ymin)
            coords.append((xx, yy))
        if len(coords) >= 2:
            draw.line(coords, fill=color, width=3)
        for xx, yy in coords:
            draw.ellipse([xx - 3, yy - 3, xx + 3, yy + 3], fill=color)


def make_task_mean_plot(task_rows: list[dict], path: Path) -> None:
    img = Image.new("RGB", (1500, 980), (247, 249, 251))
    draw = ImageDraw.Draw(img)
    draw_text(draw, (40, 28), "ViT-B vs ViT-L: task mean score over first 15 epochs", fnt=FONT_TITLE)
    draw_text(draw, (40, 66), "Scores are normalized to 0-1 where higher is better. ViT-B is restricted to the first 15 epochs/checkpoints.", fill=COLORS["muted"], fnt=FONT_LABEL)

    panels = {
        "classification": (40, 110),
        "regression": (770, 110),
        "detection": (40, 535),
        "segmentation": (770, 535),
    }
    for task, origin in panels.items():
        series = {}
        for model in ["vitb", "vitl"]:
            pts = [(r["epoch"], r["score"]) for r in task_rows if r["task"] == task and r["model"] == model]
            series[model] = sorted(pts)
        plot_series_panel(draw, origin, (690, 390), f"{task} mean", series, TASKS[task]["label"])

    lx, ly = 1160, 42
    for i, model in enumerate(["vitb", "vitl"]):
        draw.line([(lx + i * 120, ly), (lx + 36 + i * 120, ly)], fill=COLORS[model], width=5)
        draw_text(draw, (lx + 44 + i * 120, ly - 9), model.upper(), fnt=FONT_LABEL)
    img.save(path)


def make_dataset_delta_plot(dataset_rows: list[dict], path: Path) -> None:
    rows = dataset_rows
    img = Image.new("RGB", (1500, 760), (247, 249, 251))
    draw = ImageDraw.Draw(img)
    draw_text(draw, (40, 28), "Epoch 15 difference: ViT-L minus ViT-B", fnt=FONT_TITLE)
    draw_text(draw, (40, 66), "Positive bars mean ViT-L is better on the selected task metric.", fill=COLORS["muted"], fnt=FONT_LABEL)

    px0, py0, px1, py1 = 110, 120, 1450, 610
    vals = [r["delta_l_minus_b"] for r in rows]
    max_abs = max(abs(v) for v in vals) * 1.15
    max_abs = max(max_abs, 0.02)
    zero_x = px0 + (px1 - px0) / 2
    draw.line([(zero_x, py0), (zero_x, py1)], fill=COLORS["axis"], width=2)
    for i in range(-4, 5):
        x = zero_x + (px1 - px0) * i / 8
        draw.line([(x, py0), (x, py1)], fill=COLORS["grid"], width=1)
        val = max_abs * i / 4
        draw_text(draw, (int(x), py1 + 10), f"{val:+.2f}", fill=COLORS["muted"], fnt=FONT_SMALL, anchor="ma")

    bar_h = 34
    gap = 13
    y = py0 + 8
    for r in rows:
        v = r["delta_l_minus_b"]
        x = zero_x + (px1 - px0) * v / (2 * max_abs)
        color = COLORS["positive"] if v >= 0 else COLORS["negative"]
        draw.rounded_rectangle([min(zero_x, x), y, max(zero_x, x), y + bar_h], radius=5, fill=color)
        label = f"{r['task']}/{r['dataset']}"
        draw_text(draw, (px0 - 10, y + bar_h // 2 - 8), label, fill=COLORS["text"], fnt=FONT_SMALL, anchor="ra")
        draw_text(draw, (x + (8 if v >= 0 else -8), y + bar_h // 2 - 8), f"{v:+.3f}", fill=color, fnt=FONT_SMALL_BOLD, anchor="la" if v >= 0 else "ra")
        y += bar_h + gap
    draw_text(draw, ((px0 + px1) // 2, 700), "Delta on normalized score scale", fill=COLORS["muted"], fnt=FONT_LABEL, anchor="ma")
    img.save(path)


def make_task_delta_plot(task_rows: list[dict], path: Path) -> None:
    img = Image.new("RGB", (1500, 980), (247, 249, 251))
    draw = ImageDraw.Draw(img)
    draw_text(draw, (40, 28), "ViT-L minus ViT-B: task mean delta over first 15 epochs", fnt=FONT_TITLE)
    draw_text(draw, (40, 66), "Delta uses normalized task score, so +0.01 is roughly +1 percentage point for percent metrics.", fill=COLORS["muted"], fnt=FONT_LABEL)
    panels = {
        "classification": (40, 110),
        "regression": (770, 110),
        "detection": (40, 535),
        "segmentation": (770, 535),
    }
    for task, origin in panels.items():
        by_model = {
            m: {r["epoch"]: r["score"] for r in task_rows if r["task"] == task and r["model"] == m}
            for m in ["vitb", "vitl"]
        }
        deltas = [(e, by_model["vitl"][e] - by_model["vitb"][e]) for e in sorted(by_model["vitb"])]
        plot_series_panel(draw, origin, (690, 390), f"{task} delta", {"vitl": deltas}, "L - B", y_range=(-0.08, 0.08))
    img.save(path)


def write_markdown(path: Path, dataset_rows: list[dict], overall_rows: list[dict], ckpts: list[int]) -> None:
    def pct(x: float) -> str:
        return f"{x * 100:.2f}"

    lines = [
        "# ViT-B vs ViT-L comparison",
        "",
        f"Compared checkpoints: `{', '.join(map(str, ckpts))}`.",
        "ViT-B is restricted to these first 15 epoch/checkpoint points. Scores are normalized to 0-1 where higher is better.",
        "",
        "## Epoch 15 task means",
        "",
        "| task | metric | ViT-B | ViT-L | L-B |",
        "|---|---|---:|---:|---:|",
    ]
    for r in overall_rows:
        lines.append(
            f"| {r['task']} | {r['metric']} | {pct(r['vitb_epoch15_mean'])} | {pct(r['vitl_epoch15_mean'])} | {pct(r['delta_l_minus_b'])} |"
        )

    lines.extend(["", "## Epoch 15 per dataset", "", "| task | dataset | metric | ViT-B | ViT-L | L-B |", "|---|---|---|---:|---:|---:|"])
    for r in dataset_rows:
        lines.append(f"| {r['task']} | {r['dataset']} | {r['metric']} | {pct(r['vitb_epoch15'])} | {pct(r['vitl_epoch15'])} | {pct(r['delta_l_minus_b'])} |")

    best = max(dataset_rows, key=lambda r: r["delta_l_minus_b"])
    worst = min(dataset_rows, key=lambda r: r["delta_l_minus_b"])
    worst_label = "Largest ViT-L drop" if worst["delta_l_minus_b"] < 0 else "Smallest ViT-L gain"
    task_deltas = {r["task"]: r["delta_l_minus_b"] for r in overall_rows if r["task"] != "all_task_mean"}
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Largest ViT-L gain at epoch 15: `{best['task']}/{best['dataset']}` ({pct(best['delta_l_minus_b'])} points).",
            f"- {worst_label} at epoch 15: `{worst['task']}/{worst['dataset']}` ({pct(worst['delta_l_minus_b'])} points).",
            f"- Task mean deltas: classification {pct(task_deltas['classification'])}, regression {pct(task_deltas['regression'])}, detection {pct(task_deltas['detection'])}, segmentation {pct(task_deltas['segmentation'])} points.",
            "",
            "## Figures",
            "",
            "- `task_mean_score_vs_epoch.png`",
            "- `task_delta_vitl_minus_vitb.png`",
            "- `dataset_delta_epoch15.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpts = discover_ckpts()
    rows = load_rows(ckpts)
    task_rows, dataset_rows, overall_rows = aggregate(rows)

    write_csv(OUTPUT_DIR / "per_dataset_scores.csv", rows, ["model", "epoch", "ckpt", "task", "dataset", "metric", "raw_value", "score"])
    write_csv(OUTPUT_DIR / "task_mean_scores.csv", task_rows, ["model", "task", "epoch", "ckpt", "score"])
    write_csv(OUTPUT_DIR / "epoch15_dataset_delta.csv", dataset_rows, ["task", "dataset", "metric", "vitb_epoch15", "vitl_epoch15", "delta_l_minus_b"])
    write_csv(OUTPUT_DIR / "epoch15_task_summary.csv", overall_rows, ["task", "metric", "vitb_epoch15_mean", "vitl_epoch15_mean", "delta_l_minus_b"])

    make_task_mean_plot(task_rows, OUTPUT_DIR / "task_mean_score_vs_epoch.png")
    make_task_delta_plot(task_rows, OUTPUT_DIR / "task_delta_vitl_minus_vitb.png")
    make_dataset_delta_plot(dataset_rows, OUTPUT_DIR / "dataset_delta_epoch15.png")
    write_markdown(OUTPUT_DIR / "summary.md", dataset_rows, overall_rows, ckpts)
    print(f"Wrote comparison outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
