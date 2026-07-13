#!/usr/bin/env python3
"""Plot uint16/lossless-uint16/uint8 compression and downstream summaries."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "00_reports" / "compression_uint16_lossless_uint8_20260707"

RAW_UINT16 = Path("/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle")
LOSSLESS_UINT16 = Path("/mnt/huawei_deepcad/compression/deflate_lossless")
UINT8_DEFLATE = Path("/mnt/huawei_deepcad/compression/uint8_deflate")

CLASS_AVG = ROOT / "outputs/03_comparisons/uint8_vs_16bit_sklearn_fairB_20260602/classification_avg_best_scores.csv"
DENSE_DET = ROOT / "outputs/03_comparisons/uint8_vs_16bit_dense_fair_20260603/merged/dense_detection_best_scores.csv"
DENSE_SEG_AVG = ROOT / "outputs/03_comparisons/uint8_vs_16bit_dense_fair_20260603/merged/dense_segmentation_avg_best_scores.csv"
RETR_AVG = ROOT / "outputs/03_comparisons/uint8_vs_16bit_retrieval_fair_20260605/merged/retrieval_avg_best_scores.csv"

ONE_TB_BUILD_LOG = Path("/mnt/huawei_deepcad/compression/build.log")
TEN_TB_BUILD_LOG = ROOT / "outputs/06_data_prep_transfer/compression_10tb_lossless_uint16_20260608/build_lossless.log"
TEN_TB_STATS_SUMMARY = (
    ROOT
    / "outputs/06_data_prep_transfer/compression_10tb_lossless_uint16_20260608/resumable_packwds_stats_3ch_20260609_114220/summary.txt"
)

COLORS = {
    "raw": "#204B57",
    "lossless": "#7BAF9E",
    "uint8": "#E76F51",
    "positive": "#2A9D8F",
    "negative": "#C8553D",
    "muted": "#64748B",
}


def tar_stats(path: Path) -> tuple[int, int]:
    files = sorted(path.glob("*.tar")) if path.exists() else []
    return len(files), sum(p.stat().st_size for p in files)


def fmt_bytes_gb(n: int | float) -> str:
    if not math.isfinite(float(n)):
        return "NA"
    return f"{n / 1e9:.2f}"


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def first_row(rows: list[dict[str, str]], **matches: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(k) == v for k, v in matches.items()):
            return row
    raise KeyError(f"No row in CSV matched {matches}")


def parse_payload_summary(log_path: Path) -> dict[str, str]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="replace")
    out: dict[str, str] = {}
    patterns = {
        "source_tif_payload_gb": r"source tif payload\s*:\s*([0-9.]+) GB",
        "deflate_lossless_gb": r"deflate \(lossless\)\s*:\s*([0-9.]+) GB",
        "deflate_lossless_ratio": r"deflate \(lossless\)\s*:\s*[0-9.]+ GB \(([0-9.]+)%\)",
        "uint8_deflate_gb": r"uint8 \+ deflate\s*:\s*([0-9.]+) GB",
        "uint8_deflate_ratio": r"uint8 \+ deflate\s*:\s*[0-9.]+ GB \(([0-9.]+)%\)",
        "elapsed_sec": r"elapsed:\s*([0-9]+)s",
        "shards_ok": r"shards:\s*ok=([0-9]+)",
        "decode_fail": r"decode_fail=([0-9]+)",
        "verify_fail": r"verify_fail=([0-9]+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            out[key] = m.group(1)
    return out


def parse_stats_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(errors="replace")
    out: dict[str, str] = {}
    for key in ["shards_total", "shards_done", "shards_missing", "samples", "bad"]:
        m = re.search(rf"^{key}=([0-9]+)", text, re.MULTILINE)
        if m:
            out[key] = m.group(1)
    return out


def storage_rows() -> list[dict]:
    specs = [
        ("Raw uint16 TIFF", RAW_UINT16, "raw"),
        ("Lossless uint16 TIFF DEFLATE", LOSSLESS_UINT16, "lossless"),
        ("uint8 + TIFF DEFLATE", UINT8_DEFLATE, "uint8"),
    ]
    raw_shards, raw_bytes = tar_stats(RAW_UINT16)
    rows: list[dict] = []
    for name, path, kind in specs:
        shards, size = tar_stats(path)
        ratio = size / raw_bytes if raw_bytes else math.nan
        factor = raw_bytes / size if size else math.nan
        rows.append(
            {
                "version": name,
                "kind": kind,
                "path": str(path),
                "shards": shards,
                "bytes": size,
                "gb": size / 1e9 if size else math.nan,
                "gib": size / 2**30 if size else math.nan,
                "ratio_vs_raw": ratio,
                "compression_factor_vs_raw": factor,
                "raw_shards": raw_shards,
            }
        )
    return rows


def load_downstream_rows() -> list[dict]:
    rows: list[dict] = []

    cls = read_csv_dict(CLASS_AVG)
    for model in ["B", "L"]:
        for metric, label in [
            ("accuracy", "Cls Acc"),
            ("balanced_accuracy", "Cls BalAcc"),
        ]:
            row = first_row(cls, model_size=model, metric=metric)
            rows.append(make_perf_row("classification", model, label, float(row["16-bit"]), float(row["uint8"])))

    det = read_csv_dict(DENSE_DET)
    for model in ["B", "L"]:
        v16 = float(first_row(det, model_size=model, metric="patch_f1", precision="16-bit")["best_score"])
        v8 = float(first_row(det, model_size=model, metric="patch_f1", precision="uint8")["best_score"])
        rows.append(make_perf_row("detection", model, "LIVECell F1", v16, v8))

    seg = read_csv_dict(DENSE_SEG_AVG)
    for model in ["B", "L"]:
        row = first_row(seg, model_size=model, metric="mIoU")
        rows.append(make_perf_row("segmentation", model, "Seg mIoU", float(row["16-bit"]), float(row["uint8"])))

    retr = read_csv_dict(RETR_AVG)
    for model in ["B", "L"]:
        for metric, label in [
            ("recall_at_1", "Ret R@1"),
            ("map_at_10", "Ret mAP@10"),
            ("nmi", "Cluster NMI"),
        ]:
            row = first_row(retr, model_size=model, metric=metric)
            rows.append(make_perf_row("retrieval/clustering", model, label, float(row["16-bit"]) * 100.0, float(row["uint8"]) * 100.0))

    return rows


def make_perf_row(task: str, model: str, metric: str, v16: float, v8: float) -> dict:
    # Lossless uint16 is plotted equal to raw uint16 because pixel values are identical.
    return {
        "task": task,
        "model_size": model,
        "model": f"ViT-{model}",
        "metric": metric,
        "uint16": float(v16),
        "lossless_uint16": float(v16),
        "uint8": float(v8),
        "uint8_minus_uint16": float(v8) - float(v16),
        "unit": "score_points",
        "lossless_note": "same pixel values as uint16; plotted equal, not an independent retrain",
    }


def plot_storage(rows: list[dict], path: Path) -> None:
    labels = ["Raw\nuint16", "Lossless\nuint16", "uint8"]
    sizes = [r["gb"] for r in rows]
    ratios = [r["ratio_vs_raw"] * 100.0 for r in rows]
    factors = [r["compression_factor_vs_raw"] for r in rows]
    colors = [COLORS["raw"], COLORS["lossless"], COLORS["uint8"]]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    for ax in (ax0, ax1):
        ax.set_facecolor("white")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    x = np.arange(len(labels))
    bars = ax0.bar(x, sizes, color=colors, width=0.62)
    ax0.set_title("On-disk size for the same 1TB shard set", fontsize=12, fontweight="bold")
    ax0.set_ylabel("GB, sum of .tar files")
    ax0.set_xticks(x, labels)
    ax0.set_ylim(0, max(sizes) * 1.18)
    for bar, size, factor in zip(bars, sizes, factors):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sizes) * 0.025,
            f"{size:.0f} GB\n{factor:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bars = ax1.bar(x, ratios, color=colors, width=0.62)
    ax1.axhline(100, color="#334155", lw=1, alpha=0.5)
    ax1.set_title("Size retained vs raw uint16", fontsize=12, fontweight="bold")
    ax1.set_ylabel("% of raw uint16")
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0, 112)
    for bar, ratio in zip(bars, ratios):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{ratio:.1f}%", ha="center", fontsize=9)

    fig.suptitle("Compression: uint16 vs lossless uint16 vs uint8", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_downstream_bars(rows: list[dict], path: Path) -> None:
    labels = [f"{r['model']} {r['metric']}" for r in rows]
    x = np.arange(len(rows))
    width = 0.26
    v16 = np.array([r["uint16"] for r in rows])
    vl = np.array([r["lossless_uint16"] for r in rows])
    v8 = np.array([r["uint8"] for r in rows])

    fig, ax = plt.subplots(figsize=(15, 6.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar(x - width, v16, width, color=COLORS["raw"], label="uint16")
    ax.bar(x, vl, width, color=COLORS["lossless"], label="lossless uint16", hatch="//", edgecolor="#2F5D50", linewidth=0.7)
    ax.bar(x + width, v8, width, color=COLORS["uint8"], label="uint8")
    ax.set_title("Downstream scores: uint16 / lossless uint16 / uint8", fontsize=15, fontweight="bold", pad=42)
    ax.set_ylabel("Score points; retrieval and clustering metrics are x100")
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylim(45, 101)
    ax.legend(ncols=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.045))
    ax.text(
        0.01,
        0.03,
        "Lossless uint16 bars are plotted equal to uint16 because TIFF DEFLATE keeps identical pixel values; no separate lossless retrain is implied.",
        transform=ax.transAxes,
        fontsize=9,
        color=COLORS["muted"],
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_downstream_delta(rows: list[dict], path: Path) -> None:
    labels = [f"{r['model']} {r['metric']}" for r in rows]
    vals = np.array([r["uint8_minus_uint16"] for r in rows])
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in vals]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.axvline(0, color="#111827", lw=1.2)
    ax.barh(y, vals, color=colors, height=0.68)
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("uint8 - uint16, score points")
    ax.set_title("Fair comparison deltas: uint8 is essentially tied with uint16", fontsize=14, fontweight="bold")
    pad = max(abs(vals).max() * 0.16, 0.08)
    ax.set_xlim(vals.min() - pad, vals.max() + pad)
    for yi, v in zip(y, vals):
        ha = "left" if v >= 0 else "right"
        offset = 0.025 if v >= 0 else -0.025
        ax.text(v + offset, yi, f"{v:+.2f}", va="center", ha=ha, fontsize=8.5)

    legend = [
        Patch(facecolor=COLORS["positive"], label="uint8 higher"),
        Patch(facecolor=COLORS["negative"], label="uint16 higher"),
    ]
    ax.legend(handles=legend, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def write_readme(storage: list[dict], downstream: list[dict]) -> None:
    one_tb = parse_payload_summary(ONE_TB_BUILD_LOG)
    ten_tb = parse_payload_summary(TEN_TB_BUILD_LOG)
    ten_stats = parse_stats_summary(TEN_TB_STATS_SUMMARY)

    storage_table = []
    for row in storage:
        storage_table.append(
            [
                row["version"],
                str(row["shards"]),
                f"{row['gb']:.2f}",
                f"{row['ratio_vs_raw'] * 100:.1f}%",
                f"{row['compression_factor_vs_raw']:.2f}x",
                f"`{row['path']}`",
            ]
        )

    downstream_table = []
    for row in downstream:
        downstream_table.append(
            [
                row["model"],
                row["metric"],
                f"{row['uint16']:.2f}",
                f"{row['lossless_uint16']:.2f}",
                f"{row['uint8']:.2f}",
                f"{row['uint8_minus_uint16']:+.2f}",
            ]
        )

    lines = [
        "# Compression: uint16 / lossless uint16 / uint8",
        "",
        "This report plots the storage and downstream fair-comparison results for the compression experiment.",
        "",
        "Important interpretation: lossless uint16 TIFF DEFLATE keeps exactly the same uint16 pixel values as raw uint16. In the downstream score plot, the lossless uint16 bar is therefore shown equal to uint16 as an expected pixel-equivalent reference; it is not a separate retraining run.",
        "",
        "## Files",
        "",
        "- `compression_size_summary.png`",
        "- `uint16_lossless_uint8_downstream_bars.png`",
        "- `uint8_downstream_delta_summary.png`",
        "- `storage_summary.csv`",
        "- `downstream_summary.csv`",
        "",
        "## 1TB on-disk size",
        "",
        md_table(["version", "shards", "size GB", "ratio vs raw", "factor", "path"], storage_table),
        "",
    ]

    if one_tb:
        lines += [
            "Payload summary from `/mnt/huawei_deepcad/compression/build.log`: "
            f"source TIFF payload {one_tb.get('source_tif_payload_gb', 'NA')} GB; "
            f"lossless {one_tb.get('deflate_lossless_gb', 'NA')} GB "
            f"({one_tb.get('deflate_lossless_ratio', 'NA')}%); "
            f"uint8+DEFLATE {one_tb.get('uint8_deflate_gb', 'NA')} GB "
            f"({one_tb.get('uint8_deflate_ratio', 'NA')}%).",
            "",
        ]

    lines += [
        "## Downstream representative scores",
        "",
        md_table(["model", "metric", "uint16", "lossless uint16", "uint8", "uint8 - uint16"], downstream_table),
        "",
    ]

    if ten_tb:
        lines += [
            "## Full 10TB lossless uint16 run",
            "",
            f"- Build log: source TIFF payload {ten_tb.get('source_tif_payload_gb', 'NA')} GB -> lossless {ten_tb.get('deflate_lossless_gb', 'NA')} GB ({ten_tb.get('deflate_lossless_ratio', 'NA')}%), factor {float(ten_tb['source_tif_payload_gb']) / float(ten_tb['deflate_lossless_gb']):.2f}x.",
            f"- Shards ok: {ten_tb.get('shards_ok', ten_stats.get('shards_done', 'NA'))}; samples: {ten_stats.get('samples', 'NA')}; decode_fail: {ten_tb.get('decode_fail', 'NA')}; verify_fail: {ten_tb.get('verify_fail', 'NA')}.",
            "",
        ]

    lines += [
        "## Source CSVs",
        "",
        f"- `{CLASS_AVG.relative_to(ROOT)}`",
        f"- `{DENSE_DET.relative_to(ROOT)}`",
        f"- `{DENSE_SEG_AVG.relative_to(ROOT)}`",
        f"- `{RETR_AVG.relative_to(ROOT)}`",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    storage = storage_rows()
    downstream = load_downstream_rows()

    write_csv(
        OUT / "storage_summary.csv",
        storage,
        ["version", "kind", "path", "shards", "bytes", "gb", "gib", "ratio_vs_raw", "compression_factor_vs_raw", "raw_shards"],
    )
    write_csv(
        OUT / "downstream_summary.csv",
        downstream,
        [
            "task",
            "model_size",
            "model",
            "metric",
            "uint16",
            "lossless_uint16",
            "uint8",
            "uint8_minus_uint16",
            "unit",
            "lossless_note",
        ],
    )

    plot_storage(storage, OUT / "compression_size_summary.png")
    plot_downstream_bars(downstream, OUT / "uint16_lossless_uint8_downstream_bars.png")
    plot_downstream_delta(downstream, OUT / "uint8_downstream_delta_summary.png")
    write_readme(storage, downstream)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
