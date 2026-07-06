#!/usr/bin/env python
"""Summarize Cellpose-trained decoder benchmark-test results against CPSAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CPSAM = {
    "pannuke": ["outputs/instance_seg/full_eval/cpsam_zs_full/results.json"],
    "tissuenet": ["outputs/instance_seg/test_eval/tissuenet/cpsam/results.json"],
    "conic": ["outputs/instance_seg/test_eval/conic/cpsam/results.json"],
    "bbbc038": ["outputs/instance_seg/test_eval/bbbc038/cpsam/results.json"],
    "livecell": ["outputs/instance_seg/test_eval/livecell/cpsam/results.json"],
    "monuseg": ["outputs/instance_seg/monuseg_test/cpsam_test/results.json"],
}


def _metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("test") or data.get("metrics")


def _fmt(metric: dict | None) -> str:
    if not metric:
        return "NA"
    return f"{metric.get('AJI', float('nan')):.3f}/{metric.get('bPQ', float('nan')):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/instance_seg/cellpose_decoder_finetune_topckpts_local8_benchmark")
    parser.add_argument("--metric", default="bPQ", choices=["bPQ", "AJI", "AP50", "AP", "AP75"])
    args = parser.parse_args()

    root = Path(args.root)
    datasets = sorted(CPSAM)
    print("| dataset | best cellpose-trained decoder | AJI/bPQ | CPSAM AJI/bPQ | delta bPQ |")
    print("|---|---|---:|---:|---:|")
    decoder_avgs = []
    cpsam_avgs = []
    for dataset in datasets:
        rows = []
        for p in root.glob(f"*/{dataset}/results.json"):
            m = _metrics(p)
            if m and args.metric in m:
                rows.append((m[args.metric], p.parts[-3], m))
        rows.sort(reverse=True, key=lambda x: x[0])
        best = rows[0] if rows else None
        cps = None
        for raw in CPSAM[dataset]:
            cps = _metrics(Path(raw))
            if cps:
                break
        delta = "NA"
        if best and cps and "bPQ" in best[2] and "bPQ" in cps:
            delta = f"{best[2]['bPQ'] - cps['bPQ']:+.3f}"
            decoder_avgs.append(best[2]["bPQ"])
            cpsam_avgs.append(cps["bPQ"])
        print(
            f"| {dataset} | {best[1] if best else 'NA'} | "
            f"{_fmt(best[2] if best else None)} | {_fmt(cps)} | {delta} |"
        )
    if decoder_avgs and cpsam_avgs:
        print(
            f"\nAvg bPQ: decoder={sum(decoder_avgs)/len(decoder_avgs):.3f}, "
            f"CPSAM={sum(cpsam_avgs)/len(cpsam_avgs):.3f}, "
            f"delta={sum(decoder_avgs)/len(decoder_avgs) - sum(cpsam_avgs)/len(cpsam_avgs):+.3f}"
        )


if __name__ == "__main__":
    main()
