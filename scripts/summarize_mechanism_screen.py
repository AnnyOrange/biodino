#!/usr/bin/env python3
"""Aggregate partial or complete mechanism-screen evaluations reproducibly."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _finite_float(value: str | None) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def collect(root: Path) -> dict[str, object]:
    by_variant: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for summary_path in root.glob("*/bio_classification/*/*/summary.csv"):
        variant = summary_path.relative_to(root).parts[0]
        with summary_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                score = _finite_float(row.get("balanced_accuracy"))
                if score is not None:
                    by_variant[variant]["classification"].append(
                        {"dataset": row["dataset"], "balanced_accuracy": score}
                    )
    # Segmentation launchers append their probe policy to ``bio_eval`` (for
    # example ``bio_eval__best__...``), so accept both the bare and tagged
    # directory forms.
    for result_path in root.glob("*/bio_segmentation/bio_eval*/*/*/results.json"):
        variant = result_path.relative_to(root).parts[0]
        result = json.loads(result_path.read_text())
        test_metrics = result.get("test", {})
        score = test_metrics.get("mIoU")
        if isinstance(score, (int, float)):
            by_variant[variant]["segmentation"].append(
                {"dataset": result_path.parent.parent.name, "mIoU": float(score)}
            )

    summary: dict[str, object] = {}
    for variant, groups in sorted(by_variant.items()):
        cls = groups["classification"]
        seg = groups["segmentation"]
        summary[variant] = {
            "classification": cls,
            "classification_mean_balanced_accuracy": _mean(
                [item["balanced_accuracy"] for item in cls]
            ),
            "segmentation": seg,
            "segmentation_mean_mIoU": _mean([item["mIoU"] for item in seg]),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = collect(args.eval_root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text)
        print(args.output)


if __name__ == "__main__":
    main()
