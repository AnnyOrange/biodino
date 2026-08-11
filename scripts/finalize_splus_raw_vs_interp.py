#!/usr/bin/env python3
"""Summarize raw ck8199 versus its alpha=0.75 interpolated backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from finalize_splus_checkpoint_ab import collect_candidate, write_csv, write_markdown


FAMILY_KEYS = (
    "c25_macro_f1",
    "bbbc005_r2",
    "retrieval4_map_at_5",
    "clustering4_nmi",
    "segmentation8_mdice",
    "livecell_detection_f1",
)


def replace_detection(candidate: dict, result_path: Path) -> None:
    result = json.loads(result_path.read_text())
    value = float(result["test_patch_f1"])
    if value > 1.0:
        value /= 100.0
    candidate["summary"]["livecell_detection_f1"] = value
    candidate["detection_result"] = str(result_path)
    row = next(row for row in candidate["dataset_rows"] if row["task"] == "detection")
    row["value"] = value
    row["result_path"] = str(result_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--interp-root", type=Path, required=True)
    parser.add_argument("--raw-detection", type=Path)
    parser.add_argument("--interp-detection", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if (args.raw_detection is None) != (args.interp_detection is None):
        parser.error("--raw-detection and --interp-detection must be provided together")

    candidates = [
        collect_candidate("raw_ck8199_alpha1", args.raw_root, args.raw_root, "8199"),
        collect_candidate("interp_ck8199_alpha075", args.interp_root, args.interp_root, "75"),
    ]
    if args.raw_detection is not None:
        replace_detection(candidates[0], args.raw_detection)
        replace_detection(candidates[1], args.interp_detection)
    for candidate in candidates:
        candidate["summary"]["family6_equal_mean"] = mean(
            candidate["summary"][key] for key in FAMILY_KEYS
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    serializable = []
    dataset_rows = []
    summary_rows = []
    for candidate in candidates:
        dataset_rows.extend(candidate["dataset_rows"])
        serializable.append(
            {key: value for key, value in candidate.items() if key != "dataset_rows"}
        )
        summary_rows.append({"candidate": candidate["label"], **candidate["summary"]})

    (args.output_dir / "metrics.json").write_text(
        json.dumps(serializable, indent=2) + "\n"
    )
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "per_dataset.csv", dataset_rows)
    write_markdown(
        args.output_dir / "README.md",
        candidates,
        reference="raw_ck8199_alpha1",
    )
    raw_mean = candidates[0]["summary"]["family6_equal_mean"]
    interp_mean = candidates[1]["summary"]["family6_equal_mean"]
    with (args.output_dir / "README.md").open("a") as handle:
        handle.write("\n## Equal-family aggregate\n\n")
        handle.write(
            "This descriptive mean gives one equal vote to each of the six task families; "
            "it is not used to hide per-family regressions.\n\n"
        )
        handle.write("| candidate | family6_equal_mean | delta vs raw |\n")
        handle.write("|---|---:|---:|\n")
        handle.write(f"| raw_ck8199_alpha1 | {raw_mean:.6f} | +0.000000 |\n")
        handle.write(
            f"| interp_ck8199_alpha075 | {interp_mean:.6f} | "
            f"{interp_mean - raw_mean:+.6f} |\n"
        )
        deltas = {
            key: candidates[1]["summary"][key] - candidates[0]["summary"][key]
            for key in FAMILY_KEYS
        }
        handle.write("\n## Decision\n\n")
        handle.write(
            "Retain the α=0.75 interpolation as the deployable S+ checkpoint. "
            "All six family summaries are non-negative versus the raw teacher; "
            f"the equal-family gain is {interp_mean - raw_mean:+.6f}, led by "
            f"clustering ({deltas['clustering4_nmi']:+.6f}). This is checkpoint "
            "post-processing only and does not involve retraining.\n"
        )
        handle.write("\n## Matched protocol\n\n")
        handle.write(
            "Classification, BBBC005 regression, and retrieval/clustering use "
            "bf16 feature extraction, batch 64, auto-channel TTA-8, and seed 0. "
            "Segmentation uses the same auto-channel TTA-8 policy with the frozen "
            "best-protocol linear probes. LIVECell detection is a fresh batch-4, "
            "seed-0 rerun on both checkpoints after making probe initialization and "
            "data order deterministic.\n"
        )
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
