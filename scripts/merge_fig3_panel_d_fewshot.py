#!/usr/bin/env python3
"""Merge independently written Fig. 3 Panel D probe results safely."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/04_figures/fig3_representation_20260812"
KEY_COLUMNS = ("model_key", "label_fraction", "seed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--source", type=Path, action="append", required=True, help="A staging raw-result CSV.")
    return parser.parse_args()


def _success_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS) | {"score", "error"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)}")
    frame = frame[frame["error"].fillna("").eq("")].copy()
    if frame.empty:
        raise ValueError(f"{path} has no successful result rows")
    return frame


def _key(row: pd.Series) -> tuple[str, float, int]:
    return str(row["model_key"]), float(row["label_fraction"]), int(row["seed"])


def main() -> None:
    args = parse_args()
    raw_path = args.output_dir / "panel_d_fewshot_curves.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    merged = pd.read_csv(raw_path)
    existing = {_key(row): row for _, row in merged.iterrows() if str(row.get("error", "") or "") == ""}
    additions: list[pd.Series] = []

    for source in args.source:
        for _, row in _success_rows(source).iterrows():
            key = _key(row)
            old = existing.get(key)
            if old is not None:
                old_score = float(old["score"])
                new_score = float(row["score"])
                if not np.isclose(old_score, new_score, rtol=0.0, atol=1e-12):
                    raise ValueError(f"Conflicting result for {key}: {old_score} vs {new_score} ({source})")
                continue
            additions.append(row)
            existing[key] = row

    if additions:
        merged = pd.concat([merged, pd.DataFrame(additions)], ignore_index=True)
    merged.sort_values(list(KEY_COLUMNS), inplace=True)
    merged.to_csv(raw_path, index=False)

    successful = merged[merged["error"].fillna("").eq("")].copy()
    summary = (
        successful.groupby(
            ["dataset", "split", "metric", "model_key", "model", "model_family", "label_fraction", "label_percent"],
            as_index=False,
        )
        .agg(
            score=("score", "mean"),
            score_std=("score", "std"),
            n_seeds=("seed", "nunique"),
            n_train=("n_train", "mean"),
            n_test=("n_test", "first"),
            accuracy=("accuracy", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
        )
        .sort_values(["model", "label_fraction"])
    )
    summary["score_std"] = summary["score_std"].fillna(0.0)
    summary["n_train"] = summary["n_train"].round().astype(int)
    summary_path = args.output_dir / "panel_d_fewshot_summary.csv"
    summary.to_csv(summary_path, index=False)

    metadata_path = args.output_dir / "panel_d_fewshot_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update(
        {
            "raw_results": str(raw_path),
            "summary": str(summary_path),
            "models_completed": sorted(successful["model_key"].unique().tolist()),
            "merge_sources": [str(source) for source in args.source],
            "rows_added_last_merge": len(additions),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(
        f"[panel-d-merge] added={len(additions)} total_successes={len(successful)} "
        f"wrote {raw_path} and {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
