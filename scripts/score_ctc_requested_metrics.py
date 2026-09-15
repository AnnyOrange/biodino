#!/usr/bin/env python3
"""Run the production requested-only CTC scorer on an existing result directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dinov3.eval.bio_tracking.ctc_metrics_requested import score_requested_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics, diagnostics = score_requested_metrics(
        args.result_dir,
        args.ground_truth_dir,
        args.vendor,
        threads=args.threads,
    )
    payload = {"metrics": metrics, "diagnostics": diagnostics}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
