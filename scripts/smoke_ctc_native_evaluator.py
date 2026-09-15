#!/usr/bin/env python3
"""Oracle-format smoke test for the pinned native CTC evaluator."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "outputs/02_eval_runtime/py-ctcmetrics"
COMMIT = "59481c48a62d4376fe34bed3e3606b4ec4d60972"
sys.path.insert(0, str(VENDOR))
from ctc_metrics.scripts.evaluate import evaluate_sequence  # noqa: E402


def native(value):
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ctc-native-smoke-") as temp:
        root = Path(temp); gt = root / "01_GT"; res = root / "01_RES"
        (gt / "TRA").mkdir(parents=True); (gt / "SEG").mkdir(); res.mkdir()
        for frame, shift in enumerate((0, 1)):
            mask = np.zeros((32, 32), dtype=np.uint16)
            mask[8 + shift:16 + shift, 10:18] = 1
            tifffile.imwrite(gt / "TRA" / f"man_track{frame:03d}.tif", mask)
            tifffile.imwrite(gt / "SEG" / f"man_seg{frame:03d}.tif", mask)
            tifffile.imwrite(res / f"mask{frame:03d}.tif", mask)
        (gt / "TRA" / "man_track.txt").write_text("1 0 1 0\n")
        (res / "res_track.txt").write_text("1 0 1 0\n")
        metrics = native(evaluate_sequence(str(res), str(gt), metrics=["Valid", "DET", "SEG", "TRA"], threads=1))
    passed = metrics.get("Valid") == 1 and all(abs(float(metrics[key]) - 1.0) < 1e-12 for key in ("DET", "SEG", "TRA"))
    report = {"status": "PASS" if passed else "FAIL", "evaluator_commit": COMMIT,
              "test": "two-frame oracle CTC-format sequence", "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
