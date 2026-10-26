#!/usr/bin/env python3
"""Verify FM14 MoNuSeg cached masks against the locked 30/7/14 sample order."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from dinov3.eval.bio_segmentation.datasets.monuseg import MoNuSegDataset


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/mnt/huawei_deepcad/benchmark/segmentation/monuseg/extracted")
LOCK = ROOT / "outputs/02_eval_inputs/monuseg37_v4_20261012/manifest.json"
AUDIT = ROOT / "outputs/02_eval_runs/fm14_segmentation_five_seed_20261019/audit.json"
OUT = ROOT / "outputs/00_reports/fm14_monuseg_identity_20261021.json"
PARTS = ("train", "val", "test")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_masks(manifest: dict) -> dict[str, list[np.ndarray]]:
    masks = {}
    for part in PARTS:
        rows = [row for row in manifest["samples"] if row["split"] == part]
        dataset = MoNuSegDataset(
            [str(DATA / row["image"]) for row in rows],
            [str(DATA / row["xml"]) for row in rows],
            size=(768, 768), resize_mode="pad", augment=False, do_normalize=False,
        )
        masks[part] = [dataset[index][1].numpy().astype(np.int16) for index in range(len(rows))]
    return masks


def cache_masks(path: Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return [row for chunk in range(int(archive["num_chunks"]))
                for row in archive[f"sem_masks_{chunk:04d}"]]


def main() -> None:
    locked = json.loads(LOCK.read_text())
    if locked["counts"] != {"train": 30, "val": 7, "test": 14}:
        raise RuntimeError("Unexpected locked MoNuSeg counts")
    masks = expected_masks(locked)
    cells = [cell for cell in json.loads(AUDIT.read_text())["cells"]
             if cell["dataset"] == "monuseg"]
    if len(cells) != 14 or len({cell["model"] for cell in cells}) != 14:
        raise RuntimeError("Expected exactly 14 distinct FM14 MoNuSeg cache records")

    results = []
    for cell in cells:
        parts = {}
        for part in PARTS:
            path = Path(cell["caches"][part])
            actual = cache_masks(path)
            mismatches = [index for index, (found, target) in enumerate(zip(actual, masks[part]))
                          if not np.array_equal(found, target)]
            if len(actual) != len(masks[part]):
                mismatches.extend(range(min(len(actual), len(masks[part])),
                                        max(len(actual), len(masks[part]))))
            parts[part] = {"cache": str(path), "cache_sha256": sha256(path),
                           "samples": len(actual), "mismatched_indices": mismatches}
        results.append({"model": cell["model"], "parts": parts,
                        "all_match": all(not part["mismatched_indices"] for part in parts.values())})
        print(cell["model"], results[-1]["all_match"], flush=True)

    payload = {"status": "PASS" if all(row["all_match"] for row in results) else "FAIL",
               "locked_manifest": str(LOCK), "locked_manifest_sha256": sha256(LOCK),
               "cache_inventory": str(AUDIT), "cache_inventory_sha256": sha256(AUDIT),
               "basis": "All cached semantic masks match the locked ordered sample IDs after fixed 768-pad rasterization",
               "models": results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(payload["status"], str(OUT), flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
